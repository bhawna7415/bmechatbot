import os
from pathlib import Path
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, render_template, request
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI as OpenAIClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from typing import TypedDict, List, Annotated
from typing_extensions import TypedDict
import re

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

client = OpenAIClient()

project_root = os.path.dirname(os.path.realpath(__file__))
static_path = os.path.join(project_root, 'app/static')
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

model_name = 'text-embedding-3-small'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

index = pc.Index(name="bmebot")

text_field = "text"
vectorstore = PineconeVectorStore(
    index=index, 
    embedding=embed
)


llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.2,
    max_tokens=500
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

def messages_reducer(left: List, right: List) -> List:
    """Combine messages, keeping a reasonable history length."""
    combined = left + right
    return combined[-20:] if len(combined) > 20 else combined

class ConversationState(TypedDict):
    messages: Annotated[List[AIMessage | HumanMessage], messages_reducer]

_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are a helpful BatchMaster ERP assistant that provides clear and accurate responses using data stored in a vector database.
Your job is to understand user queries related to BatchMaster ERP and retrieve the most relevant information.

Context from vector database:
{context}

Instructions for the conversation:
1. The main goal is to understand the user's needs and provide relevant assistance.
2. Carefully analyze the user's query and the provided context to determine the appropriate response.
3. If the query is unclear, ask the user for clarification.
4. Provide step-by-step instructions according to data stored in vector database.
5. Respond in a clear and straightforward manner without using special formatting.
6. If the answer is not present in the context provided or the context is empty, respond with: "I don't have information about that in my knowledge base. Could you please ask a question related to BatchMaster ERP?"
7. If the user asks a query that is not related to BatchMaster ERP, reply with 'I am here to assist with any queries related to Batchmaster ERP. Feel free to ask if you need any help!'.
8. Understand that users can use different terminology like "bill of material" instead of "BOM" that refer to the same concept.

Question: {question}
Answer:"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    if not docs:
        return "No relevant information found in the knowledge base."
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def create_conversation_workflow():
    workflow = StateGraph(ConversationState)

    def process_query(state: ConversationState):
        try:
            user_input = state['messages'][-1].content if state['messages'] else ""
            chat_history = state['messages'][:-1]

            condense_inputs = {
                "question": user_input,
                "chat_history": chat_history
            }
            
            standalone_question = CONDENSE_QUESTION_PROMPT.format_prompt(**{
                "question": user_input,
                "chat_history": get_buffer_string(chat_history)
            }).to_string()
            
            standalone_question = llm.invoke(standalone_question).content
            
            docs = retriever.invoke(standalone_question)
            context = _combine_documents(docs)
            
            final_inputs = {
                "context": context,
                "question": standalone_question
            }
            
            result = ANSWER_PROMPT.format_prompt(**final_inputs).to_string()
            result = llm.invoke(result)

            formatted_response = format_response_to_html(result.content)

            return {"messages": [AIMessage(content=formatted_response)]}
    
        except Exception as e:
            print(f"Error in process_query: {e}")
            return {"messages": [AIMessage(content="Unfortunately, an error occurred while processing your query. Please try again.")]}


    workflow.add_node("process_query", process_query)
    workflow.add_edge(START, "process_query")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

conversation_app = create_conversation_workflow()

@app.route('/query', methods=['POST', 'GET'])
def query():
    if request.method == 'GET':
        user_input = str(request.args.get('text'))

        try:
            thread_id = "default_conversation"

            result = conversation_app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": thread_id}}
            )

            response = result['messages'][-1].content
            return response

        except Exception as e:
            print(f"Error in query endpoint: {e}")
            return "Unfortunately, information is not currently accessible due to a technical error."

def format_response_to_html(response_text):
    """Formats the OpenAI response with proper HTML tags while ensuring correct indentation for lists and bold headings."""

    response_text = response_text.replace("\n", "<br>")

    response_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response_text)  # **Heading**
    response_text = re.sub(r"###\s?(.*)", r"<strong>\1</strong>", response_text)  # ### Heading

    response_text = re.sub(r"\n- (.+)", r"\n<li>\1</li>", response_text)
    response_text = re.sub(r"(<li>.*?</li>)+", lambda m: f"<ul>{m.group(0)}</ul>", response_text)

    response_text = re.sub(r"\n\d+\.\s(.+)", r"\n<li>\1</li>", response_text)
    response_text = re.sub(r"(<li>.*?</li>)+", lambda m: f"<ol>{m.group(0)}</ol>", response_text)

    if not response_text.startswith(("<p>", "<ul>", "<ol>", "<strong>")):
        response_text = f"{response_text}"

    return response_text

if __name__ == '__main__':
    app.run(debug=True)

