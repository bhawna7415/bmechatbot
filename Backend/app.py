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
from flask import Flask, render_template, request, Response, jsonify, url_for, stream_with_context
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI as OpenAIClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from typing import TypedDict, List, Annotated
import re
import urllib.parse
from flask_cors import CORS
import base64, hashlib
import time 
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from flask import Flask, request, jsonify
from datetime import datetime, timedelta

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

client = OpenAIClient()

FRONTEND_URL=os.getenv("FRONTEND_URL")
project_root = os.path.dirname(os.path.realpath(__file__))
static_path = os.path.join(project_root, 'app/static')
app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

#Authentication
DECRYPTION_KEY = "BatchMasterAIHelp"

def openssl_key_iv(password: str, salt: bytes, key_size=32, iv_size=16):
    """Derive key and IV from password and salt (OpenSSL EVP_BytesToKey)."""
    d = d_i = b""
    while len(d) < key_size + iv_size:
        d_i = hashlib.md5(d_i + password.encode() + salt).digest()
        d += d_i
    return d[:key_size], d[key_size:key_size + iv_size]

def decrypt_openssl(enc_base64: str, password: str) -> str:
    """Decrypt AES-CBC encrypted token (OpenSSL compatible)."""
    enc = base64.b64decode(enc_base64)
    if not enc.startswith(b"Salted__"):
        raise ValueError("Invalid encrypted token format")

    salt = enc[8:16]
    key, iv = openssl_key_iv(password, salt, 32, 16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(enc[16:]), AES.block_size)
    return decrypted.decode("utf-8")


def verify_payload(payload: str) -> bool:
    try:
        decrypted_payload = decrypt_openssl(payload, DECRYPTION_KEY)
        print("Decrypted payload:", decrypted_payload)

        if "$" not in decrypted_payload:
            print("Invalid token format")
            return False

        base, payload_time_str = decrypted_payload.split("$", 1)
        if base != "AISecurity":
            print("Invalid token prefix")
            return False

        payload_time = datetime.strptime(payload_time_str.strip(), "%I:%M:%S %p")
        today = datetime.utcnow().date()
        payload_datetime = datetime.combine(today, payload_time.time())

        now = datetime.utcnow()

        if payload_datetime > now + timedelta(minutes=2):
            payload_datetime -= timedelta(days=1)

        expiry_time = payload_datetime + timedelta(minutes=4)
        print("Payload time:", payload_datetime)
        print("Expiry time:", expiry_time)
        print("Now:", now)

        return now <= expiry_time

    except Exception as e:
        print("Verification error:", e)
        return False

@app.route("/auth", methods=["POST"])
def authenticate():
    data = request.get_json()
    if not data or "payload" not in data:
        return jsonify({"status": "fail", "message": "Missing payload"})

    encrypted_payload = data["payload"]

    if verify_payload(encrypted_payload):
        return jsonify({"status": "success", "message": "Authentication passed"})
    else:
        return jsonify({"status": "fail", "message": "Authentication failed"})


embedding_model_name = os.getenv('EMBEDDING_MODEL')

if not embedding_model_name:
    raise ValueError("EMBEDDING_MODEL not set in .env")

embed = OpenAIEmbeddings(
    model=embedding_model_name,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

index = pc.Index(name="bmebot")

text_field = "text"
vectorstore = PineconeVectorStore(
    index=index, 
    embedding=embed
)

llm_model_name = os.getenv('LLM_MODEL')
if not llm_model_name:
    raise ValueError("LLM_MODEL not set in .env")

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name=llm_model_name,
    temperature=0,
    max_tokens=1000
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 12},
    return_source_documents=True
)

def messages_reducer(left: List, right: List) -> List:
    """Combine messages, keeping a reasonable history length."""
    combined = left + right
    return combined[-20:] if len(combined) > 20 else combined

class ConversationState(TypedDict):
    messages: Annotated[List[AIMessage | HumanMessage], messages_reducer]
    
_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question in English only.

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
4. Provide step-by-step instructions based strictly on the data stored in the vector database, and always include any preconditions, validations, exceptions, or warnings mentioned in the context.
5. Respond in a clear and straightforward manner without using special formatting.
6. If PDF link is found in the context related to user query, then provide a clickable link in response.
7. If the context contains video URLs, ask the user if they would like to receive video links. If the user replies "yes," then provide the video link as clickable URLs in response.
8. If the answer is not present in the context provided or the context is empty, respond with: "I don't have information about that in my knowledge base. Could you please ask a question related to BatchMaster ERP?"
9. If the user asks a query that is not related to BatchMaster ERP, reply with 'I am here to assist with any queries related to Batchmaster ERP. Feel free to ask if you need any help!'.
10. Understand that users can use different terminology like "bill of material" instead of "BOM" that refer to the same concept.

Always provide me response in english language only.
Question: {question}
Answer:"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

def format_document_with_metadata(doc):
    content = doc.page_content
    metadata = doc.metadata if hasattr(doc, 'metadata') else {}

    if "pdf_name" in metadata and metadata["pdf_name"]:
        pdf_name = metadata["pdf_name"]
        pdf_link = url_for('static', filename=f"pdfs/{pdf_name}.pdf", _external=True)
        content += f"\nPDF LINK: {pdf_link} "

    if "url" in metadata and metadata["url"]:
        content += f"\nVIDEO URL: {metadata['url']} "

    return content


def _combine_documents(docs, document_separator="\n\n"):
    if not docs:
        return "No relevant information found in the knowledge base."
    
    doc_strings = [format_document_with_metadata(doc) for doc in docs]
    return document_separator.join(doc_strings)


def format_response_to_html(response_text):
    import re
    response_text = response_text.replace("\n", "<br>")
    response_text = re.sub(r'\((https?://[^\s<>]+)\)', r'\1', response_text)
    markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    def replace_markdown_link(match):
        link_text = match.group(1).lower()
        url = match.group(2)
        
        if 'pdf' in url.lower() or link_text in ['here', 'click here']:
            return f'<a href="{url}" data-url="{url}" class="kbvideo" target="_blank">Click here</a>'
        else:
            return f'<a href="{url}" data-url="{url}" class="kbvideo" target="_blank">{match.group(1)}</a>'
    
    response_text = re.sub(markdown_pattern, replace_markdown_link, response_text)

    url_pattern = r'(?<!href=")(?<!data-url=")(https?://[^\s<.]+(?:\.[^\s<.]+)+(?:/[^\s<.])?(?:\.[^\s<.]+)?)'
    response_text = re.sub(url_pattern, r'<a href="\1" data-url="\1" class="kbvideo" target="_blank">Click here</a>', response_text)

    response_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", response_text)
    response_text = re.sub(r"###\s?(.*)", r"<strong>\1</strong>", response_text)
    
    response_text = re.sub(r"<br>-\s(.+)", r"<li>\1</li>", response_text)
    response_text = re.sub(r"((?:<li>.*?</li>)+)", r"<ul>\1</ul>", response_text, flags=re.DOTALL)
    response_text = re.sub(r'(^|<br>)\d+\.\s(.+?)(?=(<br>\d+\.|\Z))', r'\1<li>\2</li>', response_text)
    response_text = re.sub(r'((?:<li>.*?</li>)+)', r'<ul>\1</ul>', response_text, flags=re.DOTALL)
    
    return response_text


def _has_complete_url_pattern(text):
    """Check if text contains complete markdown link pattern [text](url)"""
    import re
    markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    return bool(re.search(markdown_pattern, text))

def _might_be_in_url(text):
    """Check if we might be in the middle of a URL"""
    if '[' in text and '](' not in text:
        return True
    if 'http' in text and not ('http://' in text or 'https://' in text):
        return True
    return False

def format_chunk_for_streaming(chunk_text):
    """Enhanced chunk formatting that properly handles URLs"""
    chunk_text = chunk_text.replace("\n", "<br>")
    
    import re
    markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    def replace_markdown_link(match):
        link_text = match.group(1).lower()
        url = match.group(2)
        
        if 'pdf' in url.lower() or link_text in ['here', 'click here']:
            return f'<a href="{url}" data-url="{url}" class="kbvideo" target="_blank">Click here</a>'
        else:
            return f'<a href="{url}" data-url="{url}" class="kbvideo" target="_blank">{match.group(1)}</a>'
    
    chunk_text = re.sub(markdown_pattern, replace_markdown_link, chunk_text)
    
    url_pattern = r'(?<!href=")(?<!data-url=")(https?://[^\s<.]+(?:\.[^\s<.]+)+(?:/[^\s<.])?(?:\.[^\s<.]+)?)'
    chunk_text = re.sub(url_pattern, r'<a href="\1" data-url="\1" class="kbvideo" target="_blank">Click here</a>', chunk_text)
    
    chunk_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', chunk_text)

    
    return chunk_text.strip()

def process_query(state: ConversationState):
    
    timings = {}  
    step_start = time.time()
    try:
        user_input = state['messages'][-1].content if state['messages'] else ""
        chat_history = state['messages'][:-1]
        timings["extract_input"] = time.time() - step_start

        step_start = time.time()
        standalone_question = CONDENSE_QUESTION_PROMPT.format_prompt(**{
            "question": user_input,
            "chat_history": get_buffer_string(chat_history)
        }).to_string()
        standalone_question = llm.invoke(standalone_question).content
        timings["condense_question"] = time.time() - step_start

        step_start = time.time()
        docs = retriever.invoke(standalone_question)
        context = _combine_documents(docs)
        timings["retriever"] = time.time() - step_start

        step_start = time.time()
        result = ANSWER_PROMPT.format_prompt(**{
            "context": context,
            "question": standalone_question
        }).to_string()
        result = llm.invoke(result)
        timings["generate_answer"] = time.time() - step_start

        step_start = time.time()
        formatted_response = format_response_to_html(result.content)
        timings["format_response"] = time.time() - step_start

        total_time = sum(timings.values())
        print("\n[LOG] process_query step timings (seconds):")
        for step, t in timings.items():
            print(f"   {step}: {t:.2f}")
        print(f"   TOTAL: {total_time:.2f}\n")

        return {"messages": [AIMessage(content=formatted_response)]}

    except Exception as e:
        print(f"Error in process_query: {e}")
        return {"messages": [AIMessage(content="Unfortunately, an error occurred while processing your query. Please try again.")]}

def process_query_streaming(state: ConversationState, stream_callback=None):
    timings = {}
    step_start = time.time()
    try:
        user_input = state['messages'][-1].content if state['messages'] else ""
        chat_history = state['messages'][:-1]
        timings["extract_input"] = time.time() - step_start

        step_start = time.time()
        standalone_question = CONDENSE_QUESTION_PROMPT.format_prompt(**{
            "question": user_input,
            "chat_history": get_buffer_string(chat_history)
        }).to_string()
        standalone_question = llm.invoke(standalone_question).content
        timings["condense_question"] = time.time() - step_start

        step_start = time.time()
        docs = retriever.invoke(standalone_question)
        context = _combine_documents(docs)
        timings["retriever"] = time.time() - step_start

        step_start = time.time()
        prompt = ANSWER_PROMPT.format_prompt(**{
            "context": context,
            "question": standalone_question
        })
        
        if stream_callback:
            full_response = ""
            chunk_buffer = ""
            word_count = 0
            
            for chunk in llm.stream(prompt.to_messages()):
                if chunk and hasattr(chunk, 'content') and chunk.content:
                    chunk_buffer += chunk.content
                    word_count += len(chunk.content.split())
                    
                    has_complete_url = self._has_complete_url_pattern(chunk_buffer)
                    
                    should_send = (
                        len(chunk_buffer) >= 100 or  
                        (any(punct in chunk_buffer for punct in ['.', '!', '?']) and not self._might_be_in_url(chunk_buffer)) or
                        word_count >= 15 or
                        has_complete_url
                    )
                    
                    if should_send and not self._buffer_contains_incomplete_url(chunk_buffer):
                        formatted_chunk = format_chunk_for_streaming(chunk_buffer)
                        if formatted_chunk.strip():
                            stream_callback(formatted_chunk)
                        full_response += chunk_buffer
                        chunk_buffer = ""
                        word_count = 0
          
            if chunk_buffer.strip():
                formatted_chunk = format_chunk_for_streaming(chunk_buffer)
                stream_callback(formatted_chunk)
                full_response += chunk_buffer
                
            result_content = full_response
        else:
            result = llm.invoke(prompt.to_messages())
            result_content = result.content
            
        timings["generate_answer"] = time.time() - step_start

        step_start = time.time()
        formatted_response = format_response_to_html(result_content)
        timings["format_response"] = time.time() - step_start

        total_time = sum(timings.values())
        print(f"\n[LOG] process_query timings: {total_time:.2f}s total\n")

        return {"messages": [AIMessage(content=formatted_response)]}

    except Exception as e:
        print(f"Error in process_query_streaming: {e}")
        error_msg = "Unfortunately, an error occurred while processing your query. Please try again."
        if stream_callback:
            stream_callback(error_msg)
        return {"messages": [AIMessage(content=error_msg)]}


def _buffer_contains_incomplete_url(text):
    """Check if buffer contains incomplete URL that should not be sent yet"""
    if '[' in text and '](' not in text:
        return True
    if '](' in text and text.count('](') > text.count('])'):
        return True
    return False

def create_conversation_workflow():
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("process_query", process_query)
    workflow.add_edge(START, "process_query")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

conversation_app = create_conversation_workflow()


BM_HELP_KEY=os.getenv("BM_HELP_KEY")
Decrypt_key="BMEAISecurity9.0"

if BM_HELP_KEY:
    try:
        decrypted_key = decrypt_openssl(BM_HELP_KEY, Decrypt_key)
        print("Decrypted BM_HELP_KEY:", decrypted_key)
    except Exception as e:
        print("Error decrypting BM_HELP_KEY:", e)
else:
    print("BM_HELP_KEY not found in .env")



@app.route('/query_stream', methods=['POST'])
def query_stream():
    enc_header = request.headers.get("Authorization")
    if not enc_header:
        return {"error": "Missing Authorization header"}

    try:
        decrypted_header = decrypt_openssl(enc_header, Decrypt_key)
        key_part, time_part = decrypted_header.split("$$", 1)

        if key_part != decrypted_key:
            return {"error": "Authentication failed: key mismatch"}

        payload_time = datetime.strptime(time_part.strip(), "%I:%M:%S %p").time()
        today = datetime.utcnow().date()
        payload_datetime = datetime.combine(today, payload_time)

        now = datetime.utcnow()

        if abs((now - payload_datetime).total_seconds()) > 1200:
            return {"error": "Authentication failed: timestamp expired"}

    except Exception as e:
        print("Auth decryption failed:", e)
        return {"error": "Invalid auth format"}

    data = request.get_json()
    if not data or "text" not in data:
        return {"error": "Missing 'text' in request body"}
    
    user_input = data['text'].strip()
    thread_id = "default-thread"

    def generate():
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state_snapshot = conversation_app.get_state(config)
            current_messages = state_snapshot.values.get('messages', []) if state_snapshot.values else []
            new_state = ConversationState(messages=current_messages + [HumanMessage(content=user_input)])
            
            chunks_sent = 0
            def stream_chunk(chunk):
                nonlocal chunks_sent
                chunks_sent += 1
                print(f"[STREAM {chunks_sent}] Yielding chunk: {chunk}")
                return f"data: {chunk}\n\n"
            
            full_response = ""
            chunk_buffer = ""
            word_count = 0
            
            user_input_content = new_state['messages'][-1].content if new_state['messages'] else ""
            chat_history = new_state['messages'][:-1]
            
            standalone_question = CONDENSE_QUESTION_PROMPT.format_prompt(**{
                "question": user_input_content,
                "chat_history": get_buffer_string(chat_history)
            }).to_string()
            standalone_question = llm.invoke(standalone_question).content
            
            docs = retriever.invoke(standalone_question)
            context = _combine_documents(docs)
            
        
            prompt = ANSWER_PROMPT.format_prompt(**{
                "context": context,
                "question": standalone_question
            })
            
            for chunk in llm.stream(prompt.to_messages()):
                if chunk and hasattr(chunk, 'content') and chunk.content:
                    chunk_buffer += chunk.content
                    word_count += len(chunk.content.split())
                    
                    has_complete_url = _has_complete_url_pattern(chunk_buffer)
                    
                    should_send = (
                        len(chunk_buffer) >= 100 or  
                        (any(punct in chunk_buffer for punct in ['.', '!', '?']) and not _might_be_in_url(chunk_buffer)) or
                        word_count >= 15 or
                        has_complete_url
                    )
                    
                    if should_send and not _buffer_contains_incomplete_url(chunk_buffer):
                        formatted_chunk = format_chunk_for_streaming(chunk_buffer)
                        if formatted_chunk.strip():
                            yield stream_chunk(formatted_chunk)
                        full_response += chunk_buffer
                        chunk_buffer = ""
                        word_count = 0
                        
            if chunk_buffer.strip():
                formatted_chunk = format_chunk_for_streaming(chunk_buffer)
                yield stream_chunk(formatted_chunk)
                full_response += chunk_buffer
            
            final_state = ConversationState(
                messages=current_messages + [
                    HumanMessage(content=user_input_content),
                    AIMessage(content=format_response_to_html(full_response))
                ]
            )
            conversation_app.update_state(config, final_state)
            
            print(f"[STREAM COMPLETE] Total chunks sent: {chunks_sent}")
            yield "data: [END]\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: [ERROR] {str(e)}\n\n"

    response = Response(
        stream_with_context(generate)(),
        mimetype='text/event-stream',
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": FRONTEND_URL,
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )
    return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)






