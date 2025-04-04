import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def process_word_docs(directory_path):
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = "bmebot"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=1536, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  
        )


    index = pc.Index(index_name)

  
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

 
    for filename in os.listdir(directory_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing: {file_path}")

            try:
              
                loader = Docx2txtLoader(file_path)
                documents = loader.load()

             
                docs = text_splitter.split_documents(documents)

                PineconeVectorStore.from_documents(
                    docs, 
                    embeddings, 
                    index_name=index_name
                )
                print(f"Uploaded {filename} successfully!")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print("All Word documents processed and uploaded!")


directory_path = "Video_scripts"
process_word_docs(directory_path)
