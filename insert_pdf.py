import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore  
from pinecone import Pinecone as PineconeClient, ServerlessSpec  
from dotenv import load_dotenv

load_dotenv()

def process_pdfs_from_directory(directory_path):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Missing Pinecone API key.")
    pc = PineconeClient(api_key=pinecone_api_key)
    
    index_name = "bmebot"

    if index_name not in pc.list_indexes().names():
        print(f"Creating index: {index_name}")
        pc.create_index(
            name=index_name, 
            dimension=1536,  
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Using existing index: {index_name}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing: {file_path}")
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()

                docs = text_splitter.split_documents(documents)
                print(f"Split into {len(docs)} chunks")

                PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
                print(f"Uploaded {filename} successfully!")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("All PDFs processed and uploaded!")

directory_path = "BME_Web_help_files"
process_pdfs_from_directory(directory_path)
