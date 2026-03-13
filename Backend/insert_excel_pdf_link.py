import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def store_pdf_names(file_path):
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = "bmebot"  
    index = pc.Index(index_name)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print(f"Processing PDF names from Excel file: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        pdf_names_column = "Pdf Name" 
        
        if pdf_names_column not in df.columns:
            print(f"Error: Column '{pdf_names_column}' not found in Excel file.")
            return
        documents = []
        for _, row in df.iterrows():
            pdf_name = str(row[pdf_names_column]).strip()
            
            if not pdf_name:
                continue
                
            print(f"Processing PDF name: {pdf_name}")
            metadata = {
                "pdf_name": pdf_name,
                "source": "pdf_catalog",
                "filename": os.path.basename(file_path)
            }
            
            doc = Document(page_content=f"PDF Document: {pdf_name}", metadata=metadata)
            documents.append(doc)
       
        PineconeVectorStore.from_documents(
            documents, 
            embeddings, 
            index_name=index_name
        )
        
        print(f"Successfully stored {len(documents)} PDF names in the vector database!")
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")

pdf_names_excel_path = "Pdf URL.xlsx"  
store_pdf_names(pdf_names_excel_path)