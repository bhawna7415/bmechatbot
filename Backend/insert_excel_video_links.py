import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def process_excel_file(file_path):
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = "bmebot"  
    
    index = pc.Index(index_name)
  
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print(f"Processing Excel file: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
        
        df.columns = df.columns.str.strip()
        
        documents = []
        
        for _, row in df.iterrows():
            content = f"Video Titles: {row['Video Titles']}\n"
            content += f"Screen Name: {row['Screen Name']}\n"
            content += f"Screen Code: {row['Screen Code']}\n"
            content += f"Category: {row['Category']}\n"
            
            # Add URL if it exists
            if 'Sample URL' in row and isinstance(row['Sample URL'], str) and row['Sample URL'].strip():
                content += f"URL: {row['Sample URL']}\n"
            
            # Create metadata
            metadata = {
                "video_titles": str(row["Video Titles"]),
                "screen_name": str(row["Screen Name"]),
                "screen_code": str(row["Screen Code"]),
                "category": str(row["Category"]),
                "source": "excel_video_catalog",
                "filename": os.path.basename(file_path)
            }
            
            if 'Sample URL' in row and isinstance(row['Sample URL'], str) and row['Sample URL'].strip():
                metadata["url"] = row['Sample URL']
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        PineconeVectorStore.from_documents(
            documents, 
            embeddings, 
            index_name=index_name
        )
        
        print(f"Successfully uploaded {len(documents)} video records from Excel file!")
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")

excel_file_path = "BME Web Videos List.xlsx" 
process_excel_file(excel_file_path)