from src.utils import load_pdf_data, text_chunk_splitter, download_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import pinecone
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


extracted_data = load_pdf_data("/data")
text_chunks = text_chunk_splitter(extracted_data)
embedding = download_embedding_model()

pinecone.Pinecone(
   api_key=os.getenv("PINECONE_API_KEY"),  
   environment=os.getenv("PINECONE_ENV"),  
)
index_name = "med-chatbot-hybrid"

docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embedding, index_name=index_name)