
#from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


#extract data func
def load_pdf_data(data):
    loader = DirectoryLoader(data,
                    glob='*.pdf',
                    loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents

#create chunks of text
def text_chunk_splitter(extracted_data):
    text_split = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150)
    text_chunk = text_split.split_documents(extracted_data)

    return text_chunk

#download embedding model
def download_embedding_asy_model():
    embedding= HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-v3")

    return embedding
