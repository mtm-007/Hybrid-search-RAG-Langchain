from flask import Flask, render_template, jsonify, request
from src.utils import download_embedding_model
import pinecone
from langchain_pinecone import PineconeVectorStore
#from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
#from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from src.prompt import *

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embedding = download_embedding_model()

pinecone.Pinecone(
   api_key=os.getenv("PINECONE_API_KEY"),  
   environment=os.getenv("PINECONE_ENV"),  
)
index_name = "medical-chatbot-vector-index"

docsearch = PineconeVectorStore.from_existing_index(index_name, embedding)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model= "./model/gguf/llama-2-7b-chat.Q3_K_M.gguf",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.7})

QA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = docsearch.as_retriever(search_kwargs={'k':3}),
    return_source_documents = True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')


if __name__== '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
