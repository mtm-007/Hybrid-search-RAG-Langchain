# RAG_MedChatBot
End to End RAG based medical chatbot using Opensource LLM -- Llama-2


# Embedding model choice
- the embedding model embedds both the query and the chunked text and compare their similarity, the lenth of `query text vs chunked text` has a huge impact
- So architectural difference decision comes to use symmetric or asymmetric embedding models, asymmetric models works best for difference in text length thus short query vs longer paragraph like result (or retrieved context)
- Based on that switching to asymmetric embedding model is the best choice based on those facts --> from `all-MiniLM-L6-v2` to `msmarco-distilbert-base-v3` 

