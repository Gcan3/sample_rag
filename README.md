# Simple RAG application on a custom pdf file data.

Llm: Mistral-7B-Instruct-v0.3
Vector db: Chroma
Embedder: langchain SentenceTransformerEmbeddings

Please create an .env file and place your hugginface token key

Order of Repo:
1. run ingest.py
2. storage folder gets created for chroma db
3. streamlit run rag.py
