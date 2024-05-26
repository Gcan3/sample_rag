# import app and rag needed libraries
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Simple UI for streamlit
st.set_page_config(page_title="Microplastic QA", page_icon="🔍", layout="wide")
st.title("Microplastic QA")

# load API key from .env file
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Template for the queries
prompt_template = """
You are given a research document for context.

Context: {context}

You must answer the question given to you based on the context provided. Do not use any external resources.
If you do not know the answer, please respond with "I don't know".

Question: {question}

Answer:

"""
# Create the prompt template
prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

# Call the embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

# Get the vector storage
load_vector_store = Chroma(persist_directory="storage/microplastic_cosine", embedding_function=embeddings)

# Load the retriever
retriever = load_vector_store.as_retriever(search_kwargs={"k": 5})

# Load the LLM
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.5, "max_length": 700}
)

chain_type_kwargs = {"prompt": prompt}

# Create the QA chain
def qa_chain():
    qa = RetrievalQA.from_chain_type(
        chain_type="stuff",
        retriever=retriever,
        llm=llm,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    return qa

qa = qa_chain()

# Main function for the whole process with streamlit
def main():
    # Ask the user for the question
    text_query = st.text_input("Enter your question here:")
    
    generate_response = st.button("Generate Response")
    
    st.subheader("Response:")
    if generate_response and text_query:
        with st.spinner("Generating response..."):
            response = qa(text_query)
            if response:
                st.write(response)
                st.success("Response generated successfully.")
            else:
                st.error("No response generated.")

if __name__ == "__main__":
    main()