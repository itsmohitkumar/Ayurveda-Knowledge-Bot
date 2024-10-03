import boto3
import streamlit as st

# Import necessary modules from LangChain
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configuration variables
DATA_DIRECTORY = "data"
FAISS_INDEX_PATH = "faiss_index"
TITAN_MODEL_ID = "amazon.titan-embed-text-v1"
CLAUDE_MODEL_ID = "ai21.j2-mid-v1"
LLAMA_MODEL_ID = "meta.llama3-70b-instruct-v1:0"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Initialize the Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id=TITAN_MODEL_ID, client=bedrock)

# Prompt template for the LLM responses
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def load_and_chunk_documents():
    """
    Load PDF documents from the specified directory and split them into manageable chunks.
    
    Returns:
        List of chunked documents.
    """
    loader = PyPDFDirectoryLoader(DATA_DIRECTORY)
    documents = loader.load()

    # Split documents into chunks with overlapping text for enhanced context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def create_and_save_vector_store(chunked_documents):
    """
    Create a FAISS vector store from the provided documents and save it locally.
    
    Args:
        chunked_documents (list): List of chunked documents.
    """
    vectorstore_faiss = FAISS.from_documents(chunked_documents, bedrock_embeddings)
    vectorstore_faiss.save_local(FAISS_INDEX_PATH)

def initialize_llm(model_id):
    """
    Create and return a Bedrock LLM instance based on the specified model ID.
    
    Args:
        model_id (str): The model identifier for the LLM.
        
    Returns:
        LLM instance.
    """
    llm = BedrockLLM(model_id=model_id, client=bedrock)
    return llm

def generate_response(llm, vectorstore_faiss, query):
    """
    Generate a response from the LLM based on the user's query and the context retrieved from the vector store.
    
    Args:
        llm: The language model instance.
        vectorstore_faiss: The FAISS vector store.
        query (str): The user's query.
        
    Returns:
        str: The generated response from the LLM.
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa.invoke({"query": query})  # Use the new invoke method
    return answer['result']

def main():
    """
    Main function to run the Streamlit application for querying PDF documents.
    """
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                chunked_documents = load_and_chunk_documents()
                create_and_save_vector_store(chunked_documents)
                st.success("Vector store updated successfully!")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(FAISS_INDEX_PATH, bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = initialize_llm(CLAUDE_MODEL_ID)
            response = generate_response(llm, faiss_index, user_question)
            st.write(response)
            st.success("Claude response generated!")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(FAISS_INDEX_PATH, bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = initialize_llm(LLAMA_MODEL_ID)
            response = generate_response(llm, faiss_index, user_question)
            st.write(response)
            st.success("Llama2 response generated!")

if __name__ == "__main__":
    main()
