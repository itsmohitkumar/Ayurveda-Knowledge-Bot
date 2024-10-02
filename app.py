# Step 1: Import Libraries
import os
import boto3
import pinecone
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Step 2: Load Environment Variables
load_dotenv()

aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
region_name = os.getenv("region_name")
pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Add your Pinecone API key
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")  # Add your Pinecone environment

# Step 3: Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

# Step 4: Define Prompt Template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

# Step 5: Initialize Bedrock Client
bedrock = boto3.client(
    service_name="bedrock-runtime", 
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

# Step 6: Get Embeddings Model
bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Step 7: Function to Load Documents
def get_documents():
    loader = PyPDFDirectoryLoader("Data")  # Load all PDFs from the 'Data' directory
    documents = loader.load()
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=500
    )
    docs = text_spliter.split_documents(documents)
    return docs

# Step 8: Function to Create and Save Vector Store
def get_vector_store(docs):
    # Create a new Pinecone index if it doesn't exist
    index_name = "ayurveda-chatbot"
    
    # Create the index (if it doesn't exist)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=bedrock_embedding.output_dim)
    
    index = pinecone.Index(index_name)
    
    # Upsert documents into Pinecone
    vectors = [(f"doc-{i}", bedrock_embedding.embed(doc), {"text": doc}) for i, doc in enumerate(docs)]
    index.upsert(vectors)
    
    print("Vector store created and documents upserted.")

# Step 9: Function to Get Language Model
def get_llm():
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock)
    return llm

# Step 10: Create Prompt Template Instance
PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# Step 11: Function to Get LLM Response
def get_llm_response(llm, query):
    index = pinecone.Index("ayurveda-chatbot")

    # Query Pinecone for the most similar documents
    query_vector = bedrock_embedding.embed(query)
    results = index.query(query_vector, top_k=3, include_metadata=True)

    # Prepare the context for the prompt
    context = "\n".join([item['metadata']['text'] for item in results['matches']])
    
    # Generate response using the language model
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=None,  # Pinecone already handles the retrieval
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    response = qa({"context": context, "question": query})
    return response['result']

# Step 12: Main Function to Run Chatbot
def main():
    print("Ayurveda Chatbot")
    print("Type 'exit' to quit the chat.")
    
    while True:
        user_question = input("Ask a question about Ayurveda: ")
        
        if user_question.lower() == 'exit':
            break
        
        llm = get_llm()
        
        response = get_llm_response(llm, user_question)
        print("\nAssistant:", response)

# Step 13: Run the Main Function
if __name__ == "__main__":
    main()
