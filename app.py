import os
import json
import time
import boto3
import logging
from typing import List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from fastapi.responses import JSONResponse

# Load environment variables from .env file
load_dotenv()

# Load configuration from config.json
try:
    with open("src/config.json") as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    raise RuntimeError("Configuration file 'src/config.json' not found.")

# Configuration variables from config.json
DATA_DIRECTORY = config.get("DATA_DIRECTORY")
FAISS_INDEX_PATH = config.get("FAISS_INDEX_PATH")
TITAN_MODEL_ID = config.get("TITAN_MODEL_ID")
LLAMA_MODEL_ID = config.get("LLAMA_MODEL_ID")
CHUNK_SIZE = config.get("CHUNK_SIZE")
CHUNK_OVERLAP = config.get("CHUNK_OVERLAP")
MAX_THREADS = config.get("MAX_THREADS")
LOG_LEVEL = config.get("LOG_LEVEL", "INFO").upper()

# Validate configuration variables
required_configs = [DATA_DIRECTORY, FAISS_INDEX_PATH, TITAN_MODEL_ID, LLAMA_MODEL_ID, CHUNK_SIZE, CHUNK_OVERLAP, MAX_THREADS]
if any(config is None for config in required_configs):
    raise ValueError("Missing required configuration in config.json")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your specific needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Prompt template for tax-related queries in India
prompt_template = """
You are a knowledgeable Indian tax advisor. Use the following pieces of context to provide a detailed answer to the question at the end. 
Provide at least 250 words with detailed explanations, practical examples where applicable, and include an analysis at the end.

Examples:
1. 
Human: What are the different types of income tax in India?
Assistant: In India, income tax is categorized into various heads based on the source of income. The primary types are: 
   - **Salaries**: Income earned from employment.
   - **House Property**: Income from rental properties.
   - **Business or Profession**: Profits from business activities or professional services.
   - **Capital Gains**: Profits from the sale of capital assets such as stocks or real estate.
   - **Other Sources**: Includes interest income, dividends, and other miscellaneous income. 
Each category is subject to specific tax rates and exemptions under the Income Tax Act, 1961.

Analysis: Understanding the different types of income tax is crucial for individuals and businesses to comply with tax regulations and optimize their tax liabilities.

2.
Human: Can you explain the Goods and Services Tax (GST) framework in India?
Assistant: The Goods and Services Tax (GST) is a comprehensive indirect tax levied on the supply of goods and services in India, implemented on July 1, 2017. It subsumes various indirect taxes such as Value Added Tax (VAT), Central Excise Duty, and Service Tax. The GST structure consists of three components: 
   - **CGST**: Central Goods and Services Tax, collected by the central government.
   - **SGST**: State Goods and Services Tax, collected by the state government.
   - **IGST**: Integrated Goods and Services Tax, applied to inter-state transactions.
The GST aims to simplify the tax structure, enhance compliance, and eliminate the cascading effect of multiple taxes.

Analysis: The GST framework represents a significant reform in India's tax system, promoting a unified market and fostering ease of doing business.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Pydantic model for the question input
class QuestionRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "What is the capital of France?"})
    aws_access_key_id: Optional[str] = Field(None, json_schema_extra={"example": "your_access_key_id"})
    aws_secret_access_key: Optional[str] = Field(None, json_schema_extra={"example": "your_secret_access_key"})
    aws_default_region: Optional[str] = Field(None, json_schema_extra={"example": "your_region"})

# PDF Document Processor
class PDFDocumentProcessor:
    def __init__(self, data_directory: str):
        self.data_directory = data_directory

    def load_and_chunk_documents(self) -> List[str]:
        start_time = time.time()
        try:
            loader = PyPDFDirectoryLoader(self.data_directory)
            logger.info(f"Loading PDFs from {self.data_directory}...")
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            
            # Parallel chunking
            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                chunked_documents = list(executor.map(text_splitter.split_documents, [documents]))

            logger.info(f"Document loading and chunking completed in {time.time() - start_time:.2f} seconds.")
            return [chunk for sublist in chunked_documents for chunk in sublist]
        except FileNotFoundError:
            logger.error(f"Data directory '{self.data_directory}' not found.")
            raise HTTPException(status_code=404, detail="Data directory not found")
        except Exception as e:
            logger.error(f"Error loading and chunking documents: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error processing documents")

# FAISS Manager
class FAISSManager:
    def __init__(self, index_path: str, embeddings):
        self.index_path = index_path
        self.embeddings = embeddings

    def create_and_save_vector_store(self, chunked_documents: List[str]):
        try:
            vectorstore_faiss = FAISS.from_documents(chunked_documents, self.embeddings)
            vectorstore_faiss.save_local(self.index_path)
            logger.info(f"FAISS index created and saved to {self.index_path}.")
        except Exception as e:
            logger.error(f"Error creating and saving FAISS vector store: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error creating FAISS vector store")

    def load_vector_store(self):
        try:
            logger.info(f"Loading FAISS index from {self.index_path}...")
            return FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        except FileNotFoundError:
            logger.error(f"FAISS index file '{self.index_path}' not found.")
            raise HTTPException(status_code=404, detail="FAISS index not found")
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error loading FAISS vector store")

# LLM Service
class LLMService:
    def __init__(self, model_id: str, client):
        self.model_id = model_id
        self.client = client

    def initialize_llm(self):
        try:
            logger.info(f"Initializing LLM with model ID: {self.model_id}")
            return BedrockLLM(model_id=self.model_id, client=self.client)
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error initializing LLM")

    def generate_response(self, llm, vectorstore_faiss, query: str):
        try:
            start_time = time.time()
            logger.info(f"Generating response for query: '{query}'")
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            result = qa.invoke({"query": query})
            logger.info(f"Response generated in {time.time() - start_time:.2f} seconds.")
            return result['result']
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error generating response")

# Middleware to log requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Completed request: {request.method} {request.url} - Status code: {response.status_code}")
    return response

# Custom error handler for validation errors
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error for request: {request.url} - {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body
        },
    )

# General exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"An unexpected error occurred: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred"})

# Endpoint to ask a question
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        start_time = time.time()
        logger.info(f"Received question: '{request.question}'")

        # Initialize AWS client with keys from the request if provided
        aws_access_key_id = request.aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = request.aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_default_region = request.aws_default_region or os.getenv("AWS_DEFAULT_REGION")

        if not all([aws_access_key_id, aws_secret_access_key, aws_default_region]):
            raise HTTPException(status_code=400, detail="AWS keys are required.")

        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_default_region,
        )

        # Load FAISS index
        faiss_manager = FAISSManager(FAISS_INDEX_PATH, BedrockEmbeddings(model_id=TITAN_MODEL_ID, client=bedrock_client))
        faiss_index = faiss_manager.load_vector_store()

        # Initialize LLM
        llm_service = LLMService(model_id=LLAMA_MODEL_ID, client=bedrock_client)
        llm = llm_service.initialize_llm()

        # Generate response
        response = llm_service.generate_response(llm, faiss_index, request.question)

        logger.info(f"Question processed in {time.time() - start_time:.2f} seconds.")
        return {"answer": response}

    except HTTPException as e:
        logger.error(f"HTTP exception: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing request")
