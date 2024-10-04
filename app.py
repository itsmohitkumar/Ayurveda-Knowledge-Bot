import os
import json
import time
import boto3
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Load configuration from config.json
with open("src/config.json") as config_file:
    config = json.load(config_file)

# Configuration variables from config.json
DATA_DIRECTORY = config["DATA_DIRECTORY"]
FAISS_INDEX_PATH = config["FAISS_INDEX_PATH"]
TITAN_MODEL_ID = config["TITAN_MODEL_ID"]
LLAMA_MODEL_ID = config["LLAMA_MODEL_ID"]
CHUNK_SIZE = config["CHUNK_SIZE"]
CHUNK_OVERLAP = config["CHUNK_OVERLAP"]
MAX_THREADS = config["MAX_THREADS"]
LOG_LEVEL = config["LOG_LEVEL"]

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
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

# Pydantic model for the question input
class QuestionRequest(BaseModel):
    question: str
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_default_region: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "question": "What is the capital of France?",
                "aws_access_key_id": "your_access_key_id",
                "aws_secret_access_key": "your_secret_access_key",
                "aws_default_region": "your_region"
            }
        }

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

            logger.info(f"Document loading and chunking completed in {time.time() - start_time} seconds.")
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
            logger.info(f"Response generated in {time.time() - start_time} seconds.")
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
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for request: {request.url} - {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body
        },
    )

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

        if not aws_access_key_id or not aws_secret_access_key or not aws_default_region:
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

        logger.info(f"Question processed in {time.time() - start_time} seconds.")
        return {"answer": response}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing request")
