from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
import boto3
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List
import time
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Configuration variables
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "data")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss-index")
TITAN_MODEL_ID = os.getenv("TITAN_MODEL_ID", "amazon.titan-embed-text-v1")
LLAMA_MODEL_ID = os.getenv("LLAMA_MODEL_ID", "meta.llama3-70b-instruct-v1:0")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
MAX_THREADS = int(os.getenv("MAX_THREADS", 4))

# Initialize FastAPI app
app = FastAPI()

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

    class Config:
        schema_extra = {
            "example": {
                "question": "What is the capital of France?"
            }
        }

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

# Dependency injection function for the Bedrock client
def get_bedrock_client():
    try:
        logger.info("Initializing Bedrock client...")
        return boto3.client(service_name="bedrock-runtime")
    except Exception as e:
        logger.error(f"Error initializing Bedrock client: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

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

# Startup event to load FAISS index and process documents
@app.on_event("startup")
def on_startup():
    try:
        logger.info("Starting up application...")
        bedrock_client = get_bedrock_client()
        document_processor = PDFDocumentProcessor(DATA_DIRECTORY)
        chunked_documents = document_processor.load_and_chunk_documents()

        faiss_manager = FAISSManager(FAISS_INDEX_PATH, BedrockEmbeddings(model_id=TITAN_MODEL_ID, client=bedrock_client))
        faiss_manager.create_and_save_vector_store(chunked_documents)

        logger.info("Startup process completed successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Startup process failed")

@app.post("/ask")
async def ask_question(request: QuestionRequest, bedrock_client = Depends(get_bedrock_client)):
    try:
        start_time = time.time()
        logger.info(f"Received question: '{request.question}'")

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
