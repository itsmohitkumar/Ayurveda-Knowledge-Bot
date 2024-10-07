import boto3
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from langchain_aws import BedrockEmbeddings
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from src.chatbot.config import DATA_DIRECTORY, FAISS_INDEX_PATH, TITAN_MODEL_ID, LLAMA_MODEL_ID, LOG_LEVEL
from src.chatbot.services import FAISSManager, PDFDocumentProcessor, LLMService

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Lifespan event triggered. Automatically running the /create_index endpoint...")
        await create_index()  # Automatically trigger create_index during startup
    except Exception as e:
        logger.error(f"Error during lifespan event: {e}", exc_info=True)
    yield  # Continue with the application lifecycle

app = FastAPI(lifespan=lifespan)

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

# Pydantic model for the question input
class QuestionRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "What is the new tax laws??"})
    aws_access_key_id: Optional[str] = Field(None, json_schema_extra={"example": "your_access_key_id"})
    aws_secret_access_key: Optional[str] = Field(None, json_schema_extra={"example": "your_secret_access_key"})
    aws_default_region: Optional[str] = Field(None, json_schema_extra={"example": "your_region"})

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

# Endpoint to create FAISS index from PDF documents
@app.post("/create_index")
async def create_index():
    try:
        logger.info("Creating FAISS index...")

        # Load and chunk PDF documents
        processor = PDFDocumentProcessor(data_directory=DATA_DIRECTORY)
        chunked_documents = processor.load_and_chunk_documents()

        # Load embeddings and create FAISS index
        embeddings = BedrockEmbeddings(model_id=TITAN_MODEL_ID)
        faiss_manager = FAISSManager(index_path=FAISS_INDEX_PATH, embeddings=embeddings)
        faiss_manager.create_and_save_vector_store(chunked_documents)

        return {"message": "FAISS index created successfully."}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error creating FAISS index")

# Question answering endpoint
@app.post("/answer")
async def answer_question(request: QuestionRequest):
    try:
        logger.info(f"Received question: {request.question}")

        # Validate AWS credentials
        if request.aws_access_key_id and request.aws_secret_access_key and request.aws_default_region:
            logger.info("AWS credentials provided in the request.")
            # Initialize Boto3 client using provided credentials
            client = boto3.Session(
                aws_access_key_id=request.aws_access_key_id,
                aws_secret_access_key=request.aws_secret_access_key,
                region_name=request.aws_default_region
            ).client("bedrock-runtime")
        else:
            # Initialize Boto3 client using environment variables or credentials from AWS CLI
            client = boto3.client("bedrock-runtime")
            logger.info("Using AWS credentials from environment variables or AWS CLI configuration.")

        # Load FAISS index
        embeddings = BedrockEmbeddings(model_id=TITAN_MODEL_ID)
        faiss_manager = FAISSManager(index_path=FAISS_INDEX_PATH, embeddings=embeddings)
        vectorstore_faiss = faiss_manager.load_vector_store()

        # Initialize LLM
        llm_service = LLMService(model_id=LLAMA_MODEL_ID, client=client)
        llm = llm_service.initialize_llm()

        # Generate response
        response = llm_service.generate_response(llm=llm, vectorstore_faiss=vectorstore_faiss, query=request.question)
        return {"answer": response}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing question")
