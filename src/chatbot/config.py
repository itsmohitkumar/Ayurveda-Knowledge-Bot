import json
from dotenv import load_dotenv

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
required_configs = [
    DATA_DIRECTORY, FAISS_INDEX_PATH, TITAN_MODEL_ID, LLAMA_MODEL_ID, 
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_THREADS
]

if any(config is None for config in required_configs):
    raise ValueError("Missing required configuration in config.json")

# Set LangChain tracing and API key from the config file
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Indian-Tax-Advisor"

# Retrieve and validate the API key from the environment variables
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key is None:
    raise ValueError("LANGCHAIN_API_KEY is not set in the environment variables.")

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

