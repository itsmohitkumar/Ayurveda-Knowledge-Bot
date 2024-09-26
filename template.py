from pathlib import Path
import logging

# Configure logging for better visibility of actions
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# List of files and folders required for the Bedrock Knowledge Base and Multimodal chatbot project
list_of_files = [
    "src/__init__.py",                # Initialization file for the source package
    "src/data_loader.py",             # File for data loading (e.g., loading documents, media files)
    "src/bedrock_knowledge_base.py",  # File for setting up and managing Bedrock Knowledge Base
    "src/embedding_service.py",       # File for handling embedding models (e.g., using Titan or Claude embeddings)
    "src/multimodal_retrieval.py",    # File for implementing multimodal retrieval logic
    "src/chatbot_interface.py",       # File for handling chatbot interface and queries
    "src/utils.py",                   # File for utility functions (e.g., logging, config parsing)
    ".env",                           # Environment file for storing sensitive data like API keys
    "setup.py",                       # Setup file for packaging the project
    "pyproject.toml",                 # Configuration file for Python project dependencies and settings
    "Dockerfile",                     # Dockerfile for containerizing the application
    ".gitignore",                     # File to ignore unnecessary files in Git
    ".dockerignore",                  # File to ignore unnecessary files in Docker builds
    "notebooks/experiments.ipynb",    # Jupyter notebook for experiments with the chatbot and retrieval system
    "data/.gitkeep",                  # Keeps the data folder in Git (for data files)
    "tests/test_data_loader.py",      # Test case for data loader functionality
    "tests/test_bedrock_kb.py",       # Test case for Bedrock Knowledge Base integration
    "tests/test_embedding_service.py",# Test case for embedding service
    "tests/test_multimodal_retrieval.py", # Test case for multimodal retrieval logic
    "tests/test_chatbot_interface.py",# Test case for chatbot interface functionality
    "tests/test_utils.py",            # Test case for utility functions
    "app.py",                         # Entry point for the chatbot application
    "static/.gitkeep",                # Keeps the static folder in Git (for CSS, JS)
    "templates/index.html",           # HTML template for the web interface
    "README.md",                      # README file for project documentation
    ".github/workflows/cicd-main.yml"      # GitHub Actions workflow for CI/CD automation
]

# Loop over the list of files and create the necessary directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    # Create directories if they don't exist
    if filedir != Path("."):  # Only create directory if it's not the current directory
        if not filedir.exists():
            filedir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {filedir}")

    # Create the file if it doesn't exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()  # This creates an empty file
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File {filepath} already exists and is not empty")
