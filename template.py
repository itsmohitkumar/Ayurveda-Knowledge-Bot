from pathlib import Path
import logging

# Configure logging for better visibility of actions
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# List of files and folders required for the Ayurveda Chatbot project
list_of_files = [
    "src/chatbot/__init__.py",                # Initialization file for the source package
    "src/chatbot/data_loader.py",             # File for data loading (e.g., loading PDF documents)
    "src/chatbot/pinecone_service.py",        # File for handling Pinecone vector store
    "src/chatbot/chatbot.py",                  # Main chatbot logic and interactions
    "src/chatbot/utils.py",                   # File for utility functions (e.g., logging, config parsing)
    "src/__init__.py",                        # Initialization file for the source package
    "src/config.json",                        # Configuration file for the chatbot
    ".env",                                   # Environment file for storing sensitive data like API keys
    "setup.py",                               # Setup file for packaging the project
    "pyproject.toml",                         # Configuration file for Python project dependencies and settings
    "notebooks/chatbot_interaction.ipynb",    # Jupyter notebook for experimenting with the chatbot
    "data/.gitkeep",                          # Keeps the data folder in Git (for data files)
    "tests/test_chatbot.py",                  # Test case for chatbot functionality
    "README.md",                              # README file for project documentation
    ".gitignore",                             # File to ignore unnecessary files in Git
    "LICENSE"                                 # MIT License file
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

# Step to create the MIT License file
mit_license = """MIT License

Copyright (c) [2024] [Mohit Kumar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

...

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Write the MIT License to the LICENSE file
license_filepath = Path("LICENSE")
if not license_filepath.exists() or license_filepath.stat().st_size == 0:
    with open(license_filepath, "w") as license_file:
        license_file.write(mit_license.replace("[year]", "2024").replace("[your name]", "Angela"))
    logging.info(f"Created MIT License file: {license_filepath}")
else:
    logging.info(f"File {license_filepath} already exists and is not empty")
