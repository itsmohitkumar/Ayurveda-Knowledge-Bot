import subprocess
import os
import time
from threading import Thread

# Function to run the FastAPI app
def run_fastapi():
    try:
        # Assuming the FastAPI app is in app.py
        print("Starting FastAPI server...")
        subprocess.run(["uvicorn", "src.chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"])
    except Exception as e:
        print(f"Error running FastAPI: {e}")

# Function to update and run the React app
def run_react():
    try:
        # Set working directory to the frontend React folder
        os.chdir('frontend-react')

        # Install dependencies if needed
        print("Installing npm dependencies...")
        subprocess.run(["npm", "install"])

        # Fix vulnerabilities and update deprecated options
        print("Fixing vulnerabilities and updating dependencies...")
        #subprocess.run(["npm", "audit", "fix"])
        #subprocess.run(["npm", "audit", "fix", "--force"])

        # Start the React app using npm start
        print("Starting React app...")
        subprocess.run(["npm", "start"])
    except Exception as e:
        print(f"Error running React app: {e}")
    finally:
        # Return to the original directory
        os.chdir('..')

# Main function to run both FastAPI and React concurrently
def main():
    # Create threads to run both FastAPI and React
    fastapi_thread = Thread(target=run_fastapi)
    react_thread = Thread(target=run_react)

    # Start both threads
    fastapi_thread.start()
    time.sleep(5)  # Give FastAPI a moment to start
    react_thread.start()

    # Wait for both threads to finish
    fastapi_thread.join()
    react_thread.join()

if __name__ == "__main__":
    main()
