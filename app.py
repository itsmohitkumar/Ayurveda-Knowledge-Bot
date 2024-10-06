import subprocess
import os
import time
from threading import Thread

# Function to run the FastAPI app
def run_fastapi():
    try:
        # Check if uvicorn is available in the environment
        print("Starting FastAPI server...")
        process = subprocess.Popen(
            ["uvicorn", "src.chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Stream FastAPI output to the terminal
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")
        
        # Check if FastAPI process has completed
        process.stdout.close()
        process.wait()

    except Exception as e:
        print(f"Error running FastAPI: {e}")

# Function to run the React app
def run_react():
    try:
        # Set working directory to the frontend React folder
        print("Starting React app...")
        os.chdir('frontend-react')

        # Install dependencies if needed (can be skipped if already installed)
        subprocess.run(["npm", "install"])

        # Run the React app using npm start
        subprocess.run(["npm", "start"])

    except Exception as e:
        print(f"Error running React app: {e}")

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
