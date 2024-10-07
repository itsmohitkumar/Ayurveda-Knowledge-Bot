# ----- Stage 1: Build the React frontend -----
    FROM node:16-alpine AS frontend-build

    # Set the working directory for the frontend
    WORKDIR /frontend
    
    # Copy package.json and package-lock.json
    COPY frontend/package*.json ./
    
    # Install npm dependencies
    RUN npm install
    
    # Copy the rest of the frontend code
    COPY frontend/ .
    
    # Build the React app
    RUN npm run build
    
    # ----- Stage 2: Set up the FastAPI backend -----
    FROM python:3.10-slim AS backend
    
    # Set the working directory for the backend
    WORKDIR /app
    
    # Copy backend requirements
    COPY backend/requirements.txt .
    
    # Install Python dependencies
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy the backend source code
    COPY backend/ .
    
    # Copy the frontend build output from the previous stage
    COPY --from=frontend-build /frontend/build ./frontend/build
    
    # Expose the backend port
    EXPOSE 8000
    
    # Command to run the FastAPI app
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    