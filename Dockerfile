# ----------------------
# Stage 1: Build React App
# ----------------------

    FROM node:18-alpine AS frontend-build

    # Set working directory for React app
    WORKDIR /frontend
    
    # Copy only package.json and package-lock.json to install dependencies
    COPY frontend-react/package*.json ./
    
    # Install Node.js dependencies
    RUN npm install
    
    # Copy the rest of the React app code
    COPY frontend-react/ .
    
    # Build the React app for production
    RUN npm run build
    
    # ----------------------
    # Stage 2: Set up FastAPI backend
    # ----------------------
    
    FROM python:3.10-slim AS backend
    
    # Set environment variables
    ENV LANGCHAIN_TRACING_V2=true
    ENV LANGCHAIN_PROJECT=YouTube-Echo
    
    # Uncomment if you want to set a default value for LANGCHAIN_API_KEY
    # ENV LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
    
    # Set working directory for FastAPI app
    WORKDIR /app
    
    # Copy the requirements file and install Python dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy only the backend code
    COPY ./app.py .
    COPY ./src ./src
    COPY ./src/config.json .
    
    # Expose the FastAPI port
    EXPOSE 8000
    
    # ----------------------
    # Stage 3: Combine and run both frontend and backend
    # ----------------------
    
    FROM nginx:alpine AS production
    
    # Copy the built React frontend to NGINX
    COPY --from=frontend-build /frontend/build /usr/share/nginx/html
    
    # Copy the FastAPI backend (from the backend stage)
    COPY --from=backend /app /app
    
    # Copy the NGINX configuration file
    COPY nginx.conf /etc/nginx/nginx.conf
    
    # Expose port 80 for NGINX
    EXPOSE 80
    
    # ----------------------
    # Stage 4: Run the FastAPI app
    # ----------------------
    
    # Start NGINX and FastAPI
    CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & nginx -g 'daemon off;'"]
    