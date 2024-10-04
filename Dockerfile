# ----------------------
# Stage 1: Build React App
# ----------------------

FROM node:18-alpine AS frontend-build

# Set working directory for React app
WORKDIR /frontend
    
# Copy React package.json and package-lock.json to install dependencies
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
    
FROM python:3.10-slim

# Set environment variables
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_PROJECT=YouTube-Echo

# Accept the LANGCHAIN_API_KEY as a build argument
ARG LANGCHAIN_API_KEY

# Set it as an environment variable
ENV LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
    
# Set working directory for FastAPI app
WORKDIR /app
    
# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
    
# Copy only the backend code (excluding the frontend code and other unnecessary files)
COPY ./app.py .
COPY ./src ./src
COPY ./src/config.json .
    
# Copy the built React frontend from Stage 1 to the backend static folder
COPY --from=frontend-build /frontend/build /app/static
    
# Expose the FastAPI port
EXPOSE 8000
    
# ----------------------
# Stage 3: Run the FastAPI app
# ----------------------

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    