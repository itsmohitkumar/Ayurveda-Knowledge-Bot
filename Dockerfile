# Stage 1: Build the React app
FROM node:18-alpine AS build-frontend

# Set working directory for React app
WORKDIR /app/frontend-react

# Copy the React app package.json and install dependencies
COPY frontend-react/package.json frontend-react/package-lock.json ./
RUN npm install

# Copy the rest of the React app files and build the app
COPY frontend-react/ ./
RUN npm run build

# Stage 2: Set up the FastAPI backend with Python 3.11.6
FROM python:3.11.6-slim AS build-backend

# Set working directory for FastAPI backend
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI source code
COPY app.py ./
COPY src/ ./src/

# Copy the built React app from the first stage to the FastAPI static folder
COPY --from=build-frontend /app/frontend-react/build ./frontend_build

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
