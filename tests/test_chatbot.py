import pytest
from fastapi.testclient import TestClient
from app import app  # Use the absolute import here

client = TestClient(app)

def test_ask_question():
    response = client.post("/ask", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    assert "answer" in response.json()  # Adjust based on the expected response
