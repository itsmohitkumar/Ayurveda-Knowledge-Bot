import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.chatbot.main import app 

# Mocking a document class for the test
class MockDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

@pytest.fixture
def client():
    return TestClient(app)

def test_create_index_success(client):
    with patch('app.PDFDocumentProcessor') as mock_processor:
        # Mock the load_and_chunk_documents method to return mock data
        mock_processor.return_value.load_and_chunk_documents.return_value = [
            MockDocument("chunk1", metadata={"source": "test.pdf"}),
            MockDocument("chunk2", metadata={"source": "test.pdf"})
        ]
        
        # Make a request to the /create_index endpoint
        response = client.post("/create_index")
        
        # Assert the response status code and message
        assert response.status_code == 200
        assert response.json() == {"message": "FAISS index created successfully."}