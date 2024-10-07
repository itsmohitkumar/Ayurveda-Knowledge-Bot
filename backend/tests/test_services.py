import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from src.chatbot.services import PDFDocumentProcessor, FAISSManager, LLMService

DATA_DIRECTORY = "test_data"
INDEX_PATH = "test_index"
MODEL_ID = "test_model_id"
CLIENT = MagicMock()

@pytest.fixture
def pdf_processor():
    return PDFDocumentProcessor(data_directory=DATA_DIRECTORY)

@pytest.fixture
def faiss_manager():
    embeddings = MagicMock()
    return FAISSManager(index_path=INDEX_PATH, embeddings=embeddings)

@pytest.fixture
def llm_service():
    return LLMService(model_id=MODEL_ID, client=CLIENT)

def test_load_and_chunk_documents_success(pdf_processor):
    with patch('src.chatbot.services.PyPDFDirectoryLoader') as mock_loader:
        mock_loader.return_value.load.return_value = ["doc1", "doc2"]
        with patch('src.chatbot.services.RecursiveCharacterTextSplitter') as mock_splitter:
            mock_splitter.return_value.split_documents.side_effect = lambda docs: docs
            
            result = pdf_processor.load_and_chunk_documents()
            assert result == ["doc1", "doc2"]

def test_load_and_chunk_documents_file_not_found(pdf_processor):
    with patch('src.chatbot.services.PyPDFDirectoryLoader', side_effect=FileNotFoundError):
        with pytest.raises(HTTPException) as excinfo:
            pdf_processor.load_and_chunk_documents()
        assert excinfo.value.status_code == 404

def test_create_and_save_vector_store_success(faiss_manager):
    with patch('src.chatbot.services.FAISS.from_documents') as mock_from_docs:
        mock_vectorstore = MagicMock()
        mock_from_docs.return_value = mock_vectorstore
        
        faiss_manager.create_and_save_vector_store(["chunk1", "chunk2"])
        mock_vectorstore.save_local.assert_called_once_with(INDEX_PATH)

def test_load_vector_store_success(faiss_manager):
    with patch('src.chatbot.services.FAISS.load_local') as mock_load:
        mock_load.return_value = MagicMock()
        result = faiss_manager.load_vector_store()
        assert result is not None

def test_initialize_llm_success(llm_service):
    with patch('src.chatbot.services.BedrockLLM'):
        llm_service.initialize_llm()

def test_generate_response_success(llm_service):
    llm = MagicMock()
    vectorstore_faiss = MagicMock()
    with patch('src.chatbot.services.RetrievalQA.from_chain_type') as mock_from_chain:
        mock_from_chain.return_value.invoke.return_value = {'result': 'response text'}
        response = llm_service.generate_response(llm, vectorstore_faiss, "query")
        assert response == 'response text'

def test_generate_response_error(llm_service):
    llm = MagicMock()
    vectorstore_faiss = MagicMock()
    with patch('src.chatbot.services.RetrievalQA.from_chain_type', side_effect=Exception):
        with pytest.raises(HTTPException):
            llm_service.generate_response(llm, vectorstore_faiss, "query")

if __name__ == "__main__":
    pytest.main()
