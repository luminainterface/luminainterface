import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import json
from lumina_core.api.main import app

client = TestClient(app)

@pytest.fixture
def mock_ollama():
    with patch("lumina_core.llm.ollama_bridge.OllamaBridge") as mock:
        mock_instance = mock.return_value
        mock_instance.generate_stream = AsyncMock()
        mock_instance.generate_stream.return_value = [
            {"response": "Hello"},
            {"response": " there"},
            {"response": "!"}
        ]
        mock_instance.tokens_used = 10
        yield mock_instance

@pytest.fixture
def mock_qdrant():
    with patch("lumina_core.memory.qdrant_store.QdrantStore") as mock:
        mock_instance = mock.return_value
        mock_instance.get_similar_messages = AsyncMock()
        mock_instance.get_similar_messages.return_value = []
        mock_instance.upsert_messages = AsyncMock()
        mock_instance.get_metrics = AsyncMock()
        mock_instance.get_metrics.return_value = {"vectors": 10, "conversations": 5}
        mock_instance.encoder.encode = AsyncMock()
        mock_instance.encoder.encode.return_value = [0.1, 0.2, 0.3]
        yield mock_instance

def test_chat_endpoint(mock_ollama, mock_qdrant):
    response = client.post(
        "/chat",
        json={"message": "Hello, world!"}
    )
    assert response.status_code == 200
    
    # Check streaming response
    chunks = []
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8').replace('data: ', ''))
            if 'chunk' in data:
                chunks.append(data['chunk'])
            elif 'done' in data:
                break
    
    assert ''.join(chunks) == "Hello there!"
    mock_ollama.generate_stream.assert_called_once()
    mock_qdrant.upsert_messages.assert_called_once()

def test_metrics_endpoint(mock_qdrant, mock_ollama):
    response = client.get("/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "conversations" in data
    assert "tokens_used" in data
    assert "qdrant_vectors" in data
    assert data["qdrant_vectors"] == 10
    assert data["conversations"] == 5
    assert data["tokens_used"] == 10

def test_health_check_all_ok(mock_ollama, mock_qdrant):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert all(status == "ok" for status in data["services"].values())

def test_health_check_ollama_error(mock_ollama, mock_qdrant):
    mock_ollama.generate_stream.side_effect = Exception("Ollama error")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["services"]["ollama"] == "error"
    assert data["services"]["qdrant"] == "ok"
    assert data["services"]["embeddings"] == "ok"

def test_health_check_qdrant_error(mock_ollama, mock_qdrant):
    mock_qdrant.get_metrics.side_effect = Exception("Qdrant error")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["services"]["ollama"] == "ok"
    assert data["services"]["qdrant"] == "error"
    assert data["services"]["embeddings"] == "ok"

def test_health_check_embeddings_error(mock_ollama, mock_qdrant):
    mock_qdrant.encoder.encode.side_effect = Exception("Embeddings error")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["services"]["ollama"] == "ok"
    assert data["services"]["qdrant"] == "ok"
    assert data["services"]["embeddings"] == "error" 