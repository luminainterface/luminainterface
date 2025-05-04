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

def test_chat_completions(mock_ollama, mock_qdrant):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi2",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "phi2"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "Hello there!"
    assert "usage" in data

def test_chat_completions_streaming(mock_ollama, mock_qdrant):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi2",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "stream": True
        }
    )
    assert response.status_code == 200
    
    chunks = []
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8').replace('data: ', ''))
            if data == "[DONE]":
                break
            if "choices" in data:
                chunks.append(data["choices"][0]["delta"].get("content", ""))
    
    assert ''.join(chunks) == "Hello there!"

def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "phi2" 