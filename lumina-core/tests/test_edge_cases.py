import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import json
import time
from lumina_core.api.main import app

client = TestClient(app)

@pytest.fixture
def mock_ollama():
    with patch("lumina_core.llm.ollama_bridge.OllamaBridge") as mock:
        mock_instance = mock.return_value
        mock_instance.generate_stream = AsyncMock()
        mock_instance.generate_stream.return_value = [
            {"response": "Test response"}
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
        yield mock_instance

@pytest.fixture
def mock_redis():
    with patch("redis.asyncio.Redis") as mock:
        mock_instance = mock.return_value
        mock_instance.get = AsyncMock(return_value=None)
        mock_instance.set = AsyncMock()
        mock_instance.flushdb = AsyncMock()
        yield mock_instance

def test_cache_hit_miss(mock_ollama, mock_qdrant, mock_redis):
    """Test embedding cache hit/miss behavior."""
    # First request (cache miss)
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi2",
            "messages": [{"role": "user", "content": "Test message"}]
        }
    )
    assert response1.status_code == 200
    
    # Second request with same content (should hit cache)
    response2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi2",
            "messages": [{"role": "user", "content": "Test message"}]
        }
    )
    assert response2.status_code == 200
    
    # Verify Redis was called for both requests
    assert mock_redis.get.call_count == 2
    assert mock_redis.set.call_count == 1  # Only set on cache miss

def test_streaming_vs_non_streaming(mock_ollama, mock_qdrant):
    """Test both streaming and non-streaming responses."""
    # Non-streaming
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi2",
            "messages": [{"role": "user", "content": "Test"}],
            "stream": False
        }
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["object"] == "chat.completion"
    assert "choices" in data1
    
    # Streaming
    response2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi2",
            "messages": [{"role": "user", "content": "Test"}],
            "stream": True
        }
    )
    assert response2.status_code == 200
    assert response2.headers["content-type"] == "text/event-stream"
    
    chunks = []
    for line in response2.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8').replace('data: ', ''))
            if data == "[DONE]":
                break
            if "choices" in data:
                chunks.append(data["choices"][0]["delta"].get("content", ""))
    
    assert ''.join(chunks) == "Test response"

def test_rate_limiting(mock_ollama, mock_qdrant):
    """Test rate limiting behavior."""
    # Make multiple requests in quick succession
    responses = []
    for _ in range(5):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "phi2",
                "messages": [{"role": "user", "content": "Test"}]
            }
        )
        responses.append(response)
    
    # At least one should be rate limited
    assert any(r.status_code == 429 for r in responses)

def test_large_prompt(mock_ollama, mock_qdrant):
    """Test handling of large prompts."""
    # Create a large prompt (>8k tokens)
    large_prompt = "Test " * 2000  # Approximately 8k tokens
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi2",
            "messages": [{"role": "user", "content": large_prompt}]
        }
    )
    
    # Should handle large prompts gracefully
    assert response.status_code in [200, 413]  # 413 if we implement size limits

def test_health_check_redis(mock_ollama, mock_qdrant, mock_redis):
    """Test health check includes Redis status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert "services" in data
    assert "redis" in data["services"]
    assert data["services"]["redis"] == "ok"
    assert "openai_shim" in data
    assert data["openai_shim"] is True

def test_error_handling(mock_ollama, mock_qdrant):
    """Test error handling for various scenarios."""
    # Test invalid model
    response1 = client.post(
        "/v1/chat/completions",
        json={
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Test"}]
        }
    )
    assert response1.status_code == 400
    
    # Test invalid message format
    response2 = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi2",
            "messages": [{"invalid": "format"}]
        }
    )
    assert response2.status_code == 422
    
    # Test empty messages
    response3 = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi2",
            "messages": []
        }
    )
    assert response3.status_code == 422 