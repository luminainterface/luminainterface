import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import json
import time
import statistics
import psutil
import gc
from lumina_core.api.main import app
import aiohttp
import os

client = TestClient(app)

class PerformanceMetrics:
    def __init__(self):
        self.latencies = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_memory = psutil.Process().memory_info().rss
        self.peak_memory = self.start_memory

    def add_latency(self, latency_ms):
        self.latencies.append(latency_ms)
        self.peak_memory = max(self.peak_memory, psutil.Process().memory_info().rss)

    def add_cache_hit(self):
        self.cache_hits += 1

    def add_cache_miss(self):
        self.cache_misses += 1

    @property
    def p50_latency(self):
        return statistics.median(self.latencies) if self.latencies else 0

    @property
    def p95_latency(self):
        return statistics.quantiles(self.latencies, n=20)[18] if self.latencies else 0

    @property
    def cache_hit_rate(self):
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0

    @property
    def memory_delta_mb(self):
        return (self.peak_memory - self.start_memory) / (1024 * 1024)

    def print_summary(self):
        print(f"\nPerformance Summary:")
        print(f"  Latency (ms):")
        print(f"    P50: {self.p50_latency:.2f}")
        print(f"    P95: {self.p95_latency:.2f}")
        print(f"  Cache:")
        print(f"    Hit Rate: {self.cache_hit_rate:.1f}%")
        print(f"    Hits: {self.cache_hits}")
        print(f"    Misses: {self.cache_misses}")
        print(f"  Memory:")
        print(f"    Delta: {self.memory_delta_mb:.1f} MB")

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

async def make_streaming_request(session, prompt, metrics):
    """Helper to make a streaming request with timing."""
    start_time = time.time()
    async with session.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "phi2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
    ) as response:
        assert response.status == 200
        chunks = []
        async for line in response.content:
            if line:
                data = json.loads(line.decode('utf-8').replace('data: ', ''))
                if data == "[DONE]":
                    break
                if "choices" in data:
                    chunks.append(data["choices"][0]["delta"].get("content", ""))
        
        latency_ms = (time.time() - start_time) * 1000
        metrics.add_latency(latency_ms)
        return ''.join(chunks)

@pytest.mark.asyncio
async def test_concurrent_streams(mock_ollama, mock_qdrant, mock_redis):
    """Test multiple concurrent streaming requests with performance metrics."""
    num_requests = 10
    prompts = [f"Test prompt {i}" for i in range(num_requests)]
    metrics = PerformanceMetrics()
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_streaming_request(session, prompt, metrics) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        # Verify all requests completed
        assert len(responses) == num_requests
        assert all(r == "Test response" for r in responses)
        
        # Verify Redis was called for each request
        assert mock_redis.get.call_count == num_requests
        assert mock_redis.set.call_count == num_requests  # Each prompt is unique
        
        # Print performance summary
        metrics.print_summary()

@pytest.mark.asyncio
async def test_redis_failure_fallback(mock_ollama, mock_qdrant):
    """Test graceful degradation when Redis is unavailable."""
    # Simulate Redis failure by unsetting REDIS_URL
    original_redis_url = os.environ.get("REDIS_URL")
    try:
        if "REDIS_URL" in os.environ:
            del os.environ["REDIS_URL"]
        
        # Make a request - should fall back to LRU cache
        async with aiohttp.ClientSession() as session:
            response = await make_streaming_request(session, "Test prompt", PerformanceMetrics())
            assert response == "Test response"
        
        # Check health endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["services"]["redis"] == "down"
        assert health_data["status"] == "degraded"
        
    finally:
        # Restore original Redis URL
        if original_redis_url:
            os.environ["REDIS_URL"] = original_redis_url

@pytest.mark.asyncio
async def test_concurrent_cache_contention(mock_ollama, mock_qdrant, mock_redis):
    """Test cache behavior under concurrent load with same prompts."""
    num_requests = 10
    prompt = "Same prompt for all requests"
    metrics = PerformanceMetrics()
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_streaming_request(session, prompt, metrics) for _ in range(num_requests)]
        responses = await asyncio.gather(*tasks)
        
        # Verify all requests completed
        assert len(responses) == num_requests
        assert all(r == "Test response" for r in responses)
        
        # Verify Redis was called for each request
        assert mock_redis.get.call_count == num_requests
        # But only one set operation (first request)
        assert mock_redis.set.call_count == 1
        
        # Track cache hits/misses
        for _ in range(num_requests - 1):
            metrics.add_cache_hit()
        metrics.add_cache_miss()
        
        # Print performance summary
        metrics.print_summary()

@pytest.mark.asyncio
async def test_memory_usage(mock_ollama, mock_qdrant, mock_redis):
    """Test memory usage under load."""
    num_requests = 20
    prompts = [f"Test prompt {i}" for i in range(num_requests)]
    metrics = PerformanceMetrics()
    
    # Force garbage collection before test
    gc.collect()
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_streaming_request(session, prompt, metrics) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        # Verify all requests completed
        assert len(responses) == num_requests
        assert all(r == "Test response" for r in responses)
        
        # Print performance summary
        metrics.print_summary()
        
        # Verify memory usage is reasonable
        assert metrics.memory_delta_mb < 100  # Should use less than 100MB

@pytest.mark.asyncio
async def test_ollama_failure_handling(mock_ollama, mock_qdrant, mock_redis):
    """Test handling of Ollama service failures."""
    # Simulate Ollama failure
    mock_ollama.generate_stream.side_effect = Exception("Ollama service unavailable")
    
    async with aiohttp.ClientSession() as session:
        try:
            await make_streaming_request(session, "Test prompt", PerformanceMetrics())
            assert False, "Should have raised an exception"
        except aiohttp.ClientResponseError as e:
            assert e.status == 500
        
        # Check health endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["services"]["ollama"] == "error"
        assert health_data["status"] == "degraded"

@pytest.mark.asyncio
async def test_qdrant_failure_handling(mock_ollama, mock_qdrant, mock_redis):
    """Test handling of Qdrant service failures."""
    # Simulate Qdrant failure
    mock_qdrant.get_similar_messages.side_effect = Exception("Qdrant service unavailable")
    
    async with aiohttp.ClientSession() as session:
        try:
            await make_streaming_request(session, "Test prompt", PerformanceMetrics())
            assert False, "Should have raised an exception"
        except aiohttp.ClientResponseError as e:
            assert e.status == 500
        
        # Check health endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["services"]["qdrant"] == "error"
        assert health_data["status"] == "degraded" 