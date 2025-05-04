import pytest
from fastapi.testclient import TestClient
import time
import os

from lumina_core.api.main import app

client = TestClient(app)

# Test API key
TEST_API_KEY = "test-key-123"
os.environ["LUMINA_API_KEY"] = TEST_API_KEY

def test_chat_rate_limit():
    """Test that chat endpoint respects rate limits."""
    # Make 10 requests (should all succeed)
    for _ in range(10):
        response = client.post(
            "/chat",
            json={"message": "test"},
            headers={"X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 200
    
    # 11th request should be rate limited
    response = client.post(
        "/chat",
        json={"message": "test"},
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 429
    assert "Too many requests" in response.text

def test_admin_rate_limit():
    """Test that admin endpoints respect rate limits."""
    # Make 5 requests (should all succeed)
    for _ in range(5):
        response = client.post(
            "/admin/prune",
            headers={"X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 200
    
    # 6th request should be rate limited
    response = client.post(
        "/admin/prune",
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 429
    assert "Too many requests" in response.text

def test_metrics_rate_limit():
    """Test that metrics endpoint respects rate limits."""
    # Make 30 requests (should all succeed)
    for _ in range(30):
        response = client.get(
            "/metrics/summary",
            headers={"X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 200
    
    # 31st request should be rate limited
    response = client.get(
        "/metrics/summary",
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 429
    assert "Too many requests" in response.text

def test_rate_limit_reset():
    """Test that rate limits reset after the time window."""
    # Make 10 requests to hit the limit
    for _ in range(10):
        response = client.post(
            "/chat",
            json={"message": "test"},
            headers={"X-API-Key": TEST_API_KEY}
        )
        assert response.status_code == 200
    
    # Wait for rate limit window to expire
    time.sleep(61)
    
    # Should be able to make requests again
    response = client.post(
        "/chat",
        json={"message": "test"},
        headers={"X-API-Key": TEST_API_KEY}
    )
    assert response.status_code == 200

def test_different_api_keys():
    """Test that rate limits are tracked per API key."""
    # Make 10 requests with first key
    for _ in range(10):
        response = client.post(
            "/chat",
            json={"message": "test"},
            headers={"X-API-Key": "key1"}
        )
        assert response.status_code == 200
    
    # Should be able to make requests with different key
    response = client.post(
        "/chat",
        json={"message": "test"},
        headers={"X-API-Key": "key2"}
    )
    assert response.status_code == 200 