import pytest
from fastapi.testclient import TestClient
import os
from lumina_core.api.main import app

client = TestClient(app)

def test_health_check_no_auth():
    """Health check should be public."""
    response = client.get("/health")
    assert response.status_code == 200

def test_protected_endpoint_no_key():
    """Protected endpoints should require API key."""
    response = client.post("/chat", json={"message": "test"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"

def test_protected_endpoint_invalid_key():
    """Invalid API key should be rejected."""
    response = client.post(
        "/chat",
        json={"message": "test"},
        headers={"X-API-Key": "invalid-key"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"

def test_protected_endpoint_valid_key():
    """Valid API key should be accepted."""
    # Set test API key
    os.environ["LUMINA_API_KEY"] = "test-key"
    
    response = client.post(
        "/chat",
        json={"message": "test"},
        headers={"X-API-Key": "test-key"}
    )
    assert response.status_code == 200
    
    # Clean up
    del os.environ["LUMINA_API_KEY"]

def test_metrics_endpoint_auth():
    """Metrics endpoint should require API key."""
    response = client.get("/metrics/summary")
    assert response.status_code == 401
    
    # Set test API key
    os.environ["LUMINA_API_KEY"] = "test-key"
    
    response = client.get(
        "/metrics/summary",
        headers={"X-API-Key": "test-key"}
    )
    assert response.status_code == 200
    
    # Clean up
    del os.environ["LUMINA_API_KEY"] 