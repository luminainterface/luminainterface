import json
import asyncio
import time
import pytest
import httpx
import respx
from unittest.mock import AsyncMock, patch
from starlette.testclient import TestClient
from importlib import reload

# Import the FastAPI app
from lumina_core.masterchat.main import app, redis_client

# Mock Redis client
@pytest.fixture(autouse=True)
def mock_redis():
    with patch('lumina_core.masterchat.main.redis_client') as mock:
        mock.xadd = AsyncMock(return_value="mocked_id")
        mock.xgroup_create = AsyncMock()
        mock.xreadgroup = AsyncMock(return_value=[])
        mock.xack = AsyncMock()
        yield mock

# ---------- Fixtures ----------
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# ---------- Tests ----------
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["planner_alive"] is True

def test_tasks_endpoint(client, mock_redis):
    # Test task submission
    task_data = {
        "crawl": ["test_node"],
        "hops": 0,
        "max_nodes": 5
    }
    r = client.post("/tasks", json=task_data)
    assert r.status_code == 200
    response = r.json()
    assert response["status"] == "ok"
    assert response["task"]["crawl"] == task_data["crawl"]
    assert response["task"]["hops"] == task_data["hops"]
    assert response["task"]["max_nodes"] == task_data["max_nodes"]
    
    # Verify Redis was called
    mock_redis.xadd.assert_called_once()

def test_invalid_task_data(client):
    # Test with invalid task data
    invalid_data = {
        "crawl": "not_a_list",  # Should be a list
        "hops": "not_an_int",   # Should be an integer
        "max_nodes": "invalid"   # Should be an integer
    }
    r = client.post("/tasks", json=invalid_data)
    assert r.status_code == 422  # Validation error

def test_planner_logs_endpoint(client):
    r = client.get("/planner/logs")
    assert r.status_code == 200
    # The response should be a stream of SSE events
    assert "text/event-stream" in r.headers["content-type"]

def test_chat_endpoint(client):
    chat_data = {
        "message": "test message"
    }
    r = client.post("/masterchat/chat", json=chat_data)
    # Since LLM planner is not implemented yet, we expect a 501
    assert r.status_code == 501
    assert "LLM planner not implemented yet" in r.json()["detail"]

def test_task_with_default_values(client, mock_redis):
    # Test task submission with only required fields
    task_data = {
        "crawl": ["test_node"]
    }
    r = client.post("/tasks", json=task_data)
    assert r.status_code == 200
    response = r.json()
    assert response["status"] == "ok"
    assert response["task"]["crawl"] == task_data["crawl"]
    assert response["task"]["hops"] == 0  # Default value
    assert response["task"]["max_nodes"] == 5  # Default value

def test_task_with_custom_values(client, mock_redis):
    # Test task submission with custom values
    task_data = {
        "crawl": ["test_node"],
        "hops": 2,
        "max_nodes": 10
    }
    r = client.post("/tasks", json=task_data)
    assert r.status_code == 200
    response = r.json()
    assert response["status"] == "ok"
    assert response["task"]["crawl"] == task_data["crawl"]
    assert response["task"]["hops"] == task_data["hops"]
    assert response["task"]["max_nodes"] == task_data["max_nodes"]

def test_task_with_empty_crawl_list(client):
    # Test task submission with empty crawl list
    task_data = {
        "crawl": []
    }
    r = client.post("/tasks", json=task_data)
    assert r.status_code == 422  # Validation error

def test_task_with_negative_values(client):
    # Test task submission with negative values
    task_data = {
        "crawl": ["test_node"],
        "hops": -1,
        "max_nodes": -5
    }
    r = client.post("/tasks", json=task_data)
    assert r.status_code == 422  # Validation error 