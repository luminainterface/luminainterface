import pytest
import asyncio
from typing import Generator
from unittest.mock import MagicMock, patch

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_redis():
    """Create a mock Redis client."""
    mock_redis = MagicMock()
    mock_redis.get = MagicMock(return_value=None)
    mock_redis.set = MagicMock(return_value=True)
    mock_redis.delete = MagicMock(return_value=True)
    mock_redis.flushdb = MagicMock(return_value=True)
    return mock_redis

@pytest.fixture
def test_qdrant():
    """Create a mock Qdrant client."""
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection = MagicMock(return_value=None)
    mock_qdrant.create_collection = MagicMock(return_value=True)
    mock_qdrant.delete_collection = MagicMock(return_value=True)
    mock_qdrant.upsert = MagicMock(return_value=True)
    mock_qdrant.search = MagicMock(return_value=[])
    return mock_qdrant

@pytest.fixture(autouse=True)
async def setup_teardown(test_redis, test_qdrant):
    """Setup and teardown for each test."""
    # Mock the Redis and Qdrant clients in the main module
    with patch('graph_api.main.redis_client', test_redis), \
         patch('graph_api.main.qdrant_client', test_qdrant):
        yield 