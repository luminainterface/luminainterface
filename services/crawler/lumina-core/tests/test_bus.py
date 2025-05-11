"""Tests for the Redis Stream bus helper."""

import pytest
import asyncio
import json
from datetime import datetime
from typing import List, Dict
import redis.asyncio as aioredis
from lumina_core.common.bus import BusClient, StreamMessage

# Test configuration
TEST_STREAM = "test:bus"
TEST_GROUP = "test-group"
TEST_CONSUMER = "test-consumer"
TEST_REDIS_URL = "redis://localhost:6379"

@pytest.fixture
async def redis_client():
    """Create a Redis client for testing."""
    client = aioredis.from_url(TEST_REDIS_URL, decode_responses=True)
    yield client
    # Cleanup
    await client.delete(TEST_STREAM)
    await client.close()

@pytest.fixture
async def bus_client():
    """Create a bus client for testing."""
    client = BusClient(redis_url=TEST_REDIS_URL)
    await client.connect()
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_publish_and_consume(bus_client: BusClient, redis_client: aioredis.Redis):
    """Test basic publish and consume functionality."""
    # Test data
    test_data = {
        "test": "data",
        "number": 42,
        "nested": {"key": "value"}
    }
    
    # Track received messages
    received_messages: List[StreamMessage] = []
    
    async def message_handler(msg: StreamMessage):
        received_messages.append(msg)
    
    # Start consumer
    consumer_task = asyncio.create_task(
        bus_client.consume(
            TEST_STREAM,
            TEST_GROUP,
            TEST_CONSUMER,
            message_handler,
            block_ms=1000
        )
    )
    
    try:
        # Publish message
        msg_id = await bus_client.publish(TEST_STREAM, test_data)
        assert msg_id is not None
        
        # Wait for consumer to process
        await asyncio.sleep(1)
        
        # Verify message was received
        assert len(received_messages) == 1
        msg = received_messages[0]
        assert msg.stream == TEST_STREAM
        assert msg.data == test_data
        assert "timestamp" in msg.data
        
        # Verify message was acknowledged
        pending = await redis_client.xpending(TEST_STREAM, TEST_GROUP)
        assert len(pending) == 0
        
    finally:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

@pytest.mark.asyncio
async def test_retry_logic(bus_client: BusClient, redis_client: aioredis.Redis):
    """Test retry logic for connection issues."""
    # Temporarily break Redis connection
    await redis_client.close()
    
    # Attempt publish (should retry)
    with pytest.raises(Exception) as exc_info:
        await bus_client.publish(TEST_STREAM, {"test": "retry"})
    assert "Max retries exceeded" in str(exc_info.value)
    
    # Restore Redis
    await redis_client.ping()

@pytest.mark.asyncio
async def test_stream_info(bus_client: BusClient, redis_client: aioredis.Redis):
    """Test stream information retrieval."""
    # Publish some messages
    for i in range(3):
        await bus_client.publish(TEST_STREAM, {"count": i})
    
    # Get stream info
    info = await bus_client.get_stream_info(TEST_STREAM)
    
    assert info["length"] == 3
    assert "last_generated_id" in info
    assert "groups" in info

@pytest.mark.asyncio
async def test_consumer_lag(bus_client: BusClient, redis_client: aioredis.Redis):
    """Test consumer lag monitoring."""
    # Publish message
    await bus_client.publish(TEST_STREAM, {"test": "lag"})
    
    # Create consumer group but don't consume
    try:
        await redis_client.xgroup_create(TEST_STREAM, TEST_GROUP, id="0", mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
    
    # Check lag
    lag = await bus_client.get_consumer_lag(TEST_STREAM, TEST_GROUP)
    assert lag == 1  # One unprocessed message

@pytest.mark.asyncio
async def test_maxlen_trimming(bus_client: BusClient, redis_client: aioredis.Redis):
    """Test stream length trimming."""
    maxlen = 3
    
    # Publish more messages than maxlen
    for i in range(5):
        await bus_client.publish(TEST_STREAM, {"count": i}, maxlen=maxlen)
    
    # Verify stream length
    info = await bus_client.get_stream_info(TEST_STREAM)
    assert info["length"] == maxlen
    
    # Verify oldest messages were trimmed
    entries = await redis_client.xrange(TEST_STREAM, "-", "+")
    assert len(entries) == maxlen
    # Verify we kept the newest messages
    assert json.loads(entries[0][1]["data"])["count"] == 2
    assert json.loads(entries[-1][1]["data"])["count"] == 4

@pytest.mark.asyncio
async def test_error_handling(bus_client: BusClient, redis_client: aioredis.Redis):
    """Test error handling in consumer."""
    # Track errors
    errors = []
    
    async def error_handler(msg: StreamMessage):
        if "error" in msg.data:
            raise ValueError("Test error")
        errors.append(msg)
    
    # Start consumer
    consumer_task = asyncio.create_task(
        bus_client.consume(
            TEST_STREAM,
            TEST_GROUP,
            TEST_CONSUMER,
            error_handler,
            block_ms=1000
        )
    )
    
    try:
        # Publish valid message
        await bus_client.publish(TEST_STREAM, {"valid": True})
        
        # Publish message that will cause error
        await bus_client.publish(TEST_STREAM, {"error": True})
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Verify valid message was processed
        assert len(errors) == 1
        assert errors[0].data["valid"] is True
        
        # Verify error message is still pending
        pending = await redis_client.xpending(TEST_STREAM, TEST_GROUP)
        assert len(pending) == 1
        
    finally:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 