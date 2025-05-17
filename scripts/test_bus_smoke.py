#!/usr/bin/env python3
"""Smoke test for the Redis Stream bus helper.

This script demonstrates basic publish/consume functionality and can be used
to verify the bus is working correctly in a development environment.

Usage:
    python scripts/test_bus_smoke.py

Requirements:
    - Redis server running on localhost:6379
    - lumina_core package installed
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import List

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lumina_core.common.bus import BusClient
from lumina_core.common.stream_message import StreamMessage

# Test configuration
TEST_STREAM = "test:smoke"
TEST_GROUP = "smoke-group"
TEST_CONSUMER = "smoke-consumer"
TEST_REDIS_URL = "redis://localhost:6379"

async def run_smoke_test():
    """Run the smoke test."""
    print("🚀 Starting bus smoke test...")
    
    # Create bus client
    bus = BusClient(redis_url=TEST_REDIS_URL)
    await bus.connect()
    
    try:
        # Track received messages
        received: List[StreamMessage] = []
        
        async def message_handler(msg: StreamMessage):
            """Handle received messages."""
            received.append(msg)
            print(f"📥 Received message: {msg.data}")
        
        # Start consumer
        print(f"👂 Starting consumer on {TEST_STREAM}...")
        consumer_task = asyncio.create_task(
            bus.consume(
                TEST_STREAM,
                TEST_GROUP,
                TEST_CONSUMER,
                message_handler,
                block_ms=1000
            )
        )
        
        try:
            # Publish test messages
            test_messages = [
                {"type": "greeting", "text": "Hello, Stream!"},
                {"type": "number", "value": 42},
                {"type": "nested", "data": {"key": "value"}}
            ]
            
            print(f"📤 Publishing {len(test_messages)} messages...")
            for msg in test_messages:
                msg_id = await bus.publish(TEST_STREAM, msg)
                print(f"  Published to {msg_id}")
            
            # Wait for consumer to process
            print("⏳ Waiting for messages to be processed...")
            await asyncio.sleep(2)
            
            # Verify results
            print("\n📊 Results:")
            print(f"  Messages published: {len(test_messages)}")
            print(f"  Messages received: {len(received)}")
            
            if len(received) == len(test_messages):
                print("✅ Test passed! All messages were received.")
            else:
                print("❌ Test failed! Not all messages were received.")
                sys.exit(1)
            
            # Show stream info
            info = await bus.get_stream_info(TEST_STREAM)
            print("\n📈 Stream Info:")
            print(f"  Length: {info['length']}")
            print(f"  Groups: {len(info['groups'])}")
            
        finally:
            # Cleanup
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
            
    finally:
        await bus.close()

if __name__ == "__main__":
    try:
        asyncio.run(run_smoke_test())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1) 