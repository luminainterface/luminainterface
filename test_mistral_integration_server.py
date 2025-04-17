#!/usr/bin/env python3
"""
Mistral Integration Server Test Script

This script tests the MistralIntegrationServer by:
1. Starting the server
2. Connecting to it with a WebSocket client
3. Sending test messages
4. Displaying responses
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import websockets
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestClient")

# Import local modules
try:
    from src.api.mistral_integration_server import MistralIntegrationServer
    SERVER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing server module: {e}")
    SERVER_AVAILABLE = False


async def test_client(host: str = "localhost", port: int = 8765):
    """
    Test WebSocket client for the Mistral Integration Server
    
    Args:
        host: Server host
        port: Server port
    """
    # WebSocket URI
    uri = f"ws://{host}:{port}"
    logger.info(f"Connecting to {uri}")
    
    # Connect to server
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to server")
            
            # Send client info
            await send_message(websocket, {
                "type": "client_info",
                "client_type": "test_client",
                "version": "1.0.0"
            })
            
            # Receive welcome message
            welcome = await recv_message(websocket)
            logger.info(f"Received welcome: {welcome}")
            
            # Check if Mistral is available
            mistral_available = welcome.get("system_info", {}).get("mistral_available", False)
            if not mistral_available:
                logger.warning("Mistral system is not available")
            
            # Send ping
            await send_message(websocket, {
                "type": "ping",
                "ping_id": "test-ping-1"
            })
            
            # Receive pong
            pong = await recv_message(websocket)
            logger.info(f"Received pong: {pong}")
            
            # Send chat message if Mistral is available
            if mistral_available:
                logger.info("Sending test chat message")
                await send_message(websocket, {
                    "type": "chat_message",
                    "message": "Hello, Mistral! What can you do?",
                    "message_id": "test-message-1"
                })
                
                # May receive processing started message
                processing = await recv_message(websocket)
                logger.info(f"Received processing notification: {processing}")
                
                # Receive chat response
                response = await recv_message(websocket)
                logger.info(f"Received response: {response}")
                
                # Get system stats
                await send_message(websocket, {
                    "type": "get_stats"
                })
                
                # Receive stats
                stats = await recv_message(websocket)
                logger.info(f"Received stats: {stats}")
            
            logger.info("Test completed successfully")
    except Exception as e:
        logger.error(f"Error in WebSocket client: {e}")


async def send_message(websocket, message: Dict[str, Any]):
    """Send a message to the server"""
    await websocket.send(json.dumps(message))
    logger.debug(f"Sent: {message}")


async def recv_message(websocket) -> Dict[str, Any]:
    """Receive a message from the server"""
    message = await websocket.recv()
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        logger.warning(f"Received non-JSON message: {message}")
        return {"type": "error", "message": "Failed to parse server response"}


async def run_test():
    """Run the integration test"""
    # Create server instance
    if not SERVER_AVAILABLE:
        logger.error("Server module not available")
        return False
    
    try:
        # Create server instance
        server = MistralIntegrationServer(
            host="localhost",
            port=8765,
            api_key=os.getenv("MISTRAL_API_KEY", "")
        )
        
        # Start server
        logger.info("Starting server...")
        server_task = asyncio.create_task(server.start())
        
        # Wait for server to start
        await asyncio.sleep(1)
        
        # Run test client
        logger.info("Starting test client...")
        await test_client()
        
        # Stop server
        logger.info("Stopping server...")
        await server.stop()
        
        # Done
        logger.info("Test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in test: {e}")
        return False


if __name__ == "__main__":
    # Make sure API key is set
    if not os.getenv("MISTRAL_API_KEY"):
        print("Warning: MISTRAL_API_KEY environment variable not set.")
        print("The test will run but Mistral features will be unavailable.")
    
    # Run the test
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1) 