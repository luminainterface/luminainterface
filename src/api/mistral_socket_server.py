#!/usr/bin/env python3
"""
WebSocket Server for Mistral Integration

This module provides a WebSocket server that integrates the Mistral AI system
with a frontend interface, enabling real-time communication.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MistralSocketServer")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import Mistral integration
try:
    from src.api.mistral_integration_fixed import MistralIntegration
    logger.info("Mistral Integration imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Mistral Integration: {e}")
    sys.exit(1)


class MistralSocketServer:
    """
    WebSocket server for Mistral Integration
    
    This class provides a WebSocket interface to the Mistral Integration system,
    allowing frontend applications to communicate with the system in real-time.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        api_key: Optional[str] = None,
        model: str = "mistral-large-latest"
    ):
        """
        Initialize the WebSocket server
        
        Args:
            host: Hostname to bind the server to
            port: Port to bind the server to
            api_key: Mistral API key (uses MISTRAL_API_KEY env var if not provided)
            model: Mistral model to use
        """
        self.host = host
        self.port = port
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        
        # Initialize Mistral integration
        self.mistral = MistralIntegration(
            api_key=self.api_key,
            model=self.model
        )
        
        # Track active connections
        self.active_connections: Set[WebSocketServerProtocol] = set()
        
        # Server state
        self.running = False
        self.server = None
    
    async def handler(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """
        Handle WebSocket connections
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        # Register connection
        self.active_connections.add(websocket)
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected from {websocket.remote_address}")
        
        try:
            # Send initial system information
            await self._send_system_info(websocket)
            
            # Handle messages
            async for message in websocket:
                try:
                    await self._process_message(websocket, message)
                except Exception as e:
                    logger.error(f"Error processing message from client {client_id}: {e}")
                    await self._send_error(websocket, f"Error processing message: {str(e)}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Client {client_id} disconnected: {e}")
        finally:
            # Unregister connection
            self.active_connections.remove(websocket)
    
    async def _process_message(self, websocket: WebSocketServerProtocol, message: str) -> None:
        """
        Process a message from a client
        
        Args:
            websocket: WebSocket connection
            message: Client message as JSON string
        """
        try:
            # Parse message
            data = json.loads(message)
            message_type = data.get("type", "")
            
            if message_type == "query":
                # Process query through Mistral integration
                user_message = data.get("message", "")
                if not user_message:
                    await self._send_error(websocket, "Empty message")
                    return
                
                # Process the message
                logger.info(f"Processing query: {user_message[:50]}...")
                result = self.mistral.process_message(user_message)
                
                # Send response
                response = {
                    "type": "response",
                    "input": result["input"],
                    "response": result["combined_response"] or result["llm_response"] or "No response generated",
                    "has_error": result["error"] is not None,
                    "error": result["error"]
                }
                await websocket.send(json.dumps(response))
            
            elif message_type == "adjust_weights":
                # Adjust weights
                llm_weight = data.get("llm_weight")
                nn_weight = data.get("nn_weight")
                
                if llm_weight is None and nn_weight is None:
                    await self._send_error(websocket, "No weights provided")
                    return
                
                # Adjust weights
                new_weights = self.mistral.adjust_weights(
                    llm_weight=llm_weight,
                    nn_weight=nn_weight
                )
                
                # Send response
                response = {
                    "type": "weights_updated",
                    "weights": new_weights
                }
                await websocket.send(json.dumps(response))
            
            elif message_type == "get_system_info":
                # Send system information
                await self._send_system_info(websocket)
            
            else:
                # Unknown message type
                await self._send_error(websocket, f"Unknown message type: {message_type}")
        
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON")
    
    async def _send_system_info(self, websocket: WebSocketServerProtocol) -> None:
        """
        Send system information to a client
        
        Args:
            websocket: WebSocket connection
        """
        stats = self.mistral.get_system_stats()
        info = {
            "type": "system_info",
            "mistral_available": stats["mistral_available"],
            "processor_available": stats["processor_available"],
            "model": stats["model"],
            "weights": stats["weights"]
        }
        await websocket.send(json.dumps(info))
    
    async def _send_error(self, websocket: WebSocketServerProtocol, message: str) -> None:
        """
        Send an error message to a client
        
        Args:
            websocket: WebSocket connection
            message: Error message
        """
        error = {
            "type": "error",
            "message": message
        }
        await websocket.send(json.dumps(error))
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcast a message to all connected clients
        
        Args:
            message: Message to broadcast
        """
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        await asyncio.gather(*[
            client.send(message_json)
            for client in self.active_connections
        ])
    
    async def start(self) -> None:
        """Start the WebSocket server"""
        self.running = True
        self.server = await websockets.serve(self.handler, self.host, self.port)
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        
        # Keep server running
        await self.server.wait_closed()
    
    def stop(self) -> None:
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            self.running = False
            logger.info("WebSocket server stopped")


async def main():
    """Run the WebSocket server"""
    # Get API key from environment
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.warning("MISTRAL_API_KEY environment variable not set")
    
    # Create and start server
    server = MistralSocketServer(api_key=api_key)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    finally:
        server.stop()


if __name__ == "__main__":
    asyncio.run(main()) 