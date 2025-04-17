#!/usr/bin/env python3
"""
Mistral WebSocket Server

This module provides a WebSocket server for the Mistral integration system,
allowing client applications to connect and interact with the system.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Callable, List, Optional, Set, Coroutine, Union

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

# Configure logging
logger = logging.getLogger("MistralWebSocketServer")

# Type definitions
MessageHandler = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]
ConnectHandler = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]
DisconnectHandler = Callable[[str], Coroutine[Any, Any, None]]


class MistralWebSocketServer:
    """
    Mistral WebSocket Server
    
    This class provides a WebSocket server that handles client connections,
    message routing, and event handling for the Mistral integration system.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize the WebSocket server
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.client_info: Dict[str, Dict[str, Any]] = {}
        self.message_handlers: Dict[str, List[MessageHandler]] = {}
        self.connect_handlers: List[ConnectHandler] = []
        self.disconnect_handlers: List[DisconnectHandler] = []
        self.server = None
        self.task = None
        self.running = False
    
    async def start(self):
        """Start the WebSocket server"""
        if self.running:
            logger.warning("Server is already running")
            return
        
        # Start server
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )
        self.running = True
    
    async def stop(self):
        """Stop the WebSocket server"""
        if not self.running:
            logger.warning("Server is not running")
            return
        
        # Close all client connections
        close_tasks = []
        for client_id, websocket in self.clients.items():
            logger.info(f"Closing connection to client {client_id}")
            close_tasks.append(websocket.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Clear client data
        self.clients.clear()
        self.client_info.clear()
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        
        self.running = False
        logger.info("WebSocket server stopped")
    
    def add_message_handler(self, message_type: str, handler: MessageHandler):
        """
        Add a message handler for a specific message type
        
        Args:
            message_type: Message type to handle
            handler: Handler function
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.debug(f"Added handler for message type: {message_type}")
    
    def add_connect_handler(self, handler: ConnectHandler):
        """
        Add a connect handler
        
        Args:
            handler: Handler function
        """
        self.connect_handlers.append(handler)
        logger.debug("Added connect handler")
    
    def add_disconnect_handler(self, handler: DisconnectHandler):
        """
        Add a disconnect handler
        
        Args:
            handler: Handler function
        """
        self.disconnect_handlers.append(handler)
        logger.debug("Added disconnect handler")
    
    def get_client_count(self) -> int:
        """
        Get the number of connected clients
        
        Returns:
            int: Number of connected clients
        """
        return len(self.clients)
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]) -> bool:
        """
        Send a message to a specific client
        
        Args:
            client_id: Client ID
            message: Message to send
            
        Returns:
            bool: True if the message was sent, False otherwise
        """
        if client_id not in self.clients:
            logger.warning(f"Client {client_id} not found")
            return False
        
        websocket = self.clients[client_id]
        try:
            await websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {e}")
            return False
    
    async def broadcast(self, message: Dict[str, Any], exclude_clients: Optional[List[str]] = None):
        """
        Broadcast a message to all connected clients
        
        Args:
            message: Message to broadcast
            exclude_clients: List of client IDs to exclude
        """
        if exclude_clients is None:
            exclude_clients = []
        
        tasks = []
        for client_id in self.clients:
            if client_id not in exclude_clients:
                tasks.append(self.send_to_client(client_id, message))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            logger.debug(f"Broadcast message to {success_count}/{len(tasks)} clients")
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle a client connection
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        # Generate client ID
        client_id = str(uuid.uuid4())
        
        # Store client connection
        self.clients[client_id] = websocket
        
        try:
            # Wait for initial message with client info
            try:
                initial_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                client_info = json.loads(initial_message)
                self.client_info[client_id] = client_info
            except (asyncio.TimeoutError, json.JSONDecodeError):
                logger.warning(f"Client {client_id} did not send valid client info")
                self.client_info[client_id] = {"type": "unknown"}
            
            # Process connect handlers
            response_data = {"type": "welcome"}
            for handler in self.connect_handlers:
                try:
                    handler_response = await handler(client_id, self.client_info.get(client_id, {}))
                    if handler_response:
                        response_data.update(handler_response)
                except Exception as e:
                    logger.error(f"Error in connect handler: {e}")
            
            # Send welcome message
            await websocket.send(json.dumps(response_data))
            
            # Process messages
            async for message in websocket:
                await self._process_message(client_id, message)
        
        except ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        
        finally:
            # Remove client
            self.clients.pop(client_id, None)
            client_info = self.client_info.pop(client_id, None)
            
            # Process disconnect handlers
            for handler in self.disconnect_handlers:
                try:
                    await handler(client_id)
                except Exception as e:
                    logger.error(f"Error in disconnect handler: {e}")
    
    async def _process_message(self, client_id: str, message: str):
        """
        Process a message from a client
        
        Args:
            client_id: Client ID
            message: Message string
        """
        try:
            # Parse message
            data = json.loads(message)
            
            # Get message type
            message_type = data.get("type")
            if not message_type:
                logger.warning(f"Message from client {client_id} has no type")
                await self.send_to_client(client_id, {
                    "type": "error",
                    "message": "Message has no type"
                })
                return
            
            # Process message handlers
            if message_type in self.message_handlers:
                for handler in self.message_handlers[message_type]:
                    try:
                        response = await handler(client_id, data)
                        if response:
                            await self.send_to_client(client_id, response)
                    except Exception as e:
                        logger.error(f"Error in message handler for type {message_type}: {e}")
                        await self.send_to_client(client_id, {
                            "type": "error",
                            "message": f"Error processing message: {str(e)}",
                            "original_type": message_type
                        })
            else:
                logger.warning(f"No handler for message type: {message_type}")
                await self.send_to_client(client_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
        
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client {client_id}")
            await self.send_to_client(client_id, {
                "type": "error",
                "message": "Invalid JSON message"
            })
        
        except Exception as e:
            logger.error(f"Error processing message from client {client_id}: {e}")
            await self.send_to_client(client_id, {
                "type": "error",
                "message": f"Server error: {str(e)}"
            })


async def test_server():
    """Test the WebSocket server"""
    server = MistralWebSocketServer(host="localhost", port=8765)
    
    # Add message handlers
    async def echo_handler(client_id, data):
        message = data.get("message", "")
        return {"type": "echo", "message": message}
    
    server.add_message_handler("echo", echo_handler)
    
    # Add connect handler
    async def connect_handler(client_id, client_info):
        logger.info(f"Client connected: {client_id} - {client_info}")
        return {"message": "Welcome to the Mistral WebSocket Server!"}
    
    server.add_connect_handler(connect_handler)
    
    # Start server
    await server.start()
    
    try:
        # Run for 1 hour
        await asyncio.sleep(3600)
    finally:
        # Stop server
        await server.stop()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run test server
    asyncio.run(test_server()) 