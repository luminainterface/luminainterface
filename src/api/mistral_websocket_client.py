#!/usr/bin/env python3
"""
WebSocket Client for Mistral Integration

This module provides a client for connecting to the Mistral WebSocket server.
It can be used in both applications and scripts.
"""

import json
import logging
import asyncio
import sys
import os
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path

# Add the project root to sys.path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Websockets library
try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    from websockets.exceptions import ConnectionClosed
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("WebSockets library not available. Please install it with: pip install websockets")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/mistral_websocket_client.log")
    ]
)
logger = logging.getLogger("MistralWebSocketClient")


class MistralWebSocketClient:
    """
    WebSocket client for connecting to the Mistral WebSocket server
    
    This class provides a simple interface for applications to connect to the
    Mistral WebSocket server, send messages, and receive responses.
    """
    
    def __init__(self, url: str = "ws://localhost:8765"):
        """
        Initialize the WebSocket client
        
        Args:
            url: WebSocket server URL
        """
        self.url = url
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connected = False
        self.client_info = {
            "user_agent": f"MistralWebSocketClient/1.0.0 Python/{sys.version.split()[0]}",
            "version": "1.0.0",
            "type": "python_client"
        }
        
        # Callbacks
        self.message_callbacks: Dict[str, List[Callable]] = {}
        self.on_connect_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None
        
        # Background task
        self.receive_task: Optional[asyncio.Task] = None
        
        # Message queue for pending messages before connection
        self.message_queue: List[Dict[str, Any]] = []
    
    async def connect(self) -> bool:
        """
        Connect to the WebSocket server
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets library not available")
            return False
        
        if self.connected:
            return True
        
        try:
            # Connect to server
            self.websocket = await websockets.connect(self.url)
            self.connected = True
            logger.info(f"Connected to {self.url}")
            
            # Start receiving messages
            self.receive_task = asyncio.create_task(self._receive_messages())
            
            # Send initial connect message
            await self.send_message("connect", {"client_info": self.client_info})
            
            # Send any queued messages
            if self.message_queue:
                for message in self.message_queue:
                    await self._send_raw(message)
                self.message_queue.clear()
            
            # Call the connect callback if set
            if self.on_connect_callback:
                try:
                    self.on_connect_callback()
                except Exception as e:
                    logger.error(f"Error in connect callback: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to {self.url}: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if not self.connected or not self.websocket:
            return
        
        # Cancel receive task
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
            self.receive_task = None
        
        # Close connection
        try:
            await self.websocket.close()
            logger.info(f"Disconnected from {self.url}")
        except Exception as e:
            logger.error(f"Error disconnecting from {self.url}: {e}")
        finally:
            self.connected = False
            self.websocket = None
            
            # Call the disconnect callback if set
            if self.on_disconnect_callback:
                try:
                    self.on_disconnect_callback()
                except Exception as e:
                    logger.error(f"Error in disconnect callback: {e}")
    
    async def send_message(self, message_type: str, data: Dict[str, Any] = None) -> bool:
        """
        Send a message to the server
        
        Args:
            message_type: Message type
            data: Message data
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if data is None:
            data = {}
        
        # Create message
        message = {"type": message_type, **data}
        
        # Send message
        if self.connected and self.websocket:
            return await self._send_raw(message)
        else:
            # Queue message for later
            self.message_queue.append(message)
            logger.debug(f"Queued message of type '{message_type}' for later")
            return False
    
    async def _send_raw(self, message: Dict[str, Any]) -> bool:
        """
        Send a raw message to the server
        
        Args:
            message: Message data
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.connected or not self.websocket:
            return False
        
        try:
            # Convert message to JSON
            message_str = json.dumps(message)
            
            # Send message
            await self.websocket.send(message_str)
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            # Connection might be closed
            if isinstance(e, ConnectionClosed):
                self.connected = False
                self.websocket = None
                # Call the disconnect callback if set
                if self.on_disconnect_callback:
                    try:
                        self.on_disconnect_callback()
                    except Exception as inner_e:
                        logger.error(f"Error in disconnect callback: {inner_e}")
            return False
    
    async def _receive_messages(self):
        """Receive and process messages from the server"""
        if not self.connected or not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                try:
                    # Parse JSON message
                    data = json.loads(message)
                    
                    # Get message type
                    message_type = data.get("type")
                    if not message_type:
                        logger.warning(f"Received message without type: {message}")
                        continue
                    
                    # Call callbacks for this message type
                    if message_type in self.message_callbacks:
                        for callback in self.message_callbacks[message_type]:
                            try:
                                callback(data)
                            except Exception as e:
                                logger.error(f"Error in callback for message type '{message_type}': {e}")
                    
                    # Log message (debug only)
                    logger.debug(f"Received message of type '{message_type}'")
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON message: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except ConnectionClosed:
            logger.info("Connection closed")
            self.connected = False
            self.websocket = None
            # Call the disconnect callback if set
            if self.on_disconnect_callback:
                try:
                    self.on_disconnect_callback()
                except Exception as e:
                    logger.error(f"Error in disconnect callback: {e}")
        except Exception as e:
            logger.error(f"Error in receive task: {e}")
            self.connected = False
            self.websocket = None
            # Call the disconnect callback if set
            if self.on_disconnect_callback:
                try:
                    self.on_disconnect_callback()
                except Exception as inner_e:
                    logger.error(f"Error in disconnect callback: {inner_e}")
    
    def add_message_callback(self, message_type: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Add a callback for a specific message type
        
        Args:
            message_type: Message type to listen for
            callback: Function to call when message is received
        """
        if message_type not in self.message_callbacks:
            self.message_callbacks[message_type] = []
        self.message_callbacks[message_type].append(callback)
    
    def remove_message_callback(self, message_type: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Remove a callback for a specific message type
        
        Args:
            message_type: Message type
            callback: Callback function to remove
            
        Returns:
            bool: True if callback was removed, False otherwise
        """
        if message_type in self.message_callbacks and callback in self.message_callbacks[message_type]:
            self.message_callbacks[message_type].remove(callback)
            return True
        return False
    
    def set_connect_callback(self, callback: Callable[[], None]):
        """
        Set the callback for when connection is established
        
        Args:
            callback: Function to call on connect
        """
        self.on_connect_callback = callback
    
    def set_disconnect_callback(self, callback: Callable[[], None]):
        """
        Set the callback for when connection is closed
        
        Args:
            callback: Function to call on disconnect
        """
        self.on_disconnect_callback = callback

    # Convenience methods for common operations
    
    async def send_chat_message(self, message: str, message_id: str = None) -> bool:
        """
        Send a chat message to the server
        
        Args:
            message: Message text
            message_id: Optional message ID for tracking
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        data = {"message": message}
        if message_id:
            data["message_id"] = message_id
        return await self.send_message("chat_message", data)
    
    async def adjust_weights(self, llm_weight: Optional[float] = None, nn_weight: Optional[float] = None) -> bool:
        """
        Adjust the weights of the Mistral system
        
        Args:
            llm_weight: LLM weight (0-1)
            nn_weight: NN weight (0-1)
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        data = {}
        if llm_weight is not None:
            data["llm_weight"] = llm_weight
        if nn_weight is not None:
            data["nn_weight"] = nn_weight
        return await self.send_message("adjust_weights", data)
    
    async def get_stats(self) -> bool:
        """
        Request system stats from the server
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        return await self.send_message("get_stats")
    
    async def ping(self, ping_id: str = None) -> bool:
        """
        Send a ping message to the server
        
        Args:
            ping_id: Optional ping ID for tracking
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        data = {}
        if ping_id:
            data["ping_id"] = ping_id
        return await self.send_message("ping", data)


# Simple example usage
async def main():
    """Example usage of the MistralWebSocketClient"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create client
    client = MistralWebSocketClient(url="ws://localhost:8765")
    
    # Add callbacks
    def on_chat_response(data):
        print(f"\nServer: {data.get('response', '')}")
    
    def on_welcome(data):
        print(f"Connected to Mistral WebSocket Server")
        print(f"Mistral available: {data.get('mistral_available', False)}")
    
    def on_error(data):
        print(f"Error: {data.get('message', 'Unknown error')}")
    
    # Register callbacks
    client.add_message_callback("chat_response", on_chat_response)
    client.add_message_callback("welcome", on_welcome)
    client.add_message_callback("error", on_error)
    
    # Connect to server
    if not await client.connect():
        print("Failed to connect to server")
        return
    
    # Main loop
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            # Send chat message
            await client.send_chat_message(user_input)
            print("(Processing...)")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Disconnect
        await client.disconnect()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 