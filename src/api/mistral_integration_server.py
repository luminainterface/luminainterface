#!/usr/bin/env python3
"""
Mistral Integration Server

This module provides a server that integrates the Mistral enhanced system with
a WebSocket server, allowing client applications to utilize the Mistral system.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the project root to sys.path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/mistral_integration_server.log")
    ]
)
logger = logging.getLogger("MistralIntegrationServer")

# Import local modules
try:
    from src.api.mistral_websocket_server import MistralWebSocketServer
    from src.mistral_integration_fixed import MistralEnhancedSystem
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    IMPORTS_AVAILABLE = False


class MistralIntegrationServer:
    """
    Mistral Integration Server
    
    This class integrates the Mistral Enhanced System with a WebSocket server
    to provide a complete solution for client applications.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        api_key: Optional[str] = None,
        data_dir: str = "data/mistral",
        llm_weight: float = 0.7,
        nn_weight: float = 0.6
    ):
        """
        Initialize the integration server
        
        Args:
            host: Server host
            port: Server port
            api_key: Mistral API key (can also be set via MISTRAL_API_KEY env var)
            data_dir: Data directory for Mistral system
            llm_weight: Initial LLM weight (0-1)
            nn_weight: Initial neural network weight (0-1)
        """
        self.host = host
        self.port = port
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self.data_dir = data_dir
        self.llm_weight = llm_weight
        self.nn_weight = nn_weight
        
        # Initialize components
        self.mistral_system = None
        self.websocket_server = None
        self.running = False
        
        logger.info(f"MistralIntegrationServer initialized with host={host}, port={port}")
    
    async def start(self) -> bool:
        """
        Start the integration server
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not IMPORTS_AVAILABLE:
            logger.error("Cannot start server: Required modules not available")
            return False
        
        if self.running:
            logger.warning("Server is already running")
            return True
        
        # Initialize Mistral system
        try:
            self.mistral_system = MistralEnhancedSystem(
                api_key=self.api_key,
                llm_weight=self.llm_weight,
                nn_weight=self.nn_weight
            )
            logger.info("MistralEnhancedSystem initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MistralEnhancedSystem: {e}")
            return False
        
        # Initialize WebSocket server
        self.websocket_server = MistralWebSocketServer(
            host=self.host,
            port=self.port
        )
        
        # Register message handlers
        self._register_message_handlers()
        
        # Start WebSocket server
        try:
            await self.websocket_server.start()
            self.running = True
            logger.info(f"Mistral Integration Server started on ws://{self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            if self.mistral_system:
                self.mistral_system.close()
                self.mistral_system = None
            return False
    
    async def stop(self):
        """Stop the integration server"""
        if not self.running:
            return
        
        # Stop WebSocket server
        if self.websocket_server:
            await self.websocket_server.stop()
        
        # Clean up Mistral system
        if self.mistral_system:
            self.mistral_system.close()
            self.mistral_system = None
        
        self.running = False
        logger.info("Mistral Integration Server stopped")
    
    def _register_message_handlers(self):
        """Register message handlers for the WebSocket server"""
        # Connect handler
        self.websocket_server.add_connect_handler(self._handle_connect)
        
        # Message handlers
        self.websocket_server.add_message_handler("chat_message", self._handle_chat_message)
        self.websocket_server.add_message_handler("adjust_weights", self._handle_adjust_weights)
        self.websocket_server.add_message_handler("get_stats", self._handle_get_stats)
        self.websocket_server.add_message_handler("ping", self._handle_ping)
    
    async def _handle_connect(self, client_id: str, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle client connection
        
        Args:
            client_id: Client ID
            client_info: Client information
            
        Returns:
            Dict: Response data
        """
        logger.info(f"Client connected: {client_id} - {client_info}")
        
        # Return system status
        return {
            "type": "welcome",
            "message": "Connected to Mistral Integration Server",
            "system_info": {
                "mistral_available": self.mistral_system is not None,
                "llm_weight": self.mistral_system.central_node.llm_weight if self.mistral_system else self.llm_weight,
                "nn_weight": self.mistral_system.central_node.nn_weight if self.mistral_system else self.nn_weight,
                "server_version": "1.0.0"
            }
        }
    
    async def _handle_chat_message(self, client_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle chat message
        
        Args:
            client_id: Client ID
            data: Message data
            
        Returns:
            Dict: Response data
        """
        if not self.mistral_system:
            return {
                "type": "error",
                "message": "Mistral system not available"
            }
        
        # Extract message
        user_message = data.get("message", "").strip()
        if not user_message:
            return {
                "type": "error",
                "message": "Empty message"
            }
        
        try:
            # Notify client that processing has started
            await self.websocket_server.send_to_client(client_id, {
                "type": "processing_started",
                "message_id": data.get("message_id", "")
            })
            
            # Process message (use async wrapper for sync API)
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.mistral_system.process_message, user_message
            )
            
            # Return response
            return {
                "type": "chat_response",
                "message_id": data.get("message_id", ""),
                "response": response,
                "system_stats": {
                    "llm_weight": self.mistral_system.central_node.llm_weight,
                    "nn_weight": self.mistral_system.central_node.nn_weight
                }
            }
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return {
                "type": "error",
                "message": f"Error processing message: {str(e)}"
            }
    
    async def _handle_adjust_weights(self, client_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle weight adjustment
        
        Args:
            client_id: Client ID
            data: Message data
            
        Returns:
            Dict: Response data
        """
        if not self.mistral_system:
            return {
                "type": "error",
                "message": "Mistral system not available"
            }
        
        # Extract weights
        llm_weight = data.get("llm_weight")
        nn_weight = data.get("nn_weight")
        
        # Validate weights
        if llm_weight is not None:
            try:
                llm_weight = float(llm_weight)
                if not (0 <= llm_weight <= 1):
                    return {
                        "type": "error",
                        "message": "LLM weight must be between 0 and 1"
                    }
            except (ValueError, TypeError):
                return {
                    "type": "error",
                    "message": "Invalid LLM weight"
                }
        
        if nn_weight is not None:
            try:
                nn_weight = float(nn_weight)
                if not (0 <= nn_weight <= 1):
                    return {
                        "type": "error",
                        "message": "NN weight must be between 0 and 1"
                    }
            except (ValueError, TypeError):
                return {
                    "type": "error",
                    "message": "Invalid NN weight"
                }
        
        # Update weights
        if llm_weight is not None:
            self.mistral_system.central_node.set_llm_weight(llm_weight)
        
        if nn_weight is not None:
            self.mistral_system.central_node.set_nn_weight(nn_weight)
        
        # Return updated status
        return {
            "type": "weights_updated",
            "llm_weight": self.mistral_system.central_node.llm_weight,
            "nn_weight": self.mistral_system.central_node.nn_weight
        }
    
    async def _handle_get_stats(self, client_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle stats request
        
        Args:
            client_id: Client ID
            data: Message data
            
        Returns:
            Dict: Response data
        """
        if not self.mistral_system:
            return {
                "type": "error",
                "message": "Mistral system not available"
            }
        
        try:
            # Get system stats
            system_stats = await asyncio.get_event_loop().run_in_executor(
                None, self.mistral_system.get_system_stats
            )
            
            # Return stats
            return {
                "type": "system_stats",
                "stats": system_stats
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "type": "error",
                "message": f"Error getting system stats: {str(e)}"
            }
    
    async def _handle_ping(self, client_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle ping message
        
        Args:
            client_id: Client ID
            data: Message data
            
        Returns:
            Dict: Response data
        """
        # Return pong response
        return {
            "type": "pong",
            "timestamp": asyncio.get_event_loop().time(),
            "ping_id": data.get("ping_id", "")
        }


async def main():
    """Main entry point for the integration server"""
    # Get API key from environment
    api_key = os.getenv("MISTRAL_API_KEY", "")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create and start server
    server = MistralIntegrationServer(
        host="0.0.0.0",  # Listen on all interfaces
        port=8765,
        api_key=api_key
    )
    
    # Set up signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def shutdown_signal():
        logging.info("Shutdown signal received")
        shutdown_event.set()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, shutdown_signal)
    
    if await server.start():
        print(f"Mistral Integration Server running at ws://0.0.0.0:8765")
        print("Press Ctrl+C to stop")
        
        # Keep the server running until shutdown signal
        await shutdown_event.wait()
    
    # Graceful shutdown
    await server.stop()


if __name__ == "__main__":
    # Run the server
    asyncio.run(main()) 