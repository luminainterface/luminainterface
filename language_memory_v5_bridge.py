#!/usr/bin/env python3
"""
Language Memory V5 Bridge - Connects the Language Memory System with the V5 Visualization System

This module serves as the integration point between the Language Memory API and the
V5 Fractal Echo Visualization system, facilitating data exchange and synchronization.
"""

import os
import sys
import json
import logging
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Any, Callable, Union
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LM-V5-Bridge")

class MemoryAPISocketProvider:
    """
    Socket provider for Language Memory API integration with V5 system.
    Implements the socket interface for the V5 Fractal Echo Visualization system.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Memory API Socket Provider.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.connected = False
        self.mock_mode = self.config.get("mock_mode", False)
        self.message_handlers = {}
        self.message_queue = Queue()
        self.node_id = "language_memory_api"
        self.interface_type = "service"
        self.connections = {}
        self.processing_thread = None
        self.running = False
        
        logger.info(f"Initialized Memory API Socket Provider (mock_mode={self.mock_mode})")
    
    def connect(self) -> bool:
        """
        Connect to the Language Memory API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.mock_mode:
            logger.info("Mock mode enabled, using simulated Language Memory API")
            self.connected = True
            return True
            
        try:
            # Import the actual Language Memory API client
            from language_memory_api_client import LanguageMemoryClient
            
            # Initialize the client
            self.api_client = LanguageMemoryClient()
            connected = self.api_client.connect()
            
            if connected:
                logger.info("Successfully connected to Language Memory API")
                self.connected = True
                return True
            else:
                logger.error("Failed to connect to Language Memory API")
                return False
                
        except ImportError:
            logger.error("Failed to import Language Memory API client, falling back to mock mode")
            self.mock_mode = True
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Error connecting to Language Memory API: {e}")
            return False
    
    def start_processing(self) -> None:
        """Start the message processing thread"""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            logger.warning("Message processing thread already running")
            return
            
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._process_messages,
            daemon=True,
            name="LM-V5-Bridge-Thread"
        )
        self.processing_thread.start()
        logger.info("Started message processing thread")
    
    def stop_processing(self) -> None:
        """Stop the message processing thread"""
        self.running = False
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)
            logger.info("Stopped message processing thread")
    
    def _process_messages(self) -> None:
        """Process messages from the queue"""
        while self.running:
            try:
                if self.message_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                    
                message = self.message_queue.get(block=False)
                self._handle_message(message)
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    def _handle_message(self, message: Dict[str, Any]) -> None:
        """
        Handle a message from the queue
        
        Args:
            message: Message dictionary with type, data, etc.
        """
        msg_type = message.get("type", "unknown")
        if msg_type in self.message_handlers:
            try:
                handler = self.message_handlers[msg_type]
                handler(message)
            except Exception as e:
                logger.error(f"Error in message handler for {msg_type}: {e}")
        else:
            logger.warning(f"No handler registered for message type: {msg_type}")
    
    def register_message_handler(self, msg_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific message type
        
        Args:
            msg_type: Message type to handle
            handler: Function to call when message of this type arrives
        """
        self.message_handlers[msg_type] = handler
        logger.debug(f"Registered handler for message type: {msg_type}")
    
    def send_message(self, msg_type: str, data: Any, target_id: Optional[str] = None) -> bool:
        """
        Send a message to connected V5 components
        
        Args:
            msg_type: Type of message
            data: Message data (will be serialized)
            target_id: Optional target node ID
            
        Returns:
            bool: True if message was sent, False otherwise
        """
        message = {
            "type": msg_type,
            "source": self.node_id,
            "timestamp": time.time(),
            "data": data
        }
        
        if target_id:
            message["target"] = target_id
        
        # If we have specific connections, send to them
        if self.connections:
            for conn_id, conn in self.connections.items():
                if target_id is None or conn_id == target_id:
                    try:
                        conn.receive_message(message)
                        logger.debug(f"Sent {msg_type} message to {conn_id}")
                        return True
                    except Exception as e:
                        logger.error(f"Error sending message to {conn_id}: {e}")
            
            return False
        else:
            # If no connections, put in our own queue (allows testing)
            self.message_queue.put(message)
            return True
    
    def connect_to(self, other_socket: Any) -> bool:
        """
        Connect to another socket
        
        Args:
            other_socket: Another socket object implementing the V5 socket interface
            
        Returns:
            bool: True if connection successful
        """
        if not hasattr(other_socket, 'node_id') or not hasattr(other_socket, 'receive_message'):
            logger.error("Cannot connect to invalid socket object")
            return False
            
        self.connections[other_socket.node_id] = other_socket
        logger.info(f"Connected to {other_socket.node_id}")
        return True
    
    def receive_message(self, message: Dict[str, Any]) -> None:
        """
        Receive a message from another socket
        
        Args:
            message: Message dictionary
        """
        if not isinstance(message, dict):
            logger.error(f"Received invalid message: {message}")
            return
            
        self.message_queue.put(message)
    
    # Language Memory specific methods
    def get_available_topics(self) -> List[str]:
        """
        Get list of available topics in the Language Memory system
        
        Returns:
            List of topic names
        """
        if self.mock_mode:
            # Return mock data
            return ["neural_networks", "consciousness", "language_processing", 
                   "visualization_techniques", "memory_systems"]
        
        if not self.connected:
            logger.warning("Not connected to Language Memory API")
            return []
            
        try:
            topics = self.api_client.get_topics()
            return topics
        except Exception as e:
            logger.error(f"Error getting topics: {e}")
            return []
    
    def get_memory_by_id(self, memory_id: str) -> Dict[str, Any]:
        """
        Get a specific memory by ID
        
        Args:
            memory_id: Unique identifier of the memory
            
        Returns:
            Memory data or empty dict if not found
        """
        if self.mock_mode:
            # Return mock data
            return {
                "id": memory_id,
                "content": f"This is mock memory content for ID {memory_id}",
                "timestamp": time.time(),
                "topic": "neural_networks",
                "tags": ["mock", "test", "memory"],
                "connections": [
                    {"id": "conn1", "strength": 0.8, "type": "association"},
                    {"id": "conn2", "strength": 0.5, "type": "sequence"}
                ]
            }
        
        if not self.connected:
            logger.warning("Not connected to Language Memory API")
            return {}
            
        try:
            memory = self.api_client.get_memory(memory_id)
            return memory
        except Exception as e:
            logger.error(f"Error getting memory {memory_id}: {e}")
            return {}
    
    def search_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search memories by query string
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        if self.mock_mode:
            # Return mock data
            return [
                {
                    "id": f"mock_id_{i}",
                    "content": f"Mock result {i} for query: {query}",
                    "relevance": 1.0 - (i * 0.1),
                    "timestamp": time.time() - (i * 3600)
                }
                for i in range(min(5, limit))
            ]
        
        if not self.connected:
            logger.warning("Not connected to Language Memory API")
            return []
            
        try:
            results = self.api_client.search(query, limit=limit)
            return results
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def store_memory(self, content: str, topic: str = None, 
                    tags: List[str] = None) -> str:
        """
        Store a new memory
        
        Args:
            content: Memory content
            topic: Optional topic
            tags: Optional list of tags
            
        Returns:
            ID of the stored memory, or empty string on failure
        """
        if self.mock_mode:
            # Just return a fake ID in mock mode
            import uuid
            memory_id = str(uuid.uuid4())
            logger.info(f"Stored mock memory with ID: {memory_id}")
            return memory_id
        
        if not self.connected:
            logger.warning("Not connected to Language Memory API")
            return ""
            
        try:
            memory_id = self.api_client.store_memory(
                content=content,
                topic=topic,
                tags=tags or []
            )
            return memory_id
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return ""
    
    def get_connections(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Get connections for a specific memory
        
        Args:
            memory_id: Memory ID
            
        Returns:
            List of connection objects
        """
        if self.mock_mode:
            # Return mock connections
            return [
                {"target_id": f"mock_conn_{i}", "strength": 0.9 - (i * 0.2), 
                 "type": ["association", "sequence", "causation"][i % 3]}
                for i in range(5)
            ]
        
        if not self.connected:
            logger.warning("Not connected to Language Memory API")
            return []
            
        try:
            connections = self.api_client.get_connections(memory_id)
            return connections
        except Exception as e:
            logger.error(f"Error getting connections for {memory_id}: {e}")
            return []

    def get_status(self):
        """
        Get the status of the Memory API Socket Provider
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "connected": self.connected,
            "mock_mode": self.mock_mode,
            "handlers_registered": len(self.message_handlers)
        }

class LanguageMemoryV5Bridge:
    """
    Bridge between Language Memory system and V5 Visualization system.
    Handles data conversion, synchronization and event handling.
    """
    
    def __init__(self, config=None):
        """
        Initialize the bridge
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.mock_mode = self.config.get("mock_mode", False)
        self.socket = MemoryAPISocketProvider(self.config)
        self.connected = False
        self.fractal_processor = None
        self.topic_cache = {}
        self.memory_cache = {}
        
        # Register message handlers
        self._register_handlers()
        
        logger.info("Initialized Language Memory V5 Bridge")
    
    def _register_handlers(self):
        """Register message handlers for incoming socket messages"""
        self.socket.register_message_handler("get_topics", self._handle_get_topics)
        self.socket.register_message_handler("search_memories", self._handle_search)
        self.socket.register_message_handler("get_memory", self._handle_get_memory)
        self.socket.register_message_handler("store_memory", self._handle_store_memory)
        self.socket.register_message_handler("get_connections", self._handle_get_connections)
        self.socket.register_message_handler("generate_fractal", self._handle_generate_fractal)
    
    def connect(self) -> bool:
        """
        Connect to the Language Memory API
        
        Returns:
            bool: True if connection successful
        """
        if self.connected:
            return True
            
        # Connect the socket
        connected = self.socket.connect()
        if connected:
            # Start processing messages
            self.socket.start_processing()
            self.connected = True
            logger.info("Language Memory V5 Bridge connected")
        
        return connected
    
    def disconnect(self) -> None:
        """Disconnect from the Language Memory API"""
        if self.connected:
            self.socket.stop_processing()
            self.connected = False
            logger.info("Language Memory V5 Bridge disconnected")
    
    # Event handlers
    def _handle_get_topics(self, message: Dict[str, Any]) -> None:
        """Handle get_topics message"""
        try:
            topics = self.socket.get_available_topics()
            
            # Cache the topics
            self.topic_cache = {topic: {"last_updated": time.time()} for topic in topics}
            
            # Send response
            self.socket.send_message(
                "topics_result",
                {"topics": topics},
                target_id=message.get("source")
            )
        except Exception as e:
            logger.error(f"Error handling get_topics: {e}")
            self.socket.send_message(
                "error",
                {"error": str(e), "original_request": "get_topics"},
                target_id=message.get("source")
            )
    
    def _handle_search(self, message: Dict[str, Any]) -> None:
        """Handle search_memories message"""
        try:
            data = message.get("data", {})
            query = data.get("query", "")
            limit = data.get("limit", 10)
            
            results = self.socket.search_memories(query, limit)
            
            # Cache results
            for result in results:
                if "id" in result:
                    self.memory_cache[result["id"]] = {
                        "data": result,
                        "last_accessed": time.time()
                    }
            
            # Send response
            self.socket.send_message(
                "search_results",
                {"query": query, "results": results},
                target_id=message.get("source")
            )
        except Exception as e:
            logger.error(f"Error handling search: {e}")
            self.socket.send_message(
                "error",
                {"error": str(e), "original_request": "search_memories"},
                target_id=message.get("source")
            )
    
    def _handle_get_memory(self, message: Dict[str, Any]) -> None:
        """Handle get_memory message"""
        try:
            data = message.get("data", {})
            memory_id = data.get("id", "")
            
            memory = self.socket.get_memory_by_id(memory_id)
            
            # Cache the memory
            if memory:
                self.memory_cache[memory_id] = {
                    "data": memory,
                    "last_accessed": time.time()
                }
            
            # Send response
            self.socket.send_message(
                "memory_result",
                {"id": memory_id, "memory": memory},
                target_id=message.get("source")
            )
        except Exception as e:
            logger.error(f"Error handling get_memory: {e}")
            self.socket.send_message(
                "error",
                {"error": str(e), "original_request": "get_memory"},
                target_id=message.get("source")
            )
    
    def _handle_store_memory(self, message: Dict[str, Any]) -> None:
        """Handle store_memory message"""
        try:
            data = message.get("data", {})
            content = data.get("content", "")
            topic = data.get("topic")
            tags = data.get("tags", [])
            
            memory_id = self.socket.store_memory(content, topic, tags)
            
            # Send response
            self.socket.send_message(
                "memory_stored",
                {"id": memory_id, "success": bool(memory_id)},
                target_id=message.get("source")
            )
        except Exception as e:
            logger.error(f"Error handling store_memory: {e}")
            self.socket.send_message(
                "error",
                {"error": str(e), "original_request": "store_memory"},
                target_id=message.get("source")
            )
    
    def _handle_get_connections(self, message: Dict[str, Any]) -> None:
        """Handle get_connections message"""
        try:
            data = message.get("data", {})
            memory_id = data.get("id", "")
            
            connections = self.socket.get_connections(memory_id)
            
            # Send response
            self.socket.send_message(
                "connections_result",
                {"id": memory_id, "connections": connections},
                target_id=message.get("source")
            )
        except Exception as e:
            logger.error(f"Error handling get_connections: {e}")
            self.socket.send_message(
                "error",
                {"error": str(e), "original_request": "get_connections"},
                target_id=message.get("source")
            )
    
    def _handle_generate_fractal(self, message: Dict[str, Any]) -> None:
        """Handle generate_fractal message (convert memory to fractal pattern)"""
        try:
            data = message.get("data", {})
            memory_id = data.get("memory_id", "")
            
            # First, get the memory data
            memory = None
            if memory_id in self.memory_cache:
                memory = self.memory_cache[memory_id]["data"]
            else:
                memory = self.socket.get_memory_by_id(memory_id)
                
            if not memory:
                raise ValueError(f"Memory with ID {memory_id} not found")
            
            # Convert memory to fractal pattern
            fractal_pattern = self._memory_to_fractal(memory)
            
            # Send response
            self.socket.send_message(
                "fractal_pattern",
                {
                    "memory_id": memory_id,
                    "pattern": fractal_pattern
                },
                target_id=message.get("source")
            )
        except Exception as e:
            logger.error(f"Error handling generate_fractal: {e}")
            self.socket.send_message(
                "error",
                {"error": str(e), "original_request": "generate_fractal"},
                target_id=message.get("source")
            )
    
    def _memory_to_fractal(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a memory to a fractal pattern for visualization
        
        Args:
            memory: Memory data
            
        Returns:
            Fractal pattern data
        """
        # This is where the magic happens - convert semantic memory to visual pattern
        # In production code, this would use sophisticated algorithms
        
        if self.mock_mode:
            # Generate a mock fractal pattern with random properties
            # Use consistent seed for the same topic
            random.seed(memory.get("topic", "unknown"))
            
            # Start with basic pattern structure
            pattern = {
                "source_id": memory.get("id", "unknown"),
                "pattern_type": "memory_fractal",
                "core_frequency": random.uniform(0.1, 0.9),
                "amplitude": random.uniform(0.5, 1.0),
                "complexity": random.uniform(0.3, 0.8),
                "color_palette": [
                    f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
                    for _ in range(5)
                ],
                "recursion_depth": random.randint(3, 7),
                "symmetry_type": random.choice(["radial", "bilateral", "spiral", "chaotic"]),
                "nodes": []
            }
            
            # Generate nodes based on content
            content = memory.get("content", "")
            words = content.split()[:10]  # Take first 10 words for demo
            
            for i, word in enumerate(words):
                pattern["nodes"].append({
                    "id": f"node_{i}",
                    "label": word,
                    "size": len(word) / 10.0,
                    "position": {
                        "x": random.uniform(-1, 1),
                        "y": random.uniform(-1, 1),
                        "z": random.uniform(-0.5, 0.5)
                    },
                    "color": pattern["color_palette"][i % len(pattern["color_palette"])],
                    "connections": []
                })
            
            # Add connections between nodes
            for i in range(len(pattern["nodes"])):
                node = pattern["nodes"][i]
                # Connect to 1-3 other nodes
                for _ in range(random.randint(1, min(3, len(pattern["nodes"])))):
                    target_idx = random.randint(0, len(pattern["nodes"]) - 1)
                    if target_idx != i:
                        connection = {
                            "target": f"node_{target_idx}",
                            "strength": random.uniform(0.1, 1.0),
                            "type": random.choice(["semantic", "temporal", "causal"])
                        }
                        node["connections"].append(connection)
            
            return pattern
        
        # In a real implementation, we would use more sophisticated algorithms
        # to convert the semantic content to visual patterns
        raise NotImplementedError("Real fractal conversion not implemented")

    # Public API for direct calls from V5 components
    def get_topics(self) -> List[str]:
        """Get list of available topics"""
        return self.socket.get_available_topics()
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by query"""
        return self.socket.search_memories(query, limit)
    
    def get_memory(self, memory_id: str) -> Dict[str, Any]:
        """Get memory by ID"""
        return self.socket.get_memory_by_id(memory_id)
    
    def store_memory(self, content: str, topic: str = None, tags: List[str] = None) -> str:
        """Store a new memory"""
        return self.socket.store_memory(content, topic, tags)
    
    def get_connections(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get connections for a memory"""
        return self.socket.get_connections(memory_id)
    
    def generate_fractal(self, memory_id: str) -> Dict[str, Any]:
        """Generate fractal visualization for a memory"""
        memory = self.get_memory(memory_id)
        if not memory:
            return None
        return self._memory_to_fractal(memory)

    def get_status(self):
        """
        Get the status of the bridge
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {"available": True}

# Singleton instance for easy access
bridge_instance = None

def get_bridge(mock_mode: bool = False) -> LanguageMemoryV5Bridge:
    """
    Get the singleton bridge instance
    
    Args:
        mock_mode: Whether to use mock data
        
    Returns:
        Bridge instance
    """
    global bridge_instance
    if bridge_instance is None:
        bridge_instance = LanguageMemoryV5Bridge(mock_mode=mock_mode)
    
    return bridge_instance

# Test the bridge if run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create bridge
    bridge = LanguageMemoryV5Bridge({"mock_mode": True})
    
    # Connect to Memory API
    bridge.connect()
    
    # Get available topics
    topics = bridge.get_topics()
    print(f"Available topics: {topics}")
    
    # Search for memories
    results = bridge.search("consciousness")
    print(f"Search results: {json.dumps(results, indent=2)}")
    
    # Generate fractal pattern
    pattern = bridge.generate_fractal("consciousness")
    print(f"Generated pattern for 'consciousness': {json.dumps(pattern, indent=2)}")
    
    # Get status
    status = bridge.get_status()
    print(f"Bridge status: {json.dumps(status, indent=2)}")
    
    # Cleanup
    bridge.disconnect()
    print("Test complete") 