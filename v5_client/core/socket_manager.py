"""
Client Socket Manager for V5 PySide6 Client

This module provides the ClientSocketManager class that manages communication
between UI components and the backend services.
"""

import uuid
import time
import threading
import logging
import random
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PySide6
try:
    from PySide6.QtCore import QObject, Signal
    USING_PYSIDE6 = True
except ImportError:
    from PyQt5.QtCore import QObject, pyqtSignal as Signal
    USING_PYSIDE6 = False

class ClientSocketManager(QObject):
    """Socket manager for the V5 PySide6 client"""
    
    # Signals
    message_received = Signal(dict)
    plugin_added = Signal(dict)
    plugin_removed = Signal(str)
    connection_status_changed = Signal(bool, str)
    
    def __init__(self, mock_mode=False):
        """
        Initialize the socket manager
        
        Args:
            mock_mode: Use mock mode for testing without backend services
        """
        super().__init__()
        
        self.mock_mode = mock_mode
        self.plugins = {}
        self.message_handlers = {}
        self.response_handlers = {}
        self.memory_bridge = None
        self.discovery_thread = None
        self.connected = False
        
        # Try to import the V5 socket components
        if not mock_mode:
            try:
                # Try to import from src.v5 if available
                from src.v5.node_socket import NodeSocket, QtSocketAdapter
                from src.v5.frontend_socket_manager import FrontendSocketManager
                
                # Create the V5 socket manager
                self.v5_socket_manager = FrontendSocketManager()
                self.connected = True
                
                # Create a node socket for the client
                self.client_socket = NodeSocket("v5_client", "frontend")
                
                logger.info("V5 socket components initialized")
                
                # Signal connection status change
                self.connection_status_changed.emit(True, "Connected to V5 socket system")
                
            except ImportError:
                logger.warning("V5 socket components not available, using mock mode")
                self.mock_mode = True
        
        if self.mock_mode:
            # Mock socket manager
            self.v5_socket_manager = None
            self.client_socket = None
            logger.info("Using mock socket manager")
    
    def register_message_handler(self, message_type, handler):
        """
        Register a handler for a specific message type
        
        Args:
            message_type: Type of message to handle
            handler: Function to call when message is received
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.debug(f"Registered handler for message type: {message_type}")
    
    def deregister_message_handler(self, message_type, handler=None):
        """
        Deregister a handler for a specific message type
        
        Args:
            message_type: Type of message to handle
            handler: Function to deregister (if None, deregister all)
        """
        if message_type not in self.message_handlers:
            return
        
        if handler is None:
            # Remove all handlers
            self.message_handlers[message_type] = []
        else:
            # Remove specific handler
            self.message_handlers[message_type] = [
                h for h in self.message_handlers[message_type] if h != handler
            ]
        
        logger.debug(f"Deregistered handler for message type: {message_type}")
    
    def send_message(self, message):
        """
        Send a message through the socket
        
        Args:
            message: Message dictionary to send
            
        Returns:
            Request ID for the message
        """
        # Ensure message has a request ID
        if "request_id" not in message:
            message["request_id"] = str(uuid.uuid4())
        
        request_id = message["request_id"]
        
        if self.mock_mode:
            # Handle mock message sending
            logger.debug(f"Sending mock message: {message}")
            
            # Start a thread to handle the mock response
            threading.Thread(
                target=self._handle_mock_response,
                args=(message,),
                daemon=True
            ).start()
            
            return request_id
        
        # Send through V5 socket
        if self.client_socket:
            try:
                self.client_socket.send_message(message)
                logger.debug(f"Sent message: {message}")
            except Exception as e:
                logger.error(f"Error sending message: {e}")
        
        return request_id
    
    def _handle_mock_response(self, message):
        """
        Handle mock responses for testing without backend
        
        Args:
            message: Original message
        """
        # Wait a bit to simulate network delay
        time.sleep(random.uniform(0.1, 0.5))
        
        message_type = message.get("type", "")
        request_id = message.get("request_id", "")
        
        # Generate mock response based on message type
        response = None
        
        if message_type == "request_pattern_data":
            response = self._generate_mock_pattern_data(message)
        elif message_type == "get_available_topics":
            response = self._generate_mock_topics()
        elif message_type == "search_memories":
            response = self._generate_mock_search_results(message)
        elif message_type == "store_conversation":
            response = {"status": "success", "request_id": request_id}
        elif message_type == "get_node_consciousness_data":
            response = self._generate_mock_consciousness_data()
        
        # If no specific response handler, use a default response
        if response is None:
            response = {
                "status": "success",
                "request_id": request_id,
                "type": f"{message_type}_response",
                "data": {"mock": True}
            }
        
        # Emit the response
        if response:
            # Call response handlers
            if request_id in self.response_handlers:
                for handler in self.response_handlers.get(request_id, []):
                    try:
                        handler(response)
                    except Exception as e:
                        logger.error(f"Error in response handler: {e}")
                
                # Remove the handler after calling
                del self.response_handlers[request_id]
            
            # Call message type handlers
            response_type = response.get("type", "")
            for handler in self.message_handlers.get(response_type, []):
                try:
                    handler(response)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
            
            # Emit the signal
            self.message_received.emit(response)
    
    def register_response_handler(self, request_id, handler):
        """
        Register a handler for a specific request response
        
        Args:
            request_id: ID of the request
            handler: Function to call when response is received
        """
        if request_id not in self.response_handlers:
            self.response_handlers[request_id] = []
        
        self.response_handlers[request_id].append(handler)
        logger.debug(f"Registered handler for request ID: {request_id}")
    
    def start_plugin_discovery(self, interval=30):
        """
        Start plugin discovery process
        
        Args:
            interval: Discovery interval in seconds
        """
        if self.mock_mode:
            # Start mock discovery thread
            if self.discovery_thread is None or not self.discovery_thread.is_alive():
                self.discovery_thread = threading.Thread(
                    target=self._mock_discovery_thread,
                    args=(interval,),
                    daemon=True
                )
                self.discovery_thread.start()
                logger.info(f"Started mock plugin discovery thread with interval {interval}s")
            return
        
        # Start V5 plugin discovery
        if self.v5_socket_manager:
            self.v5_socket_manager.start_plugin_discovery(interval)
            logger.info(f"Started V5 plugin discovery with interval {interval}s")
    
    def _mock_discovery_thread(self, interval):
        """
        Mock plugin discovery thread
        
        Args:
            interval: Discovery interval in seconds
        """
        while True:
            try:
                # Generate mock plugins
                self._discover_mock_plugins()
            except Exception as e:
                logger.error(f"Error in mock discovery thread: {e}")
            
            # Wait for next interval
            time.sleep(interval)
    
    def _discover_mock_plugins(self):
        """Discover mock plugins for testing"""
        # Mock plugin data
        mock_plugins = [
            {
                "plugin_id": "mock_pattern_processor",
                "name": "Mock Pattern Processor",
                "description": "Processes fractal patterns for visualization",
                "version": "1.0.0",
                "ui_components": ["fractal_view", "pattern_controls"],
                "subscription_mode": "push"
            },
            {
                "plugin_id": "mock_neural_state",
                "name": "Mock Neural State Provider",
                "description": "Provides neural network state data",
                "version": "1.0.0",
                "ui_components": ["neural_state_view", "state_controls"],
                "subscription_mode": "request-response"
            },
            {
                "plugin_id": "mock_consciousness_metrics",
                "name": "Mock Consciousness Metrics",
                "description": "Provides neural node consciousness metrics",
                "version": "1.0.0",
                "ui_components": ["consciousness_view", "metrics_controls"],
                "subscription_mode": "dual"
            }
        ]
        
        # Add plugins that don't already exist
        for plugin_data in mock_plugins:
            plugin_id = plugin_data["plugin_id"]
            
            if plugin_id not in self.plugins:
                self.plugins[plugin_id] = plugin_data
                logger.info(f"Discovered mock plugin: {plugin_id}")
                
                # Emit signal
                self.plugin_added.emit(plugin_data)
    
    def get_plugins(self):
        """
        Get all available plugins
        
        Returns:
            Dictionary of plugin data
        """
        if self.mock_mode:
            return self.plugins
        
        if self.v5_socket_manager:
            return self.v5_socket_manager.get_plugin_descriptors()
        
        return {}
    
    def get_plugin(self, plugin_id):
        """
        Get a specific plugin
        
        Args:
            plugin_id: ID of the plugin to get
            
        Returns:
            Plugin data or None if not found
        """
        if self.mock_mode:
            return self.plugins.get(plugin_id)
        
        if self.v5_socket_manager:
            return self.v5_socket_manager.get_plugin(plugin_id)
        
        return None
    
    def set_memory_bridge(self, bridge):
        """
        Set the memory bridge
        
        Args:
            bridge: Language Memory Bridge instance
        """
        self.memory_bridge = bridge
        logger.info(f"Memory bridge {'set' if bridge else 'cleared'}")
    
    def get_memory_bridge(self):
        """
        Get the memory bridge
        
        Returns:
            Language Memory Bridge instance
        """
        return self.memory_bridge
    
    def cleanup(self):
        """Clean up resources before shutdown"""
        # Clean up any threads or resources
        if self.discovery_thread and self.discovery_thread.is_alive():
            # There's no clean way to stop a thread in Python,
            # but marking as daemon ensures it won't prevent shutdown
            pass
        
        # Clear handlers
        self.message_handlers.clear()
        self.response_handlers.clear()
        
        logger.info("Socket manager cleaned up")
    
    # Mock data generation methods
    def _generate_mock_pattern_data(self, message):
        """Generate mock pattern data"""
        content = message.get("content", {})
        pattern_style = content.get("pattern_style", "neural")
        fractal_depth = content.get("fractal_depth", 5)
        
        # Generate pattern nodes
        nodes = []
        for i in range(20 + fractal_depth * 5):
            nodes.append({
                "id": f"node_{i}",
                "x": random.uniform(0, 1),
                "y": random.uniform(0, 1),
                "size": random.uniform(0.1, 0.3),
                "color": [
                    random.randint(50, 255),
                    random.randint(50, 200),
                    random.randint(100, 255)
                ],
                "connections": random.sample(
                    [f"node_{j}" for j in range(20 + fractal_depth * 5) if j != i],
                    min(5, fractal_depth)
                )
            })
        
        return {
            "status": "success",
            "request_id": message.get("request_id", ""),
            "type": "pattern_data_updated",
            "data": {
                "pattern_style": pattern_style,
                "fractal_depth": fractal_depth,
                "nodes": nodes,
                "metrics": {
                    "fractal_dimension": round(1.2 + random.uniform(0, 0.8), 2),
                    "complexity_index": random.randint(60, 98),
                    "pattern_coherence": random.randint(70, 99),
                    "entropy_level": random.choice(["Low", "Medium", "High"])
                },
                "insights": {
                    "detected_patterns": [
                        "Recursive branching structure",
                        "Hierarchical organization",
                        f"Depth-{fractal_depth} resonance"
                    ]
                }
            }
        }
    
    def _generate_mock_topics(self):
        """Generate mock memory topics"""
        topics = [
            {"id": "neural_networks", "name": "Neural Networks", "count": random.randint(20, 50)},
            {"id": "consciousness", "name": "Consciousness", "count": random.randint(15, 40)},
            {"id": "pattern_recognition", "name": "Pattern Recognition", "count": random.randint(10, 30)},
            {"id": "language_processing", "name": "Language Processing", "count": random.randint(25, 45)},
            {"id": "memory_systems", "name": "Memory Systems", "count": random.randint(15, 35)}
        ]
        
        return {
            "status": "success",
            "type": "topics_list_response",
            "data": {
                "topics": topics,
                "total_topics": len(topics)
            }
        }
    
    def _generate_mock_search_results(self, message):
        """Generate mock memory search results"""
        content = message.get("content", {})
        query = content.get("query", "")
        max_results = content.get("max_results", 5)
        
        # Generate memories based on query
        memories = []
        for i in range(max_results):
            memories.append({
                "id": f"memory_{i}",
                "text": f"This is a memory about {query} with some additional context and details "
                        f"for testing purposes. Item {i+1} of {max_results}.",
                "timestamp": time.time() - random.randint(0, 86400 * 30),
                "relevance": round(random.uniform(0.5, 0.95), 2),
                "topic": random.choice(["Neural Networks", "Consciousness", "Pattern Recognition"]),
                "source": random.choice(["conversation", "system", "user"])
            })
        
        return {
            "status": "success",
            "request_id": message.get("request_id", ""),
            "type": "search_memories_response",
            "data": {
                "memories": memories,
                "query": query,
                "total_results": max_results
            }
        }
    
    def _generate_mock_consciousness_data(self):
        """Generate mock consciousness metrics data"""
        # Generate metrics for a set of nodes
        nodes = {}
        for i in range(20):
            nodes[f"node_{i}"] = {
                "consciousness_index": round(random.uniform(0.3, 0.9), 2),
                "integration_score": round(random.uniform(0.4, 0.95), 2),
                "differentiation_score": round(random.uniform(0.3, 0.85), 2),
                "resonance_frequency": round(random.uniform(0.5, 15.0), 1),
                "activation_level": round(random.uniform(0.1, 1.0), 2)
            }
        
        # Generate overall metrics
        overall = {
            "global_consciousness": round(random.uniform(0.3, 0.8), 2),
            "phi_value": round(random.uniform(0.2, 0.6), 2),
            "harmony_index": round(random.uniform(0.4, 0.9), 2),
            "complexity_measure": round(random.uniform(0.5, 0.95), 2),
            "status": random.choice(["stable", "evolving", "fluctuating"])
        }
        
        return {
            "status": "success",
            "type": "node_consciousness_data",
            "data": {
                "timestamp": time.time(),
                "nodes": nodes,
                "overall": overall
            }
        } 