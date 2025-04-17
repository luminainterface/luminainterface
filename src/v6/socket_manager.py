#!/usr/bin/env python
"""
V6 Socket Manager

Handles socket communications between UI components and backend systems,
implementing the node socket architecture for the V6 Portal of Contradiction.
"""

import json
import logging
import threading
import time
from queue import Queue
from enum import Enum

logger = logging.getLogger("V6SocketManager")

class SubscriptionMode(Enum):
    """Socket subscription modes"""
    PUSH = "push"
    PULL = "pull"
    PUSH_PULL = "push_pull"

class V6SocketManager:
    """
    Socket manager for V6 Portal of Contradiction
    
    Implements the Node Socket Architecture described in the frontend documentation,
    allowing UI components to communicate with backend plugins through a standardized
    message protocol.
    """
    
    def __init__(self, mock_mode=True):
        # Configuration
        self.mock_mode = mock_mode
        self.connected = True
        
        # Message handling
        self.message_queue = Queue()
        self.response_queues = {}
        self.handlers = {}
        self.subscriptions = {}
        
        # Plugin management
        self.plugins = {}
        self.plugin_descriptors = {}
        
        # WebSocket connections
        self.websocket_connections = {}
        
        # UI Component mapping
        self.ui_component_mapping = {}
        
        # Start message processing thread
        self.processing_thread = threading.Thread(
            target=self._process_messages,
            daemon=True,
            name="SocketProcessingThread"
        )
        self.processing_thread.start()
        
        logger.info(f"V6 Socket Manager initialized (mock_mode={mock_mode})")
    
    def register_plugin(self, plugin):
        """Register a backend plugin with the socket manager"""
        try:
            # Get plugin descriptor
            descriptor = plugin.get_socket_descriptor()
            plugin_id = descriptor.get("plugin_id")
            
            if not plugin_id:
                logger.error(f"Invalid plugin descriptor: missing plugin_id")
                return False
            
            # Store plugin and descriptor
            self.plugins[plugin_id] = plugin
            self.plugin_descriptors[plugin_id] = descriptor
            
            # Register message types
            message_types = descriptor.get("message_types", [])
            for msg_type in message_types:
                if msg_type not in self.handlers:
                    self.handlers[msg_type] = []
                self.handlers[msg_type].append(plugin_id)
            
            # Setup UI component mapping
            ui_components = descriptor.get("ui_components", [])
            for component in ui_components:
                if component not in self.ui_component_mapping:
                    self.ui_component_mapping[component] = []
                self.ui_component_mapping[component].append(plugin_id)
            
            logger.info(f"Registered plugin: {plugin_id} with {len(message_types)} message types")
            return True
            
        except Exception as e:
            logger.error(f"Error registering plugin: {e}")
            return False
    
    def register_handler(self, event, handler):
        """Register a UI event handler"""
        if event not in self.subscriptions:
            self.subscriptions[event] = []
        
        self.subscriptions[event].append(handler)
        logger.debug(f"Registered handler for event: {event}")
        return True
    
    def emit(self, event, data=None):
        """Emit an event with data"""
        if not data:
            data = {}
            
        # Create message
        message = {
            "type": event,
            "timestamp": self._get_timestamp(),
            "content": data
        }
        
        logger.debug(f"Emitting event: {event}")
        
        # Add to queue for processing
        self.message_queue.put(message)
        return True
    
    def emit_to_plugin(self, plugin_id, event, data=None):
        """Emit an event to a specific plugin"""
        if plugin_id not in self.plugins:
            logger.warning(f"Plugin not found: {plugin_id}")
            return False
            
        if not data:
            data = {}
            
        # Create message
        message = {
            "type": event,
            "plugin_id": plugin_id,
            "timestamp": self._get_timestamp(),
            "content": data
        }
        
        logger.debug(f"Emitting event to plugin {plugin_id}: {event}")
        
        # Add to queue for processing
        self.message_queue.put(message)
        return True
    
    def request(self, event, data=None, timeout=5.0):
        """Send a request and wait for response"""
        if not data:
            data = {}
            
        # Create unique request ID
        request_id = f"req_{int(time.time())}_{id(data)}"
        
        # Create message
        message = {
            "type": event,
            "request_id": request_id,
            "timestamp": self._get_timestamp(),
            "content": data
        }
        
        # Create response queue
        response_queue = Queue()
        self.response_queues[request_id] = response_queue
        
        # Send request
        self.message_queue.put(message)
        
        # Wait for response
        try:
            logger.debug(f"Waiting for response to request: {request_id}")
            response = response_queue.get(timeout=timeout)
            return response
        except Exception as e:
            logger.error(f"Error waiting for response: {e}")
            return None
        finally:
            # Clean up
            if request_id in self.response_queues:
                del self.response_queues[request_id]
    
    def connect_websocket(self, url, component):
        """Connect to a WebSocket endpoint"""
        if self.mock_mode:
            logger.info(f"Mock WebSocket connection to {url} for {component}")
            self.websocket_connections[url] = {
                "component": component,
                "connected": True,
                "mock": True
            }
            return True
            
        # In real implementation, this would establish actual WebSocket connection
        logger.warning("Real WebSocket connections not implemented")
        return False
    
    def connect_to_endpoint(self, endpoint, component):
        """Connect a UI component to a backend endpoint"""
        if component in self.ui_component_mapping:
            plugin_ids = self.ui_component_mapping[component]
            logger.info(f"Component {component} connected to plugins: {plugin_ids}")
            return True
        else:
            logger.warning(f"No plugins registered for component: {component}")
            return False
    
    def is_connected(self):
        """Check if socket manager is connected"""
        return self.connected
    
    def _process_messages(self):
        """Message processing thread"""
        logger.debug("Starting message processing thread")
        
        while True:
            try:
                # Get next message from queue
                message = self.message_queue.get()
                
                # Process message
                self._process_message(message)
                
                # Mark as done
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _process_message(self, message):
        """Process a single message"""
        message_type = message.get("type")
        plugin_id = message.get("plugin_id")
        request_id = message.get("request_id")
        content = message.get("content", {})
        
        # Handle request/response
        if request_id and request_id in self.response_queues:
            self.response_queues[request_id].put(message)
            return
        
        # Route to handler plugins
        if message_type in self.handlers:
            plugin_ids = self.handlers[message_type]
            
            # If plugin_id specified, only route to that plugin
            if plugin_id and plugin_id in plugin_ids:
                plugin_ids = [plugin_id]
            
            for pid in plugin_ids:
                if pid in self.plugins:
                    try:
                        plugin = self.plugins[pid]
                        plugin.handle_message(message_type, content)
                    except Exception as e:
                        logger.error(f"Error in plugin {pid} handler: {e}")
        
        # Notify UI event subscribers
        if message_type in self.subscriptions:
            for handler in self.subscriptions[message_type]:
                try:
                    handler(content)
                except Exception as e:
                    logger.error(f"Error in UI handler for {message_type}: {e}")
    
    def _get_timestamp(self):
        """Get ISO8601 timestamp"""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    def shutdown(self):
        """Shutdown the socket manager"""
        logger.info("Shutting down socket manager")
        self.connected = False

class MockPlugin:
    """Mock plugin for development and testing"""
    
    def __init__(self, plugin_id, message_types, ui_components):
        self.plugin_id = plugin_id
        self.message_types = message_types
        self.ui_components = ui_components
        self.subscription_mode = "push"
        
        logger.info(f"Initialized mock plugin: {plugin_id}")
    
    def get_socket_descriptor(self):
        """Get plugin socket descriptor"""
        return {
            "plugin_id": self.plugin_id,
            "message_types": self.message_types,
            "subscription_mode": self.subscription_mode,
            "ui_components": self.ui_components,
            "data_format": "json"
        }
    
    def handle_message(self, message_type, content):
        """Handle incoming message"""
        logger.debug(f"Mock plugin {self.plugin_id} handling {message_type}")
        # Mock implementation would process message here
        return True

def create_mock_plugins():
    """Create mock plugins for testing"""
    plugins = [
        MockPlugin(
            "neural_state_plugin", 
            ["neural_activity", "node_state", "network_update"],
            ["network_panel", "consciousness_panel"]
        ),
        MockPlugin(
            "pattern_processor_plugin",
            ["pattern_recognized", "pattern_analysis", "fractal_update"],
            ["fractal_pattern_panel", "network_panel"]
        ),
        MockPlugin(
            "consciousness_plugin",
            ["consciousness_metrics", "awareness_level", "integration_index"],
            ["consciousness_panel"]
        ),
        MockPlugin(
            "language_memory_plugin",
            ["memory_synthesis", "memory_query", "association_network"],
            ["memory_synthesis_panel", "conversation_panel"]
        ),
        MockPlugin(
            "v6_breath_plugin",
            ["breath_state", "breath_cycle", "breath_metrics"],
            ["breath_panel"]
        ),
        MockPlugin(
            "v6_glyph_plugin",
            ["glyph_activation", "glyph_field_update", "symbolic_state"],
            ["glyph_panel"]
        ),
        MockPlugin(
            "v6_mirror_plugin",
            ["contradiction_detected", "mirror_mode_activate", "glitch_update"],
            ["mirror_panel"]
        ),
        MockPlugin(
            "v6_echo_plugin",
            ["memory_thread", "echo_resonance", "thread_navigation"],
            ["echo_panel"]
        ),
        MockPlugin(
            "v6_mythos_plugin",
            ["myth_generation", "narrative_fragment", "symbolic_integration"],
            ["mythos_panel"]
        ),
        MockPlugin(
            "v6_embodiment_plugin",
            ["node_transition", "embodiment_state", "ritual_event"],
            ["embodiment_panel"]
        )
    ]
    
    return plugins 