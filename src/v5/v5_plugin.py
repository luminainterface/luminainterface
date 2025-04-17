"""
V5 Plugin Base Class

This module provides the base class for all V5 plugins, which implement
the V5 plugin interface for integration with the frontend.
"""

import uuid
import logging
from .node_socket import NodeSocket

# Configure logging
logger = logging.getLogger(__name__)

class V5Plugin:
    """Base class for all V5 plugins"""
    
    def __init__(self, plugin_id=None, plugin_type=None, name=None):
        """
        Initialize the plugin
        
        Args:
            plugin_id: Unique identifier for the plugin (generated if not provided)
            plugin_type: Type of plugin (e.g., "pattern_processor", "neural_state")
            name: Human-readable name for the plugin
        """
        self.node_id = plugin_id or f"{plugin_type}_{uuid.uuid4().hex[:8]}"
        self.node_type = plugin_type or "v5_plugin"
        self.name = name or f"V5 Plugin ({self.node_id})"
        
        # Initialize socket
        self.socket = NodeSocket(self.node_id, self.node_type)
        
        # Set up message handlers
        self.socket.message_handlers = {
            "get_descriptor": self._handle_get_descriptor,
            "status_request": self._handle_status_request
        }
        
        logger.info(f"Initialized V5 plugin: {self.name} ({self.node_id})")
    
    def _handle_get_descriptor(self, message):
        """Handle request to get plugin descriptor"""
        descriptor = self.get_socket_descriptor()
        
        response = {
            "type": "descriptor_response",
            "request_id": message.get("request_id"),
            "content": descriptor
        }
        
        self.socket.send_message(response)
    
    def _handle_status_request(self, message):
        """Handle request to get plugin status"""
        status = self.get_status()
        
        response = {
            "type": "status_response",
            "request_id": message.get("request_id"),
            "content": status
        }
        
        self.socket.send_message(response)
    
    def get_socket_descriptor(self):
        """
        Return socket descriptor for frontend integration
        
        Must be implemented by derived classes.
        """
        return {
            "plugin_id": self.node_id,
            "plugin_type": self.node_type,
            "name": self.name,
            "message_types": ["get_descriptor", "status_request"],
            "subscription_mode": "request-response",
            "ui_components": []
        }
    
    def get_status(self):
        """Get plugin status"""
        return {
            "status": "active",
            "plugin_id": self.node_id,
            "plugin_type": self.node_type,
            "name": self.name
        }
    
    def register_message_handler(self, message_type, handler):
        """
        Register a custom message handler
        
        Args:
            message_type: The message type to handle
            handler: The handler function
        """
        self.socket.message_handlers[message_type] = handler 