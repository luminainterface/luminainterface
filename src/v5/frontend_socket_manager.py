"""
Frontend Socket Manager for V5 Visualization System

This module provides the FrontendSocketManager class that coordinates
all plugin connections for the V5 frontend.
"""

import uuid
import logging
import threading
import json
from .node_socket import NodeSocket
from .db_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the connection discovery service
try:
    from connection_discovery import ConnectionDiscovery
    HAS_DISCOVERY = True
except ImportError:
    logger.warning("ConnectionDiscovery not available. Limited plugin discovery will be used.")
    HAS_DISCOVERY = False

# Import Qt compatibility layer
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
from src.v5.ui.qt_compat import get_widgets, get_gui, get_core

# Get required Qt classes
QSplitter = get_widgets().QSplitter
QFormLayout = get_widgets().QFormLayout

class FrontendSocketManager:
    """Manages plugin sockets for frontend integration"""
    
    def __init__(self):
        """Initialize the frontend socket manager"""
        self.plugins = {}
        self.ui_component_map = {}
        
        # Connect to discovery service if available
        if HAS_DISCOVERY:
            self.discovery = ConnectionDiscovery.get_instance()
            logger.info("Connected to connection discovery service")
        else:
            self.discovery = None
            logger.warning("Running without connection discovery service")
        
        # Create a socket for inter-plugin communication
        self.manager_socket = NodeSocket("frontend_socket_manager", "manager")
        logger.info("Frontend Socket Manager initialized")
    
    def register_plugin(self, plugin):
        """
        Register a plugin for frontend integration
        
        Args:
            plugin: The plugin instance to register
            
        Returns:
            The plugin's socket descriptor
        """
        # Get the plugin's socket descriptor
        try:
            descriptor = plugin.get_socket_descriptor()
        except AttributeError:
            logger.error(f"Plugin {plugin} does not have a get_socket_descriptor method")
            return None
        
        plugin_id = descriptor["plugin_id"]
        
        # Store the plugin
        self.plugins[plugin_id] = {
            "plugin": plugin,
            "descriptor": descriptor
        }
        
        # Map UI components to plugins
        for component in descriptor.get("ui_components", []):
            if component not in self.ui_component_map:
                self.ui_component_map[component] = []
            self.ui_component_map[component].append(plugin_id)
        
        logger.info(f"Registered plugin: {plugin_id}")
        logger.debug(f"Plugin descriptor: {descriptor}")
        
        return descriptor
    
    def get_ui_component_providers(self, component_name):
        """
        Get all plugins that provide a specific UI component
        
        Args:
            component_name: The name of the UI component
            
        Returns:
            List of plugin info dictionaries
        """
        plugin_ids = self.ui_component_map.get(component_name, [])
        return [self.plugins[plugin_id] for plugin_id in plugin_ids if plugin_id in self.plugins]
    
    def get_plugin_descriptors(self):
        """
        Get socket descriptors for all registered plugins
        
        Returns:
            Dictionary mapping plugin IDs to their descriptors
        """
        return {plugin_id: info["descriptor"] for plugin_id, info in self.plugins.items()}
    
    def connect_ui_to_plugin(self, ui_component, plugin_id):
        """
        Connect a UI component to a plugin
        
        Args:
            ui_component: The UI component to connect
            plugin_id: The ID of the plugin to connect to
            
        Returns:
            True if connection successful, False otherwise
        """
        if plugin_id not in self.plugins:
            logger.error(f"Plugin {plugin_id} not found")
            return False
            
        plugin = self.plugins[plugin_id]["plugin"]
        descriptor = self.plugins[plugin_id]["descriptor"]
        
        # Different connection strategies based on subscription mode
        subscription_mode = descriptor.get("subscription_mode", "push")
        
        try:
            if subscription_mode == "push":
                ui_component.connect_to_socket(plugin.socket, self._handle_push_message)
                logger.info(f"Connected UI component to plugin {plugin_id} in push mode")
            
            elif subscription_mode == "request-response":
                ui_component.set_request_handler(plugin.socket, self._handle_request_response)
                logger.info(f"Connected UI component to plugin {plugin_id} in request-response mode")
            
            elif subscription_mode == "dual":
                ui_component.connect_to_socket(plugin.socket, self._handle_push_message)
                ui_component.set_request_handler(plugin.socket, self._handle_request_response)
                logger.info(f"Connected UI component to plugin {plugin_id} in dual mode")
            
            else:
                logger.error(f"Unknown subscription mode: {subscription_mode}")
                return False
                
            # If plugin has a websocket, connect to that as well
            if "websocket_endpoint" in descriptor:
                ui_component.connect_to_websocket(descriptor["websocket_endpoint"])
                logger.info(f"Connected UI component to websocket endpoint: {descriptor['websocket_endpoint']}")
            
            # Try to load saved state for this component
            if hasattr(ui_component, "component_name"):
                self._load_component_state(ui_component)
            
            return True
        
        except Exception as e:
            logger.error(f"Error connecting UI component to plugin {plugin_id}: {str(e)}")
            return False
    
    def _handle_push_message(self, ui_component, message):
        """
        Handle push messages from plugins to UI components
        
        Args:
            ui_component: The UI component receiving the message
            message: The message data
        """
        try:
            # Update UI component
            ui_component.update(message["type"], message.get("data", {}))
            
            # Save state if component has a name and the message contains state data
            if hasattr(ui_component, "component_name") and "data" in message:
                self._save_component_state(ui_component, message["data"])
        except Exception as e:
            logger.error(f"Error handling push message in UI component: {str(e)}")
    
    def _handle_request_response(self, ui_component, request, callback):
        """
        Handle request-response pattern between UI and plugins
        
        Args:
            ui_component: The UI component making the request
            request: The request data
            callback: Callback function to handle the response
        """
        plugin_id = request.get("plugin_id")
        if not plugin_id:
            logger.error("Request missing plugin_id")
            return
            
        if plugin_id not in self.plugins:
            logger.error(f"Plugin {plugin_id} not found")
            return
            
        plugin = self.plugins[plugin_id]["plugin"]
        
        # Add request ID if not present
        if "request_id" not in request:
            request["request_id"] = str(uuid.uuid4())
        
        try:
            # Send the request to the plugin
            plugin.socket.send_message({
                "type": request["type"],
                "request_id": request["request_id"],
                "content": request.get("content", {})
            })
            
            # Register callback for when response comes back
            plugin.socket.register_response_handler(request["request_id"], callback)
            
        except Exception as e:
            logger.error(f"Error sending request to plugin {plugin_id}: {str(e)}")
    
    def discover_plugins(self):
        """
        Discover plugins using the connection discovery service
        
        Returns:
            List of discovered plugin nodes
        """
        if not self.discovery:
            logger.warning("Cannot discover plugins: connection discovery service not available")
            return []
            
        try:
            # Find all nodes implementing the V5 plugin interface
            nodes = self.discovery.find_nodes_by_type("v5_plugin")
            logger.info(f"Discovered {len(nodes)} V5 plugin nodes")
            
            # Register the nodes
            for node in nodes:
                if hasattr(node, "get_socket_descriptor"):
                    self.register_plugin(node)
                else:
                    logger.warning(f"Discovered node {node.node_id} does not implement the V5 plugin interface")
            
            return nodes
        
        except Exception as e:
            logger.error(f"Error discovering plugins: {str(e)}")
            return []
    
    def start_plugin_discovery(self, interval=30):
        """
        Start periodic plugin discovery
        
        Args:
            interval: Discovery interval in seconds
        """
        if not self.discovery:
            logger.warning("Cannot start plugin discovery: connection discovery service not available")
            return
            
        def discovery_thread():
            while True:
                try:
                    self.discover_plugins()
                except Exception as e:
                    logger.error(f"Error in plugin discovery: {str(e)}")
                finally:
                    threading.Event().wait(interval)
        
        thread = threading.Thread(target=discovery_thread, daemon=True)
        thread.start()
        logger.info(f"Started plugin discovery thread with interval {interval}s")
        
    def get_plugin(self, plugin_id):
        """
        Get a plugin by ID
        
        Args:
            plugin_id: The ID of the plugin to get
            
        Returns:
            The plugin instance or None if not found
        """
        if plugin_id in self.plugins:
            return self.plugins[plugin_id]["plugin"]
        return None
        
    def send_message(self, message):
        """
        Send a message to all registered plugins or a specific plugin
        
        Args:
            message: The message to send. Should be a dictionary with at least 'type' field.
                   Can include 'plugin_id' to target a specific plugin.
        
        Returns:
            True if message was sent, False otherwise
        """
        try:
            # Extract target plugin_id if specified
            target_plugin_id = message.get("plugin_id")
            
            if target_plugin_id:
                # Send to specific plugin
                if target_plugin_id in self.plugins:
                    plugin = self.plugins[target_plugin_id]["plugin"]
                    if hasattr(plugin, "socket"):
                        plugin.socket.send_message(message)
                        return True
                    else:
                        logger.error(f"Plugin {target_plugin_id} does not have a socket")
                else:
                    logger.error(f"Plugin {target_plugin_id} not found")
            else:
                # Broadcast to all plugins that can handle this message type
                msg_type = message.get("type")
                if not msg_type:
                    logger.error("Message has no type field")
                    return False
                
                sent = False
                for plugin_id, info in self.plugins.items():
                    if hasattr(info["plugin"], "socket"):
                        # Check if plugin can handle this message type
                        descriptor = info["descriptor"]
                        message_types = descriptor.get("message_types", [])
                        
                        if msg_type in message_types:
                            info["plugin"].socket.send_message(message)
                            sent = True
                
                return sent
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            
        return False
    
    def register_message_handler(self, message_type, handler):
        """
        Register a handler for a specific message type
        
        Args:
            message_type: The type of message to handle
            handler: The handler function to call
        """
        self.manager_socket.register_message_handler(message_type, handler)
    
    def establish_direct_connection(self, ui_component, plugin_id, component_name):
        """
        Establish a direct connection between a UI component and a specific plugin
        
        This method bypasses the usual component provider mechanism and directly
        establishes a connection between a UI component and a plugin.
        
        Args:
            ui_component: The UI component to connect
            plugin_id: The ID of the plugin to connect to
            component_name: The name of the UI component type
            
        Returns:
            True if connection successful, False otherwise
        """
        if plugin_id not in self.plugins:
            logger.error(f"Plugin {plugin_id} not found")
            return False
            
        plugin = self.plugins[plugin_id]["plugin"]
        descriptor = self.plugins[plugin_id]["descriptor"]
        
        # Log connection attempt
        logger.info(f"Establishing direct connection between UI component and plugin {plugin_id} for {component_name}")
        
        # Ensure component is mapped
        if component_name not in self.ui_component_map:
            self.ui_component_map[component_name] = []
        if plugin_id not in self.ui_component_map[component_name]:
            self.ui_component_map[component_name].append(plugin_id)
            logger.info(f"Added plugin {plugin_id} to UI component map for {component_name}")
        
        # Different connection strategies based on subscription mode
        subscription_mode = descriptor.get("subscription_mode", "push")
        
        try:
            if subscription_mode == "push":
                ui_component.connect_to_socket(plugin.socket, self._handle_push_message)
                logger.info(f"Directly connected UI component to plugin {plugin_id} in push mode")
            
            elif subscription_mode == "request-response":
                ui_component.set_request_handler(plugin.socket, self._handle_request_response)
                logger.info(f"Directly connected UI component to plugin {plugin_id} in request-response mode")
            
            elif subscription_mode == "dual":
                ui_component.connect_to_socket(plugin.socket, self._handle_push_message)
                ui_component.set_request_handler(plugin.socket, self._handle_request_response)
                logger.info(f"Directly connected UI component to plugin {plugin_id} in dual mode")
            
            else:
                logger.error(f"Unknown subscription mode: {subscription_mode}")
                return False
                
            # If plugin has a websocket, connect to that as well
            if "websocket_endpoint" in descriptor:
                ui_component.connect_to_websocket(descriptor["websocket_endpoint"])
                logger.info(f"Connected UI component to websocket endpoint: {descriptor['websocket_endpoint']}")
                
            return True
        
        except Exception as e:
            logger.error(f"Error directly connecting UI component to plugin {plugin_id}: {str(e)}")
            return False
    
    def _save_component_state(self, ui_component, state_data):
        """
        Save UI component state to database
        
        Args:
            ui_component: The UI component
            state_data: State data to save
        """
        try:
            component_name = getattr(ui_component, "component_name", None)
            if not component_name:
                return
            
            # Don't save for certain message types that don't represent stable state
            if state_data.get("type") in ["progress_update", "log_message", "status_update"]:
                return
            
            # Get database manager
            db_manager = DatabaseManager.get_instance()
            
            # Save visualization state
            success = db_manager.save_visualization_state(component_name, state_data)
            
            if success:
                logger.debug(f"Saved state for component: {component_name}")
        except Exception as e:
            logger.error(f"Error saving component state: {str(e)}")
    
    def _load_component_state(self, ui_component):
        """
        Load UI component state from database
        
        Args:
            ui_component: The UI component
            
        Returns:
            True if state was loaded, False otherwise
        """
        try:
            component_name = getattr(ui_component, "component_name", None)
            if not component_name:
                return False
            
            # Get database manager
            db_manager = DatabaseManager.get_instance()
            
            # Load visualization state
            state_data = db_manager.get_latest_visualization_state(component_name)
            
            if state_data:
                # Apply state to UI component
                if hasattr(ui_component, "restore_state"):
                    ui_component.restore_state(state_data)
                    logger.info(f"Restored state for component: {component_name}")
                    return True
                elif hasattr(ui_component, "update"):
                    ui_component.update("restore_state", state_data)
                    logger.info(f"Updated component state: {component_name}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error loading component state: {str(e)}")
            return False 