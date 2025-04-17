"""
LUMINA V6.5 Bridge Connector for V7 Node Consciousness System

This module provides a connector for Lumina V6.5 to integrate with all previous version
bridges (v1-2, v3-4, and v5) while connecting to the V7 Node Consciousness system.
"""

import os
import sys
import time
import logging
import threading
import importlib
import json
from queue import Queue
from typing import Dict, Any, List, Callable, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger("lumina_v7.v65_bridge_connector")

class V65BridgeConnector:
    """
    Bridge Connector for Lumina V6.5 that connects to all previous version bridges
    and interfaces with the v7 Node Consciousness system. This connector acts as
    a central hub for inter-version communication.
    
    This bridge connects:
    - V1-V2 Bridge (text interface to graphical interface)
    - V3-V4 Connector (advanced interfaces with breath, glyph processing)
    - V5 Bridge (simplified for language memory integration)
    - V7 Node Consciousness (for advanced AI capabilities)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, mock_mode: bool = False):
        """
        Initialize the V6.5 Bridge Connector
        
        Args:
            config: Configuration dictionary
            mock_mode: Enable mock mode for testing
        """
        # Initialize configuration
        self.config = config or {}
        self.mock_mode = self.config.get("mock_mode", mock_mode)
        self.config["mock_mode"] = self.mock_mode
        
        # Setup component containers
        self.bridges = {}
        self.message_handlers = {}
        self.event_queue = Queue()
        self.component_status = {
            "v1v2_bridge": "not_initialized",
            "v3v4_connector": "not_initialized",
            "v5_language_bridge": "not_initialized"
        }
        
        # Threading control
        self.running = False
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Initialize bridges
        self._initialize_bridges()
        
        # Start processing thread
        self._start_processing_thread()
        
        logger.info(f"V6.5 Bridge Connector initialized (mock_mode={self.mock_mode})")
    
    def _initialize_bridges(self):
        """Initialize all bridge components"""
        # Initialize V1-V2 Bridge
        if self.config.get("enable_v1v2_bridge", True):
            self._initialize_v1v2_bridge()
        
        # Initialize V3-V4 Connector
        if self.config.get("enable_v3v4_connector", True):
            self._initialize_v3v4_connector()
        
        # Initialize V5 Language Memory Bridge (simplified)
        if self.config.get("enable_v5_language_bridge", True):
            self._initialize_v5_language_bridge()
    
    def _initialize_v1v2_bridge(self):
        """Initialize the V1-V2 Bridge"""
        logger.info("Initializing V1-V2 Bridge")
        
        try:
            # Try to import the real bridge
            if not self.mock_mode:
                try:
                    # First try src.v6.bridges path (V6 implementation)
                    module_path = "src.v6.bridges.v1v2_bridge"
                    v1v2_module = importlib.import_module(module_path)
                    V1V2Bridge = getattr(v1v2_module, "V1V2Bridge")
                except ImportError:
                    # Fallback to direct import (might be in path)
                    module_path = "v1v2_bridge"
                    v1v2_module = importlib.import_module(module_path)
                    V1V2Bridge = getattr(v1v2_module, "V1V2Bridge")
                
                # Create bridge with configuration
                bridge_config = {
                    "mock_mode": self.mock_mode,
                    "connector_version": "v6.5"
                }
                bridge = V1V2Bridge(bridge_config)
                
                # Initialize the bridge
                if hasattr(bridge, "initialize"):
                    success = bridge.initialize()
                    if not success:
                        logger.warning("V1-V2 Bridge initialization failed, using mock bridge")
                        bridge = self._create_mock_bridge("v1v2_bridge")
                
                self.bridges["v1v2_bridge"] = bridge
                self.component_status["v1v2_bridge"] = "initialized"
                logger.info("✅ V1-V2 Bridge initialized successfully")
            else:
                # Create mock bridge
                self.bridges["v1v2_bridge"] = self._create_mock_bridge("v1v2_bridge")
                self.component_status["v1v2_bridge"] = "mock"
                logger.info("✅ V1-V2 Bridge mock initialized")
        
        except Exception as e:
            logger.error(f"Error initializing V1-V2 Bridge: {e}")
            self.bridges["v1v2_bridge"] = self._create_mock_bridge("v1v2_bridge")
            self.component_status["v1v2_bridge"] = "error"
    
    def _initialize_v3v4_connector(self):
        """Initialize the V3-V4 Connector"""
        logger.info("Initializing V3-V4 Connector")
        
        try:
            # Try to import the real connector
            if not self.mock_mode:
                try:
                    # First try src.v6.bridges path (V6 implementation)
                    module_path = "src.v6.bridges.v3v4_connector"
                    v3v4_module = importlib.import_module(module_path)
                    V3V4Connector = getattr(v3v4_module, "V3V4Connector")
                except ImportError:
                    # Fallback to direct import (might be in path)
                    module_path = "v3v4_connector"
                    v3v4_module = importlib.import_module(module_path)
                    V3V4Connector = getattr(v3v4_module, "V3V4Connector")
                
                # Create connector with configuration
                connector_config = {
                    "mock_mode": self.mock_mode,
                    "connector_version": "v6.5"
                }
                connector = V3V4Connector(connector_config)
                
                # Connect to V5 system if method exists
                if hasattr(connector, "connect_to_v5"):
                    success = connector.connect_to_v5()
                    if not success:
                        logger.warning("V3-V4 Connector failed to connect to V5, will continue with limited functionality")
                
                self.bridges["v3v4_connector"] = connector
                self.component_status["v3v4_connector"] = "initialized"
                logger.info("✅ V3-V4 Connector initialized successfully")
            else:
                # Create mock connector
                self.bridges["v3v4_connector"] = self._create_mock_bridge("v3v4_connector")
                self.component_status["v3v4_connector"] = "mock"
                logger.info("✅ V3-V4 Connector mock initialized")
        
        except Exception as e:
            logger.error(f"Error initializing V3-V4 Connector: {e}")
            self.bridges["v3v4_connector"] = self._create_mock_bridge("v3v4_connector")
            self.component_status["v3v4_connector"] = "error"
    
    def _initialize_v5_language_bridge(self):
        """Initialize the simplified V5 Language Memory Bridge"""
        logger.info("Initializing V5 Language Memory Bridge (simplified)")
        
        try:
            # Try to import the real bridge
            if not self.mock_mode:
                try:
                    # Try src.v6.bridges path (V6 implementation)
                    module_path = "src.v6.bridges.language_memory_v5_bridge"
                    v5_module = importlib.import_module(module_path)
                    LanguageMemoryV5Bridge = getattr(v5_module, "LanguageMemoryV5Bridge")
                except ImportError:
                    # Fallback to direct import (might be in path)
                    module_path = "language_memory_v5_bridge"
                    v5_module = importlib.import_module(module_path)
                    LanguageMemoryV5Bridge = getattr(v5_module, "LanguageMemoryV5Bridge")
                
                # Create bridge with configuration
                bridge_config = {
                    "mock_mode": self.mock_mode,
                    "connector_version": "v6.5",
                    "simplified": True  # Simplified version for v6.5
                }
                bridge = LanguageMemoryV5Bridge(bridge_config)
                
                # Start the bridge if method exists
                if hasattr(bridge, "start"):
                    success = bridge.start()
                    if not success:
                        logger.warning("V5 Language Memory Bridge failed to start, using mock bridge")
                        bridge = self._create_mock_bridge("v5_language_bridge")
                
                self.bridges["v5_language_bridge"] = bridge
                self.component_status["v5_language_bridge"] = "initialized"
                logger.info("✅ V5 Language Memory Bridge initialized successfully")
            else:
                # Create mock bridge
                self.bridges["v5_language_bridge"] = self._create_mock_bridge("v5_language_bridge")
                self.component_status["v5_language_bridge"] = "mock"
                logger.info("✅ V5 Language Memory Bridge mock initialized")
        
        except Exception as e:
            logger.error(f"Error initializing V5 Language Memory Bridge: {e}")
            self.bridges["v5_language_bridge"] = self._create_mock_bridge("v5_language_bridge")
            self.component_status["v5_language_bridge"] = "error"
    
    def _create_mock_bridge(self, bridge_type: str):
        """Create a mock bridge for testing"""
        return MockBridge(bridge_type)
    
    def _start_processing_thread(self):
        """Start the event processing thread"""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            logger.warning("Processing thread already running")
            return
        
        self.running = True
        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._process_events,
            daemon=True,
            name="V65BridgeConnectorThread"
        )
        self.processing_thread.start()
        logger.info("Started event processing thread")
    
    def _stop_processing_thread(self):
        """Stop the event processing thread"""
        self.running = False
        self.stop_event.set()
        
        if self.processing_thread is not None and self.processing_thread.is_alive():
            try:
                self.processing_thread.join(timeout=2.0)
                logger.info("Stopped event processing thread")
            except:
                logger.warning("Failed to join event processing thread")
    
    def _process_events(self):
        """Process events from the queue"""
        while self.running and not self.stop_event.is_set():
            try:
                if self.event_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                
                event = self.event_queue.get(block=False)
                self._handle_event(event)
                self.event_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _handle_event(self, event: Dict[str, Any]):
        """Handle an event from the queue"""
        event_type = event.get("type", "unknown")
        source = event.get("source", "unknown")
        target = event.get("target", None)
        data = event.get("data", {})
        
        logger.debug(f"Handling event: {event_type} from {source} to {target}")
        
        # Check for registered handlers
        handler_key = f"{source}:{event_type}"
        if handler_key in self.message_handlers:
            try:
                for handler in self.message_handlers[handler_key]:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {handler_key}: {e}")
        
        # Route event to target if specified
        if target and target in self.bridges:
            self._route_event_to_bridge(target, event)
        # Otherwise route based on event type and source
        else:
            self._route_event_by_type(event_type, source, data)
    
    def _route_event_to_bridge(self, target: str, event: Dict[str, Any]):
        """Route an event to a specific bridge"""
        bridge = self.bridges.get(target)
        if not bridge:
            logger.warning(f"Target bridge not found: {target}")
            return
        
        try:
            # Different bridges may have different methods for handling events
            if hasattr(bridge, "handle_event"):
                bridge.handle_event(event)
            elif hasattr(bridge, "process_message"):
                bridge.process_message(event)
            elif hasattr(bridge, "send_message"):
                bridge.send_message(event.get("type"), event.get("data"))
            else:
                logger.warning(f"No suitable method found to handle event on {target}")
        except Exception as e:
            logger.error(f"Error routing event to {target}: {e}")
    
    def _route_event_by_type(self, event_type: str, source: str, data: Dict[str, Any]):
        """Route an event based on type and source"""
        # Route based on known event types
        if event_type in ["text_input", "text_command"]:
            # Text events from V1-V2 should go to V3-V4 and V5
            if source == "v1v2_bridge":
                self._send_to_bridge("v3v4_connector", event_type, data)
                self._send_to_bridge("v5_language_bridge", "process_text", data)
        
        elif event_type in ["breath_state", "breath_pattern"]:
            # Breath events from V3-V4 should go to V5 and V7
            if source == "v3v4_connector":
                self._send_to_bridge("v5_language_bridge", "breath_integration", data)
                self.emit_v7_event("breath_update", data)
        
        elif event_type in ["glyph_update", "visual_pattern"]:
            # Glyph events from V3-V4 should go to V5
            if source == "v3v4_connector":
                self._send_to_bridge("v5_language_bridge", "process_glyph", data)
        
        elif event_type in ["memory_query", "memory_update"]:
            # Memory events from V5 should go to V7
            if source == "v5_language_bridge":
                self.emit_v7_event("memory_update", data)
        
        elif event_type in ["topic_update", "language_pattern"]:
            # Language events from V5 should go to V7
            if source == "v5_language_bridge":
                self.emit_v7_event("language_update", data)
        
        elif event_type in ["node_request", "node_response"]:
            # Node events from V7 should go to V5
            if source == "v7":
                self._send_to_bridge("v5_language_bridge", "process_node_message", data)
    
    def _send_to_bridge(self, bridge_id: str, message_type: str, data: Dict[str, Any]):
        """Send a message to a specific bridge"""
        bridge = self.bridges.get(bridge_id)
        if not bridge:
            logger.warning(f"Bridge not found: {bridge_id}")
            return False
        
        try:
            # Different bridges may have different methods for sending messages
            if hasattr(bridge, "send_message"):
                bridge.send_message(message_type, data)
            elif hasattr(bridge, "process_message"):
                bridge.process_message({
                    "type": message_type,
                    "data": data,
                    "source": "v65_bridge_connector"
                })
            elif hasattr(bridge, "handle_event"):
                bridge.handle_event({
                    "type": message_type,
                    "data": data,
                    "source": "v65_bridge_connector"
                })
            else:
                logger.warning(f"No suitable method found to send message to {bridge_id}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error sending message to {bridge_id}: {e}")
            return False
    
    def emit_event(self, event_type: str, data: Dict[str, Any], source: str = "v65_bridge_connector", 
                  target: Optional[str] = None):
        """
        Emit an event to be processed by the connector
        
        Args:
            event_type: Type of the event
            data: Event data
            source: Source of the event
            target: Optional target bridge
        """
        event = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        }
        
        if target:
            event["target"] = target
        
        self.event_queue.put(event)
        return True
    
    def emit_v7_event(self, event_type: str, data: Dict[str, Any]):
        """
        Emit an event specifically for V7 Node Consciousness
        
        Args:
            event_type: Type of the event
            data: Event data
        """
        # This will be connected to V7's event system by the V6V7Connector
        logger.debug(f"Emitting V7 event: {event_type}")
        
        # Add v6.5 metadata
        enriched_data = {
            **data,
            "source_system": "v6.5_bridge",
            "timestamp": time.time()
        }
        
        # Signal that this event is intended for V7
        self.emit_event(event_type, enriched_data, source="v65_to_v7")
        return True
    
    def register_handler(self, event_type: str, handler: Callable, source: str = None):
        """
        Register a handler for a specific event type
        
        Args:
            event_type: Type of the event
            handler: Handler function
            source: Optional source filter
        """
        handler_key = f"{source or '*'}:{event_type}"
        
        if handler_key not in self.message_handlers:
            self.message_handlers[handler_key] = []
        
        self.message_handlers[handler_key].append(handler)
        logger.debug(f"Registered handler for {handler_key}")
        return True
    
    def start(self):
        """Start all bridges and the connector"""
        logger.info("Starting V6.5 Bridge Connector")
        
        success = True
        
        # Start all bridges
        for bridge_id, bridge in self.bridges.items():
            try:
                if hasattr(bridge, "start"):
                    bridge_success = bridge.start()
                    if not bridge_success:
                        logger.warning(f"Failed to start {bridge_id}")
                        success = False
                    else:
                        logger.info(f"Started {bridge_id}")
            except Exception as e:
                logger.error(f"Error starting {bridge_id}: {e}")
                success = False
        
        # Start processing thread if not already running
        if not self.processing_thread or not self.processing_thread.is_alive():
            self._start_processing_thread()
        
        logger.info(f"V6.5 Bridge Connector started (success={success})")
        return success
    
    def stop(self):
        """Stop all bridges and the connector"""
        logger.info("Stopping V6.5 Bridge Connector")
        
        # Stop processing thread
        self._stop_processing_thread()
        
        # Stop all bridges
        for bridge_id, bridge in self.bridges.items():
            try:
                if hasattr(bridge, "stop"):
                    bridge.stop()
                    logger.info(f"Stopped {bridge_id}")
            except Exception as e:
                logger.error(f"Error stopping {bridge_id}: {e}")
        
        logger.info("V6.5 Bridge Connector stopped")
        return True
    
    def get_status(self):
        """Get status of all components"""
        status = {
            "mock_mode": self.mock_mode,
            "running": self.running,
            "components": self.component_status.copy(),
            "bridges": {}
        }
        
        # Get status from each bridge if available
        for bridge_id, bridge in self.bridges.items():
            try:
                if hasattr(bridge, "get_status"):
                    status["bridges"][bridge_id] = bridge.get_status()
                else:
                    status["bridges"][bridge_id] = {"available": True}
            except Exception as e:
                logger.error(f"Error getting status from {bridge_id}: {e}")
                status["bridges"][bridge_id] = {"error": str(e)}
        
        return status
    
    def connect_to_v7(self, v7_connector):
        """
        Connect this bridge to the V7 system
        
        Args:
            v7_connector: The V7 connector instance
        
        Returns:
            bool: True if connection successful
        """
        if not v7_connector:
            logger.warning("No V7 connector provided")
            return False
        
        logger.info("Connecting V6.5 Bridge to V7 system")
        
        # Register event handlers from V7
        if hasattr(v7_connector, "register_handler") or hasattr(v7_connector, "on"):
            try:
                register_method = getattr(v7_connector, "register_handler", None) or getattr(v7_connector, "on")
                
                # Register V7 event handlers
                for event_type in ["text_input", "breath_state", "node_update", "memory_query"]:
                    register_method(event_type, lambda data: self.emit_event(
                        event_type, data, source="v7"
                    ))
                
                logger.info("Registered V7 event handlers")
            except Exception as e:
                logger.error(f"Error registering V7 event handlers: {e}")
                return False
        
        # Register this bridge's events with V7
        if hasattr(v7_connector, "register_source") or hasattr(v7_connector, "add_source"):
            try:
                register_method = getattr(v7_connector, "register_source", None) or getattr(v7_connector, "add_source")
                register_method("v6.5_bridge", self.emit_v7_event)
                logger.info("Registered as event source with V7")
            except Exception as e:
                logger.error(f"Error registering as event source with V7: {e}")
                return False
        
        logger.info("Successfully connected V6.5 Bridge to V7 system")
        return True


class MockBridge:
    """Mock bridge implementation for testing"""
    
    def __init__(self, bridge_type: str):
        """Initialize the mock bridge"""
        self.bridge_type = bridge_type
        self.running = False
        self.messages = []
    
    def start(self):
        """Start the mock bridge"""
        self.running = True
        logger.info(f"Mock {self.bridge_type} started")
        return True
    
    def stop(self):
        """Stop the mock bridge"""
        self.running = False
        logger.info(f"Mock {self.bridge_type} stopped")
        return True
    
    def get_status(self):
        """Get mock bridge status"""
        return {
            "mock": True,
            "running": self.running,
            "type": self.bridge_type,
            "messages_received": len(self.messages)
        }
    
    def handle_event(self, event):
        """Handle an event"""
        self.messages.append(event)
        logger.debug(f"Mock {self.bridge_type} received event: {event['type']}")
        return True
    
    def process_message(self, message):
        """Process a message"""
        self.messages.append(message)
        logger.debug(f"Mock {self.bridge_type} received message: {message['type']}")
        return True
    
    def send_message(self, message_type, data):
        """Send a message (just logs it)"""
        logger.debug(f"Mock {self.bridge_type} would send message: {message_type}")
        return True


# Factory function for easy instantiation
def create_v65_bridge_connector(config=None, mock_mode=False):
    """Create a new V6.5 Bridge Connector instance"""
    connector = V65BridgeConnector(config, mock_mode)
    return connector 