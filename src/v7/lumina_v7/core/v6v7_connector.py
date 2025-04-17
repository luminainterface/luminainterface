"""
V6V7 Connector

This module provides connectivity between V6 (Portal of Contradiction) and
V7 (Node Consciousness) systems, enabling backward compatibility and
integration between systems.
"""

import logging
import time
import threading
import importlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union

# Set up logging
logger = logging.getLogger("lumina_v7.v6v7_connector")

class V6V7Connector:
    """
    Connector providing integration between V6 and V7 systems.
    
    This class serves as a bridge for communication between V6 components
    (Portal of Contradiction) and V7 components (Node Consciousness),
    allowing for seamless backwards compatibility and data exchange.
    """
    
    def __init__(self, mock_mode: bool = False, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the V6-V7 connector.
        
        Args:
            mock_mode: If True, use mock implementations for missing components
            config: Configuration dictionary (optional)
        """
        # Core configuration
        self.config = {
            "mock_mode": mock_mode,
            "enable_v6": True,
            "enable_symbolic_state": True,
            "enable_breath_detection": True,
            "enable_contradiction_bridge": True,
            "v6_data_dir": "data/v6",
            "event_buffer_size": 100,
            "event_processing_interval": 0.1  # seconds
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Component registry
        self.components = {}
        self.component_status = {}
        self.event_handlers = {}
        
        # Event loop management
        self.events_buffer = []
        self.events_lock = threading.Lock()
        self.event_processing_thread = None
        self.stop_event_processing = threading.Event()
        
        # V6 integration components
        self.v6_symbolic_state_manager = None
        self.v6_language_memory = None
        self.v6_connector = None
        
        # Initialize V6 components if enabled
        if self.config["enable_v6"]:
            self._initialize_v6_components()
        
        # Start the event processing thread
        self._start_event_processing()
        
        logger.info(f"V6-V7 Connector initialized (mock_mode: {mock_mode})")
    
    def _initialize_v6_components(self) -> None:
        """Initialize V6 system components for integration"""
        # Try to import the V6 symbolic state manager
        if self.config["enable_symbolic_state"]:
            try:
                symbolic_state_module = importlib.import_module("src.v6.ui.symbolic_state_manager")
                self.v6_symbolic_state_manager = symbolic_state_module.SymbolicStateManager()
                self.component_status["v6_symbolic_state"] = "active"
                logger.info("✅ V6 Symbolic State Manager initialized")
            except ImportError:
                logger.warning("⚠️ V6 Symbolic State Manager not found")
                self.component_status["v6_symbolic_state"] = "not_found"
        
        # Try to import the V6 language memory connector
        try:
            language_memory_module = importlib.import_module("src.language_memory_v6_v10_connector")
            self.v6_language_memory = language_memory_module.LanguageMemoryAdvancedConnector()
            self.component_status["v6_language_memory"] = "active"
            logger.info("✅ V6 Language Memory Connector initialized")
        except ImportError:
            logger.warning("⚠️ V6 Language Memory Connector not found")
            self.component_status["v6_language_memory"] = "not_found"
    
    def _start_event_processing(self) -> None:
        """Start the background event processing thread"""
        self.stop_event_processing.clear()
        self.event_processing_thread = threading.Thread(
            target=self._process_events,
            daemon=True,
            name="V6V7EventProcessor"
        )
        self.event_processing_thread.start()
    
    def _process_events(self) -> None:
        """Background thread to process events in the buffer"""
        while not self.stop_event_processing.is_set():
            # Sleep briefly to avoid CPU overuse
            time.sleep(self.config["event_processing_interval"])
            
            # Process events in the buffer
            with self.events_lock:
                if not self.events_buffer:
                    continue
                
                current_events = self.events_buffer.copy()
                self.events_buffer = []
            
            # Process each event
            for event in current_events:
                try:
                    self._route_event(event)
                except Exception as e:
                    logger.error(f"Error processing event: {str(e)}")
    
    def _route_event(self, event: Dict[str, Any]) -> None:
        """
        Route an event to the appropriate handler(s).
        
        Args:
            event: The event to route
        """
        event_type = event.get("type", "unknown")
        
        # Log the event (debugging only for important events)
        if event_type not in ["heartbeat", "status_update"]:
            logger.debug(f"Routing event: {event_type}")
        
        # Check for handlers for this event type
        handlers = self.event_handlers.get(event_type, [])
        
        # Check for wildcard handlers (handle all events)
        wildcard_handlers = self.event_handlers.get("*", [])
        
        # Combine handlers
        all_handlers = handlers + wildcard_handlers
        
        # If no handlers, log and return
        if not all_handlers:
            logger.debug(f"No handlers for event type: {event_type}")
            return
        
        # Call each handler
        for handler in all_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {str(e)}")
    
    def register_component(self, component_id: str, component: Any) -> bool:
        """
        Register a component with the connector.
        
        Args:
            component_id: Unique identifier for the component
            component: The component instance
        
        Returns:
            True if registration successful, False otherwise
        """
        if component_id in self.components:
            logger.warning(f"Component {component_id} already registered")
            return False
        
        self.components[component_id] = component
        self.component_status[component_id] = "active"
        
        # If component has event handlers, register them
        if hasattr(component, "get_event_handlers"):
            try:
                handlers = component.get_event_handlers()
                if handlers and isinstance(handlers, dict):
                    for event_type, handler in handlers.items():
                        self.register_event_handler(event_type, handler)
            except Exception as e:
                logger.error(f"Error registering event handlers for {component_id}: {str(e)}")
        
        logger.info(f"Registered component: {component_id}")
        return True
    
    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a component from the connector.
        
        Args:
            component_id: The component ID
        
        Returns:
            True if unregistered, False otherwise
        """
        if component_id not in self.components:
            logger.warning(f"Component {component_id} not found")
            return False
        
        # If component has event handlers, remove them
        component = self.components[component_id]
        if hasattr(component, "get_event_handlers"):
            try:
                handlers = component.get_event_handlers()
                if handlers and isinstance(handlers, dict):
                    for event_type, handler in handlers.items():
                        self.unregister_event_handler(event_type, handler)
            except Exception as e:
                logger.error(f"Error unregistering event handlers for {component_id}: {str(e)}")
        
        # Remove the component
        del self.components[component_id]
        self.component_status[component_id] = "unregistered"
        
        logger.info(f"Unregistered component: {component_id}")
        return True
    
    def get_component(self, component_id: str) -> Optional[Any]:
        """
        Get a registered component by ID.
        
        Args:
            component_id: The component ID
        
        Returns:
            The component instance, or None if not found
        """
        return self.components.get(component_id)
    
    def register_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register an event handler for the specified event type.
        
        Args:
            event_type: The type of event to handle
            handler: The function to call when an event of this type occurs
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        if handler not in self.event_handlers[event_type]:
            self.event_handlers[event_type].append(handler)
            logger.debug(f"Registered handler for event type: {event_type}")
    
    def unregister_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unregister an event handler.
        
        Args:
            event_type: The event type
            handler: The handler function to remove
        """
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            logger.debug(f"Unregistered handler for event type: {event_type}")
    
    def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None, 
                  source: Optional[str] = None) -> None:
        """
        Emit an event to be processed by the connector.
        
        Args:
            event_type: The type of event
            data: Event data dictionary
            source: Event source identifier
        """
        # Create the event
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "source": source or "v6v7_connector",
            "data": data or {}
        }
        
        # Add to the event buffer
        with self.events_lock:
            self.events_buffer.append(event)
            
            # Limit buffer size
            if len(self.events_buffer) > self.config["event_buffer_size"]:
                self.events_buffer = self.events_buffer[-self.config["event_buffer_size"]:]
    
    def get_v6_state(self) -> Dict[str, Any]:
        """
        Get the current state of V6 components.
        
        Returns:
            Dictionary with V6 state information
        """
        state = {
            "components": self.component_status.copy(),
            "symbolic_state": None,
            "language_memory_status": None
        }
        
        # Get symbolic state if available
        if self.v6_symbolic_state_manager:
            try:
                state["symbolic_state"] = self.v6_symbolic_state_manager.get_current_state()
            except Exception as e:
                logger.error(f"Error getting symbolic state: {str(e)}")
        
        # Get language memory status if available
        if self.v6_language_memory:
            try:
                state["language_memory_status"] = {
                    "active": True,
                    "components": self.v6_language_memory.get_available_components()
                }
            except Exception as e:
                logger.error(f"Error getting language memory status: {str(e)}")
        
        return state
    
    def send_to_v6(self, component: str, action: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send a command or data to a V6 component.
        
        Args:
            component: The target V6 component (e.g., 'symbolic_state', 'language_memory')
            action: The action to perform
            data: The data to send
        
        Returns:
            Response data from the V6 component, or None if not successful
        """
        # Handle symbolic state actions
        if component == "symbolic_state" and self.v6_symbolic_state_manager:
            try:
                if action == "set_state":
                    self.v6_symbolic_state_manager.set_state(data.get("state"))
                    return {"success": True}
                elif action == "get_state":
                    return {"state": self.v6_symbolic_state_manager.get_current_state()}
                else:
                    logger.warning(f"Unknown symbolic state action: {action}")
                    return None
            except Exception as e:
                logger.error(f"Error in symbolic state action {action}: {str(e)}")
                return None
        
        # Handle language memory actions
        elif component == "language_memory" and self.v6_language_memory:
            try:
                if action == "process_text":
                    return self.v6_language_memory.process_text(data.get("text"), data.get("context", {}))
                elif action == "get_memory":
                    return self.v6_language_memory.get_memory(data.get("identifier"))
                else:
                    logger.warning(f"Unknown language memory action: {action}")
                    return None
            except Exception as e:
                logger.error(f"Error in language memory action {action}: {str(e)}")
                return None
        
        # Handle generic V6 connector actions
        elif component == "v6_connector" and self.v6_connector:
            try:
                if action == "send_message":
                    return self.v6_connector.send_message(data.get("target"), data.get("message"))
                else:
                    logger.warning(f"Unknown V6 connector action: {action}")
                    return None
            except Exception as e:
                logger.error(f"Error in V6 connector action {action}: {str(e)}")
                return None
        
        else:
            logger.warning(f"Unknown component or component not available: {component}")
            return None
    
    def synchronize_states(self) -> bool:
        """
        Synchronize states between V6 and V7 components.
        
        Returns:
            True if synchronization successful, False otherwise
        """
        success = True
        
        # Get states from V7 components
        v7_states = {}
        for component_id, component in self.components.items():
            if hasattr(component, "get_state") or hasattr(component, "get_status"):
                try:
                    if hasattr(component, "get_state"):
                        v7_states[component_id] = component.get_state()
                    else:
                        v7_states[component_id] = component.get_status()
                except Exception as e:
                    logger.error(f"Error getting state for {component_id}: {str(e)}")
                    success = False
        
        # Get V6 state
        v6_state = self.get_v6_state()
        
        # Emit synchronization event
        self.emit_event("state_synchronization", {
            "v6_state": v6_state,
            "v7_states": v7_states
        })
        
        return success
    
    def close(self) -> None:
        """
        Close the connector and stop all processing.
        """
        logger.info("Closing V6-V7 Connector")
        
        # Stop the event processing thread
        self.stop_event_processing.set()
        if self.event_processing_thread and self.event_processing_thread.is_alive():
            self.event_processing_thread.join(timeout=2.0)
        
        # Unregister all components
        component_ids = list(self.components.keys())
        for component_id in component_ids:
            self.unregister_component(component_id)
        
        logger.info("V6-V7 Connector closed")

if __name__ == "__main__":
    # If run directly, initialize and test the connector
    logging.basicConfig(level=logging.INFO)
    
    connector = V6V7Connector(mock_mode=True)
    
    # Register a test component
    class TestComponent:
        def __init__(self):
            self.name = "test_component"
        
        def get_status(self):
            return {"status": "active", "name": self.name}
        
        def get_event_handlers(self):
            return {
                "test_event": self.handle_test_event
            }
        
        def handle_test_event(self, event):
            logger.info(f"Test component handling event: {event}")
    
    test_component = TestComponent()
    connector.register_component("test", test_component)
    
    # Emit a test event
    connector.emit_event("test_event", {"message": "Hello from test event"})
    
    # Sleep briefly to allow event processing
    time.sleep(1)
    
    # Close the connector
    connector.close() 