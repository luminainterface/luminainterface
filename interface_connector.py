"""
Interface Connector for v1-v2 Integration with V5

This module provides the InterfaceConnector class that bridges older v1-v2 interfaces 
with the new V5 system, ensuring backward compatibility and data flow between systems.
"""

import logging
import json
import importlib
from typing import Dict, Any, Optional, Callable

# Set up logger
logger = logging.getLogger(__name__)

class InterfaceConnector:
    """
    Interface Connector for v1-v2 Integration with V5
    
    This class provides methods to connect the v1-v2 interfaces with the V5 system,
    specifically with the Language Memory System and the Frontend Socket Manager.
    """
    
    def __init__(self):
        """Initialize the Interface Connector."""
        self.memory_system = None
        self.socket_manager = None
        self.message_handlers = {}
        self.v1_components = {}
        self.v2_components = {}
        
        # Try to import v1-v2 components
        self._import_v1v2_components()
        
        logger.info("Interface Connector initialized")
    
    def _import_v1v2_components(self):
        """
        Import v1-v2 components if available.
        """
        # Try to import v1 components
        try:
            v1_module = importlib.import_module("v1_components")
            self.v1_components = {
                "pattern_recognizer": getattr(v1_module, "PatternRecognizer", None),
                "text_processor": getattr(v1_module, "TextProcessor", None),
                "connection_manager": getattr(v1_module, "ConnectionManager", None)
            }
            logger.info("v1 components imported successfully")
        except ImportError:
            logger.warning("v1 components not available")
        
        # Try to import v2 components
        try:
            v2_module = importlib.import_module("v2_components")
            self.v2_components = {
                "neural_interface": getattr(v2_module, "NeuralInterface", None),
                "resonance_processor": getattr(v2_module, "ResonanceProcessor", None),
                "pattern_matcher": getattr(v2_module, "PatternMatcher", None)
            }
            logger.info("v2 components imported successfully")
        except ImportError:
            logger.warning("v2 components not available")
    
    def connect_memory_system(self, memory_system):
        """
        Connect to the Language Memory System.
        
        Args:
            memory_system: The Language Memory System component
        """
        self.memory_system = memory_system
        logger.info("Connected to Language Memory System")
        
        # Set up memory access for v1-v2 components
        if self.memory_system is not None:
            # Connect v1 components to memory system if available
            for name, component_class in self.v1_components.items():
                if component_class is not None:
                    try:
                        component = component_class()
                        if hasattr(component, "set_memory_system"):
                            component.set_memory_system(self.memory_system)
                            logger.info(f"Connected v1 {name} to Memory System")
                    except Exception as e:
                        logger.error(f"Error connecting v1 {name} to Memory System: {e}")
            
            # Connect v2 components to memory system if available
            for name, component_class in self.v2_components.items():
                if component_class is not None:
                    try:
                        component = component_class()
                        if hasattr(component, "set_memory_system"):
                            component.set_memory_system(self.memory_system)
                            logger.info(f"Connected v2 {name} to Memory System")
                    except Exception as e:
                        logger.error(f"Error connecting v2 {name} to Memory System: {e}")
    
    def connect_to_socket_manager(self, socket_manager):
        """
        Connect to the Frontend Socket Manager.
        
        Args:
            socket_manager: The Frontend Socket Manager component
        """
        self.socket_manager = socket_manager
        logger.info("Connected to Frontend Socket Manager")
        
        # Register message handlers
        if self.socket_manager is not None:
            self._register_message_handlers()
    
    def _register_message_handlers(self):
        """
        Register message handlers for v1-v2 interfaces.
        """
        # Register handlers for messages from v1-v2 components
        self.message_handlers = {
            "v1_pattern_update": self._handle_v1_pattern_update,
            "v2_resonance_update": self._handle_v2_resonance_update,
            "v1v2_text_process": self._handle_v1v2_text_process,
            "v1v2_query": self._handle_v1v2_query
        }
        
        # Register handlers with socket manager
        if self.socket_manager is not None:
            for message_type, handler in self.message_handlers.items():
                try:
                    self.socket_manager.register_message_handler(message_type, handler)
                    logger.info(f"Registered handler for {message_type}")
                except Exception as e:
                    logger.error(f"Error registering handler for {message_type}: {e}")
    
    def _handle_v1_pattern_update(self, message):
        """
        Handle pattern update messages from v1 components.
        
        Args:
            message: The message containing pattern data
        
        Returns:
            Dict: Response message
        """
        logger.info(f"Handling v1 pattern update: {message.get('pattern_id', 'unknown')}")
        
        # Process pattern data
        pattern_data = message.get("pattern_data", {})
        
        # Forward to memory system if available
        if self.memory_system is not None and hasattr(self.memory_system, "update_pattern"):
            try:
                result = self.memory_system.update_pattern(pattern_data)
                logger.info(f"Pattern update forwarded to memory system: {result}")
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error forwarding pattern update to memory system: {e}")
                return {"status": "error", "error": str(e)}
        
        return {"status": "not_processed", "reason": "Memory system not available"}
    
    def _handle_v2_resonance_update(self, message):
        """
        Handle resonance update messages from v2 components.
        
        Args:
            message: The message containing resonance data
        
        Returns:
            Dict: Response message
        """
        logger.info(f"Handling v2 resonance update: {message.get('resonance_id', 'unknown')}")
        
        # Process resonance data
        resonance_data = message.get("resonance_data", {})
        
        # Forward to memory system if available
        if self.memory_system is not None and hasattr(self.memory_system, "update_resonance"):
            try:
                result = self.memory_system.update_resonance(resonance_data)
                logger.info(f"Resonance update forwarded to memory system: {result}")
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error forwarding resonance update to memory system: {e}")
                return {"status": "error", "error": str(e)}
        
        return {"status": "not_processed", "reason": "Memory system not available"}
    
    def _handle_v1v2_text_process(self, message):
        """
        Handle text processing requests from v1-v2 components.
        
        Args:
            message: The message containing text data
        
        Returns:
            Dict: Response message
        """
        logger.info("Handling v1v2 text process request")
        
        # Extract text data
        text = message.get("text", "")
        options = message.get("options", {})
        
        # Forward to memory system if available
        if self.memory_system is not None and hasattr(self.memory_system, "process_text"):
            try:
                result = self.memory_system.process_text(text, options)
                logger.info("Text processed by memory system")
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error processing text with memory system: {e}")
                return {"status": "error", "error": str(e)}
        
        return {"status": "not_processed", "reason": "Memory system not available"}
    
    def _handle_v1v2_query(self, message):
        """
        Handle query requests from v1-v2 components.
        
        Args:
            message: The message containing query data
        
        Returns:
            Dict: Response message
        """
        logger.info("Handling v1v2 query request")
        
        # Extract query data
        query = message.get("query", "")
        params = message.get("params", {})
        
        # Forward to memory system if available
        if self.memory_system is not None and hasattr(self.memory_system, "query"):
            try:
                result = self.memory_system.query(query, params)
                logger.info("Query processed by memory system")
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error processing query with memory system: {e}")
                return {"status": "error", "error": str(e)}
        
        return {"status": "not_processed", "reason": "Memory system not available"}
    
    def send_to_v1v2(self, message_type, data):
        """
        Send a message to v1-v2 components.
        
        Args:
            message_type: Type of the message
            data: Message data
        
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        logger.info(f"Sending message to v1-v2 components: {message_type}")
        
        # Check if socket manager is available
        if self.socket_manager is None:
            logger.error("Cannot send message: Socket manager not available")
            return False
        
        # Prepare the message
        message = {"type": message_type, "data": data}
        
        # Send the message through the socket manager
        try:
            self.socket_manager.send_message("v1v2_bridge", message)
            logger.info(f"Message sent to v1-v2 components: {message_type}")
            return True
        except Exception as e:
            logger.error(f"Error sending message to v1-v2 components: {e}")
            return False
    
    def get_status(self):
        """
        Get the status of the Interface Connector.
        
        Returns:
            Dict: Status information
        """
        return {
            "memory_system_connected": self.memory_system is not None,
            "socket_manager_connected": self.socket_manager is not None,
            "v1_components_available": len(self.v1_components) > 0,
            "v2_components_available": len(self.v2_components) > 0,
            "message_handlers_registered": len(self.message_handlers) > 0
        }

# Main function for testing
def main():
    """
    Main function for testing the Interface Connector.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create Interface Connector
    connector = InterfaceConnector()
    
    # Get and print status
    status = connector.get_status()
    logger.info(f"Interface Connector status: {json.dumps(status, indent=2)}")
    
    logger.info("Interface Connector test complete")

if __name__ == "__main__":
    main() 