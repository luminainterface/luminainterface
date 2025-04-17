"""
V3-V4 Connector for Integration with V5 System

This module provides the connector class that bridges v3-v4 interfaces
with the V5 system, ensuring smooth integration and data flow between
the different version components.
"""

import logging
import json
import importlib
from typing import Dict, Any, Optional, List, Callable

# Set up logger
logger = logging.getLogger(__name__)

class V3V4Connector:
    """
    V3-V4 Connector for Integration with V5
    
    This class facilitates connections between the v3-v4 interfaces
    and the V5 system, with special focus on breath state integration,
    glyph processing, and neural resonance patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, mock_mode: bool = False):
        """
        Initialize the V3-V4 Connector with optional configuration.
        
        Args:
            config: Configuration dictionary for the connector
            mock_mode: Whether to use mock mode for simulated data and connections
        """
        # Handle config or initialize empty dict
        self.config = config or {}
        
        # Set mock_mode from parameter or config
        if isinstance(config, dict) and "mock_mode" in config:
            self.mock_mode = config["mock_mode"]
        else:
            self.mock_mode = mock_mode
            
        # Add mock_mode to config if not already there
        if "mock_mode" not in self.config:
            self.config["mock_mode"] = self.mock_mode
        
        self.v5_system = None
        self.socket_manager = None
        self.message_handlers = {}
        self.v3_components = {}
        self.v4_components = {}
        self.breath_state_handlers = []
        self.glyph_update_handlers = []
        
        # Initialize component registry
        self._import_v3v4_components()
        
        logger.info(f"V3-V4 Connector initialized (mock_mode={self.mock_mode})")
    
    def _import_v3v4_components(self):
        """
        Import v3-v4 components if available.
        """
        # If mock mode, we might skip real imports
        if self.mock_mode:
            logger.info("Mock mode enabled: Using simulated v3-v4 components")
            # Initialize mock components
            self._setup_mock_components()
            return
            
        # Try to import v3 components
        try:
            v3_module = importlib.import_module("v3_components")
            self.v3_components = {
                "breath_controller": getattr(v3_module, "BreathController", None),
                "glyph_processor": getattr(v3_module, "GlyphProcessor", None),
                "neural_bridge": getattr(v3_module, "NeuralBridge", None)
            }
            logger.info("v3 components imported successfully")
        except ImportError:
            logger.warning("v3 components not available")
        
        # Try to import v4 components
        try:
            v4_module = importlib.import_module("v4_components")
            self.v4_components = {
                "advanced_glyph_system": getattr(v4_module, "AdvancedGlyphSystem", None),
                "resonance_harmonizer": getattr(v4_module, "ResonanceHarmonizer", None),
                "consciousness_interface": getattr(v4_module, "ConsciousnessInterface", None)
            }
            logger.info("v4 components imported successfully")
        except ImportError:
            logger.warning("v4 components not available")
    
    def _setup_mock_components(self):
        """Setup mock components for testing"""
        try:
            # Try to import mock components if available
            from test_connections import MockV3V4Interface
            
            # Create a mock interface
            mock_interface = MockV3V4Interface()
            
            # Set up components to use the mock interface
            class MockComponent:
                def __init__(self, interface):
                    self.interface = interface
                
                def connect_to_v5(self, v5_system):
                    logger.info("Mock component connected to V5")
                    return True
                
                def set_message_callback(self, callback):
                    logger.info("Mock component set message callback")
                    return True
                
                def process_command(self, command_data):
                    logger.info(f"Mock component processing command: {command_data}")
                    return {"status": "success", "mock": True}
                
                def update_resonance(self, resonance_data):
                    logger.info(f"Mock component updating resonance: {resonance_data}")
                    return {"status": "success", "mock": True}
                
                def update_consciousness(self, consciousness_data):
                    logger.info(f"Mock component updating consciousness: {consciousness_data}")
                    return {"status": "success", "mock": True}
            
            # Create mock component instances
            self.v3_components = {
                "breath_controller": lambda: MockComponent(mock_interface),
                "glyph_processor": lambda: MockComponent(mock_interface),
                "neural_bridge": lambda: MockComponent(mock_interface)
            }
            
            self.v4_components = {
                "advanced_glyph_system": lambda: MockComponent(mock_interface),
                "resonance_harmonizer": lambda: MockComponent(mock_interface),
                "consciousness_interface": lambda: MockComponent(mock_interface)
            }
            
            logger.info("Mock components initialized successfully")
            
        except ImportError:
            logger.warning("Mock components not available, creating empty mock components")
            # Create empty dictionaries if test_connections isn't available
            self.v3_components = {}
            self.v4_components = {}
    
    def connect_to_v5(self, v5_system=None):
        """
        Connect to the V5 System.
        
        Args:
            v5_system: The V5 system component
            
        Returns:
            bool: True if connection was successful
        """
        if self.mock_mode:
            logger.info("Mock mode: Simulating connection to V5 System")
            self.v5_system = True  # Use a placeholder value in mock mode
            return True
            
        # Use provided v5_system or try to import one
        if v5_system is not None:
            self.v5_system = v5_system
        else:
            # Try to import and connect to V5 system
            try:
                from src.v5 import v5_system_interface
                self.v5_system = v5_system_interface.get_instance()
            except ImportError:
                logger.warning("V5 system interface not available")
                return False
                
        if self.v5_system is None:
            logger.warning("Failed to connect to V5 System")
            return False
            
        logger.info("Connected to V5 System")
        
        # Connect v3-v4 components to V5
        self._connect_v3_components()
        self._connect_v4_components()
        
        return True
    
    def _connect_v3_components(self):
        """
        Connect v3 components to the V5 system.
        """
        if self.mock_mode:
            logger.info("Mock mode: Simulating v3 component connections")
            # In mock mode, we don't need to actually connect components
            # Just log the simulation for debugging purposes
            for name in ["breath_controller", "glyph_processor", "neural_bridge"]:
                logger.info(f"Mock mode: Simulated connection of v3 {name} to V5 system")
            return
            
        for name, component_class in self.v3_components.items():
            if component_class is not None:
                try:
                    component = component_class()
                    if hasattr(component, "connect_to_v5"):
                        component.connect_to_v5(self.v5_system)
                        logger.info(f"Connected v3 {name} to V5 system")
                    elif hasattr(component, "set_message_callback"):
                        component.set_message_callback(self._v3_message_callback)
                        logger.info(f"Set message callback for v3 {name}")
                except Exception as e:
                    logger.error(f"Error connecting v3 {name} to V5 system: {e}")
    
    def _connect_v4_components(self):
        """
        Connect v4 components to the V5 system.
        """
        if self.mock_mode:
            logger.info("Mock mode: Simulating v4 component connections")
            # In mock mode, we don't need to actually connect components
            # Just log the simulation for debugging purposes
            for name in ["advanced_glyph_system", "resonance_harmonizer", "consciousness_interface"]:
                logger.info(f"Mock mode: Simulated connection of v4 {name} to V5 system")
            return
            
        for name, component_class in self.v4_components.items():
            if component_class is not None:
                try:
                    component = component_class()
                    if hasattr(component, "connect_to_v5"):
                        component.connect_to_v5(self.v5_system)
                        logger.info(f"Connected v4 {name} to V5 system")
                    elif hasattr(component, "set_message_callback"):
                        component.set_message_callback(self._v4_message_callback)
                        logger.info(f"Set message callback for v4 {name}")
                except Exception as e:
                    logger.error(f"Error connecting v4 {name} to V5 system: {e}")
    
    def _v3_message_callback(self, message_type, data):
        """
        Callback for v3 component messages.
        
        Args:
            message_type: Type of the message
            data: Message data
        """
        logger.info(f"Received v3 message: {message_type}")
        self._process_v3v4_message(message_type, data, "v3")
    
    def _v4_message_callback(self, message_type, data):
        """
        Callback for v4 component messages.
        
        Args:
            message_type: Type of the message
            data: Message data
        """
        logger.info(f"Received v4 message: {message_type}")
        self._process_v3v4_message(message_type, data, "v4")
    
    def _process_v3v4_message(self, message_type, data, version):
        """
        Process messages from v3-v4 components.
        
        Args:
            message_type: Type of the message
            data: Message data
            version: The version of the sending component ("v3" or "v4")
        """
        # Map v3-v4 message types to V5 message types
        message_mapping = {
            "breath_state_update": "v5_breath_integration",
            "glyph_update": "v5_glyph_processing",
            "neural_resonance": "v5_resonance_pattern",
            "consciousness_state": "v5_consciousness_integration"
        }
        
        v5_message_type = message_mapping.get(message_type)
        if v5_message_type is None:
            logger.warning(f"No V5 mapping for {version} message type: {message_type}")
            return
        
        # Forward to V5 system if available
        if self.v5_system is not None and hasattr(self.v5_system, "process_message"):
            try:
                # Add metadata about the source
                enriched_data = {
                    **data,
                    "source_version": version,
                    "timestamp": self._get_timestamp()
                }
                
                # Process the message through V5
                result = self.v5_system.process_message(v5_message_type, enriched_data)
                logger.info(f"{version} {message_type} forwarded to V5 system: {result}")
                
                # If the message is a breath state update, notify handlers
                if message_type == "breath_state_update":
                    self._notify_breath_state_handlers(enriched_data)
                
                # If the message is a glyph update, notify handlers
                elif message_type == "glyph_update":
                    self._notify_glyph_update_handlers(enriched_data)
                
                return result
            except Exception as e:
                logger.error(f"Error forwarding {version} {message_type} to V5 system: {e}")
        else:
            logger.warning(f"Cannot forward {version} {message_type}: V5 system not available or missing process_message method")
    
    def _get_timestamp(self):
        """
        Get the current timestamp.
        
        Returns:
            float: Current timestamp
        """
        import time
        return time.time()
    
    def connect_to_socket_manager(self, socket_manager=None):
        """
        Connect to the Frontend Socket Manager.
        
        Args:
            socket_manager: The Frontend Socket Manager component
            
        Returns:
            bool: True if connection was successful
        """
        if self.mock_mode:
            logger.info("Mock mode: Simulating connection to Frontend Socket Manager")
            self.socket_manager = True  # Use a placeholder in mock mode
            self._register_message_handlers()
            return True
        
        # Use provided socket_manager or try to import one
        if socket_manager is not None:
            self.socket_manager = socket_manager
        else:
            # Try to import and connect to socket manager
            try:
                from src.v5.frontend_socket_manager import get_socket_manager
                self.socket_manager = get_socket_manager()
            except ImportError:
                logger.warning("Frontend Socket Manager not available")
                return False
        
        if self.socket_manager is None:
            logger.warning("Failed to connect to Frontend Socket Manager")
            return False
            
        logger.info("Connected to Frontend Socket Manager")
        
        # Register message handlers
        self._register_message_handlers()
        
        return True
    
    def _register_message_handlers(self):
        """
        Register message handlers for v3-v4 interfaces.
        """
        # Register handlers for messages from V5 to v3-v4 components
        self.message_handlers = {
            "v5_to_v3_breath_command": self._handle_breath_command,
            "v5_to_v4_glyph_command": self._handle_glyph_command,
            "v5_to_v3v4_resonance": self._handle_resonance_update,
            "v5_to_v3v4_consciousness": self._handle_consciousness_update
        }
        
        # In mock mode, we just log that handlers would be registered
        if self.mock_mode:
            for message_type in self.message_handlers.keys():
                logger.info(f"Mock mode: Would register handler for {message_type}")
            return
        
        # Register handlers with socket manager
        if self.socket_manager is not None and self.socket_manager is not True:  # Check it's not just a placeholder
            for message_type, handler in self.message_handlers.items():
                try:
                    self.socket_manager.register_message_handler(message_type, handler)
                    logger.info(f"Registered handler for {message_type}")
                except Exception as e:
                    logger.error(f"Error registering handler for {message_type}: {e}")
    
    def _handle_breath_command(self, message):
        """
        Handle breath command messages from V5 to v3 components.
        
        Args:
            message: The message containing breath command data
        
        Returns:
            Dict: Response message
        """
        logger.info(f"Handling breath command: {message.get('command_id', 'unknown')}")
        
        # Extract command data
        command_data = message.get("command_data", {})
        
        # Forward to v3 breath controller if available
        breath_controller_class = self.v3_components.get("breath_controller")
        if breath_controller_class is not None:
            try:
                controller = breath_controller_class()
                if hasattr(controller, "process_command"):
                    result = controller.process_command(command_data)
                    logger.info(f"Breath command forwarded to v3 controller: {result}")
                    return {"status": "success", "result": result}
                else:
                    logger.warning("v3 breath controller missing process_command method")
            except Exception as e:
                logger.error(f"Error forwarding breath command to v3 controller: {e}")
                return {"status": "error", "error": str(e)}
        
        return {"status": "not_processed", "reason": "v3 breath controller not available"}
    
    def _handle_glyph_command(self, message):
        """
        Handle glyph command messages from V5 to v4 components.
        
        Args:
            message: The message containing glyph command data
        
        Returns:
            Dict: Response message
        """
        logger.info(f"Handling glyph command: {message.get('command_id', 'unknown')}")
        
        # Extract command data
        command_data = message.get("command_data", {})
        
        # Forward to v4 glyph system if available
        glyph_system_class = self.v4_components.get("advanced_glyph_system")
        if glyph_system_class is not None:
            try:
                glyph_system = glyph_system_class()
                if hasattr(glyph_system, "process_command"):
                    result = glyph_system.process_command(command_data)
                    logger.info(f"Glyph command forwarded to v4 system: {result}")
                    return {"status": "success", "result": result}
                else:
                    logger.warning("v4 glyph system missing process_command method")
            except Exception as e:
                logger.error(f"Error forwarding glyph command to v4 system: {e}")
                return {"status": "error", "error": str(e)}
        
        return {"status": "not_processed", "reason": "v4 glyph system not available"}
    
    def _handle_resonance_update(self, message):
        """
        Handle resonance update messages from V5 to v3-v4 components.
        
        Args:
            message: The message containing resonance data
        
        Returns:
            Dict: Response message
        """
        logger.info(f"Handling resonance update: {message.get('resonance_id', 'unknown')}")
        
        # Extract resonance data
        resonance_data = message.get("resonance_data", {})
        target_version = message.get("target_version", "any")
        
        # Determine target components based on version
        target_components = []
        if target_version in ["v3", "any"]:
            neural_bridge_class = self.v3_components.get("neural_bridge")
            if neural_bridge_class is not None:
                target_components.append((neural_bridge_class, "v3 neural bridge"))
        
        if target_version in ["v4", "any"]:
            harmonizer_class = self.v4_components.get("resonance_harmonizer")
            if harmonizer_class is not None:
                target_components.append((harmonizer_class, "v4 resonance harmonizer"))
        
        # Forward to target components
        results = []
        for component_class, component_name in target_components:
            try:
                component = component_class()
                if hasattr(component, "update_resonance"):
                    result = component.update_resonance(resonance_data)
                    logger.info(f"Resonance update forwarded to {component_name}: {result}")
                    results.append({"component": component_name, "status": "success", "result": result})
                else:
                    logger.warning(f"{component_name} missing update_resonance method")
                    results.append({"component": component_name, "status": "error", "reason": "missing method"})
            except Exception as e:
                logger.error(f"Error forwarding resonance update to {component_name}: {e}")
                results.append({"component": component_name, "status": "error", "error": str(e)})
        
        if not results:
            return {"status": "not_processed", "reason": "No compatible components available"}
        
        return {"status": "processed", "results": results}
    
    def _handle_consciousness_update(self, message):
        """
        Handle consciousness update messages from V5 to v3-v4 components.
        
        Args:
            message: The message containing consciousness data
        
        Returns:
            Dict: Response message
        """
        logger.info(f"Handling consciousness update: {message.get('consciousness_id', 'unknown')}")
        
        # Extract consciousness data
        consciousness_data = message.get("consciousness_data", {})
        
        # Forward to v4 consciousness interface if available
        interface_class = self.v4_components.get("consciousness_interface")
        if interface_class is not None:
            try:
                interface = interface_class()
                if hasattr(interface, "update_consciousness"):
                    result = interface.update_consciousness(consciousness_data)
                    logger.info(f"Consciousness update forwarded to v4 interface: {result}")
                    return {"status": "success", "result": result}
                else:
                    logger.warning("v4 consciousness interface missing update_consciousness method")
            except Exception as e:
                logger.error(f"Error forwarding consciousness update to v4 interface: {e}")
                return {"status": "error", "error": str(e)}
        
        return {"status": "not_processed", "reason": "v4 consciousness interface not available"}
    
    def register_breath_state_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Register a handler for breath state updates.
        
        Args:
            handler: Handler function that takes a breath state data dictionary
        """
        self.breath_state_handlers.append(handler)
        logger.info(f"Registered breath state handler: {len(self.breath_state_handlers)} total")
    
    def register_glyph_update_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Register a handler for glyph updates.
        
        Args:
            handler: Handler function that takes a glyph update data dictionary
        """
        self.glyph_update_handlers.append(handler)
        logger.info(f"Registered glyph update handler: {len(self.glyph_update_handlers)} total")
    
    def _notify_breath_state_handlers(self, breath_state_data):
        """
        Notify all registered breath state handlers.
        
        Args:
            breath_state_data: Breath state data
        """
        for handler in self.breath_state_handlers:
            try:
                handler(breath_state_data)
            except Exception as e:
                logger.error(f"Error in breath state handler: {e}")
    
    def _notify_glyph_update_handlers(self, glyph_data):
        """
        Notify all registered glyph update handlers.
        
        Args:
            glyph_data: Glyph update data
        """
        for handler in self.glyph_update_handlers:
            try:
                handler(glyph_data)
            except Exception as e:
                logger.error(f"Error in glyph update handler: {e}")
    
    def send_to_v3v4(self, message_type: str, data: Dict[str, Any], target_version: str = "any"):
        """
        Send a message to v3-v4 components.
        
        Args:
            message_type: Type of the message
            data: Message data
            target_version: Target version ("v3", "v4", or "any")
        
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        logger.info(f"Sending message to {target_version} components: {message_type}")
        
        # Check if socket manager is available
        if self.socket_manager is None:
            logger.error("Cannot send message: Socket manager not available")
            return False
        
        # Prepare the message
        message = {
            "type": message_type,
            "data": data,
            "target_version": target_version
        }
        
        # Send the message through the socket manager
        try:
            self.socket_manager.send_message("v3v4_bridge", message)
            logger.info(f"Message sent to {target_version} components: {message_type}")
            return True
        except Exception as e:
            logger.error(f"Error sending message to {target_version} components: {e}")
            return False
    
    def get_status(self):
        """
        Get the status of the V3-V4 Connector.
        
        Returns:
            Dict: Status information
        """
        return {
            "mock_mode": self.mock_mode,
            "v5_system_connected": self.v5_system is not None,
            "socket_manager_connected": self.socket_manager is not None,
            "v3_components_available": {k: v is not None for k, v in self.v3_components.items()},
            "v4_components_available": {k: v is not None for k, v in self.v4_components.items()},
            "message_handlers_registered": len(self.message_handlers) > 0,
            "breath_state_handlers": len(self.breath_state_handlers),
            "glyph_update_handlers": len(self.glyph_update_handlers)
        }

# Main function for testing
def main():
    """
    Main function for testing the V3-V4 Connector.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create V3-V4 Connector
    connector = V3V4Connector()
    
    # Get and print status
    status = connector.get_status()
    logger.info(f"V3-V4 Connector status: {json.dumps(status, indent=2)}")
    
    logger.info("V3-V4 Connector test complete")

if __name__ == "__main__":
    main() 