"""
Bridge System Initialization

This script initializes and configures the bridge system, connecting all version bridges,
the V5 Visualization System, and the Language Memory System.
"""

import logging
import argparse
import json
import importlib
import sys
from typing import Dict, Any, Optional

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/bridge_config.json") -> Dict[str, Any]:
    """
    Load the bridge configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_path} not found. Using default configuration.")
        return {
            "log_level": "INFO",
            "mock_mode": True,
            "v1v2_config": {},
            "v3v4_config": {},
            "v5_language_config": {
                "mock_mode": True
            }
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        sys.exit(1)

def import_module_class(module_name: str, class_name: str) -> Optional[type]:
    """
    Import a class from a module.
    
    Args:
        module_name: Name of the module
        class_name: Name of the class
        
    Returns:
        Optional[type]: The imported class or None if not found
    """
    try:
        module = importlib.import_module(module_name)
        class_obj = getattr(module, class_name, None)
        if class_obj is None:
            logger.warning(f"Class {class_name} not found in module {module_name}")
        return class_obj
    except ImportError as e:
        logger.warning(f"Module {module_name} not found: {e}")
        return None

def initialize_bridges(config: Dict[str, Any]):
    """
    Initialize all bridge components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict[str, Any]: Dictionary containing all initialized components
    """
    components = {}
    
    # Import Version Bridge Manager
    version_bridge_manager_class = import_module_class("version_bridge_manager", "VersionBridgeManager")
    if version_bridge_manager_class is not None:
        components["version_bridge_manager"] = version_bridge_manager_class(config)
        logger.info("Version Bridge Manager initialized")
    else:
        logger.error("Failed to import Version Bridge Manager")
        sys.exit(1)
    
    # Import V5 System (optional)
    v5_system_module = config.get("v5_system_module", "v5_system")
    v5_system_class = config.get("v5_system_class", "V5System")
    v5_system_class_obj = import_module_class(v5_system_module, v5_system_class)
    if v5_system_class_obj is not None:
        try:
            components["v5_system"] = v5_system_class_obj(config.get("v5_config", {}))
            logger.info("V5 System initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize V5 System: {e}")
            components["v5_system"] = None
    
    # Import Language Memory System (optional)
    language_memory_module = config.get("language_memory_module", "language_memory")
    language_memory_class = config.get("language_memory_class", "LanguageMemorySystem")
    language_memory_class_obj = import_module_class(language_memory_module, language_memory_class)
    if language_memory_class_obj is not None:
        try:
            components["language_memory_system"] = language_memory_class_obj(config.get("language_memory_config", {}))
            logger.info("Language Memory System initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Language Memory System: {e}")
            components["language_memory_system"] = None
    
    # Import Socket Manager (optional)
    socket_manager_module = config.get("socket_manager_module", "socket_manager")
    socket_manager_class = config.get("socket_manager_class", "SocketManager")
    socket_manager_class_obj = import_module_class(socket_manager_module, socket_manager_class)
    if socket_manager_class_obj is not None:
        try:
            components["socket_manager"] = socket_manager_class_obj(config.get("socket_manager_config", {}))
            logger.info("Socket Manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Socket Manager: {e}")
            components["socket_manager"] = None
    
    return components

def connect_components(components: Dict[str, Any]):
    """
    Connect all components to each other.
    
    Args:
        components: Dictionary containing all initialized components
    """
    version_bridge_manager = components.get("version_bridge_manager")
    if version_bridge_manager is None:
        logger.error("Version Bridge Manager not found in components")
        return
    
    # Connect to V5 System
    v5_system = components.get("v5_system")
    if v5_system is not None:
        version_bridge_manager.connect_to_v5_system(v5_system)
    
    # Connect to Language Memory System
    language_memory_system = components.get("language_memory_system")
    if language_memory_system is not None:
        version_bridge_manager.connect_to_language_memory_system(language_memory_system)
    
    # Connect to Socket Manager
    socket_manager = components.get("socket_manager")
    if socket_manager is not None:
        version_bridge_manager.connect_to_socket_manager(socket_manager)
    
    logger.info("All components connected")

def setup_message_handlers(components: Dict[str, Any], config: Dict[str, Any]):
    """
    Set up message handlers for the Version Bridge Manager.
    
    Args:
        components: Dictionary containing all initialized components
        config: Configuration dictionary
    """
    version_bridge_manager = components.get("version_bridge_manager")
    if version_bridge_manager is None:
        logger.error("Version Bridge Manager not found in components")
        return
    
    # Register handlers for v1v2 messages
    for message_type in ["text_update", "resonance_update", "pattern_update", "memory_query"]:
        version_bridge_manager.register_message_handler("v1v2", message_type, 
            lambda source, msg_type, data: version_bridge_manager.broadcast_message(source, msg_type, data))
    
    # Register handlers for v3v4 messages
    for message_type in ["breath_state_update", "glyph_update", "neural_resonance"]:
        version_bridge_manager.register_message_handler("v3v4", message_type, 
            lambda source, msg_type, data: version_bridge_manager.broadcast_message(source, msg_type, data))
    
    # Register handlers for v5_language messages
    for message_type in ["memory_update", "topic_update", "fractal_pattern"]:
        version_bridge_manager.register_message_handler("v5_language", message_type, 
            lambda source, msg_type, data: version_bridge_manager.broadcast_message(source, msg_type, data))
    
    logger.info("Message handlers set up")

def main():
    """
    Main function to initialize and run the bridge system.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Initialize and run the bridge system")
    parser.add_argument("-c", "--config", default="config/bridge_config.json", help="Path to configuration file")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override mock mode from command line
    if args.mock:
        config["mock_mode"] = True
        if "v5_language_config" in config:
            config["v5_language_config"]["mock_mode"] = True
    
    # Initialize bridge components
    components = initialize_bridges(config)
    
    # Connect components
    connect_components(components)
    
    # Set up message handlers
    setup_message_handlers(components, config)
    
    # Print status
    version_bridge_manager = components.get("version_bridge_manager")
    if version_bridge_manager is not None:
        status = version_bridge_manager.get_status()
        logger.info(f"Bridge system status: {json.dumps(status, indent=2)}")
    
    logger.info("Bridge system initialized and ready")
    
    # Keep the script running
    try:
        logger.info("Press Ctrl+C to exit")
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Bridge system shutting down")

if __name__ == "__main__":
    main() 