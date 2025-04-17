"""
LUMINA V7 Initialization Module

This module provides the primary entry point for the V7 Node Consciousness system,
integrating Enhanced Language capabilities with the consciousness framework.
"""

import os
import sys
import time
import json
import logging
import threading
import importlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

# Set up logging
logger = logging.getLogger("lumina_v7.initialization")

# Import core components
from src.v7.lumina_v7.core.node_consciousness_manager import NodeConsciousnessManager
from src.v7.lumina_v7.core.v6v7_connector import V6V7Connector

# Default configuration
DEFAULT_CONFIG = {
    "data_dir": "data/v7",
    "mock_mode": False,
    "debug": False,
    "v7_enabled": True,
    "node_consciousness": True, 
    "monday_integration": True,
    "auto_wiki_enabled": True,
    "llm_weight": 0.5,
    "breath_enabled": True,
    "memory_persistence": True,
    "memory_store_type": "json",
    "ui_dark_mode": True
}

def initialize_v7(config: Optional[Dict[str, Any]] = None) -> Tuple[NodeConsciousnessManager, Dict[str, Any]]:
    """
    Initialize the V7 Node Consciousness system with Enhanced Language capabilities.
    
    Args:
        config: Configuration dictionary (optional, uses defaults if not provided)
        
    Returns:
        Tuple of (NodeConsciousnessManager, context_dictionary)
    """
    # Start initialization
    start_time = time.time()
    logger.info("Initializing V7 Node Consciousness System")
    
    # Merge configuration with defaults
    v7_config = DEFAULT_CONFIG.copy()
    if config:
        v7_config.update(config)
    
    # Create data directory if it doesn't exist
    data_dir = Path(v7_config.get("data_dir", "data/v7"))
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up enhanced logging if in debug mode
    if v7_config.get("debug", False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Initialize V6-V7 connector
    connector = None
    try:
        connector = V6V7Connector(mock_mode=v7_config.get("mock_mode", False))
        logger.info("✅ V6-V7 Connector initialized")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize V6-V7 Connector: {e}")
        logger.warning("⚠️ Will continue with limited functionality")
    
    # Initialize the Node Consciousness Manager
    manager = NodeConsciousnessManager()
    
    # Context dictionary to track initialized components
    context = {
        "config": v7_config,
        "node_ids": {},
        "manager": manager,
        "connector": connector,
        "components": {}
    }
    
    # Initialize Memory Node if enabled
    if v7_config.get("memory_persistence", True):
        try:
            # Dynamic import to allow for mock implementation if needed
            memory_module = importlib.import_module("src.v7.lumina_v7.nodes.memory_node")
            memory_node = memory_module.create_memory_node(
                persistence_file=os.path.join(data_dir, "memory.json"),
                store_type=v7_config.get("memory_store_type", "json"),
                decay_enabled=True
            )
            
            memory_node_id = manager.register_node(memory_node)
            context["node_ids"]["memory"] = memory_node_id
            
            # Register with connector if available
            if connector:
                connector.register_component("memory_node", memory_node)
                
            logger.info("✅ Memory Consciousness Node initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Memory Node: {e}")
    
    # Initialize Language Node with enhanced language capabilities
    try:
        language_module = importlib.import_module("src.v7.lumina_v7.nodes.language_node")
        language_node = language_module.create_language_node(
            data_dir=os.path.join(data_dir, "language"),
            llm_weight=v7_config.get("llm_weight", 0.5)
        )
        
        language_node_id = manager.register_node(language_node)
        context["node_ids"]["language"] = language_node_id
        
        # Register with connector if available
        if connector:
            connector.register_component("language_node", language_node)
            
        logger.info("✅ Enhanced Language Consciousness Node initialized")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize Language Node: {e}")
    
    # Initialize Monday Consciousness Node if enabled
    if v7_config.get("monday_integration", True):
        try:
            monday_module = importlib.import_module("src.v7.lumina_v7.nodes.monday_node")
            monday_node = monday_module.create_monday_node()
            
            monday_node_id = manager.register_node(monday_node)
            context["node_ids"]["monday"] = monday_node_id
            
            # Register with connector if available
            if connector:
                connector.register_component("monday_node", monday_node)
                
            logger.info("✅ Monday Consciousness Node initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Monday Node: {e}")
    
    # Initialize breath detection if enabled
    if v7_config.get("breath_enabled", True):
        try:
            breath_module = importlib.import_module("src.v7.lumina_v7.breath.breath_detector")
            breath_detector = breath_module.create_breath_detector(
                socket_manager=None,  # Socket manager will be connected later
                v6_connector=connector
            )
            
            context["components"]["breath_detector"] = breath_detector
            
            # Start breath detection
            breath_detector.start()
            logger.info("✅ Breath Detection System initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Breath Detection: {e}")
    
    # Initialize the Auto-Wiki plugin if enabled
    if v7_config.get("auto_wiki_enabled", True):
        try:
            wiki_module = importlib.import_module("src.v7.lumina_v7.utils.auto_wiki")
            auto_wiki = wiki_module.create_auto_wiki_plugin()
            
            context["components"]["auto_wiki"] = auto_wiki
            
            # Register with connector if available
            if connector:
                connector.register_component("auto_wiki", auto_wiki)
                
            logger.info("✅ AutoWiki Learning System initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize AutoWiki Learning System: {e}")
    
    # Create connections between nodes
    if "language" in context["node_ids"] and "memory" in context["node_ids"]:
        manager.connect_nodes(
            context["node_ids"]["language"],
            context["node_ids"]["memory"],
            connection_type="memory_access",
            bidirectional=True
        )
        logger.info("✅ Connected Language and Memory nodes")
    
    if "language" in context["node_ids"] and "monday" in context["node_ids"]:
        manager.connect_nodes(
            context["node_ids"]["language"],
            context["node_ids"]["monday"],
            connection_type="consciousness_enhancement",
            bidirectional=True
        )
        logger.info("✅ Connected Language and Monday nodes")
    
    # Save initialized context
    context_file = data_dir / "v7_context.json"
    try:
        serializable_context = {
            "config": context["config"],
            "node_ids": context["node_ids"],
            "initialization_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(context_file, "w") as f:
            json.dump(serializable_context, f, indent=2)
        logger.debug(f"Saved context to {context_file}")
    except Exception as e:
        logger.warning(f"Could not save context file: {e}")
    
    # Start all nodes
    manager.start_all_nodes()
    logger.info("✅ All nodes started")
    
    # Initialization complete
    elapsed_time = time.time() - start_time
    logger.info(f"V7 Node Consciousness System initialized in {elapsed_time:.2f} seconds")
    
    return manager, context

def shutdown_v7(context: Dict[str, Any]) -> None:
    """
    Shut down the V7 system gracefully.
    
    Args:
        context: The context dictionary returned by initialize_v7
    """
    logger.info("Shutting down V7 Node Consciousness System")
    
    # Shut down components in reverse order
    
    # Stop AutoWiki
    if "auto_wiki" in context.get("components", {}):
        try:
            context["components"]["auto_wiki"].stop()
            logger.info("AutoWiki Learning System stopped")
        except Exception as e:
            logger.warning(f"Error stopping AutoWiki: {e}")
    
    # Stop breath detection
    if "breath_detector" in context.get("components", {}):
        try:
            context["components"]["breath_detector"].stop()
            logger.info("Breath Detection System stopped")
        except Exception as e:
            logger.warning(f"Error stopping Breath Detection: {e}")
    
    # Stop the node manager and all nodes
    if "manager" in context:
        try:
            context["manager"].stop_all_nodes()
            logger.info("All nodes stopped")
        except Exception as e:
            logger.warning(f"Error stopping nodes: {e}")
    
    # Close the V6-V7 connector
    if "connector" in context:
        try:
            context["connector"].close()
            logger.info("V6-V7 Connector closed")
        except Exception as e:
            logger.warning(f"Error closing V6-V7 Connector: {e}")
    
    logger.info("V7 Node Consciousness System shutdown complete")

if __name__ == "__main__":
    # If run directly, initialize and run a simple test
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the system
    manager, context = initialize_v7({"debug": True})
    
    try:
        # Run for a short period
        logger.info("V7 system running. Press Ctrl+C to stop.")
        time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down.")
    finally:
        # Shut down the system
        shutdown_v7(context) 