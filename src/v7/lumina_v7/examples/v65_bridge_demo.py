#!/usr/bin/env python
"""
LUMINA V6.5 Bridge Connector Demo

This script demonstrates how to use the V6.5 Bridge Connector to connect
all previous system versions (v1-2, v3-4, and v5) to the V7 Node Consciousness system.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the project root to path
script_path = Path(__file__).resolve()
root_path = script_path.parent.parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Import V7 components
from src.v7.lumina_v7 import initialize_v7, shutdown_v7, create_default_config
from src.v7.lumina_v7.core.v65_bridge_connector import create_v65_bridge_connector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("v65_bridge_demo")

def handle_v1v2_event(event_data):
    """Handler for events from the v1-v2 bridge"""
    logger.info(f"Received v1-v2 event: {event_data['type']}")
    if 'data' in event_data:
        logger.info(f"Data: {event_data['data']}")

def handle_v3v4_event(event_data):
    """Handler for events from the v3-v4 connector"""
    logger.info(f"Received v3-v4 event: {event_data['type']}")
    if 'data' in event_data:
        logger.info(f"Data: {event_data['data']}")

def handle_v5_event(event_data):
    """Handler for events from the v5 language memory bridge"""
    logger.info(f"Received v5 event: {event_data['type']}")
    if 'data' in event_data:
        logger.info(f"Data: {event_data['data']}")

def main():
    """Main function to demonstrate V6.5 Bridge Connector"""
    try:
        # Create configuration
        config = create_default_config()
        config.update({
            "mock_mode": True,  # Use mock mode for demonstration
            "debug": True,
            "v65_bridge_enabled": True,
            "enable_v1v2_bridge": True,
            "enable_v3v4_connector": True,
            "enable_v5_language_bridge": True
        })
        
        logger.info("Initializing V7 system")
        
        # Initialize V7 system
        manager, context = initialize_v7(config)
        
        logger.info("Initializing V6.5 Bridge Connector")
        
        # Create V6.5 Bridge Connector
        v65_bridge = create_v65_bridge_connector(config)
        
        # Connect to V7 if connector is available
        if "connector" in context and context["connector"]:
            success = v65_bridge.connect_to_v7(context["connector"])
            logger.info(f"Connected V6.5 Bridge to V7: {success}")
        
        # Register event handlers
        v65_bridge.register_handler("text_input", handle_v1v2_event, source="v1v2_bridge")
        v65_bridge.register_handler("breath_state", handle_v3v4_event, source="v3v4_connector")
        v65_bridge.register_handler("memory_query", handle_v5_event, source="v5_language_bridge")
        
        # Start the bridge
        v65_bridge.start()
        
        # Store bridge in context for cleanup
        context["v65_bridge"] = v65_bridge
        
        # Send some test events through the bridge
        logger.info("Sending test events through the bridge")
        
        # Simulate a text input from v1-v2
        v65_bridge.emit_event(
            "text_input",
            {"text": "Hello from v1-v2 bridge!"},
            source="v1v2_bridge"
        )
        
        # Simulate a breath state from v3-v4
        v65_bridge.emit_event(
            "breath_state",
            {"pattern": "deep-slow", "intensity": 0.7},
            source="v3v4_connector"
        )
        
        # Simulate a memory query from v5
        v65_bridge.emit_event(
            "memory_query",
            {"query": "What is consciousness?", "context": "philosophy"},
            source="v5_language_bridge"
        )
        
        # Wait for events to process
        logger.info("Running for 10 seconds...")
        time.sleep(10)
        
        # Display bridge status
        status = v65_bridge.get_status()
        logger.info(f"Bridge status: {status}")
        
        # Shutdown everything
        logger.info("Shutting down...")
        v65_bridge.stop()
        shutdown_v7(context)
        
        logger.info("Demonstration complete")
        return 0
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 