#!/usr/bin/env python
"""
Test script for the VersionBridgeManager
"""

import os
import sys
import time
import logging
import argparse

# Import our bridge manager
from version_bridge_manager import VersionBridgeManager

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Version Bridge Manager")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def configure_logging(debug=False):
    """Configure logging for the test"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )
    return logging.getLogger("test_bridge_manager")

def main():
    """Main function to test the bridge manager"""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    logger = configure_logging(args.debug)
    
    logger.info("Starting Version Bridge Manager test")
    
    # Create bridge manager
    logger.info("Creating bridge manager")
    bridge_config = {
        "mock_mode": True if args.mock else False,
        "debug": args.debug
    }
    bridge_manager = VersionBridgeManager(bridge_config)
    
    # Start bridge manager
    logger.info("Starting bridge manager")
    success = bridge_manager.start()
    
    if not success:
        logger.error("Failed to start bridge manager")
        return 1
    
    logger.info("Bridge manager started successfully")
    
    # Get status
    status = bridge_manager.get_status()
    logger.info(f"Bridge manager status: {status}")
    
    # List components
    logger.info("Initialized components:")
    for name, component in bridge_manager.components.items():
        logger.info(f"  - {name}")
    
    # Test getting a component
    language_bridge = bridge_manager.get_component("language_memory_v5_bridge")
    if language_bridge:
        logger.info("Successfully retrieved Language Memory V5 Bridge component")
    else:
        logger.warning("Language Memory V5 Bridge component not found")
    
    v3v4_connector = bridge_manager.get_component("v3v4_connector")
    if v3v4_connector:
        logger.info("Successfully retrieved V3V4 Connector component")
    else:
        logger.warning("V3V4 Connector component not found")
    
    # Keep running for 5 seconds to see monitoring thread in action
    logger.info("Running for 5 seconds...")
    time.sleep(5)
    
    # Stop bridge manager
    logger.info("Stopping bridge manager")
    bridge_manager.stop()
    logger.info("Bridge manager stopped")
    
    logger.info("Test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 