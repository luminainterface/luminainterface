#!/usr/bin/env python3
"""
Simple test script to verify the basic functionality of the bridge system.
"""

import sys
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the bridge manager
try:
    from version_bridge_manager import VersionBridgeManager
    logger.info("Successfully imported VersionBridgeManager")
except ImportError as e:
    logger.error(f"Error importing VersionBridgeManager: {e}")
    sys.exit(1)

def run_simple_test():
    """Run a simple test of the bridge system."""
    logger.info("Running simple test of the bridge system")
    
    # Initialize bridge manager with mock mode
    config = {
        "mock_mode": True,
        "log_level": "DEBUG"
    }
    
    try:
        # Initialize Version Bridge Manager
        manager = VersionBridgeManager(config)
        logger.info("VersionBridgeManager initialized successfully")
        
        # Get status
        status = manager.get_status()
        logger.info(f"Bridge system status: {json.dumps(status, indent=2)}")
        
        # Print available bridges
        logger.info(f"Available bridges: {list(manager.bridges.keys())}")
        
        print("\n" + "="*80)
        print("BRIDGE SYSTEM TEST RESULTS")
        print("="*80)
        print(f"\nBridge Manager: {'✓ SUCCESS' if manager else '✗ FAILED'}")
        print(f"Status: {json.dumps(status, indent=2)}")
        print(f"Available Bridges: {list(manager.bridges.keys())}")
        print("\n" + "="*80)
        
        return True
    except Exception as e:
        logger.error(f"Error during simple test: {e}")
        return False

if __name__ == "__main__":
    success = run_simple_test()
    sys.exit(0 if success else 1) 