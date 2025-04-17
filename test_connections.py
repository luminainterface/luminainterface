#!/usr/bin/env python3
"""
Test script to verify connections between v1-v5 through the bridge system.

This script:
1. Initializes the Version Bridge Manager
2. Connects mock components for each version
3. Tests message routing between all versions
4. Verifies all connections are working correctly
"""

import sys
import os
import json
import time
import logging
import traceback
from typing import Dict, Any

# Configure logging to console
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
logger.debug(f"Python path: {sys.path}")

# Import the VersionBridgeManager
try:
    logger.debug("Attempting to import VersionBridgeManager...")
    import version_bridge_manager
    from version_bridge_manager import VersionBridgeManager
    logger.info("Successfully imported VersionBridgeManager")
except ImportError as e:
    logger.error(f"Error importing VersionBridgeManager: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)

class MockV1V2Interface:
    """Mock implementation of the v1-v2 interface for testing."""
    
    def __init__(self, name="v1v2_interface"):
        self.name = name
        self.connected = False
        self.received_messages = []
        self.connection_failures = []
        logger.info(f"Initialized mock {self.name}")
    
    def send_to_v1v2(self, message_type, data):
        """Simulate sending a message to v1-v2 interface."""
        logger.info(f"Mock {self.name} received message: {message_type}")
        self.received_messages.append({
            "type": message_type,
            "data": data,
            "timestamp": time.time()
        })
        return True
    
    def get_status(self):
        """Get the status of the mock component."""
        return {
            "name": self.name,
            "connected": self.connected,
            "messages_received": len(self.received_messages),
            "errors": self.connection_failures
        }

class MockV3V4Interface:
    """Mock implementation of the v3-v4 interface for testing."""
    
    def __init__(self, name="v3v4_interface"):
        self.name = name
        self.connected = False
        self.received_messages = []
        self.connection_failures = []
        logger.info(f"Initialized mock {self.name}")
    
    def send_to_v3v4(self, message_type, data):
        """Simulate sending a message to v3-v4 interface."""
        logger.info(f"Mock {self.name} received message: {message_type}")
        self.received_messages.append({
            "type": message_type,
            "data": data,
            "timestamp": time.time()
        })
        return True
    
    def get_status(self):
        """Get the status of the mock component."""
        return {
            "name": self.name,
            "connected": self.connected,
            "messages_received": len(self.received_messages),
            "errors": self.connection_failures
        }

class MockV5Interface:
    """Mock implementation of the V5 interface for testing."""
    
    def __init__(self, name="v5_interface"):
        self.name = name
        self.connected = False
        self.received_messages = []
        self.connection_failures = []
        logger.info(f"Initialized mock {self.name}")
    
    def send_message(self, message_type, data):
        """Simulate sending a message to V5 interface."""
        logger.info(f"Mock {self.name} received message: {message_type}")
        self.received_messages.append({
            "type": message_type,
            "data": data,
            "timestamp": time.time()
        })
        return True
    
    def get_status(self):
        """Get the status of the mock component."""
        return {
            "name": self.name,
            "connected": self.connected,
            "messages_received": len(self.received_messages),
            "errors": self.connection_failures
        }

def create_mock_interfaces():
    """Create mock interfaces for all versions."""
    logger.info("Creating mock interfaces for all versions")
    
    interfaces = {
        "v1v2": MockV1V2Interface(),
        "v3v4": MockV3V4Interface(),
        "v5_language": MockV5Interface()
    }
    
    return interfaces

def inject_mock_interfaces(manager, interfaces):
    """Inject mock interfaces into the Version Bridge Manager."""
    logger.info("Injecting mock interfaces into Version Bridge Manager")
    
    # Print initial bridges to debug
    logger.debug(f"Initial bridges in manager: {manager.bridges}")
    
    # Replace bridges with mock interfaces
    for bridge_name, interface in interfaces.items():
        manager.bridges[bridge_name] = interface
        interface.connected = True
        logger.info(f"Injected {bridge_name} mock interface")
    
    # Print final bridges to debug
    logger.debug(f"Final bridges in manager: {manager.bridges}")
    
    return manager

def test_message_routing(manager, interfaces):
    """Test message routing between all versions."""
    logger.info("Testing message routing between all versions")
    
    # Create a dictionary to store results
    results = {}
    
    try:
        # Test sending messages from v1v2 to other versions
        logger.info("Testing v1v2 -> v3v4 routing")
        v1v2_to_v3v4 = manager.relay_message("v1v2", "v3v4", "text_update", {
            "text": "Test message from v1v2 to v3v4",
            "timestamp": time.time()
        })
        results["v1v2_to_v3v4"] = v1v2_to_v3v4
        
        logger.info("Testing v1v2 -> v5_language routing")
        v1v2_to_v5 = manager.relay_message("v1v2", "v5_language", "text_update", {
            "text": "Test message from v1v2 to v5_language",
            "timestamp": time.time()
        })
        results["v1v2_to_v5"] = v1v2_to_v5
        
        # Test sending messages from v3v4 to other versions
        logger.info("Testing v3v4 -> v1v2 routing")
        v3v4_to_v1v2 = manager.relay_message("v3v4", "v1v2", "breath_state_update", {
            "state": "deep",
            "timestamp": time.time()
        })
        results["v3v4_to_v1v2"] = v3v4_to_v1v2
        
        logger.info("Testing v3v4 -> v5_language routing")
        v3v4_to_v5 = manager.relay_message("v3v4", "v5_language", "neural_resonance", {
            "resonance": 0.85,
            "timestamp": time.time()
        })
        results["v3v4_to_v5"] = v3v4_to_v5
        
        # Test sending messages from v5_language to other versions
        logger.info("Testing v5_language -> v1v2 routing")
        v5_to_v1v2 = manager.relay_message("v5_language", "v1v2", "memory_update", {
            "memory": "Test memory",
            "timestamp": time.time()
        })
        results["v5_to_v1v2"] = v5_to_v1v2
        
        logger.info("Testing v5_language -> v3v4 routing")
        v5_to_v3v4 = manager.relay_message("v5_language", "v3v4", "fractal_pattern", {
            "pattern": "Test pattern",
            "timestamp": time.time()
        })
        results["v5_to_v3v4"] = v5_to_v3v4
        
        # Test broadcasting from each version
        logger.info("Testing broadcast from v1v2")
        v1v2_broadcast = manager.broadcast_message("v1v2", "broadcast_test", {
            "text": "Broadcast from v1v2",
            "timestamp": time.time()
        })
        results["v1v2_broadcast"] = bool(v1v2_broadcast)  # Convert dict to boolean success
        
        logger.info("Testing broadcast from v3v4")
        v3v4_broadcast = manager.broadcast_message("v3v4", "broadcast_test", {
            "text": "Broadcast from v3v4",
            "timestamp": time.time()
        })
        results["v3v4_broadcast"] = bool(v3v4_broadcast)  # Convert dict to boolean success
        
        logger.info("Testing broadcast from v5_language")
        v5_broadcast = manager.broadcast_message("v5_language", "broadcast_test", {
            "text": "Broadcast from v5_language",
            "timestamp": time.time()
        })
        results["v5_broadcast"] = bool(v5_broadcast)  # Convert dict to boolean success
        
    except Exception as e:
        logger.error(f"Error during message routing test: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return results

def verify_message_reception(interfaces):
    """Verify that all interfaces received the expected messages."""
    logger.info("Verifying message reception")
    
    all_received = True
    
    # Check v1v2 interface
    v1v2 = interfaces["v1v2"]
    v1v2_received = len(v1v2.received_messages)
    logger.info(f"v1v2 interface received {v1v2_received} messages")
    if v1v2_received < 3:  # At least messages from v3v4, v5, and broadcast
        logger.error(f"v1v2 interface expected at least 3 messages, got {v1v2_received}")
        all_received = False
    
    # Print received messages for debugging
    for i, msg in enumerate(v1v2.received_messages):
        logger.debug(f"v1v2 message {i+1}: {msg['type']}")
    
    # Check v3v4 interface
    v3v4 = interfaces["v3v4"]
    v3v4_received = len(v3v4.received_messages)
    logger.info(f"v3v4 interface received {v3v4_received} messages")
    if v3v4_received < 3:  # At least messages from v1v2, v5, and broadcast
        logger.error(f"v3v4 interface expected at least 3 messages, got {v3v4_received}")
        all_received = False
    
    # Print received messages for debugging
    for i, msg in enumerate(v3v4.received_messages):
        logger.debug(f"v3v4 message {i+1}: {msg['type']}")
    
    # Check v5_language interface
    v5 = interfaces["v5_language"]
    v5_received = len(v5.received_messages)
    logger.info(f"v5_language interface received {v5_received} messages")
    if v5_received < 3:  # At least messages from v1v2, v3v4, and broadcast
        logger.error(f"v5_language interface expected at least 3 messages, got {v5_received}")
        all_received = False
    
    # Print received messages for debugging
    for i, msg in enumerate(v5.received_messages):
        logger.debug(f"v5_language message {i+1}: {msg['type']}")
    
    return all_received

def print_test_summary(routing_results, interfaces, all_received):
    """Print a summary of the test results."""
    print("\n" + "="*80)
    print("CONNECTION TEST RESULTS")
    print("="*80)
    
    # Print routing results
    print("\nMessage Routing:")
    for route, success in routing_results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {route:<20}: {status}")
    
    # Print interface status
    print("\nInterface Status:")
    for name, interface in interfaces.items():
        status = interface.get_status()
        print(f"  {name}:")
        print(f"    Connected: {'✓ YES' if status['connected'] else '✗ NO'}")
        print(f"    Messages Received: {status['messages_received']}")
        if status['errors']:
            print(f"    Errors: {len(status['errors'])}")
    
    # Print overall result
    print("\nOverall Connection Status:")
    all_routes_success = all(routing_results.values())
    if all_routes_success and all_received:
        print("  ✓ SUCCESS: All connections between v1-v5 are working correctly!")
    else:
        print("  ✗ FAILED: Some connections between v1-v5 are not working correctly.")
        if not all_routes_success:
            print("    - Message routing failed for some routes")
        if not all_received:
            print("    - Some interfaces did not receive the expected messages")
    
    print("\n" + "="*80)

def main():
    """Main entry point for the connection test."""
    logger.info("Starting connection test")
    
    try:
        # Initialize Version Bridge Manager with mock mode
        logger.info("Initializing Version Bridge Manager")
        config = {
            "mock_mode": True,
            "log_level": "DEBUG"
        }
        
        # Check if VersionBridgeManager is available
        bridge_manager_dir = dir(version_bridge_manager)
        logger.debug(f"Contents of version_bridge_manager module: {bridge_manager_dir}")
        
        manager = VersionBridgeManager(config)
        logger.info("Version Bridge Manager initialized successfully")
        
        # Create and inject mock interfaces
        interfaces = create_mock_interfaces()
        manager = inject_mock_interfaces(manager, interfaces)
        
        # Test message routing
        routing_results = test_message_routing(manager, interfaces)
        
        # Verify message reception
        all_received = verify_message_reception(interfaces)
        
        # Print test summary
        print_test_summary(routing_results, interfaces, all_received)
        
        # Return success/failure
        all_routes_success = all(routing_results.values())
        return 0 if (all_routes_success and all_received) else 1
        
    except Exception as e:
        logger.error(f"Error during connection test: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 