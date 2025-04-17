"""
Bridge System Integration Test

This script tests the basic functionality of the bridge system by:
1. Initializing the Version Bridge Manager
2. Testing mock components
3. Verifying message routing between bridges
4. Checking the status of all components
"""

import logging
import json
import os
import time
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the bridge manager
try:
    from version_bridge_manager import VersionBridgeManager
    logger.info("Successfully imported VersionBridgeManager")
except ImportError as e:
    logger.error(f"Error importing VersionBridgeManager: {e}")
    sys.exit(1)

class MockV5System:
    """Mock implementation of the V5 System for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.connected = False
        logger.info("MockV5System initialized")
    
    def connect(self):
        self.connected = True
        logger.info("MockV5System connected")
        return True
    
    def get_status(self):
        return {
            "status": "active" if self.connected else "inactive",
            "version": "v5.0.0-mock",
            "plugins": ["core", "visualization", "language_memory"]
        }

class MockLanguageMemorySystem:
    """Mock implementation of the Language Memory System for testing."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.connected = False
        self.topics = ["consciousness", "paradox", "resonance", "patterns", "symbols"]
        logger.info("MockLanguageMemorySystem initialized")
    
    def connect(self):
        self.connected = True
        logger.info("MockLanguageMemorySystem connected")
        return True
    
    def get_topics(self):
        return self.topics
    
    def search(self, query, options=None):
        options = options or {}
        return {
            "query": query,
            "results": [
                {"text": f"Memory about {query}", "score": 0.95},
                {"text": f"Another memory related to {query}", "score": 0.85}
            ],
            "total_results": 2
        }
    
    def get_status(self):
        return {
            "status": "active" if self.connected else "inactive",
            "version": "lms-1.0.0-mock",
            "topics_count": len(self.topics)
        }

class TestMessageHandler:
    """Test message handler to verify message routing."""
    
    def __init__(self):
        self.received_messages = []
        logger.info("TestMessageHandler initialized")
    
    def handle_message(self, source, message_type, data):
        self.received_messages.append({
            "source": source,
            "message_type": message_type,
            "data": data,
            "timestamp": time.time()
        })
        logger.info(f"Received message from {source}: {message_type}")
        return True

def test_bridge_manager_initialization():
    """Test basic initialization of the Version Bridge Manager."""
    
    logger.info("Testing bridge manager initialization")
    
    # Initialize with mock mode
    config = {
        "mock_mode": True,
        "log_level": "INFO",
        "v5_language_config": {
            "mock_mode": True
        }
    }
    
    try:
        manager = VersionBridgeManager(config)
        logger.info("Bridge manager initialized successfully")
        
        # Check initial status
        status = manager.get_status()
        logger.info(f"Initial status: {json.dumps(status, indent=2)}")
        
        return manager
    except Exception as e:
        logger.error(f"Error initializing bridge manager: {e}")
        return None

def test_connect_components(manager):
    """Test connecting components to the bridge manager."""
    
    logger.info("Testing component connections")
    
    if manager is None:
        logger.error("Bridge manager is None, skipping test")
        return False
    
    try:
        # Create mock components
        v5_system = MockV5System()
        language_memory_system = MockLanguageMemorySystem()
        
        # Connect components
        manager.connect_to_v5_system(v5_system)
        manager.connect_to_language_memory_system(language_memory_system)
        
        # Check status after connections
        status = manager.get_status()
        logger.info(f"Status after connections: {json.dumps(status, indent=2)}")
        
        return True
    except Exception as e:
        logger.error(f"Error connecting components: {e}")
        return False

def test_message_routing(manager):
    """Test message routing between bridges."""
    
    logger.info("Testing message routing")
    
    if manager is None:
        logger.error("Bridge manager is None, skipping test")
        return False
    
    try:
        # Create test message handler
        test_handler = TestMessageHandler()
        
        # Test bridge availability
        if "v1v2" in manager.bridges:
            # Register message handler for v1v2 bridge
            if manager.register_message_handler("v1v2", "test_message", 
                                              lambda source, msg_type, data: test_handler.handle_message(source, msg_type, data)):
                logger.info("Registered message handler for v1v2 bridge")
                
                # Send message to v1v2 bridge
                result = manager.send_message("v1v2", "test_message", {
                    "text": "Test message from bridge test",
                    "timestamp": time.time()
                })
                logger.info(f"Message sent to v1v2 bridge: {result}")
        else:
            logger.warning("v1v2 bridge not available, skipping message test")
        
        # Test broadcasting
        if len(manager.bridges) > 0:
            results = manager.broadcast_message("test", "broadcast_test", {
                "text": "Test broadcast message",
                "timestamp": time.time()
            })
            logger.info(f"Broadcast results: {results}")
        
        # Check received messages
        logger.info(f"Received messages: {len(test_handler.received_messages)}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing message routing: {e}")
        return False

def main():
    """Main test function."""
    
    logger.info("Starting bridge system integration test")
    
    # Test bridge manager initialization
    manager = test_bridge_manager_initialization()
    if manager is None:
        logger.error("Failed to initialize bridge manager, aborting test")
        return False
    
    # Test connecting components
    if not test_connect_components(manager):
        logger.error("Failed to connect components, aborting test")
        return False
    
    # Test message routing
    if not test_message_routing(manager):
        logger.error("Failed to test message routing, aborting test")
        return False
    
    logger.info("Bridge system integration test completed successfully")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 