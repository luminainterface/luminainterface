"""
Tests for the Version Bridge System
"""

import unittest
import threading
import time
from queue import Queue
from typing import Dict, Any
from src.v7.lumina_v7.core.version_bridge_system import VersionBridgeSystem

class MockSystem:
    """Mock system for testing"""
    def __init__(self, version: str):
        self.version = version
        self.received_data = []
    
    def get_version(self) -> str:
        return self.version
    
    def receive_data(self, data: Dict[str, Any]):
        self.received_data.append(data)

class TestVersionBridgeSystem(unittest.TestCase):
    """Test cases for Version Bridge System"""
    
    def setUp(self):
        """Set up test environment"""
        self.bridge = VersionBridgeSystem(mock_mode=True)  # Enable mock mode for testing
        self.mock_v5 = MockSystem("5.0.0")
        self.mock_v6 = MockSystem("6.0.0")
        self.mock_v7 = MockSystem("7.0.0")
    
    def tearDown(self):
        """Clean up test environment"""
        if self.bridge.running:
            self.bridge.stop()
    
    def test_initialization(self):
        """Test system initialization"""
        self.assertFalse(self.bridge.running)
        self.assertEqual(len(self.bridge.connections), 0)
        self.assertEqual(len(self.bridge.versions), 0)
    
    def test_connect_versions(self):
        """Test connecting different versions"""
        # Connect v5
        success = self.bridge.connect_version("v5", self.mock_v5)
        self.assertTrue(success)
        self.assertIn("v5", self.bridge.connections)
        self.assertEqual(self.bridge.versions["v5"], "5.0.0-mock")  # Note the mock suffix
        
        # Connect v6
        success = self.bridge.connect_version("v6", self.mock_v6)
        self.assertTrue(success)
        self.assertIn("v6", self.bridge.connections)
        
        # Test compatibility matrix
        self.assertIn("v6", self.bridge.compatibility_matrix["v5"])
    
    def test_compatibility_check(self):
        """Test version compatibility checking"""
        # Connect versions with explicit version numbers
        self.mock_v5 = MockSystem("5.0.0")
        self.mock_v6 = MockSystem("6.0.0")
        self.mock_v7 = MockSystem("7.0.0")
        
        self.bridge.connect_version("v5", self.mock_v5)
        self.bridge.connect_version("v6", self.mock_v6)
        self.bridge.connect_version("v7", self.mock_v7)
        
        # Check compatibility (within 2 major versions)
        self.assertTrue(self.bridge.check_compatibility("v5", "v6"))
        self.assertTrue(self.bridge.check_compatibility("v6", "v7"))
        self.assertFalse(self.bridge.check_compatibility("v5", "v7"))  # Should be incompatible (v5 to v7)
    
    def test_data_transfer(self):
        """Test data transfer between compatible versions"""
        self.bridge.connect_version("v5", self.mock_v5)
        self.bridge.connect_version("v6", self.mock_v6)
        
        test_data = {"type": "test", "content": "Hello"}
        success = self.bridge.send_data("v5", "v6", test_data)
        self.assertTrue(success)
        
        # Check if data was received with metadata
        self.assertEqual(len(self.mock_v6.received_data), 1)
        received = self.mock_v6.received_data[0]
        self.assertEqual(received["content"], "Hello")
        self.assertEqual(received["source_version"], "5.0.0-mock")  # Note the mock suffix
        self.assertEqual(received["target_version"], "6.0.0-mock")  # Note the mock suffix
    
    def test_message_handlers(self):
        """Test message handler registration and execution"""
        # Connect both source and target versions
        self.bridge.connect_version("v5", self.mock_v5)
        self.bridge.connect_version("v6", self.mock_v6)
        
        handler_called = [False]
        def test_handler(data):
            handler_called[0] = True
            
        # Register handler
        self.bridge.register_message_handler("v6", "test_type", test_handler)
        
        # Send message that should trigger handler
        test_data = {"type": "test_type", "content": "Test"}
        self.bridge.send_data("v5", "v6", test_data)
        
        # Give handler time to execute
        time.sleep(0.1)
        self.assertTrue(handler_called[0])
    
    def test_broadcast(self):
        """Test broadcasting to compatible versions"""
        self.bridge.connect_version("v5", self.mock_v5)
        self.bridge.connect_version("v6", self.mock_v6)
        self.bridge.connect_version("v7", self.mock_v7)
        
        test_data = {"type": "broadcast", "content": "Hello all"}
        results = self.bridge.broadcast("v6", test_data)
        
        # v6 should broadcast to both v5 and v7
        self.assertTrue(results.get("v5"))
        self.assertTrue(results.get("v7"))
    
    def test_system_lifecycle(self):
        """Test system start/stop functionality"""
        self.bridge.connect_version("v5", self.mock_v5)
        
        # Start system
        success = self.bridge.start()
        self.assertTrue(success)
        self.assertTrue(self.bridge.running)
        
        # Verify thread is running
        self.assertIsNotNone(self.bridge.processing_threads.get("v5"))
        self.assertTrue(self.bridge.processing_threads["v5"].is_alive())
        
        # Stop system
        self.bridge.stop()
        self.assertFalse(self.bridge.running)
        time.sleep(0.1)  # Give thread time to stop
        
        # Verify thread is stopped
        self.assertIsNone(self.bridge.processing_threads.get("v5"))
    
    def test_status_reporting(self):
        """Test system status reporting"""
        self.bridge.connect_version("v5", self.mock_v5)
        self.bridge.connect_version("v6", self.mock_v6)
        
        status = self.bridge.get_status()
        self.assertIn("v5", status["connected_versions"])
        self.assertIn("v6", status["connected_versions"])
        self.assertEqual(status["versions"]["v5"], "5.0.0-mock")  # Note the mock suffix
        self.assertEqual(status["versions"]["v6"], "6.0.0-mock")  # Note the mock suffix
        self.assertFalse(status["running"])

    def test_multiple_message_handlers(self):
        """Test multiple message handlers for different message types"""
        self.bridge.connect_version("v5", self.mock_v5)
        self.bridge.connect_version("v6", self.mock_v6)
        
        # Create handlers for different message types
        handler1_called = [False]
        handler2_called = [False]
        
        def handler1(data):
            handler1_called[0] = True
            
        def handler2(data):
            handler2_called[0] = True
            
        # Register handlers
        self.bridge.register_message_handler("v6", "type1", handler1)
        self.bridge.register_message_handler("v6", "type2", handler2)
        
        # Send messages
        self.bridge.send_data("v5", "v6", {"type": "type1", "content": "Test1"})
        self.bridge.send_data("v5", "v6", {"type": "type2", "content": "Test2"})
        
        # Give handlers time to execute
        time.sleep(0.1)
        self.assertTrue(handler1_called[0])
        self.assertTrue(handler2_called[0])
    
    def test_incompatible_data_transfer(self):
        """Test data transfer between incompatible versions"""
        self.bridge.connect_version("v5", self.mock_v5)
        self.bridge.connect_version("v7", self.mock_v7)
        
        test_data = {"type": "test", "content": "Hello"}
        success = self.bridge.send_data("v5", "v7", test_data)
        self.assertFalse(success)  # Should fail due to incompatibility
        self.assertEqual(len(self.mock_v7.received_data), 0)  # No data should be received
    
    def test_queue_overflow(self):
        """Test handling of queue overflow"""
        self.bridge.connect_version("v5", self.mock_v5)
        self.bridge.connect_version("v6", self.mock_v6)
        
        # Start the system
        self.bridge.start()
        
        # Send multiple messages
        for i in range(100):
            self.bridge.send_data("v5", "v6", {"type": "test", "content": f"Message {i}"})
        
        # Give system time to process
        time.sleep(0.5)
        
        # Check that all messages were received
        self.assertEqual(len(self.mock_v6.received_data), 100)
        
        # Stop the system
        self.bridge.stop()
    
    def test_reconnect_version(self):
        """Test reconnecting a version"""
        # Connect v5 and v6
        success = self.bridge.connect_version("v5", self.mock_v5)
        self.assertTrue(success)
        success = self.bridge.connect_version("v6", self.mock_v6)
        self.assertTrue(success)
        
        # Try to reconnect v5 with different version
        mock_v5_new = MockSystem("5.1.0")
        success = self.bridge.connect_version("v5", mock_v5_new)
        self.assertTrue(success)
        
        # Verify version was updated
        self.assertEqual(self.bridge.versions["v5"], "5.1.0-mock")
        
        # Verify compatibility matrix was updated
        self.assertIn("v6", self.bridge.compatibility_matrix["v5"])

if __name__ == "__main__":
    unittest.main() 