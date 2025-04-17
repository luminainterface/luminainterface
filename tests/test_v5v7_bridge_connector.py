"""
Tests for the V5-V7 Bridge Connector
"""

import unittest
from unittest.mock import MagicMock
from src.v7.lumina_v7.version_bridge_system import VersionBridgeSystem

class MockSystem:
    """Mock system for testing"""
    def __init__(self, version):
        self.version = version
        self.data_received = []
        self.messages_received = []
    
    def get_version(self):
        return f"{self.version}.0.0-mock"
    
    def send_data(self, data):
        self.data_received.append(data)
        return True
    
    def receive_message(self, message):
        self.messages_received.append(message)

class TestV5V7BridgeConnector(unittest.TestCase):
    """Test cases for V5-V7 Bridge Connector"""
    
    def setUp(self):
        """Set up test environment"""
        self.bridge = VersionBridgeSystem(mock_mode=True)
        
        # Initialize mock systems
        self.v5_system = MockSystem("v5")
        self.v7_system = MockSystem("v7")
        
        # Connect systems to bridge
        self.bridge.connect_version("v5", self.v5_system)
        self.bridge.connect_version("v7", self.v7_system)
        
        # Start the bridge
        self.bridge.start()
    
    def tearDown(self):
        """Clean up test environment"""
        self.bridge.stop()
    
    def test_initialization(self):
        """Test bridge initialization"""
        status = self.bridge.get_status()
        self.assertTrue(status["is_running"])
        self.assertEqual(len(status["connected_versions"]), 2)
        self.assertIn("v5", status["connected_versions"])
        self.assertIn("v7", status["connected_versions"])
    
    def test_compatibility_status(self):
        """Test version compatibility check"""
        compatibility = self.bridge.get_compatibility_matrix()
        
        # V5 and V7 should not be directly compatible
        self.assertNotIn("v7", compatibility.get("v5", []))
        self.assertNotIn("v5", compatibility.get("v7", []))
    
    def test_data_translation_v5_to_v7(self):
        """Test data translation from V5 to V7"""
        test_data = {"type": "neural_pattern", "data": [1, 2, 3]}
        
        # Direct translation between incompatible versions should fail
        result = self.bridge.translate_data("v5", "v7", test_data)
        self.assertFalse(result)
        self.assertEqual(len(self.v7_system.data_received), 0)
    
    def test_data_translation_v7_to_v5(self):
        """Test data translation from V7 to V5"""
        test_data = {"type": "consciousness_state", "data": {"level": 5}}
        
        # Direct translation between incompatible versions should fail
        result = self.bridge.translate_data("v7", "v5", test_data)
        self.assertFalse(result)
        self.assertEqual(len(self.v5_system.data_received), 0)
    
    def test_message_handling(self):
        """Test message handler registration and execution"""
        handler_called = [False]
        
        def test_handler(message):
            handler_called[0] = True
        
        self.bridge.register_message_handler("v7", test_handler)
        
        # Send message from v5 to v7 (should not trigger handler due to incompatibility)
        self.bridge.send_message("v5", "v7", "test_message")
        self.assertFalse(handler_called[0])
        self.assertEqual(len(self.v7_system.messages_received), 0)
    
    def test_status_reporting(self):
        """Test status reporting functionality"""
        status = self.bridge.get_status()
        required_keys = [
            "is_running",
            "connected_versions",
            "active_handlers",
            "translation_stats",
            "error_count"
        ]
        for key in required_keys:
            self.assertIn(key, status)

if __name__ == '__main__':
    unittest.main() 