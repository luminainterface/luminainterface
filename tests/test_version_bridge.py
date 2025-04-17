#!/usr/bin/env python3
"""
Tests for LUMINA v7.5 Version Bridge
"""

import unittest
import asyncio
from src.v7_5.version_bridge import VersionBridge
from src.v7_5.signal_system import SignalBus

class TestVersionBridge(unittest.TestCase):
    """Test cases for VersionBridge"""
    
    def setUp(self):
        self.bus = SignalBus()
        self.bridge = VersionBridge(self.bus)
        self.received_messages = []
        
    async def message_handler(self, data):
        self.received_messages.append(data)
        
    async def test_initialize(self):
        """Test bridge initialization"""
        await self.bridge.initialize()
        self.assertTrue(self.bridge.is_initialized)
        self.assertTrue(self.bridge.is_connected)
        
    async def test_version_compatibility(self):
        """Test version compatibility checking"""
        # Test valid versions
        self.assertTrue(self.bridge.is_version_compatible("v7.5", "v7.0"))
        self.assertTrue(self.bridge.is_version_compatible("v7.5", "v6.0"))
        self.assertTrue(self.bridge.is_version_compatible("v7.5", "v5.0"))
        
        # Test invalid versions
        self.assertFalse(self.bridge.is_version_compatible("v7.5", "v4.0"))
        self.assertFalse(self.bridge.is_version_compatible("v7.5", "v8.0"))
        
    async def test_message_transformation(self):
        """Test message transformation between versions"""
        # Test v7.5 to v7.0 transformation
        message = {"type": "data", "content": "test"}
        transformed = await self.bridge.transform_message("v7.5", "v7.0", message)
        self.assertIsNotNone(transformed)
        self.assertEqual(transformed["type"], "data")
        
        # Test v7.0 to v7.5 transformation
        message = {"type": "data", "content": "test"}
        transformed = await self.bridge.transform_message("v7.0", "v7.5", message)
        self.assertIsNotNone(transformed)
        self.assertEqual(transformed["type"], "data")
        
    async def test_message_routing(self):
        """Test message routing between versions"""
        await self.bridge.initialize()
        
        # Register a handler for v7.0 messages
        self.bridge.register_handler("v7.0", self.message_handler)
        
        # Send a message from v7.5 to v7.0
        await self.bridge.route_message("v7.5", "v7.0", {"type": "test", "content": "hello"})
        await asyncio.sleep(0.1)  # Allow time for processing
        
        self.assertEqual(len(self.received_messages), 1)
        self.assertEqual(self.received_messages[0]["content"], "hello")
        
    async def test_error_handling(self):
        """Test error handling in message processing"""
        # Test invalid version
        with self.assertRaises(ValueError):
            await self.bridge.transform_message("v7.5", "invalid", {})
            
        # Test incompatible versions
        with self.assertRaises(ValueError):
            await self.bridge.transform_message("v7.5", "v4.0", {})
            
    async def test_cleanup(self):
        """Test bridge cleanup"""
        await self.bridge.initialize()
        await self.bridge.cleanup()
        self.assertFalse(self.bridge.is_connected)
        self.assertFalse(self.bridge.is_initialized)

if __name__ == '__main__':
    unittest.main() 