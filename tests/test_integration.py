#!/usr/bin/env python3
"""
Integration Tests for LUMINA v7.5
"""

import unittest
import asyncio
from src.v7_5.lumina_core import LUMINACore
from src.v7_5.signal_system import SignalBus
from src.v7_5.version_bridge import VersionBridge
from src.v7_5.central_node import CentralNode

class TestIntegration(unittest.TestCase):
    """Integration tests for LUMINA v7.5 components"""
    
    def setUp(self):
        self.bus = SignalBus()
        self.core = LUMINACore()
        self.bridge = VersionBridge(self.bus)
        self.node = CentralNode(self.bus)
        
    async def test_system_initialization(self):
        """Test full system initialization"""
        # Initialize components
        await self.bus.initialize()
        await self.core.initialize()
        await self.bridge.initialize()
        await self.node.initialize()
        
        # Verify initialization
        self.assertTrue(self.bus._running)
        self.assertTrue(self.core.is_initialized)
        self.assertTrue(self.bridge.is_initialized)
        self.assertTrue(self.node._initialized)
        
    async def test_component_registration(self):
        """Test component registration in central node"""
        await self.node.initialize()
        
        # Register components
        self.assertTrue(self.node.register_component(self.core))
        self.assertTrue(self.node.register_component(self.bridge))
        
        # Verify registration
        self.assertIn("core", self.node.components)
        self.assertIn("version_bridge", self.node.components)
        
    async def test_cross_version_communication(self):
        """Test communication between different versions"""
        await self.core.initialize()
        await self.bridge.initialize()
        
        # Register handlers
        received_messages = []
        async def message_handler(data):
            received_messages.append(data)
            
        self.bridge.register_handler("v7.0", message_handler)
        
        # Send message from v7.5 to v7.0
        await self.core.emit_signal("version.message", {
            "source": "v7.5",
            "target": "v7.0",
            "data": {"type": "test", "content": "hello"}
        })
        
        await asyncio.sleep(0.1)  # Allow time for processing
        self.assertEqual(len(received_messages), 1)
        self.assertEqual(received_messages[0]["content"], "hello")
        
    async def test_error_recovery(self):
        """Test system error recovery"""
        await self.core.initialize()
        await self.bridge.initialize()
        
        # Simulate an error
        await self.core.emit_signal("error", {
            "component": "test",
            "error": "Test error"
        })
        
        # Verify error handling
        self.assertTrue(self.core.is_initialized)  # System should remain initialized
        self.assertTrue(self.bridge.is_initialized)
        
    async def test_system_cleanup(self):
        """Test system cleanup"""
        await self.core.initialize()
        await self.bridge.initialize()
        await self.node.initialize()
        
        # Clean up components
        await self.core.cleanup()
        await self.bridge.cleanup()
        await self.node.cleanup()
        
        # Verify cleanup
        self.assertFalse(self.core.is_initialized)
        self.assertFalse(self.bridge.is_initialized)
        self.assertFalse(self.node._initialized)

if __name__ == '__main__':
    unittest.main() 