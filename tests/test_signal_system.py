#!/usr/bin/env python3
"""
Tests for LUMINA v7.5 Signal System
"""

import unittest
import asyncio
from src.v7_5.signal_system import SignalBus
from src.v7_5.signal_component import SignalComponent

class TestSignalBus(unittest.TestCase):
    """Test cases for SignalBus"""
    
    def setUp(self):
        self.bus = SignalBus()
        self.received_signals = []
        
    async def signal_handler(self, data):
        self.received_signals.append(data)
        
    async def test_initialize(self):
        """Test signal bus initialization"""
        result = await self.bus.initialize()
        self.assertTrue(result)
        self.assertTrue(self.bus._running)
        
    async def test_register_handler(self):
        """Test handler registration"""
        self.bus.register_handler("test_signal", self.signal_handler)
        self.assertIn("test_signal", self.bus._handlers)
        
    async def test_emit_signal(self):
        """Test signal emission"""
        self.bus.register_handler("test_signal", self.signal_handler)
        await self.bus.emit("test_signal", "test_data")
        await asyncio.sleep(0.1)  # Allow time for processing
        self.assertIn("test_data", self.received_signals)
        
    async def test_cleanup(self):
        """Test signal bus cleanup"""
        await self.bus.initialize()
        await self.bus.cleanup()
        self.assertFalse(self.bus._running)

class TestSignalComponent(unittest.TestCase):
    """Test cases for SignalComponent"""
    
    def setUp(self):
        self.bus = SignalBus()
        self.component = SignalComponent("test_component", self.bus)
        self.received_signals = []
        
    async def signal_handler(self, data):
        self.received_signals.append(data)
        
    async def test_initialize(self):
        """Test component initialization"""
        await self.component.initialize()
        self.assertTrue(self.component.is_initialized)
        self.assertTrue(self.component.is_connected)
        
    async def test_register_handler(self):
        """Test handler registration"""
        self.component.register_handler("test_signal", self.signal_handler)
        self.assertIn("test_signal", self.component._handlers)
        
    async def test_emit_signal(self):
        """Test signal emission"""
        self.component.register_handler("test_signal", self.signal_handler)
        await self.component.emit_signal("test_signal", "test_data")
        await asyncio.sleep(0.1)  # Allow time for processing
        self.assertIn("test_data", self.received_signals)
        
    async def test_cleanup(self):
        """Test component cleanup"""
        await self.component.initialize()
        await self.component.cleanup()
        self.assertFalse(self.component.is_connected)
        self.assertFalse(self.component.is_initialized)

if __name__ == '__main__':
    unittest.main() 