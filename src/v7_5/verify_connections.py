#!/usr/bin/env python3
"""
LUMINA v7.5 Connection Verification
"""

import asyncio
import logging
from .signal_system import SignalBus
from .version_bridge import VersionBridge
from .lumina_core import LUMINACore
from .central_node import CentralNode

async def verify_connections():
    """Verify all component connections"""
    try:
        # Create signal bus
        bus = SignalBus()
        await bus.initialize()
        
        # Create components
        core = LUMINACore()
        bridge = VersionBridge(bus)
        node = CentralNode(bus)
        
        # Initialize components
        await core.initialize()
        await bridge.initialize()
        await node.initialize()
        
        # Register components with signal bus
        bus.register_component("core", core)
        bus.register_component("version_bridge", bridge)
        bus.register_component("central_node", node)
        
        # Verify connections
        if not core.is_initialized or not core.is_connected:
            raise RuntimeError("Core component not properly connected")
        if not bridge.is_initialized or not bridge.is_connected:
            raise RuntimeError("Version bridge not properly connected")
        if not node.is_initialized:
            raise RuntimeError("Central node not properly initialized")
            
        # Clean up
        await core.cleanup()
        await bridge.cleanup()
        await node.cleanup()
        
        return True
        
    except Exception as e:
        logging.error(f"Connection verification failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(verify_connections()) 