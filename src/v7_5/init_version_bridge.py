#!/usr/bin/env python3
"""
LUMINA v7.5 Version Bridge Initialization
"""

import asyncio
import logging
from .version_bridge import VersionBridge
from .signal_system import SignalBus

async def init_version_bridge():
    """Initialize the version bridge"""
    try:
        # Create signal bus
        bus = SignalBus()
        await bus.initialize()
        
        # Create and initialize version bridge
        bridge = VersionBridge(bus)
        await bridge.initialize()
        
        return bridge
        
    except Exception as e:
        logging.error(f"Failed to initialize version bridge: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(init_version_bridge()) 