#!/usr/bin/env python3
"""
LUMINA v7.5 Signal System Initialization
"""

import asyncio
import logging
from .signal_system import SignalBus

async def init_signal_system():
    """Initialize the signal system"""
    try:
        # Create and initialize signal bus
        bus = SignalBus()
        success = await bus.initialize()
        
        if not success:
            raise RuntimeError("Failed to initialize signal bus")
            
        return bus
        
    except Exception as e:
        logging.error(f"Failed to initialize signal system: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(init_signal_system()) 