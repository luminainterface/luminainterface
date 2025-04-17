#!/usr/bin/env python3
"""
LUMINA v7.5 Core Initialization Script
"""

import os
import sys
import logging
import asyncio
from .lumina_core import LUMINACore
from .signal_system import SignalBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'init_core.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("InitCore")

async def init_core():
    """Initialize the LUMINA core"""
    try:
        # Create and initialize signal bus
        signal_bus = SignalBus()
        await signal_bus.initialize()
        logger.info("Signal bus initialized")
        
        # Create and initialize core
        core = LUMINACore()
        await core.initialize()
        logger.info("Core initialized")
        
        return core
        
    except Exception as e:
        logger.error(f"Failed to initialize core: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(init_core())
        sys.exit(0)
    except Exception as e:
        logger.error(f"Core initialization failed: {e}")
        sys.exit(1) 