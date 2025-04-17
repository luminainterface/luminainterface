"""
Setup script for the Spiderweb system.
This script initializes the SpiderwebManager and connects all necessary version bridges.
"""

import logging
import asyncio
from spiderweb.spiderweb_manager import SpiderwebManager
from src.v11.core.spiderweb_bridge import SpiderwebBridge as V11Bridge
from src.v12.core.spiderweb_bridge import SpiderwebBridge as V12Bridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/spiderweb_setup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def setup_spiderweb():
    """Initialize and set up the Spiderweb system."""
    try:
        # Initialize SpiderwebManager
        spiderweb_manager = SpiderwebManager()
        if not spiderweb_manager.initialize():
            raise Exception("Failed to initialize SpiderwebManager")

        # Create version bridges
        v11_bridge = V11Bridge()
        v12_bridge = V12Bridge()

        # Connect V11 bridge
        v11_config = {
            "bridge": v11_bridge,
            "version": "v11",
            "compatible_versions": ["v9", "v10", "v12", "v13"]
        }
        if not spiderweb_manager.connect_version("v11", v11_config):
            raise Exception("Failed to connect V11 bridge")

        # Connect V12 bridge
        v12_config = {
            "bridge": v12_bridge,
            "version": "v12",
            "compatible_versions": ["v10", "v11", "v13", "v14"]
        }
        if not spiderweb_manager.connect_version("v12", v12_config):
            raise Exception("Failed to connect V12 bridge")

        # Start quantum and cosmic synchronization
        await v11_bridge.start_quantum_sync()
        await v12_bridge.start_cosmic_sync()

        logger.info("Spiderweb system setup completed successfully")
        return spiderweb_manager

    except Exception as e:
        logger.error(f"Error setting up Spiderweb system: {str(e)}")
        raise

async def cleanup_spiderweb(spiderweb_manager: SpiderwebManager):
    """Clean up the Spiderweb system."""
    try:
        # Get connected versions
        for version, config in spiderweb_manager.connections.items():
            bridge = config.get("bridge")
            if bridge:
                if version == "v11":
                    await bridge.stop_quantum_sync()
                elif version == "v12":
                    await bridge.stop_cosmic_sync()

        logger.info("Spiderweb system cleanup completed")
    except Exception as e:
        logger.error(f"Error cleaning up Spiderweb system: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        spiderweb_manager = asyncio.run(setup_spiderweb())
        logger.info("Spiderweb system is ready")
    except Exception as e:
        logger.error(f"Failed to set up Spiderweb system: {str(e)}")
        exit(1) 