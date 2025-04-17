#!/usr/bin/env python3
"""
LUMINA v7.5 Version Bridge
Handles communication between different versions of LUMINA
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional
from .signal_component import SignalComponent
from .signal_system import SignalBus
from .version_transform import MessageTransformer
from .bridge_monitor import BridgeMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", f"version_bridge_{os.getpid()}.log"))
    ]
)
logger = logging.getLogger("VersionBridge")

class VersionBridge(SignalComponent):
    """Bridge for handling communication between different versions"""
    
    def __init__(self, signal_bus: SignalBus):
        super().__init__("version_bridge", signal_bus)
        self.transformer = MessageTransformer()
        self.monitor = BridgeMonitor()
        self._initialized = False
        self._is_connected = False
        
    async def initialize(self) -> None:
        """Initialize the version bridge"""
        try:
            if self._initialized:
                return
                
            # Initialize base class
            await super().initialize()
                
            # Register handlers
            self.register_handler("version.message", self._handle_version_message)
            self.register_handler("version.status", self._handle_version_status)
            self.register_handler("version.error", self._handle_version_error)
            
            self._initialized = True
            self._is_connected = True
            logger.info("Version bridge initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize version bridge: {e}")
            raise
            
    async def _handle_version_message(self, data: Any) -> None:
        """Handle version-specific messages"""
        try:
            if not isinstance(data, dict):
                logger.error("Invalid message format")
                return
                
            source_version = data.get("version")
            target_version = data.get("target_version")
            message = data.get("message")
            
            if not all([source_version, target_version, message]):
                logger.error("Missing required fields in message")
                return
                
            # Transform and forward message
            transformed, error = await self.transformer.transform_message(
                source_version, target_version, message
            )
            
            if error:
                logger.error(f"Message transformation failed: {error}")
                return
                
            # Forward transformed message
            await self.emit_signal("version.message", {
                "version": target_version,
                "message": transformed
            })
            
        except Exception as e:
            logger.error(f"Error handling version message: {e}")
            
    async def _handle_version_status(self, data: Any) -> None:
        """Handle version status updates"""
        try:
            if not isinstance(data, dict):
                logger.error("Invalid status format")
                return
                
            version = data.get("version")
            status = data.get("status")
            
            if not all([version, status]):
                logger.error("Missing required fields in status")
                return
                
            # Record status in monitor
            self.monitor.record_status(version, status)
            
            # Forward status update
            await self.emit_signal("version.status", {
                "version": version,
                "status": status
            })
            
        except Exception as e:
            logger.error(f"Error handling version status: {e}")
            
    async def _handle_version_error(self, data: Any) -> None:
        """Handle version error messages"""
        try:
            if not isinstance(data, dict):
                logger.error("Invalid error format")
                return
                
            version = data.get("version")
            error = data.get("error")
            
            if not all([version, error]):
                logger.error("Missing required fields in error")
                return
                
            # Record error in monitor
            self.monitor.record_error(version, error)
            
            # Forward error message
            await self.emit_signal("version.error", {
                "version": version,
                "error": error
            })
            
        except Exception as e:
            logger.error(f"Error handling version error: {e}")
            
    async def cleanup(self) -> None:
        """Clean up version bridge resources"""
        await super().cleanup()
        self._initialized = False
        self._is_connected = False
        logger.info("Version bridge cleaned up")
        
    @property
    def is_initialized(self) -> bool:
        """Get initialization state"""
        return self._initialized
        
    @property
    def is_connected(self) -> bool:
        """Get connection state"""
        return self._is_connected

    async def register_version(self, version: str, endpoint: str) -> None:
        """Register a new version endpoint"""
        self._version_map[version] = endpoint
        self.logger.info(f"Registered version {version} at {endpoint}")
        
    async def get_version_endpoint(self, version: str) -> Optional[str]:
        """Get the endpoint for a specific version"""
        return self._version_map.get(version)
        
    async def emit_signal(self, signal_type: str, data: Any) -> None:
        """Emit a signal to connected components"""
        await self.signal_bus.broadcast(signal_type, data) 