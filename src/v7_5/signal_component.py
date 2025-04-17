#!/usr/bin/env python3
"""
LUMINA v7.5 Signal Component
Base class for all components that need to communicate via signals
"""

import os
import logging
import asyncio
from typing import Any, Callable, Dict, Optional
from .signal_system import SignalBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", f"signal_component_{os.getpid()}.log"))
    ]
)
logger = logging.getLogger("SignalComponent")

class SignalComponent:
    """Base class for components that communicate via signals"""
    
    def __init__(self, name: str, signal_bus: SignalBus):
        """Initialize the signal component"""
        self.name = name
        self.signal_bus = signal_bus
        self._handlers: Dict[str, Callable] = {}
        self._is_connected = False
        self._is_initialized = False
        
        # Configure logging
        self.logger = logging.getLogger(f"SignalComponent.{name}")
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # Add file handler
            file_handler = logging.FileHandler('logs/signal_component.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    @property
    def is_connected(self) -> bool:
        """Get the connection state"""
        return self._is_connected

    @property
    def is_initialized(self) -> bool:
        """Get the initialization state"""
        return self._is_initialized

    async def initialize(self) -> None:
        """Initialize the component"""
        if self._is_initialized:
            self.logger.warning(f"{self.name} already initialized")
            return
            
        try:
            self.logger.info(f"Initializing {self.name}")
            await self._register_default_handlers()
            self._is_initialized = True
            self._is_connected = True
            self.logger.info(f"{self.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {str(e)}")
            raise

    async def _register_default_handlers(self) -> None:
        """Register default signal handlers"""
        # Override this method in subclasses to register specific handlers
        pass
        
    def register_handler(self, signal_type: str, handler: Callable) -> None:
        """Register a signal handler"""
        self._handlers[signal_type] = handler
        self.signal_bus.register_handler(signal_type, handler)
        self.logger.debug(f"Registered handler for {signal_type} in {self.name}")
        
    def unregister_handler(self, signal_type: str) -> None:
        """Unregister a signal handler"""
        if signal_type in self._handlers:
            handler = self._handlers[signal_type]
            self.signal_bus.unregister_handler(signal_type, handler)
            del self._handlers[signal_type]
            self.logger.debug(f"Unregistered handler for {signal_type} in {self.name}")
            
    async def emit_signal(self, signal_type: str, data: Any) -> None:
        """Emit a signal through the signal bus"""
        if not self._is_connected:
            self.logger.warning(f"Cannot emit signal: {self.name} is not connected")
            return
            
        try:
            await self.signal_bus.emit(signal_type, data)
            self.logger.debug(f"Emitted signal {signal_type} from {self.name}")
        except Exception as e:
            self.logger.error(f"Failed to emit signal {signal_type}: {str(e)}")
            raise
        
    async def cleanup(self) -> None:
        """Clean up component resources"""
        # Unregister all handlers
        for signal_type in list(self._handlers.keys()):
            self.unregister_handler(signal_type)
            
        self._is_connected = False
        self._is_initialized = False
        self.logger.info(f"Component cleaned up: {self.name}") 