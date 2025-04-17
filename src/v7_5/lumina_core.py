#!/usr/bin/env python3
"""
LUMINA v7.5 Core Integration Module
Handles the integration between different system components
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
from PySide6.QtCore import QObject, Signal
from .signal_system import SignalBus, SignalComponent
from .mistral_integration import MistralIntegration
from datetime import datetime

if TYPE_CHECKING:
    from .version_bridge import VersionBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'lumina_core.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LUMINACore")

class LUMINACore(QObject):
    """Core class for LUMINA v7.5"""
    
    # Define Qt signals
    component_initialized = Signal(str)
    error_occurred = Signal(str)
    system_ready = Signal()
    system_error = Signal(str)
    
    def __init__(self):
        """Initialize the LUMINA core"""
        super().__init__()
        
        # Create signal bus and signal component
        self.signal_bus = SignalBus()
        self.signal_component = SignalComponent("core", self.signal_bus)
        
        # Set up logging and state
        self.logger = logging.getLogger("LUMINACore")
        self._is_initialized = False
        self._is_connected = False
        
        # Initialize Mistral integration
        self.mistral = MistralIntegration()
        self.conversation = []
        
        # Register core as a component
        self.signal_bus.register_component("core", self)
        
        # Register signal handlers
        self.signal_component.register_handler("process_message", self.handle_process_message)
        
    def report_error(self, error_msg: str) -> None:
        """Report an error through logging and Qt signal"""
        self.logger.error(error_msg)
        self.error_occurred.emit(error_msg)
        
    async def initialize(self):
        """Initialize core components"""
        try:
            if self._is_initialized:
                return
                
            # Initialize signal bus and signal component
            await self.signal_bus.initialize()
            await self.signal_component.initialize()
            
            # Initialize version bridge
            from .version_bridge import VersionBridge
            self.version_bridge = VersionBridge(self.signal_bus)
            await self.version_bridge.initialize()
            
            # Emit signal using Qt's signal mechanism
            self.component_initialized.emit('version_bridge')
            self.logger.info("Core initialization complete")
            
            self._is_initialized = True
            self._is_connected = True
            
            # Emit system ready signal
            self.system_ready.emit()
            
        except Exception as e:
            error_msg = f"Error during initialization: {str(e)}"
            self.report_error(error_msg)
            self.system_error.emit(error_msg)
            raise  # Re-raise the exception after reporting
            
    async def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'version_bridge'):
                await self.version_bridge.cleanup()
            await self.signal_component.cleanup()
            await self.signal_bus.cleanup()
            self._is_initialized = False
            self._is_connected = False
        except Exception as e:
            self.report_error(f"Error during cleanup: {str(e)}")
            raise
        
    def get_component(self, name: str) -> Optional[Any]:
        """Get a component by name"""
        return self.signal_bus.get_component(name)
        
    def get_status(self) -> Dict[str, bool]:
        """Get the status of all components"""
        return {
            name: component is not None 
            for name, component in self.signal_bus.components.items()
        }
        
    async def emit_signal(self, signal_type: str, data: Any) -> None:
        """Emit a signal through the signal component"""
        await self.signal_component.emit_signal(signal_type, data)
        
    def handle_message(self, source: str, data: Any):
        """Handle incoming messages from other components"""
        if isinstance(data, dict):
            if data.get("type") == "status_request":
                asyncio.create_task(self.emit_signal("status_response", {
                    "status": self.get_status()
                }))
            elif data.get("type") == "component_request":
                component = self.get_component(data.get("component"))
                asyncio.create_task(self.emit_signal("component_response", {
                    "component": data.get("component"),
                    "available": component is not None
                }))
                
    @property
    def is_initialized(self) -> bool:
        """Get initialization state"""
        return self._is_initialized
        
    @property
    def is_connected(self) -> bool:
        """Get connection state"""
        return self._is_connected 

    async def handle_process_message(self, data: Dict[str, Any]):
        """Handle message processing requests"""
        try:
            if not isinstance(data, dict):
                raise ValueError("Invalid message data format")
                
            message = data.get("content", "")
            source = data.get("source", "unknown")
            
            if not message:
                raise ValueError("Empty message content")
                
            self.logger.info(f"Processing message from {source}: {message[:50]}...")
            
            # Add message to conversation history
            self.conversation.append({"role": "user", "content": message})
            
            # Process with Mistral
            try:
                response_text = self.mistral.process_message(self.conversation)
                self.conversation.append({"role": "assistant", "content": response_text})
                
                response = {
                    "type": "message_response",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.error(f"Mistral processing error: {str(e)}")
                response = {
                    "type": "message_response", 
                    "error": f"Error processing with language model: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Emit the response
            await self.signal_component.emit_signal("message_response", response)
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            self.logger.error(error_msg)
            self.system_error.emit(error_msg)
            
            # Emit error response
            await self.signal_component.emit_signal("message_response", {
                "type": "message_response",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }) 