#!/usr/bin/env python3
"""
Signal System

This module implements the V7.5 signal system for real-time communication
between different versions of the application.
"""

import logging
from typing import Dict, Any, List, Callable
from PySide6.QtCore import QObject, Signal, Slot
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalBus(QObject):
    """Central signal bus for handling cross-version communication."""
    
    # Signals
    message_received = Signal(str, str)  # message, version
    connection_changed = Signal(bool)  # connected
    error_occurred = Signal(str)  # error message
    
    def __init__(self):
        """Initialize the signal bus."""
        super().__init__()
        self._components: Dict[str, SignalComponent] = {}
        self._connected = False
        self._message_queue: List[Dict[str, Any]] = []
        
    def register_component(self, component: 'SignalComponent'):
        """Register a signal component."""
        try:
            version = component.version
            if version in self._components:
                logger.warning(f"Component for version {version} already registered")
                return
                
            self._components[version] = component
            component.message_received.connect(self._handle_component_message)
            component.connection_changed.connect(self._handle_component_connection)
            
            logger.info(f"Registered component for version {version}")
            
        except Exception as e:
            logger.error(f"Error registering component: {e}")
            self.error_occurred.emit(str(e))
            
    def unregister_component(self, version: str):
        """Unregister a signal component."""
        try:
            if version not in self._components:
                return
                
            component = self._components.pop(version)
            component.message_received.disconnect()
            component.connection_changed.disconnect()
            
            logger.info(f"Unregistered component for version {version}")
            
        except Exception as e:
            logger.error(f"Error unregistering component: {e}")
            self.error_occurred.emit(str(e))
            
    def send_message(self, message: str, version: str):
        """Send a message to a specific version."""
        try:
            if version not in self._components:
                logger.warning(f"No component registered for version {version}")
                return
                
            component = self._components[version]
            component.send_message({
                'content': message,
                'timestamp': datetime.now().isoformat(),
                'version': version
            })
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.error_occurred.emit(str(e))
            
    def broadcast_message(self, message: str):
        """Broadcast a message to all versions."""
        try:
            for version, component in self._components.items():
                component.send_message({
                    'content': message,
                    'timestamp': datetime.now().isoformat(),
                    'version': version
                })
                
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
            self.error_occurred.emit(str(e))
            
    @Slot(str, Dict[str, Any])
    def _handle_component_message(self, version: str, message: Dict[str, Any]):
        """Handle message from a component."""
        try:
            content = message.get('content', '')
            self.message_received.emit(content, version)
            
        except Exception as e:
            logger.error(f"Error handling component message: {e}")
            self.error_occurred.emit(str(e))
            
    @Slot(str, bool)
    def _handle_component_connection(self, version: str, connected: bool):
        """Handle connection change from a component."""
        try:
            # Update overall connection status
            all_connected = all(c.connected for c in self._components.values())
            if self._connected != all_connected:
                self._connected = all_connected
                self.connection_changed.emit(all_connected)
                
        except Exception as e:
            logger.error(f"Error handling component connection: {e}")
            self.error_occurred.emit(str(e))
            
    def cleanup(self):
        """Clean up resources."""
        try:
            for component in self._components.values():
                component.cleanup()
            self._components.clear()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
class SignalComponent(QObject):
    """Component for handling version-specific signal communication."""
    
    # Signals
    message_received = Signal(str, Dict[str, Any])  # version, message
    connection_changed = Signal(str, bool)  # version, connected
    error_occurred = Signal(str)  # error message
    
    def __init__(self, version: str = 'v7.5'):
        """Initialize the signal component."""
        super().__init__()
        self._version = version
        self._connected = False
        self._handlers: Dict[str, Callable] = {}
        
    @property
    def version(self) -> str:
        """Get the component version."""
        return self._version
        
    @property
    def connected(self) -> bool:
        """Get the connection status."""
        return self._connected
        
    def set_version(self, version: str):
        """Set the component version."""
        self._version = version
        
    def register_handlers(self, handlers: Dict[str, Callable]):
        """Register message handlers."""
        self._handlers.update(handlers)
        
    def send_message(self, message: Dict[str, Any]):
        """Send a message through this component."""
        try:
            if not self._connected:
                logger.warning(f"Component {self._version} not connected")
                return
                
            # Add version info if not present
            if 'version' not in message:
                message['version'] = self._version
                
            # Process message through handlers
            message_type = message.get('type', 'message')
            if message_type in self._handlers:
                self._handlers[message_type](message)
                
            # Emit signal
            self.message_received.emit(self._version, message)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.error_occurred.emit(str(e))
            
    def connect(self):
        """Connect the component."""
        try:
            self._connected = True
            self.connection_changed.emit(self._version, True)
            
        except Exception as e:
            logger.error(f"Error connecting component: {e}")
            self.error_occurred.emit(str(e))
            
    def disconnect(self):
        """Disconnect the component."""
        try:
            self._connected = False
            self.connection_changed.emit(self._version, False)
            
        except Exception as e:
            logger.error(f"Error disconnecting component: {e}")
            self.error_occurred.emit(str(e))
            
    def cleanup(self):
        """Clean up resources."""
        try:
            self.disconnect()
            self._handlers.clear()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 