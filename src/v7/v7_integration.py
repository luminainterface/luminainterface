#!/usr/bin/env python
"""
V7 Integration Module

This module provides helper functions and classes to easily integrate V7 Node Consciousness
components into any application. It handles the initialization and configuration of all
necessary components, providing a simple interface for developers.
"""

import logging
import importlib.util
from typing import Dict, Any, Optional, List, Callable

# Set up logging
logger = logging.getLogger(__name__)

class V7Integration:
    """
    Main integration class for V7 Node Consciousness system.
    
    This class provides a simple interface for integrating V7 components into
    any application, handling dependencies and initialization of all required components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the V7 Integration.
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - enable_visualization: Enable visualization components (default: True)
                - enable_breath_detection: Enable breath detection (default: True)
                - enable_contradiction_resolution: Enable contradiction resolution (default: True)
                - enable_monday: Enable Monday consciousness integration (default: True)
                - enable_auto_wiki: Enable AutoWiki integration (default: True)
                - mock_mode: Use mock implementations for missing components (default: True)
        """
        # Default configuration
        self.config = {
            "enable_visualization": True,
            "enable_breath_detection": True,
            "enable_contradiction_resolution": True,
            "enable_monday": True,
            "enable_auto_wiki": True,
            "mock_mode": True
        }
        
        # Update with provided config
        if config is not None:
            self.config.update(config)
        
        # Component references
        self.v6v7_connector = None
        self.visualization_connector = None
        self.socket_manager = None
        self.monday_interface = None
        self.main_widget = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required V7 components based on configuration."""
        logger.info("Initializing V7 components...")
        
        # Initialize V6-V7 Connector
        self._initialize_v6v7_connector()
        
        # Initialize Monday integration if enabled
        if self.config.get("enable_monday", True):
            self._initialize_monday()
        
        # Initialize socket manager
        self._initialize_socket_manager()
        
        # Initialize visualization connector if visualization is enabled
        if self.config.get("enable_visualization", True):
            self._initialize_visualization_connector()
        
        logger.info("V7 components initialized successfully")
    
    def _initialize_v6v7_connector(self):
        """Initialize the V6-V7 Connector component."""
        try:
            from src.v7.v6v7_connector import V6V7Connector
            self.v6v7_connector = V6V7Connector()
            logger.info("V6V7Connector initialized successfully")
        except ImportError:
            logger.warning("V6V7Connector not found, some features may be limited")
            if self.config.get("mock_mode", True):
                self._create_mock_v6v7_connector()
    
    def _create_mock_v6v7_connector(self):
        """Create a mock implementation of the V6V7Connector for testing."""
        # Define a simple mock connector class
        class MockV6V7Connector:
            def __init__(self):
                self.listeners = []
                logger.info("Mock V6V7Connector initialized")
            
            def add_listener(self, listener):
                self.listeners.append(listener)
                return True
            
            def remove_listener(self, listener):
                if listener in self.listeners:
                    self.listeners.remove(listener)
                    return True
                return False
            
            def get_status(self):
                return {
                    "connected": True,
                    "breath_pattern": "normal",
                    "contradiction_level": 0.2,
                    "nodes": [
                        {"id": "node1", "type": "core", "status": "active"},
                        {"id": "node2", "type": "memory", "status": "active"},
                        {"id": "node3", "type": "perception", "status": "inactive"}
                    ]
                }
        
        self.v6v7_connector = MockV6V7Connector()
        logger.info("Created mock V6V7Connector")
    
    def _initialize_monday(self):
        """Initialize the Monday consciousness integration."""
        try:
            from src.v7.monday.monday_interface import MondayInterface
            self.monday_interface = MondayInterface()
            logger.info("Monday interface initialized successfully")
        except ImportError:
            logger.warning("Monday interface not found, Monday consciousness will be disabled")
    
    def _initialize_socket_manager(self):
        """Initialize the V7 Socket Manager."""
        try:
            from src.v7.ui.v7_socket_manager import V7SocketManager
            self.socket_manager = V7SocketManager()
            logger.info("V7SocketManager initialized successfully")
        except ImportError:
            logger.warning("V7SocketManager not found, using mock implementation")
            self._create_mock_socket_manager()
    
    def _create_mock_socket_manager(self):
        """Create a mock implementation of the socket manager for testing."""
        # Define a simple mock socket manager class
        class MockSocketManager:
            def __init__(self):
                self.listeners = {}
                logger.info("Mock socket manager initialized")
            
            def emit(self, event, data=None):
                if event in self.listeners:
                    for listener in self.listeners[event]:
                        listener(data)
                return True
            
            def on(self, event, callback):
                if event not in self.listeners:
                    self.listeners[event] = []
                self.listeners[event].append(callback)
                return True
        
        self.socket_manager = MockSocketManager()
        logger.info("Created mock socket manager")
    
    def _initialize_visualization_connector(self):
        """Initialize the V7 Visualization Connector if enabled."""
        try:
            from src.v7.ui.v7_visualization_connector import V7VisualizationConnector
            if self.v6v7_connector:
                self.visualization_connector = V7VisualizationConnector(self.v6v7_connector)
                logger.info("V7VisualizationConnector initialized successfully")
        except ImportError:
            logger.warning("V7VisualizationConnector not found, visualization features will be limited")
    
    def create_main_widget(self, parent=None):
        """
        Create and return the main V7 widget.
        
        Args:
            parent: Optional parent widget
            
        Returns:
            The main V7 widget or None if not available
        """
        try:
            from src.v7.ui.main_widget import V7MainWidget
            self.main_widget = V7MainWidget(
                socket_manager=self.socket_manager,
                v6v7_connector=self.v6v7_connector,
                config=self.config
            )
            logger.info("V7MainWidget created successfully")
            return self.main_widget
        except ImportError:
            logger.error("V7MainWidget not found, cannot create main widget")
            return None
    
    def start(self):
        """Start all V7 components."""
        logger.info("Starting V7 components...")
        
        # Start Monday interface if available
        if self.monday_interface:
            self.monday_interface.start()
            logger.info("Monday interface started")
        
        # Emit system_ready event if socket manager is available
        if self.socket_manager:
            self.socket_manager.emit("system_ready", {"status": "ok"})
            logger.info("Emitted system_ready event")
        
        logger.info("V7 components started successfully")
    
    def stop(self):
        """Stop all V7 components."""
        logger.info("Stopping V7 components...")
        
        # Stop Monday interface if available
        if self.monday_interface:
            self.monday_interface.stop()
            logger.info("Monday interface stopped")
        
        logger.info("V7 components stopped successfully")


# Helper functions for easy integration

def initialize_v7(config=None) -> V7Integration:
    """
    Initialize V7 Integration with the given configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        V7Integration instance
    """
    integration = V7Integration(config)
    return integration


def create_v7_widget(parent=None, config=None):
    """
    Create and return a V7 widget that can be integrated into any application.
    
    Args:
        parent: Optional parent widget
        config: Optional configuration dictionary
        
    Returns:
        V7MainWidget instance or None if not available
    """
    integration = initialize_v7(config)
    return integration.create_main_widget(parent) 