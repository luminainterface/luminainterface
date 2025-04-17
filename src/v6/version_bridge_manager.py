#!/usr/bin/env python
"""
V6 Version Bridge Manager

Central orchestration point for all bridge components in the V6 Portal of Contradiction,
managing connections between different system versions and components.
"""

import os
import sys
import logging
import threading
import time
import importlib
import json
from pathlib import Path

logger = logging.getLogger("V6BridgeManager")

class VersionBridgeManager:
    """
    Central orchestration for all version bridges and connectors
    
    Manages the lifecycle of all bridge components, handles configuration,
    and monitors the health of connections between different system versions.
    """
    
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            "mock_mode": False,
            "enable_v1v2_bridge": True,
            "enable_v3v4_connector": True,
            "enable_language_memory_v5_bridge": True,
            "debug": False
        }
        
        # Update with custom settings
        if config:
            self.config.update(config)
        
        # Bridge components
        self.bridge_components = {}
        self.bridge_status = {}
        
        # Initialize logging
        if self.config.get("debug"):
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"Version Bridge Manager initialized: mock_mode={self.config.get('mock_mode')}")
    
    def start(self):
        """Start all enabled bridge components"""
        logger.info("Starting bridge components...")
        
        # Initialize components
        self._initialize_components()
        
        # Start components
        for component_id, component in self.bridge_components.items():
            if not self._start_component(component_id, component):
                logger.warning(f"Failed to start component: {component_id}")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitor_components,
            daemon=True,
            name="BridgeMonitoringThread"
        )
        self.monitoring_thread.start()
        
        logger.info("Bridge components started")
        return True
    
    def stop(self):
        """Stop all bridge components"""
        logger.info("Stopping bridge components...")
        
        for component_id, component in self.bridge_components.items():
            if hasattr(component, "stop"):
                try:
                    component.stop()
                    self.bridge_status[component_id]["running"] = False
                    logger.info(f"Stopped component: {component_id}")
                except Exception as e:
                    logger.error(f"Error stopping component {component_id}: {e}")
        
        logger.info("All bridge components stopped")
        return True
    
    def get_status(self):
        """Get status of all bridge components"""
        return {
            "manager_status": "running",
            "mock_mode": self.config.get("mock_mode"),
            "components": self.bridge_status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    
    def get_component(self, component_id):
        """Get a specific bridge component"""
        return self.bridge_components.get(component_id)
    
    def _initialize_components(self):
        """Initialize all enabled bridge components"""
        # Initialize V1V2 Bridge
        if self.config.get("enable_v1v2_bridge"):
            self._initialize_v1v2_bridge()
        
        # Initialize V3V4 Connector
        if self.config.get("enable_v3v4_connector"):
            self._initialize_v3v4_connector()
        
        # Initialize Language Memory V5 Bridge
        if self.config.get("enable_language_memory_v5_bridge"):
            self._initialize_language_memory_v5_bridge()
    
    def _initialize_v1v2_bridge(self):
        """Initialize V1V2 Bridge component"""
        component_id = "v1v2_bridge"
        
        try:
            # Try to import the real component
            if not self.config.get("mock_mode"):
                try:
                    from src.v6.bridges.v1v2_bridge import V1V2Bridge
                    bridge = V1V2Bridge(self.config)
                    logger.info(f"Initialized V1V2 Bridge")
                except ImportError:
                    logger.warning(f"V1V2 Bridge module not found, using mock")
                    bridge = self._create_mock_component(component_id)
            else:
                # Use mock component in mock mode
                bridge = self._create_mock_component(component_id)
            
            # Store component
            self.bridge_components[component_id] = bridge
            self.bridge_status[component_id] = {
                "initialized": True,
                "running": False,
                "error": None,
                "mock": self.config.get("mock_mode") or isinstance(bridge, MockBridgeComponent)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing V1V2 Bridge: {e}")
            self.bridge_status[component_id] = {
                "initialized": False,
                "running": False,
                "error": str(e),
                "mock": False
            }
            return False
    
    def _initialize_v3v4_connector(self):
        """Initialize V3V4 Connector component"""
        component_id = "v3v4_connector"
        
        try:
            # Try to import the real component
            if not self.config.get("mock_mode"):
                try:
                    from src.v6.bridges.v3v4_connector import V3V4Connector
                    connector = V3V4Connector(self.config)
                    logger.info(f"Initialized V3V4 Connector")
                except ImportError:
                    logger.warning(f"V3V4 Connector module not found, using mock")
                    connector = self._create_mock_component(component_id)
            else:
                # Use mock component in mock mode
                connector = self._create_mock_component(component_id)
            
            # Store component
            self.bridge_components[component_id] = connector
            self.bridge_status[component_id] = {
                "initialized": True,
                "running": False,
                "error": None,
                "mock": self.config.get("mock_mode") or isinstance(connector, MockBridgeComponent)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing V3V4 Connector: {e}")
            self.bridge_status[component_id] = {
                "initialized": False,
                "running": False,
                "error": str(e),
                "mock": False
            }
            return False
    
    def _initialize_language_memory_v5_bridge(self):
        """Initialize Language Memory V5 Bridge component"""
        component_id = "language_memory_v5_bridge"
        
        try:
            # Try to import the real component
            if not self.config.get("mock_mode"):
                try:
                    from src.v6.bridges.language_memory_v5_bridge import LanguageMemoryV5Bridge
                    bridge = LanguageMemoryV5Bridge(self.config)
                    logger.info(f"Initialized Language Memory V5 Bridge")
                except ImportError:
                    logger.warning(f"Language Memory V5 Bridge module not found, using mock")
                    bridge = self._create_mock_component(component_id)
            else:
                # Use mock component in mock mode
                bridge = self._create_mock_component(component_id)
            
            # Store component
            self.bridge_components[component_id] = bridge
            self.bridge_status[component_id] = {
                "initialized": True,
                "running": False,
                "error": None,
                "mock": self.config.get("mock_mode") or isinstance(bridge, MockBridgeComponent)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Language Memory V5 Bridge: {e}")
            self.bridge_status[component_id] = {
                "initialized": False,
                "running": False,
                "error": str(e),
                "mock": False
            }
            return False
    
    def _start_component(self, component_id, component):
        """Start a bridge component"""
        try:
            if hasattr(component, "start"):
                component.start()
            
            self.bridge_status[component_id]["running"] = True
            logger.info(f"Started component: {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting component {component_id}: {e}")
            self.bridge_status[component_id]["error"] = str(e)
            return False
    
    def _monitor_components(self):
        """Monitor health of bridge components"""
        logger.debug("Starting bridge component monitoring")
        
        while True:
            try:
                # Check each component
                for component_id, component in self.bridge_components.items():
                    if hasattr(component, "is_healthy") and callable(component.is_healthy):
                        is_healthy = component.is_healthy()
                        
                        # Update status
                        self.bridge_status[component_id]["healthy"] = is_healthy
                        
                        # Restart component if needed
                        if not is_healthy and self.bridge_status[component_id]["running"]:
                            logger.warning(f"Component {component_id} is unhealthy, restarting...")
                            
                            # Stop component
                            if hasattr(component, "stop"):
                                component.stop()
                            
                            # Restart component
                            self._start_component(component_id, component)
                
                # Sleep before next check
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                time.sleep(60)
    
    def _create_mock_component(self, component_id):
        """Create a mock bridge component for testing"""
        return MockBridgeComponent(component_id)

class MockBridgeComponent:
    """Mock bridge component for testing"""
    
    def __init__(self, component_id):
        self.component_id = component_id
        self.is_running = False
        self.socket_manager = None
        logger.info(f"Created mock bridge component: {component_id}")
    
    def start(self):
        """Start the mock component"""
        self.is_running = True
        logger.info(f"Started mock component: {self.component_id}")
        return True
    
    def stop(self):
        """Stop the mock component"""
        self.is_running = False
        logger.info(f"Stopped mock component: {self.component_id}")
        return True
    
    def is_healthy(self):
        """Check if mock component is healthy"""
        return self.is_running
    
    def set_socket_manager(self, socket_manager):
        """Set the socket manager for communication"""
        self.socket_manager = socket_manager
        logger.info(f"Mock component {self.component_id} connected to socket manager")
        return True
    
    def process_message(self, message_type, data):
        """Process a message in the mock component"""
        logger.debug(f"Mock component {self.component_id} processing message: {message_type}")
        return {
            "status": "success",
            "component": self.component_id,
            "message_type": message_type,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        } 