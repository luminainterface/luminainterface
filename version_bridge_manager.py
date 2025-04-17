#!/usr/bin/env python
"""
Version Bridge Manager - Central manager for all version bridge connections
in the Lumina Neural Network System
"""

import os
import sys
import time
import logging
import importlib
from threading import Thread

class VersionBridgeManager:
    """
    Central manager class for all version bridges and connectors
    in the Lumina Neural Network System.
    
    This class is responsible for:
    1. Initializing all bridge components
    2. Managing the lifecycle of bridges
    3. Providing a centralized API for bridge operations
    4. Handling shutdown and cleanup
    """
    
    def __init__(self, config=None):
        """
        Initialize the Version Bridge Manager
        
        Args:
            config (dict): Configuration dictionary
        """
        self.logger = logging.getLogger("VersionBridgeManager")
        
        # Default configuration
        self.config = {
            "mock_mode": False,
            "enable_v1v2_bridge": True,
            "enable_v3v4_connector": True,
            "enable_language_memory_v5_bridge": True,
            "enable_language_modulation_bridge": True
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        self.logger.info(f"Initializing VersionBridgeManager with config: {self.config}")
        
        # Initialize components dictionary
        self.components = {}
        self.running = False
        self.threads = []
    
    def start(self):
        """
        Start all bridge components
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        self.logger.info("Starting VersionBridgeManager")
        
        try:
            # Initialize bridges
            success = self._initialize_bridges()
            if not success:
                self.logger.error("Failed to initialize bridges")
                return False
            
            # Start monitoring thread
            monitor_thread = Thread(target=self._monitor_bridges, daemon=True)
            monitor_thread.start()
            self.threads.append(monitor_thread)
            
            self.running = True
            self.logger.info("VersionBridgeManager started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting VersionBridgeManager: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def stop(self):
        """
        Stop all bridge components
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        self.logger.info("Stopping VersionBridgeManager")
        
        try:
            self.running = False
            
            # Stop all components
            for name, component in self.components.items():
                try:
                    if hasattr(component, 'stop') and callable(component.stop):
                        self.logger.info(f"Stopping component: {name}")
                        component.stop()
                except Exception as e:
                    self.logger.error(f"Error stopping component {name}: {e}")
            
            # Clear components
            self.components.clear()
            
            self.logger.info("VersionBridgeManager stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping VersionBridgeManager: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def _initialize_bridges(self):
        """
        Initialize all bridge components
        
        Returns:
            bool: True if initialized successfully, False otherwise
        """
        try:
            # Initialize Language Memory V5 Bridge
            if self.config.get("enable_language_memory_v5_bridge", True):
                self._initialize_language_memory_v5_bridge()
            
            # Initialize V1V2 Bridge
            if self.config.get("enable_v1v2_bridge", True):
                self._initialize_v1v2_bridge()
            
            # Initialize V3V4 Connector
            if self.config.get("enable_v3v4_connector", True):
                self._initialize_v3v4_connector()
            
            # Initialize Language Modulation Bridge
            if self.config.get("enable_language_modulation_bridge", True):
                self._initialize_language_modulation_bridge()
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error initializing bridges: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def _initialize_language_memory_v5_bridge(self):
        """
        Initialize the Language Memory V5 Bridge
        """
        try:
            self.logger.info("Initializing Language Memory V5 Bridge")
            
            # Import the bridge module
            from language_memory_v5_bridge import LanguageMemoryV5Bridge
            
            # Create configuration for the bridge
            bridge_config = {
                "mock_mode": self.config.get("mock_mode", False),
                "cache_timeout": 300,  # 5 minutes
                "log_level": "DEBUG" if self.config.get("debug", False) else "INFO"
            }
            
            # Initialize the bridge
            bridge = LanguageMemoryV5Bridge(bridge_config)
            
            # Connect to Memory API
            success = bridge.connect_to_memory_api()
            
            if success:
                self.logger.info("Language Memory V5 Bridge initialized successfully")
                self.components["language_memory_v5_bridge"] = bridge
            else:
                self.logger.warning("Language Memory V5 Bridge initialized in disconnected state")
                self.components["language_memory_v5_bridge"] = bridge
            
        except ImportError as e:
            self.logger.warning(f"Language Memory V5 Bridge module not found: {e}")
        except Exception as e:
            self.logger.error(f"Error initializing Language Memory V5 Bridge: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _initialize_v1v2_bridge(self):
        """
        Initialize the V1V2 Bridge
        """
        try:
            self.logger.info("Initializing V1V2 Bridge")
            
            # Import the bridge module dynamically
            try:
                v1v2_bridge_module = importlib.import_module("v1v2_bridge")
                V1V2Bridge = getattr(v1v2_bridge_module, "V1V2Bridge")
                
                # Initialize the bridge
                bridge = V1V2Bridge(
                    mock_mode=self.config.get("mock_mode", False)
                )
                
                # Initialize the bridge
                success = bridge.initialize()
                
                if success:
                    self.logger.info("V1V2 Bridge initialized successfully")
                    self.components["v1v2_bridge"] = bridge
                else:
                    self.logger.warning("V1V2 Bridge failed to initialize")
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"V1V2 Bridge module not found: {e}")
            
        except Exception as e:
            self.logger.error(f"Error initializing V1V2 Bridge: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _initialize_v3v4_connector(self):
        """
        Initialize the V3V4 Connector
        """
        try:
            self.logger.info("Initializing V3V4 Connector")
            
            # Import the connector module
            try:
                from v3v4_connector import V3V4Connector
                
                # Create configuration for the connector
                connector_config = {
                    "mock_mode": self.config.get("mock_mode", False),
                    "debug": self.config.get("debug", False)
                }
                
                # Initialize the connector
                connector = V3V4Connector(config=connector_config)
                
                # Initialize and connect to V5
                if connector.connect_to_v5():
                    self.logger.info("V3V4 Connector initialized and connected to V5 successfully")
                    self.components["v3v4_connector"] = connector
                else:
                    self.logger.warning("V3V4 Connector failed to connect to V5")
                    self.components["v3v4_connector"] = connector
            except ImportError as e:
                self.logger.warning(f"V3V4 Connector module not found: {e}")
            
        except Exception as e:
            self.logger.error(f"Error initializing V3V4 Connector: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _initialize_language_modulation_bridge(self):
        """
        Initialize the Language Modulation Bridge
        """
        try:
            self.logger.info("Initializing Language Modulation Bridge")
            
            # Import the bridge module
            try:
                language_modulation_module = importlib.import_module("language_modulation_bridge")
                LanguageModulationBridge = getattr(language_modulation_module, "LanguageModulationBridge")
                
                # Create configuration for the bridge
                bridge_config = {
                    "mock_mode": self.config.get("mock_mode", False)
                }
                
                # Initialize the bridge
                bridge = LanguageModulationBridge(bridge_config)
                
                # Connect components
                if bridge.connect_components():
                    self.logger.info("Language Modulation Bridge initialized successfully")
                    self.components["language_modulation_bridge"] = bridge
                else:
                    self.logger.warning("Language Modulation Bridge failed to connect components")
                    self.components["language_modulation_bridge"] = bridge
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Language Modulation Bridge module not found: {e}")
            
        except Exception as e:
            self.logger.error(f"Error initializing Language Modulation Bridge: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _monitor_bridges(self):
        """
        Monitor bridge components for health
        """
        self.logger.info("Starting bridge monitor thread")
        
        while self.running:
            try:
                # Check each component
                for name, component in self.components.items():
                    try:
                        # Check if component has a get_status method
                        if hasattr(component, 'get_status') and callable(component.get_status):
                            status = component.get_status()
                            self.logger.debug(f"Component {name} status: {status}")
                            
                            # If component is disconnected, try to reconnect
                            if not status.get("connected", False) and hasattr(component, 'reconnect') and callable(component.reconnect):
                                self.logger.info(f"Attempting to reconnect component: {name}")
                                component.reconnect()
                    except Exception as e:
                        self.logger.error(f"Error checking component {name}: {e}")
                
                # Sleep for 30 seconds
                time.sleep(30)
            except Exception as e:
                self.logger.error(f"Error in monitor thread: {e}")
                time.sleep(5)  # Sleep briefly before continuing
        
        self.logger.info("Bridge monitor thread stopped")
    
    def get_component(self, name):
        """
        Get a bridge component by name
        
        Args:
            name (str): Component name
            
        Returns:
            object: Component instance or None if not found
        """
        return self.components.get(name)
    
    def get_status(self):
        """
        Get status of all bridge components
        
        Returns:
            dict: Status dictionary
        """
        status = {
            "running": self.running,
            "components": {}
        }
        
        # Get status of each component
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_status') and callable(component.get_status):
                    status["components"][name] = component.get_status()
                else:
                    status["components"][name] = {"connected": "unknown"}
            except Exception as e:
                self.logger.error(f"Error getting status for component {name}: {e}")
                status["components"][name] = {"connected": "error", "error": str(e)}
        
        return status

# Test code
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create bridge manager with mock mode
    manager = VersionBridgeManager({"mock_mode": True})
    
    # Start bridge manager
    if manager.start():
        print("Bridge manager started successfully")
        
        # Get bridge manager status
        status = manager.get_status()
        print(f"Bridge manager status: {status}")
        
        try:
            # Keep script running
            print("Press Ctrl+C to stop")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping bridge manager")
            manager.stop()
            print("Bridge manager stopped")
    else:
        print("Failed to start bridge manager") 