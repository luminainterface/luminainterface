#!/usr/bin/env python
"""
V6-V7 Connector - Bridges the V6 Portal of Contradiction with V7 Node Consciousness

This connector enables the integration between the V6 Portal of Contradiction's
symbolic consciousness and paradox handling capabilities with the V7 Node 
Consciousness system's self-awareness and learning features.
"""

import os
import sys
import time
import logging
import importlib
import threading
from pathlib import Path

# Configure logging
logger = logging.getLogger("V6V7Connector")

class V6V7Connector:
    """
    Connector between V6 Portal of Contradiction and V7 Node Consciousness
    
    This connector bridges the symbolic presence features of V6 with the
    self-awareness capabilities of V7, allowing for seamless evolution
    from one version to the next.
    """
    
    def __init__(self, config=None):
        """
        Initialize the V6-V7 Connector
        
        Args:
            config (dict): Configuration settings
        """
        # Default configuration
        self.config = {
            "mock_mode": False,
            "v6_enabled": True,
            "v7_enabled": True,
            "debug": False,
            "contradiction_processor_enabled": True,
            "node_consciousness_enabled": True,
            "monday_integration_enabled": True,
            "auto_wiki_enabled": True
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Initialize components dictionary
        self.components = {}
        self.running = False
        self.message_queue = []
        self.processing_thread = None
        
        logger.info(f"V6V7Connector initialized with config: {self.config}")
    
    def initialize(self):
        """
        Initialize the connector and its components
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Discover and load V6 components
            if self.config["v6_enabled"]:
                self._discover_v6_components()
            
            # Discover and load V7 components
            if self.config["v7_enabled"]:
                self._discover_v7_components()
            
            # Start message processing thread
            self.processing_thread = threading.Thread(
                target=self._process_messages,
                daemon=True,
                name="V6V7ConnectorProcessingThread"
            )
            self.processing_thread.start()
            
            logger.info("V6V7Connector initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing V6V7Connector: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def _discover_v6_components(self):
        """Discover and load available V6 components"""
        try:
            # Check for contradiction processor
            if self.config["contradiction_processor_enabled"]:
                try:
                    from src.v6.contradiction_processor import ContradictionProcessor
                    
                    contradiction_processor = ContradictionProcessor()
                    self.components["contradiction_processor"] = contradiction_processor
                    logger.info("✅ V6 Contradiction Processor loaded")
                    
                except ImportError:
                    logger.warning("❌ V6 Contradiction Processor not found")
            
            # Check for symbolic state manager
            try:
                from src.v6.ui.symbolic_state_manager import SymbolicStateManager
                
                symbolic_manager = SymbolicStateManager()
                self.components["symbolic_state_manager"] = symbolic_manager
                logger.info("✅ V6 Symbolic State Manager loaded")
                
            except ImportError:
                logger.warning("❌ V6 Symbolic State Manager not found")
            
        except Exception as e:
            logger.error(f"Error discovering V6 components: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _discover_v7_components(self):
        """Discover and load available V7 components"""
        try:
            # Check for Node Consciousness
            if self.config["node_consciousness_enabled"]:
                try:
                    from src.v7.node_consciousness import LanguageConsciousnessNode
                    
                    # Initialize the consciousness node
                    language_consciousness = LanguageConsciousnessNode()
                    self.components["language_consciousness"] = language_consciousness
                    logger.info("✅ V7 Language Consciousness Node loaded")
                    
                except ImportError:
                    logger.warning("❌ V7 Language Consciousness Node not found")
            
            # Check for Monday integration
            if self.config["monday_integration_enabled"]:
                try:
                    import importlib
                    monday_module = importlib.import_module("src.v7.monday.monday_interface")
                    
                    if hasattr(monday_module, "MondayInterface"):
                        monday = monday_module.MondayInterface()
                        self.components["monday"] = monday
                        logger.info("✅ V7 Monday Interface loaded")
                except ImportError:
                    logger.warning("❌ V7 Monday Interface not found")
            
            # Check for AutoWiki system
            if self.config["auto_wiki_enabled"]:
                try:
                    import importlib
                    auto_wiki_module = importlib.import_module("src.v7.auto_wiki.auto_wiki_plugin")
                    
                    if hasattr(auto_wiki_module, "AutoWikiPlugin"):
                        auto_wiki = auto_wiki_module.AutoWikiPlugin()
                        self.components["auto_wiki"] = auto_wiki
                        logger.info("✅ V7 AutoWiki Plugin loaded")
                except ImportError:
                    logger.warning("❌ V7 AutoWiki Plugin not found")
            
        except Exception as e:
            logger.error(f"Error discovering V7 components: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def connect_contradiction_to_consciousness(self):
        """
        Connect the V6 Contradiction Processor to V7 Node Consciousness
        
        This enables contradictions detected in V6 to be processed by
        the consciousness node in V7.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        contradiction_processor = self.components.get("contradiction_processor")
        language_consciousness = self.components.get("language_consciousness")
        
        if contradiction_processor and language_consciousness:
            logger.info("Connecting Contradiction Processor to Language Consciousness")
            
            # Register the language consciousness as a listener for contradictions
            if hasattr(contradiction_processor, "register_listener"):
                contradiction_processor.register_listener(
                    "language_consciousness", 
                    language_consciousness.process_contradiction
                )
                logger.info("✅ Successfully connected V6 contradictions to V7 consciousness")
                return True
        
        logger.warning("❌ Failed to connect contradictions to consciousness - components missing")
        return False
    
    def connect_symbolic_to_monday(self):
        """
        Connect the V6 Symbolic State Manager to V7 Monday Interface
        
        This enables Monday to respond to symbolic state changes from V6.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        symbolic_manager = self.components.get("symbolic_state_manager")
        monday = self.components.get("monday")
        
        if symbolic_manager and monday:
            logger.info("Connecting Symbolic State Manager to Monday Interface")
            
            # Register Monday as a listener for symbolic state changes
            if hasattr(symbolic_manager, "register_listener"):
                symbolic_manager.register_listener(
                    "monday",
                    monday.process_symbolic_state
                )
                logger.info("✅ Successfully connected V6 symbolic state to Monday")
                return True
        
        logger.warning("❌ Failed to connect symbolic state to Monday - components missing")
        return False
    
    def _process_messages(self):
        """Process messages in the queue"""
        logger.info("Started message processing thread")
        
        while True:
            # Process any messages in the queue
            if self.message_queue:
                try:
                    message = self.message_queue.pop(0)
                    self._route_message(message)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # Sleep briefly to avoid CPU spinning
            time.sleep(0.1)
    
    def _route_message(self, message):
        """Route a message to the appropriate component"""
        try:
            message_type = message.get("type")
            target = message.get("target")
            data = message.get("data", {})
            
            if target in self.components:
                component = self.components[target]
                
                # Check if component has a process_message method
                if hasattr(component, "process_message"):
                    component.process_message(message_type, data)
                    return True
            
            logger.warning(f"No handler for message type '{message_type}' to target '{target}'")
            return False
            
        except Exception as e:
            logger.error(f"Error routing message: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def send_message(self, message_type, target, data=None):
        """
        Send a message to a component
        
        Args:
            message_type (str): Type of message
            target (str): Target component
            data (dict): Message data
            
        Returns:
            bool: True if message sent, False otherwise
        """
        message = {
            "type": message_type,
            "target": target,
            "data": data or {},
            "timestamp": time.time()
        }
        
        self.message_queue.append(message)
        return True
    
    def get_component(self, name):
        """
        Get a component by name
        
        Args:
            name (str): Component name
            
        Returns:
            object: Component instance or None if not found
        """
        return self.components.get(name)
    
    def get_status(self):
        """
        Get the status of all components
        
        Returns:
            dict: Status information
        """
        status = {
            "connector": {
                "running": self.running,
                "message_queue_size": len(self.message_queue)
            },
            "components": {}
        }
        
        # Get status from each component
        for name, component in self.components.items():
            if hasattr(component, "get_status"):
                try:
                    component_status = component.get_status()
                    status["components"][name] = component_status
                except Exception as e:
                    status["components"][name] = {"error": str(e)}
            else:
                status["components"][name] = {"status": "loaded", "details": "No status method"}
        
        return status
    
    def start(self):
        """
        Start the connector and all components
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        logger.info("Starting V6V7Connector components")
        
        try:
            # Start each component
            for name, component in self.components.items():
                if hasattr(component, "start"):
                    logger.info(f"Starting {name}")
                    success = component.start()
                    if not success:
                        logger.warning(f"Failed to start {name}")
            
            # Connect components
            if self.config["v6_enabled"] and self.config["v7_enabled"]:
                self.connect_contradiction_to_consciousness()
                self.connect_symbolic_to_monday()
            
            self.running = True
            logger.info("V6V7Connector started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting V6V7Connector: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def stop(self):
        """
        Stop the connector and all components
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        logger.info("Stopping V6V7Connector")
        
        try:
            # Stop each component
            for name, component in self.components.items():
                if hasattr(component, "stop"):
                    logger.info(f"Stopping {name}")
                    component.stop()
            
            self.running = False
            logger.info("V6V7Connector stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping V6V7Connector: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False 