"""
V6-V7 Bridge Component

This module provides a bidirectional bridge between V6 Portal of Contradiction
and V7 Node Consciousness, connecting symbolic state, breath detection,
and consciousness capabilities.
"""

import os
import sys
import time
import logging
import threading
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

# Set up logging
logger = logging.getLogger("V6V7Bridge")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

class V6V7Bridge:
    """
    Bidirectional bridge connecting V6 Portal of Contradiction with V7 Node Consciousness
    
    Key capabilities:
    - Symbolic state synchronization between V6 and V7
    - Breath detection integration with V6 breath state
    - Contradiction handling from V6 to V7 knowledge system
    - Node consciousness awareness for V6 components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the V6-V7 bridge"""
        # Default configuration
        self.config = {
            "mock_mode": False,
            "v6_enabled": True,
            "v7_enabled": True,
            "breath_integration_enabled": True,
            "contradiction_handling_enabled": True,
            "monday_integration_enabled": True,
            "node_consciousness_enabled": True
        }
        
        # Update with custom configuration
        if config:
            self.config.update(config)
        
        # Component registry
        self.components = {
            "v6": {},
            "v7": {}
        }
        
        # Connection status
        self.active = False
        self.message_queue = []
        self.message_thread = None
        
        # Event handlers
        self.event_handlers = {}
        
        logger.info(f"V6V7Bridge initialized with config: {self.config}")
    
    def initialize(self) -> bool:
        """Initialize the bridge and connect to all required components"""
        try:
            # First load V6 components
            if self.config["v6_enabled"]:
                self._discover_v6_components()
            
            # Then load V7 components
            if self.config["v7_enabled"]:
                self._discover_v7_components()
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Start message processing
            self._start_message_processing()
            
            # Mark as active
            self.active = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize V6V7Bridge: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def _discover_v6_components(self):
        """Discover and load V6 components"""
        try:
            # Try to load V6 symbolic state manager
            try:
                from src.v6.symbolic_state_manager import SymbolicStateManager
                manager = SymbolicStateManager()
                self.components["v6"]["symbolic_state_manager"] = manager
                logger.info("✅ V6 Symbolic State Manager loaded")
            except ImportError:
                logger.warning("❌ V6 Symbolic State Manager not found")
            
            # Try to load V6 contradiction processor
            try:
                from src.v6.contradiction_processor import ContradictionProcessor
                processor = ContradictionProcessor()
                self.components["v6"]["contradiction_processor"] = processor
                logger.info("✅ V6 Contradiction Processor loaded")
            except ImportError:
                logger.warning("❌ V6 Contradiction Processor not found")
            
            # Try to load V6 socket manager
            try:
                from src.v6.socket_manager import V6SocketManager
                socket_manager = V6SocketManager()
                self.components["v6"]["socket_manager"] = socket_manager
                logger.info("✅ V6 Socket Manager loaded")
            except ImportError:
                logger.warning("❌ V6 Socket Manager not found")
                
        except Exception as e:
            logger.error(f"Error discovering V6 components: {e}")
            raise
    
    def _discover_v7_components(self):
        """Discover and load V7 components"""
        try:
            # Try to load V7 node consciousness
            try:
                from src.v7.node_consciousness import LanguageConsciousnessNode
                node = LanguageConsciousnessNode()
                self.components["v7"]["node_consciousness"] = node
                logger.info("✅ V7 Language Consciousness Node loaded")
            except ImportError:
                logger.warning("❌ V7 Language Consciousness Node not found")
            
            # Try to load V7 breath detector
            try:
                from src.v7.breath_detector import BreathDetector
                breath_detector = BreathDetector(v6_connector=self)
                self.components["v7"]["breath_detector"] = breath_detector
                logger.info("✅ V7 Breath Detector loaded")
            except ImportError:
                logger.warning("❌ V7 Breath Detector not found")
            
            # Try to load V7 monday interface
            if self.config["monday_integration_enabled"]:
                try:
                    from src.v7.monday.monday_interface import MondayInterface
                    monday = MondayInterface()
                    self.components["v7"]["monday"] = monday
                    logger.info("✅ V7 Monday Interface loaded")
                except ImportError:
                    logger.warning("❌ V7 Monday Interface not found")
            
            # Try to load V7 socket manager
            try:
                from src.v7.ui.v7_socket_manager import V7SocketManager
                socket_manager = V7SocketManager()
                self.components["v7"]["socket_manager"] = socket_manager
                logger.info("✅ V7 Socket Manager loaded")
            except ImportError:
                logger.warning("❌ V7 Socket Manager not found")
                
        except Exception as e:
            logger.error(f"Error discovering V7 components: {e}")
            raise
    
    def _setup_event_handlers(self):
        """Set up event handlers between V6 and V7 components"""
        # Connect breath detector to V6 symbolic state manager
        if (self.config["breath_integration_enabled"] and 
            "breath_detector" in self.components["v7"] and 
            "symbolic_state_manager" in self.components["v6"]):
            
            # Set up breath phase synchronization
            v6_manager = self.components["v6"]["symbolic_state_manager"]
            v7_breath = self.components["v7"]["breath_detector"]
            
            # Connect V6 -> V7
            if hasattr(v6_manager, "breath_phase_changed"):
                v6_manager.breath_phase_changed.connect(
                    lambda phase: v7_breath.set_breath_phase(phase)
                )
                logger.info("✅ Connected V6 breath phase to V7 breath detector")
            
            # Start the breath detector
            v7_breath.start()
            logger.info("✅ Started V7 Breath Detector")
        
        # Connect contradiction processor to V7 consciousness
        if (self.config["contradiction_handling_enabled"] and
            "contradiction_processor" in self.components["v6"] and
            "node_consciousness" in self.components["v7"]):
            
            # Set up contradiction handling
            v6_processor = self.components["v6"]["contradiction_processor"]
            v7_consciousness = self.components["v7"]["node_consciousness"]
            
            # Register handlers
            if hasattr(v6_processor, "register_contradiction_handler"):
                v6_processor.register_contradiction_handler(
                    lambda contradiction: self._handle_contradiction(contradiction, v7_consciousness)
                )
                logger.info("✅ Connected V6 contradiction processor to V7 consciousness")
    
    def _handle_contradiction(self, contradiction, consciousness):
        """Handle contradictions from V6 in the V7 consciousness system"""
        if hasattr(consciousness, "process_contradiction"):
            consciousness.process_contradiction(contradiction)
            return True
        return False
    
    def _start_message_processing(self):
        """Start the message processing thread"""
        self.message_thread = threading.Thread(
            target=self._process_messages,
            daemon=True,
            name="V6V7BridgeMessageThread"
        )
        self.message_thread.start()
        logger.info("Started message processing thread")
    
    def _process_messages(self):
        """Process messages between V6 and V7 components"""
        while True:
            try:
                # Process any queued messages
                if self.message_queue:
                    message = self.message_queue.pop(0)
                    self._route_message(message)
                
                # Check breath integration
                if (self.config["breath_integration_enabled"] and
                    "breath_detector" in self.components["v7"] and
                    "symbolic_state_manager" in self.components["v6"]):
                    
                    # Get current states
                    v7_breath = self.components["v7"]["breath_detector"]
                    v6_manager = self.components["v6"]["symbolic_state_manager"]
                    
                    # Synchronize if needed (only when active)
                    if v7_breath.active and hasattr(v6_manager, "get_breath_phase"):
                        v6_phase = v6_manager.get_breath_phase()
                        v7_phase = v7_breath.current_phase
                        
                        # Only update if different
                        if v6_phase != v7_phase:
                            v7_breath.set_breath_phase(v6_phase)
                
                # Sleep to prevent high CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing messages: {e}")
                time.sleep(1)  # Wait longer on error
    
    def _route_message(self, message):
        """Route a message between V6 and V7 components"""
        message_type = message.get("type", "unknown")
        source = message.get("source", "unknown")
        target = message.get("target", "unknown")
        content = message.get("content", {})
        
        # Find target component
        if target in self.components:
            target_component = self.components[target]
            if hasattr(target_component, "handle_message"):
                target_component.handle_message(message_type, content)
                return True
        
        # Try event handlers
        if message_type in self.event_handlers:
            for handler in self.event_handlers[message_type]:
                handler(content)
            return True
            
        logger.warning(f"No handler found for message type: {message_type}, target: {target}")
        return False
    
    def send_message(self, message_type, content, source="bridge", target="unknown"):
        """Send a message through the bridge"""
        message = {
            "type": message_type,
            "source": source,
            "target": target,
            "content": content,
            "timestamp": time.time()
        }
        self.message_queue.append(message)
        return True
    
    def register_event_handler(self, event_type, handler):
        """Register an event handler for a specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        return True
    
    def get_component(self, component_name, version="v7"):
        """Get a component by name and version"""
        if version in self.components and component_name in self.components[version]:
            return self.components[version][component_name]
        return None
    
    def get_status(self):
        """Get the current status of the bridge"""
        # Collect status information
        v6_components = list(self.components["v6"].keys())
        v7_components = list(self.components["v7"].keys())
        
        status = {
            "active": self.active,
            "v6_components": v6_components,
            "v7_components": v7_components,
            "breath_integration": (
                self.config["breath_integration_enabled"] and
                "breath_detector" in self.components["v7"] and 
                "symbolic_state_manager" in self.components["v6"]
            ),
            "contradiction_handling": (
                self.config["contradiction_handling_enabled"] and
                "contradiction_processor" in self.components["v6"] and
                "node_consciousness" in self.components["v7"]
            ),
            "monday_integration": (
                self.config["monday_integration_enabled"] and
                "monday" in self.components["v7"]
            )
        }
        
        return status
    
    def start_all_components(self):
        """Start all components"""
        # Start breath detector if available
        if "breath_detector" in self.components["v7"]:
            self.components["v7"]["breath_detector"].start()
        
        # Start monday if available
        if "monday" in self.components["v7"]:
            if hasattr(self.components["v7"]["monday"], "start"):
                self.components["v7"]["monday"].start()
        
        return True
    
    def shutdown(self):
        """Shut down the bridge and all components"""
        # Stop breath detector if available
        if "breath_detector" in self.components["v7"]:
            if hasattr(self.components["v7"]["breath_detector"], "stop"):
                self.components["v7"]["breath_detector"].stop()
        
        # Stop monday if available
        if "monday" in self.components["v7"]:
            if hasattr(self.components["v7"]["monday"], "stop"):
                self.components["v7"]["monday"].stop()
        
        # Mark as inactive
        self.active = False
        return True


# Factory function to create a V6V7Bridge instance
def create_v6v7_bridge(config=None):
    """Create and initialize a V6V7Bridge instance"""
    bridge = V6V7Bridge(config)
    success = bridge.initialize()
    
    if success:
        logger.info("V6V7Bridge created and initialized successfully")
    else:
        logger.warning("V6V7Bridge initialization failed")
        
    return bridge 