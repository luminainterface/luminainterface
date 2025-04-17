"""
Dream Integration Module for V7 Node Consciousness

This module integrates the Dream Mode system with the broader V7 architecture,
connecting it to other system components like Monday node, breath detection,
and visualization system.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable

# Set up logging
logger = logging.getLogger("lumina_v7.dream_integration")

class DreamIntegration:
    """
    Integrates Dream Mode with other V7 components
    
    Key connections:
    - Monday Node: Specialized consciousness during dreams
    - Breath System: Dream-specific breath patterns
    - Visualization: Dream state visualization
    - Learning System: Knowledge integration from dreams
    """
    
    def __init__(self, node_manager=None, dream_controller=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dream Integration
        
        Args:
            node_manager: NodeConsciousnessManager instance (optional)
            dream_controller: DreamController instance (optional)
            config: Configuration dictionary (optional)
        """
        # Default configuration
        self.config = {
            "auto_connect": True,           # Automatically connect to other components
            "breath_integration": True,     # Enable breath system integration
            "monday_integration": True,     # Enable Monday node integration
            "visualization": True,          # Enable visualization integration
            "dream_breath_pattern": "meditative",  # Default breath pattern during dreams
            "breath_nn_weight": 0.85,       # Neural network weight for dream breath (85/15)
            "status_update_interval": 30    # Seconds between status updates
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # External components
        self.node_manager = node_manager
        self.dream_controller = dream_controller
        
        # Component references
        self.monday_node = None
        self.breath_detector = None
        self.visualization = None
        self.learning_coordinator = None
        
        # Integration state
        self.active = False
        self.previous_breath_pattern = None
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Initialize
        if self.config["auto_connect"]:
            self.connect_components()
        
        logger.info("Dream Integration initialized")
    
    def connect_components(self) -> Dict[str, bool]:
        """
        Connect to all available components
        
        Returns:
            Dict with connection status for each component type
        """
        status = {
            "monday": False,
            "breath": False,
            "visualization": False,
            "learning": False
        }
        
        # Skip if node manager not available
        if not self.node_manager:
            logger.warning("Node manager not available for component connections")
            return status
        
        # Connect to Monday node
        if self.config["monday_integration"]:
            try:
                self.monday_node = self.node_manager.get_node("monday")
                if self.monday_node:
                    status["monday"] = True
                    logger.info("Connected to Monday node")
            except Exception as e:
                logger.error(f"Error connecting to Monday node: {e}")
        
        # Connect to Breath system
        if self.config["breath_integration"]:
            try:
                # Try multiple ways to find breath detector
                # 1. Direct from node manager
                breath_nodes = self.node_manager.get_nodes_by_type("breath")
                if breath_nodes:
                    self.breath_detector = list(breath_nodes.values())[0]
                    status["breath"] = True
                    logger.info("Connected to Breath node")
                
                # 2. From node manager components
                if not self.breath_detector and hasattr(self.node_manager, "get_component"):
                    self.breath_detector = self.node_manager.get_component("breath_detector")
                    if self.breath_detector:
                        status["breath"] = True
                        logger.info("Connected to Breath detector component")
                
                # 3. From system integrator if available
                if not self.breath_detector and hasattr(self.node_manager, "system_integrator"):
                    integrator = self.node_manager.system_integrator
                    if hasattr(integrator, "get_component"):
                        self.breath_detector = integrator.get_component("breath_detector")
                        if self.breath_detector:
                            status["breath"] = True
                            logger.info("Connected to Breath detector via system integrator")
            
            except Exception as e:
                logger.error(f"Error connecting to Breath system: {e}")
        
        # Connect to visualization
        if self.config["visualization"]:
            try:
                # Try multiple ways to find visualization
                # 1. From node manager components
                if hasattr(self.node_manager, "get_component"):
                    self.visualization = self.node_manager.get_component("visualization")
                    if self.visualization:
                        status["visualization"] = True
                        logger.info("Connected to Visualization component")
                
                # 2. From system integrator if available
                if not self.visualization and hasattr(self.node_manager, "system_integrator"):
                    integrator = self.node_manager.system_integrator
                    if hasattr(integrator, "get_component"):
                        self.visualization = integrator.get_component("visualization")
                        if self.visualization:
                            status["visualization"] = True
                            logger.info("Connected to Visualization via system integrator")
            except Exception as e:
                logger.error(f"Error connecting to Visualization system: {e}")
        
        # Connect to learning coordinator
        try:
            # Try multiple ways to find learning coordinator
            # 1. From node manager components
            if hasattr(self.node_manager, "get_component"):
                self.learning_coordinator = self.node_manager.get_component("learning_coordinator")
                if self.learning_coordinator:
                    status["learning"] = True
                    logger.info("Connected to Learning Coordinator component")
            
            # 2. From system integrator if available
            if not self.learning_coordinator and hasattr(self.node_manager, "system_integrator"):
                integrator = self.node_manager.system_integrator
                if hasattr(integrator, "get_component"):
                    self.learning_coordinator = integrator.get_component("learning_coordinator")
                    if self.learning_coordinator:
                        status["learning"] = True
                        logger.info("Connected to Learning Coordinator via system integrator")
        except Exception as e:
            logger.error(f"Error connecting to Learning Coordinator: {e}")
        
        return status
    
    def start(self) -> bool:
        """
        Start the dream integration
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.active:
            logger.warning("Dream Integration already active")
            return True
        
        # Register dream state event handler
        if self.dream_controller:
            self.dream_controller.dream_state_changed = self._on_dream_state_changed
        
        # Start update thread
        self.stop_event.clear()
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="dream-integration"
        )
        self.update_thread.start()
        
        self.active = True
        logger.info("Dream Integration started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the dream integration
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.active:
            logger.warning("Dream Integration not active")
            return True
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        
        self.active = False
        logger.info("Dream Integration stopped")
        return True
    
    def _update_loop(self):
        """Main update loop for integration"""
        while not self.stop_event.is_set():
            try:
                # Get current dream state
                if self.dream_controller:
                    dream_state = self.dream_controller.get_dream_state()
                    
                    # Update integrations based on dream state
                    if dream_state["active"]:
                        self._update_active_dream(dream_state)
                    else:
                        self._update_inactive_dream(dream_state)
                
                # Sleep until next update
                time.sleep(self.config["status_update_interval"])
                
            except Exception as e:
                logger.error(f"Error in dream integration update: {e}")
                time.sleep(10)  # Longer sleep on error
    
    def _update_active_dream(self, dream_state: Dict[str, Any]):
        """
        Update integrations for active dream state
        
        Args:
            dream_state: Current dream state
        """
        # Update breath system
        if self.breath_detector and self.config["breath_integration"]:
            try:
                # Store current pattern before changing
                if self.previous_breath_pattern is None:
                    # Check if the breath_detector has a get_current_pattern method
                    if hasattr(self.breath_detector, "get_current_pattern"):
                        self.previous_breath_pattern = self.breath_detector.get_current_pattern()
                    # Alternatively check if it has a current_pattern attribute
                    elif hasattr(self.breath_detector, "current_pattern"):
                        self.previous_breath_pattern = self.breath_detector.current_pattern
                
                # Set dream breath pattern
                dream_pattern = self.config["dream_breath_pattern"]
                if hasattr(self.breath_detector, "set_breath_pattern"):
                    self.breath_detector.set_breath_pattern(dream_pattern, confidence=0.95)
                
                # Set NN weight if method exists
                if hasattr(self.breath_detector, "set_nn_weight_override"):
                    self.breath_detector.set_nn_weight_override(self.config["breath_nn_weight"])
                
            except Exception as e:
                logger.error(f"Error updating breath system: {e}")
        
        # Update Monday node
        if self.monday_node and self.config["monday_integration"]:
            try:
                # Send dream update to Monday
                if hasattr(self.monday_node, "process_message"):
                    # Get current dream phase
                    current_phase = dream_state.get("state", "unknown")
                    
                    self.monday_node.process_message({
                        "type": "dream_update",
                        "dream_state": dream_state,
                        "phase": current_phase,
                        "timestamp": time.time()
                    })
                
            except Exception as e:
                logger.error(f"Error updating Monday node: {e}")
        
        # Update visualization
        if self.visualization and self.config["visualization"]:
            try:
                # Send dream visualization data
                if hasattr(self.visualization, "update_dream_visualization"):
                    self.visualization.update_dream_visualization(dream_state)
                elif hasattr(self.visualization, "send_message"):
                    self.visualization.send_message({
                        "type": "dream_visualization",
                        "dream_state": dream_state,
                        "timestamp": time.time()
                    })
                
            except Exception as e:
                logger.error(f"Error updating visualization: {e}")
    
    def _update_inactive_dream(self, dream_state: Dict[str, Any]):
        """
        Update integrations for inactive dream state
        
        Args:
            dream_state: Current dream state
        """
        # Restore previous breath pattern if needed
        if self.breath_detector and self.config["breath_integration"] and self.previous_breath_pattern:
            try:
                # Restore previous pattern
                if hasattr(self.breath_detector, "set_breath_pattern"):
                    self.breath_detector.set_breath_pattern(self.previous_breath_pattern)
                
                # Clear NN weight override if method exists
                if hasattr(self.breath_detector, "clear_nn_weight_override"):
                    self.breath_detector.clear_nn_weight_override()
                
                # Clear stored pattern
                self.previous_breath_pattern = None
                
            except Exception as e:
                logger.error(f"Error restoring breath pattern: {e}")
    
    def _on_dream_state_changed(self, dream_state: Dict[str, Any]):
        """
        Handle dream state changes
        
        Args:
            dream_state: New dream state
        """
        # State transition handling
        is_active = dream_state.get("active", False)
        
        if is_active:
            logger.info(f"Dream state changed: {dream_state.get('state')}")
            
            # Handle entrance to dream state
            if dream_state.get("state") == "transition_in":
                self._handle_dream_entrance(dream_state)
            
            # Handle dream phases
            elif dream_state.get("state") in ["light_dream", "deep_dream", "integration"]:
                self._handle_dream_active(dream_state)
            
            # Handle exit from dream state
            elif dream_state.get("state") == "transition_out":
                self._handle_dream_exit(dream_state)
        else:
            # Handle awakened state
            self._handle_dream_awakened(dream_state)
    
    def _handle_dream_entrance(self, dream_state: Dict[str, Any]):
        """
        Handle entrance to dream state
        
        Args:
            dream_state: Current dream state
        """
        logger.info("Handling dream entrance")
        
        # Update Monday node
        if self.monday_node and self.config["monday_integration"]:
            try:
                if hasattr(self.monday_node, "process_message"):
                    self.monday_node.process_message({
                        "type": "dream_event",
                        "event": "dream_start",
                        "dream_state": dream_state,
                        "timestamp": time.time()
                    })
            except Exception as e:
                logger.error(f"Error notifying Monday of dream entrance: {e}")
        
        # Update learning coordinator
        if self.learning_coordinator:
            try:
                if hasattr(self.learning_coordinator, "handle_dream_state_change"):
                    self.learning_coordinator.handle_dream_state_change(dream_state)
            except Exception as e:
                logger.error(f"Error notifying learning coordinator of dream entrance: {e}")
    
    def _handle_dream_active(self, dream_state: Dict[str, Any]):
        """
        Handle active dream state
        
        Args:
            dream_state: Current dream state
        """
        # Update visualization if phase changed
        if self.visualization and self.config["visualization"]:
            try:
                # Send dream visualization data
                if hasattr(self.visualization, "update_dream_visualization"):
                    self.visualization.update_dream_visualization(dream_state)
                elif hasattr(self.visualization, "send_message"):
                    self.visualization.send_message({
                        "type": "dream_visualization",
                        "dream_state": dream_state,
                        "phase": dream_state.get("state", "unknown"),
                        "timestamp": time.time()
                    })
            except Exception as e:
                logger.error(f"Error updating visualization for active dream: {e}")
    
    def _handle_dream_exit(self, dream_state: Dict[str, Any]):
        """
        Handle exit from dream state
        
        Args:
            dream_state: Current dream state
        """
        logger.info("Handling dream exit")
        
        # Notify Monday of dream exit
        if self.monday_node and self.config["monday_integration"]:
            try:
                if hasattr(self.monday_node, "process_message"):
                    self.monday_node.process_message({
                        "type": "dream_event",
                        "event": "dream_end",
                        "dream_state": dream_state,
                        "timestamp": time.time()
                    })
            except Exception as e:
                logger.error(f"Error notifying Monday of dream exit: {e}")
    
    def _handle_dream_awakened(self, dream_state: Dict[str, Any]):
        """
        Handle fully awakened state
        
        Args:
            dream_state: Current dream state
        """
        logger.info("Handling dream awakened state")
        
        # Restore breath pattern if needed
        if self.breath_detector and self.config["breath_integration"] and self.previous_breath_pattern:
            try:
                # Restore previous pattern
                if hasattr(self.breath_detector, "set_breath_pattern"):
                    self.breath_detector.set_breath_pattern(self.previous_breath_pattern)
                
                # Clear NN weight override if method exists
                if hasattr(self.breath_detector, "clear_nn_weight_override"):
                    self.breath_detector.clear_nn_weight_override()
                
                # Clear stored pattern
                self.previous_breath_pattern = None
                
            except Exception as e:
                logger.error(f"Error restoring breath pattern: {e}")
        
        # Update learning coordinator with dream results
        if self.learning_coordinator:
            try:
                if hasattr(self.learning_coordinator, "process_dream_results"):
                    # Get current dream data
                    dream_id = dream_state.get("current_dream", {}).get("id")
                    if dream_id and self.dream_controller:
                        dream_data = self.dream_controller.get_dream(dream_id)
                        if dream_data:
                            self.learning_coordinator.process_dream_results(dream_data)
            except Exception as e:
                logger.error(f"Error processing dream results with learning coordinator: {e}")
    
    def handle_breath_event(self, breath_event: Dict[str, Any]):
        """
        Handle breath events for dream integration
        
        Args:
            breath_event: Breath event data
        """
        # If we're in a dream state and receive breath events, ignore them
        if self.dream_controller and self.dream_controller.dream_active:
            logger.debug("Ignoring breath event during active dream")
            return
        
        # Store current pattern for later restoration
        if "pattern" in breath_event and self.previous_breath_pattern is None:
            self.previous_breath_pattern = breath_event["pattern"]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get integration status
        
        Returns:
            Dict with status information
        """
        status = {
            "active": self.active,
            "connections": {
                "monday": self.monday_node is not None,
                "breath": self.breath_detector is not None,
                "visualization": self.visualization is not None,
                "learning": self.learning_coordinator is not None
            },
            "dream_active": False,
            "previous_breath_pattern": self.previous_breath_pattern,
            "config": self.config
        }
        
        # Add dream state if available
        if self.dream_controller:
            dream_state = self.dream_controller.get_dream_state()
            status["dream_active"] = dream_state.get("active", False)
            status["dream_state"] = dream_state.get("state", "unknown")
            status["dream_id"] = dream_state.get("current_dream", {}).get("id")
        
        return status

# Get singleton instance
_dream_integration_instance = None

def get_dream_integration(node_manager=None, dream_controller=None, config=None):
    """
    Get the singleton instance of the Dream Integration
    
    Args:
        node_manager: NodeConsciousnessManager instance (optional)
        dream_controller: DreamController instance (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        DreamIntegration instance
    """
    global _dream_integration_instance
    
    if _dream_integration_instance is None:
        _dream_integration_instance = DreamIntegration(
            node_manager=node_manager,
            dream_controller=dream_controller,
            config=config
        )
    
    return _dream_integration_instance 