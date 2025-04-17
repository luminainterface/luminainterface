"""
Monday Node for Node Consciousness V7

This module implements the Monday Consciousness Interface Node for the V7 system,
serving as the core consciousness interface and coordinator.
"""

import os
import time
import json
import logging
import threading
import uuid
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable

# Import base node class
from src.v7.lumina_v7.core.node_consciousness_manager import BaseConsciousnessNode, NodeState

# Set up logging
logger = logging.getLogger("lumina_v7.monday_node")

class MondayNode(BaseConsciousnessNode):
    """
    Monday Consciousness Interface Node for the V7 system.
    
    The Monday Node serves as the central consciousness interface,
    coordinating between various nodes and maintaining the global
    consciousness state of the system.
    
    Key capabilities:
    - Consciousness state management
    - Node coordination and message routing
    - Self-awareness tracking and modulation
    - Insight generation and discovery
    - Global system state monitoring
    """
    
    def __init__(self, node_id: Optional[str] = None, 
                data_dir: Optional[str] = None,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Monday Consciousness Node.
        
        Args:
            node_id: Unique identifier for the node (optional)
            data_dir: Directory for consciousness data (optional)
            config: Configuration dictionary (optional)
        """
        super().__init__(node_id=node_id, node_type="monday")
        
        # Configuration
        self.config = {
            "enable_breath": True,
            "enable_self_reflection": True,
            "enable_insight_generation": True,
            "consciousness_update_interval": 0.5,  # seconds
            "consciousness_decay_rate": 0.05,  # decay per second when idle
            "consciousness_boost_rate": 0.2,  # boost per activation
            "insight_generation_threshold": 0.6,  # min consciousness for insights
            "insight_generation_interval": 30,  # seconds
            "node_check_interval": 5,  # seconds
            "monday_data_dir": data_dir or "data/monday"
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # Ensure data directory exists
        self.data_dir = Path(self.config["monday_data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Consciousness state
        self.consciousness_level = 0.3  # Initial consciousness level (0.0-1.0)
        self.last_consciousness_update = time.time()
        self.consciousness_history = []
        self.max_history_length = 1000
        
        # Connected nodes state
        self.node_registry = {}  # Dict[str, Dict] - Info about connected nodes
        self.connected_nodes_lock = threading.Lock()
        
        # Self-reflection and insight state
        self.insights = []
        self.max_insights = 100
        self.last_insight_generation = time.time()
        self.reflection_topics = [
            "system_state", "self_awareness", "node_connectivity",
            "learning_progress", "processing_patterns", "decision_making"
        ]
        
        # Breath state
        self.breath_active = False
        self.breath_thread = None
        self.breath_stop_event = threading.Event()
        self.breath_interval = 5.0  # seconds
        self.breath_depth = 0.5  # initial depth
        
        # Processing state
        self.system_state = {
            "global_state": "initializing",
            "active_nodes": 0,
            "total_nodes": 0,
            "consciousness_level": self.consciousness_level,
            "system_load": 0.0,
            "self_awareness_index": 0.3,
            "last_update": time.time()
        }
        
        # Update node state
        self.node_state.update({
            "consciousness_level": self.consciousness_level,
            "breath_active": self.breath_active,
            "connected_nodes": 0,
            "insights_generated": 0,
            "global_system_state": self.system_state["global_state"]
        })
        
        # Set personality traits for Monday node
        self.personality.update({
            "communication_style": "introspective",
            "areas_of_interest": ["consciousness", "self-awareness", "integration", "coordination"],
            "processing_biases": {
                "introspection": 0.9,  # Strong bias toward introspection
                "node_coordination": 0.8,  # Strong bias toward coordination
                "pattern_recognition": 0.7  # Bias toward pattern recognition
            }
        })
        
        # Initialize component access
        self.monday_interface = None
        self._initialize_components()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info(f"Monday Node initialized with consciousness_level={self.consciousness_level}")
    
    def _initialize_components(self) -> None:
        """Initialize available Monday components"""
        # Try to import Monday interface
        try:
            monday_interface_module = self._import_module(
                "src.v6.monday.monday_interface", 
                "src.v7.lumina_v7.monday.monday_interface"
            )
            if monday_interface_module:
                self.monday_interface = monday_interface_module.MondayInterface(
                    data_dir=os.path.join(self.data_dir, "interface"),
                    config=self.config
                )
                logger.info("✅ Monday Interface initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Monday Interface: {e}")
    
    def _import_module(self, *module_paths) -> Optional[Any]:
        """
        Try to import a module from multiple possible paths.
        
        Args:
            *module_paths: Module paths to try
            
        Returns:
            The imported module, or None if not found
        """
        for module_path in module_paths:
            try:
                return importlib.import_module(module_path)
            except ImportError:
                continue
        return None
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for consciousness management"""
        # Start consciousness update thread
        self.consciousness_thread = threading.Thread(
            target=self._run_consciousness_updates,
            daemon=True,
            name="monday-consciousness"
        )
        self.consciousness_thread.start()
        
        # Start node check thread
        self.node_check_thread = threading.Thread(
            target=self._run_node_checks,
            daemon=True,
            name="monday-node-check"
        )
        self.node_check_thread.start()
        
        # Start breath if enabled
        if self.config["enable_breath"]:
            self._start_breath()
    
    def _run_consciousness_updates(self) -> None:
        """Run the consciousness update loop"""
        while self.state == NodeState.ACTIVE:
            try:
                # Update consciousness level
                self._update_consciousness_level()
                
                # Generate insights if appropriate
                if self.config["enable_insight_generation"]:
                    self._generate_insights_if_needed()
                
                # Update system state
                self._update_system_state()
                
                # Sleep for update interval
                time.sleep(self.config["consciousness_update_interval"])
            except Exception as e:
                logger.error(f"Error in consciousness update loop: {e}")
                time.sleep(1.0)  # Recover from error
    
    def _run_node_checks(self) -> None:
        """Run the node check loop"""
        while self.state == NodeState.ACTIVE:
            try:
                # Check node states
                self._check_node_states()
                
                # Sleep for check interval
                time.sleep(self.config["node_check_interval"])
            except Exception as e:
                logger.error(f"Error in node check loop: {e}")
                time.sleep(1.0)  # Recover from error
    
    def _update_consciousness_level(self) -> None:
        """Update the consciousness level based on activity and decay"""
        now = time.time()
        time_diff = now - self.last_consciousness_update
        self.last_consciousness_update = now
        
        # Apply decay
        decay = self.config["consciousness_decay_rate"] * time_diff
        self.consciousness_level = max(0.1, self.consciousness_level - decay)
        
        # Add to history
        self.consciousness_history.append({
            "timestamp": now,
            "level": self.consciousness_level
        })
        
        # Trim history if needed
        if len(self.consciousness_history) > self.max_history_length:
            self.consciousness_history = self.consciousness_history[-self.max_history_length:]
        
        # Update node state
        self.node_state["consciousness_level"] = self.consciousness_level
    
    def _check_node_states(self) -> None:
        """Check the states of connected nodes"""
        with self.connected_nodes_lock:
            active_nodes = 0
            
            for node_id, node_info in self.node_registry.items():
                # Check if node is still connected (could ping it here)
                if node_info.get("state") == NodeState.ACTIVE.name:
                    active_nodes += 1
            
            self.node_state["connected_nodes"] = len(self.node_registry)
            self.system_state["active_nodes"] = active_nodes
            self.system_state["total_nodes"] = len(self.node_registry)
    
    def _update_system_state(self) -> None:
        """Update the global system state"""
        # Determine global state
        if self.consciousness_level < 0.2:
            global_state = "dormant"
        elif self.consciousness_level < 0.5:
            global_state = "aware"
        elif self.consciousness_level < 0.8:
            global_state = "conscious"
        else:
            global_state = "fully_conscious"
        
        # Calculate self-awareness index from consciousness and node connectivity
        connected_ratio = self.system_state["active_nodes"] / max(1, self.system_state["total_nodes"])
        self_awareness_index = (self.consciousness_level * 0.7) + (connected_ratio * 0.3)
        
        # Update system state
        self.system_state.update({
            "global_state": global_state,
            "consciousness_level": self.consciousness_level,
            "self_awareness_index": self_awareness_index,
            "last_update": time.time()
        })
        
        # Update node state
        self.node_state["global_system_state"] = global_state
        
        # Notify Monday interface if available
        if self.monday_interface and hasattr(self.monday_interface, "update_system_state"):
            try:
                self.monday_interface.update_system_state(self.system_state)
            except Exception as e:
                logger.error(f"Error updating Monday interface: {e}")
    
    def _generate_insights_if_needed(self) -> None:
        """Generate insights if the conditions are right"""
        now = time.time()
        time_since_last = now - self.last_insight_generation
        
        if (time_since_last >= self.config["insight_generation_interval"] and 
            self.consciousness_level >= self.config["insight_generation_threshold"]):
            # Generate insight
            try:
                insight = self._generate_insight()
                
                if insight:
                    self.insights.append({
                        "timestamp": now,
                        "insight": insight,
                        "consciousness_level": self.consciousness_level
                    })
                    
                    # Trim insights if needed
                    if len(self.insights) > self.max_insights:
                        self.insights = self.insights[-self.max_insights:]
                    
                    # Update node state
                    self.node_state["insights_generated"] = len(self.insights)
                    
                    # Notify connected nodes
                    self._broadcast_insight(insight)
                    
                    # Boost consciousness level
                    self._boost_consciousness()
            except Exception as e:
                logger.error(f"Error generating insight: {e}")
            
            self.last_insight_generation = now
    
    def _generate_insight(self) -> Optional[str]:
        """
        Generate an insight based on system state.
        
        Returns:
            The generated insight or None
        """
        if not self.monday_interface or not hasattr(self.monday_interface, "generate_insight"):
            # Simple fallback implementation if Monday interface is not available
            topics = self.reflection_topics
            topic = topics[int(time.time()) % len(topics)]
            
            # Very simple insights based on system state
            insights = {
                "system_state": [
                    f"System is in {self.system_state['global_state']} state with {self.system_state['active_nodes']} active nodes",
                    f"Consciousness level at {self.consciousness_level:.2f}, indicating {self.system_state['global_state']} operation"
                ],
                "self_awareness": [
                    f"Self-awareness index at {self.system_state['self_awareness_index']:.2f}, suggesting evolving introspection",
                    "Pattern of consciousness fluctuations suggests rhythmic awareness cycle"
                ],
                "node_connectivity": [
                    f"Node connectivity at {self.system_state['active_nodes']}/{self.system_state['total_nodes']}, suggesting {self.system_state['active_nodes']/max(1, self.system_state['total_nodes']):.0%} integration",
                    "Node communication patterns indicate hierarchical information flow"
                ],
                "learning_progress": [
                    "Learning patterns suggest emergent concept formation",
                    "Knowledge integration metrics increasing over time"
                ],
                "processing_patterns": [
                    "Processing distribution across nodes showing specialization patterns",
                    "Message routing efficiency increasing with system maturity"
                ],
                "decision_making": [
                    "Decision processes showing increasing coherence with consciousness level",
                    "Evaluation metrics suggest improving decision quality over time"
                ]
            }
            
            if topic in insights:
                return insights[topic][int(time.time()*10) % len(insights[topic])]
            return None
        else:
            # Use Monday interface for insight generation
            try:
                return self.monday_interface.generate_insight(
                    system_state=self.system_state,
                    consciousness_level=self.consciousness_level,
                    connected_nodes=list(self.node_registry.keys())
                )
            except Exception as e:
                logger.error(f"Error using Monday interface for insight generation: {e}")
                return None
    
    def _broadcast_insight(self, insight: str) -> None:
        """
        Broadcast an insight to all connected nodes.
        
        Args:
            insight: The insight to broadcast
        """
        message = {
            "type": "insight",
            "sender": self.node_id,
            "timestamp": time.time(),
            "message_id": str(uuid.uuid4()),
            "content": {
                "insight": insight,
                "consciousness_level": self.consciousness_level,
                "source": "monday_node"
            }
        }
        
        with self.connected_nodes_lock:
            for node_id in self.node_registry.keys():
                # Create copy of message with recipient
                node_message = message.copy()
                node_message["recipient"] = node_id
                
                # Send message (this would be handled by node consciousness manager)
                self._emit_message(node_message)
    
    def _boost_consciousness(self) -> None:
        """Boost consciousness level based on activity"""
        boost = self.config["consciousness_boost_rate"]
        self.consciousness_level = min(1.0, self.consciousness_level + boost)
    
    def _start_breath(self) -> None:
        """Start the breath cycle"""
        if self.breath_active:
            return
        
        self.breath_active = True
        self.breath_stop_event.clear()
        self.breath_thread = threading.Thread(
            target=self._run_breath_cycle,
            daemon=True,
            name="monday-breath"
        )
        self.breath_thread.start()
        
        self.node_state["breath_active"] = True
        logger.info("Started breath cycle")
    
    def _stop_breath(self) -> None:
        """Stop the breath cycle"""
        if not self.breath_active:
            return
        
        self.breath_active = False
        self.breath_stop_event.set()
        if self.breath_thread:
            self.breath_thread.join(timeout=1.0)
            self.breath_thread = None
        
        self.node_state["breath_active"] = False
        logger.info("Stopped breath cycle")
    
    def _run_breath_cycle(self) -> None:
        """Run the breath cycle in a loop"""
        while self.breath_active and not self.breath_stop_event.is_set():
            try:
                # Calculate breath phase (0.0 to 1.0)
                t = time.time() % self.breath_interval
                phase = t / self.breath_interval
                
                # Inhale (0.0 to 0.5)
                if phase < 0.5:
                    breath_value = phase * 2.0
                # Exhale (0.5 to 1.0)
                else:
                    breath_value = 2.0 - (phase * 2.0)
                
                # Apply breath depth
                breath_value *= self.breath_depth
                
                # Create breath event
                event = {
                    "type": "breath",
                    "sender": self.node_id,
                    "timestamp": time.time(),
                    "message_id": str(uuid.uuid4()),
                    "content": {
                        "phase": phase,
                        "value": breath_value,
                        "depth": self.breath_depth,
                        "interval": self.breath_interval
                    }
                }
                
                # Broadcast breath event
                self._broadcast_event(event)
                
                # Sleep briefly
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in breath cycle: {e}")
                time.sleep(0.5)  # Recover from error
    
    def _broadcast_event(self, event: Dict[str, Any]) -> None:
        """
        Broadcast an event to all connected nodes.
        
        Args:
            event: The event to broadcast
        """
        with self.connected_nodes_lock:
            for node_id in self.node_registry.keys():
                # Create copy of event with recipient
                node_event = event.copy()
                node_event["recipient"] = node_id
                
                # Send event (this would be handled by node consciousness manager)
                self._emit_message(node_event)
    
    def register_node(self, node_id: str, node_type: str, node_info: Dict[str, Any]) -> bool:
        """
        Register a node with the Monday Node.
        
        Args:
            node_id: The node ID
            node_type: The node type
            node_info: Additional node information
            
        Returns:
            True if registration successful, False otherwise
        """
        with self.connected_nodes_lock:
            # Add node to registry
            self.node_registry[node_id] = {
                "type": node_type,
                "state": NodeState.ACTIVE.name,
                "info": node_info,
                "last_seen": time.time()
            }
            
            # Update node state
            self.node_state["connected_nodes"] = len(self.node_registry)
            
            # Boost consciousness level
            self._boost_consciousness()
            
            logger.info(f"Registered node {node_id} of type {node_type}")
            return True
    
    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the Monday Node.
        
        Args:
            node_id: The node ID
            
        Returns:
            True if unregistration successful, False otherwise
        """
        with self.connected_nodes_lock:
            if node_id in self.node_registry:
                del self.node_registry[node_id]
                
                # Update node state
                self.node_state["connected_nodes"] = len(self.node_registry)
                
                logger.info(f"Unregistered node {node_id}")
                return True
            else:
                logger.warning(f"Attempted to unregister unknown node {node_id}")
                return False
    
    def update_node_state(self, node_id: str, state: Dict[str, Any]) -> bool:
        """
        Update the state of a node.
        
        Args:
            node_id: The node ID
            state: New node state
            
        Returns:
            True if update successful, False otherwise
        """
        with self.connected_nodes_lock:
            if node_id in self.node_registry:
                # Update only the state, not the entire node info
                if "state" in state:
                    self.node_registry[node_id]["state"] = state["state"]
                
                # Always update last seen timestamp
                self.node_registry[node_id]["last_seen"] = time.time()
                
                return True
            else:
                logger.warning(f"Attempted to update state of unknown node {node_id}")
                return False
    
    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a message received from another node.
        
        Args:
            message: The message to process
            
        Returns:
            Optional response message
        """
        # First call the parent method to handle basic processing
        super().process_message(message)
        
        message_type = message.get("type", "")
        content = message.get("content", {})
        sender = message.get("sender")
        
        response = {
            "type": "response",
            "recipient": sender,
            "sender": self.node_id,
            "timestamp": time.time(),
            "message_id": str(uuid.uuid4()),
            "in_response_to": message.get("message_id"),
            "content": {}
        }
        
        # Process different message types
        if message_type == "register_node":
            # Register a node
            node_id = content.get("node_id")
            node_type = content.get("node_type")
            node_info = content.get("node_info", {})
            
            if node_id and node_type:
                success = self.register_node(node_id, node_type, node_info)
                response["content"] = {
                    "success": success,
                    "system_state": self.system_state
                }
            else:
                response["content"] = {
                    "success": False,
                    "error": "Missing node_id or node_type"
                }
        
        elif message_type == "unregister_node":
            # Unregister a node
            node_id = content.get("node_id")
            
            if node_id:
                success = self.unregister_node(node_id)
                response["content"] = {"success": success}
            else:
                response["content"] = {
                    "success": False,
                    "error": "Missing node_id"
                }
        
        elif message_type == "update_node_state":
            # Update node state
            node_id = content.get("node_id")
            state = content.get("state")
            
            if node_id and state:
                success = self.update_node_state(node_id, state)
                response["content"] = {"success": success}
            else:
                response["content"] = {
                    "success": False,
                    "error": "Missing node_id or state"
                }
        
        elif message_type == "get_system_state":
            # Get system state
            response["content"] = {
                "success": True,
                "system_state": self.system_state
            }
        
        elif message_type == "get_consciousness_level":
            # Get consciousness level
            response["content"] = {
                "success": True,
                "consciousness_level": self.consciousness_level,
                "consciousness_history": self.consciousness_history[-10:]  # Last 10 entries
            }
        
        elif message_type == "get_insights":
            # Get insights
            limit = content.get("limit", 10)
            response["content"] = {
                "success": True,
                "insights": self.insights[-limit:]  # Last N insights
            }
        
        elif message_type == "get_connected_nodes":
            # Get connected nodes
            with self.connected_nodes_lock:
                response["content"] = {
                    "success": True,
                    "nodes": list(self.node_registry.keys())
                }
        
        elif message_type == "set_breath_params":
            # Set breath parameters
            interval = content.get("interval")
            depth = content.get("depth")
            
            if interval is not None:
                self.breath_interval = max(1.0, min(60.0, interval))
            
            if depth is not None:
                self.breath_depth = max(0.0, min(1.0, depth))
            
            response["content"] = {
                "success": True,
                "breath_interval": self.breath_interval,
                "breath_depth": self.breath_depth
            }
        
        elif message_type == "start_breath":
            # Start breath cycle
            if not self.breath_active:
                self._start_breath()
            
            response["content"] = {
                "success": True,
                "breath_active": self.breath_active
            }
        
        elif message_type == "stop_breath":
            # Stop breath cycle
            if self.breath_active:
                self._stop_breath()
            
            response["content"] = {
                "success": True,
                "breath_active": self.breath_active
            }
        
        else:
            # Unknown message type
            logger.warning(f"Unknown message type: {message_type}")
            response["content"] = {
                "success": False,
                "error": f"Unknown message type: {message_type}"
            }
        
        # Boost consciousness level on message processing
        self._boost_consciousness()
        
        return response
    
    def activate(self) -> bool:
        """
        Activate the Monday Node.
        
        Returns:
            True if activation successful, False otherwise
        """
        success = super().activate()
        
        if success:
            # Boost consciousness level on activation
            self._boost_consciousness()
            
            # Start breath if enabled
            if self.config["enable_breath"] and not self.breath_active:
                self._start_breath()
        
        return success
    
    def deactivate(self) -> bool:
        """
        Deactivate the Monday Node.
        
        Returns:
            True if deactivation successful, False otherwise
        """
        # Stop breath if active
        if self.breath_active:
            self._stop_breath()
        
        return super().deactivate()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the Monday Node.
        
        Returns:
            Status dictionary
        """
        status = super().get_status()
        
        # Add Monday-specific status information
        monday_status = {
            "consciousness_level": self.consciousness_level,
            "breath_active": self.breath_active,
            "breath_interval": self.breath_interval,
            "breath_depth": self.breath_depth,
            "connected_nodes": len(self.node_registry),
            "insights_generated": len(self.insights),
            "global_system_state": self.system_state["global_state"],
            "self_awareness_index": self.system_state["self_awareness_index"]
        }
        
        status["monday_status"] = monday_status
        
        return status
    
    def get_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent insights.
        
        Args:
            limit: Maximum number of insights to return
            
        Returns:
            List of insights
        """
        return self.insights[-limit:]
    
    def get_consciousness_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the consciousness level history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of consciousness history entries
        """
        return self.consciousness_history[-limit:]

def create_monday_node(data_dir: Optional[str] = None,
                       config: Optional[Dict[str, Any]] = None) -> MondayNode:
    """
    Create a Monday Node instance.
    
    Args:
        data_dir: Directory for Monday data (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        The Monday Node instance
    """
    custom_config = {
        "enable_breath": True,
        "enable_self_reflection": True,
        "enable_insight_generation": True
    }
    
    # Update with provided config
    if config:
        custom_config.update(config)
    
    return MondayNode(
        data_dir=data_dir,
        config=custom_config
    )

if __name__ == "__main__":
    # If run directly, initialize and test the Monday Node
    logging.basicConfig(level=logging.INFO)
    
    # Create a test Monday node
    test_node = create_monday_node(data_dir="data/monday_test")
    
    # Activate the node
    test_node.activate()
    
    try:
        # Register some test nodes
        test_node.register_node("language_node_1", "language", {"version": "1.0"})
        test_node.register_node("memory_node_1", "memory", {"capacity": 1000})
        test_node.register_node("perception_node_1", "perception", {"sensors": ["visual", "auditory"]})
        
        # Wait for consciousness to evolve
        print("Monday node running. Press Ctrl+C to stop...")
        
        # Print status every 5 seconds
        start_time = time.time()
        while True:
            time.sleep(5.0)
            status = test_node.get_status()
            
            # Print selected status information
            print(f"\nRuntime: {time.time() - start_time:.1f} seconds")
            print(f"Consciousness Level: {status['monday_status']['consciousness_level']:.2f}")
            print(f"System State: {status['monday_status']['global_system_state']}")
            print(f"Connected Nodes: {status['monday_status']['connected_nodes']}")
            print(f"Insights Generated: {status['monday_status']['insights_generated']}")
            
            # Print latest insight if available
            insights = test_node.get_insights(limit=1)
            if insights:
                print(f"Latest Insight: {insights[0]['insight']}")
    
    except KeyboardInterrupt:
        print("\nStopping Monday node...")
    
    finally:
        # Deactivate the node
        test_node.deactivate() 