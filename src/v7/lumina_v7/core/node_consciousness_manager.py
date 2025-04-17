#!/usr/bin/env python3
"""
Node Consciousness Manager for LUMINA V7

This module implements the core Node Consciousness Manager which is responsible
for coordinating and managing the various consciousness nodes within the V7 system.
It provides a unified interface for node registration, inter-node communication,
and system-wide consciousness state management.
"""

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable

# Define node states
class NodeState(Enum):
    """Possible states for consciousness nodes"""
    INACTIVE = 0
    INITIALIZING = 1
    ACTIVE = 2
    PAUSED = 3
    ERROR = 4
    SHUTDOWN = 5


class BaseConsciousnessNode:
    """
    Base class for all consciousness nodes in the V7 system.
    
    This class provides the common interface and functionality that all
    consciousness nodes must implement.
    """
    
    def __init__(self, node_id: str, node_type: str):
        """
        Initialize a base consciousness node.
        
        Args:
            node_id: Unique identifier for this node
            node_type: Type of consciousness node (e.g., 'language', 'memory', etc.)
        """
        self.node_id = node_id
        self.node_type = node_type
        self.state = NodeState.INACTIVE
        self.creation_time = time.time()
        
    def activate(self) -> bool:
        """
        Activate the consciousness node.
        
        Returns:
            bool: True if activation was successful, False otherwise
        """
        self.state = NodeState.ACTIVE
        return True
        
    def deactivate(self) -> bool:
        """
        Deactivate the consciousness node.
        
        Returns:
            bool: True if deactivation was successful, False otherwise
        """
        self.state = NodeState.INACTIVE
        return True
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status information about the node.
        
        Returns:
            dict: Status information dictionary
        """
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'state': self.state.name,
            'creation_time': self.creation_time
        }
        
    def receive_message(self, message: Dict[str, Any]) -> bool:
        """
        Process a message sent to this node.
        
        Args:
            message: The message to process
            
        Returns:
            bool: True if message was successfully received, False otherwise
        """
        return False  # Base implementation does nothing
        
    def get_ui_widget(self):
        """
        Get a UI widget representing this node, if available.
        
        Returns:
            Optional UI widget for displaying node status
        """
        return None


class NodeConsciousnessManager:
    """
    Manager for the Node Consciousness system in V7.
    
    This class coordinates the various consciousness nodes, manages inter-node
    communication, and provides a unified interface for the system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Node Consciousness Manager.
        
        Args:
            config: Configuration dictionary with settings for the manager
        """
        # Default configuration
        self.default_config = {
            "auto_activate_nodes": True,
            "communication_interval": 0.1,  # seconds
            "monitor_interval": 1.0,  # seconds
            "auto_recovery": True,
            "breath_enabled": True,
            "breath_interval": 5.0,  # seconds
            "core_node_types": ["language", "memory", "monday", "attention"],
            "node_timeout": 10.0,  # seconds before considering a node unresponsive
        }
        
        # Merge with provided config
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Initialize component attributes
        self.nodes: Dict[str, BaseConsciousnessNode] = {}
        self.node_types: Dict[str, Set[str]] = {}  # Maps node types to sets of node IDs
        self.active: bool = False
        self.system_state = "initializing"
        
        # Communication channels
        self.channels: Dict[str, List[str]] = {}  # Maps channel names to lists of subscribed node IDs
        
        # Threading
        self.communication_thread = None
        self.monitor_thread = None
        self.breath_thread = None
        self.running = False
        
        # Metrics
        self.metrics = {
            "messages_processed": 0,
            "active_nodes": 0,
            "breath_cycles": 0,
            "last_breath_time": 0,
            "system_start_time": time.time()
        }
        
        # Event listeners
        self.event_listeners = {
            "node_added": [],
            "node_removed": [],
            "state_changed": [],
            "breath": [],
            "system_error": []
        }
        
        logging.info("✅ Node Consciousness Manager initialized")
        
    def start(self) -> bool:
        """
        Start the Node Consciousness Manager and activate registered nodes.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.active:
            logging.warning("⚠️ Node Consciousness Manager already active")
            return False
            
        logging.info("🚀 Starting Node Consciousness Manager")
        self.active = True
        self.system_state = "starting"
        self._trigger_event("state_changed", {"state": self.system_state})
        
        # Start threads
        self.running = True
        
        # Start communication thread
        self.communication_thread = threading.Thread(
            target=self._run_communication,
            name="NCM-Communication",
            daemon=True
        )
        self.communication_thread.start()
        
        # Start monitor thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_nodes,
            name="NCM-Monitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        # Start breath thread if enabled
        if self.config["breath_enabled"]:
            self.breath_thread = threading.Thread(
                target=self._breath_cycle,
                name="NCM-Breath",
                daemon=True
            )
            self.breath_thread.start()
        
        # Activate nodes if configured to do so
        if self.config["auto_activate_nodes"]:
            for node_id, node in self.nodes.items():
                if node.state == NodeState.INACTIVE:
                    self._activate_node(node_id)
        
        self.system_state = "active"
        self._trigger_event("state_changed", {"state": self.system_state})
        logging.info("✅ Node Consciousness Manager started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the Node Consciousness Manager and deactivate registered nodes.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.active:
            logging.warning("⚠️ Node Consciousness Manager already inactive")
            return False
            
        logging.info("🛑 Stopping Node Consciousness Manager")
        self.system_state = "stopping"
        self._trigger_event("state_changed", {"state": self.system_state})
        
        # Stop threads
        self.running = False
        
        # Deactivate all nodes
        for node_id, node in self.nodes.items():
            if node.state != NodeState.INACTIVE:
                self._deactivate_node(node_id)
        
        # Wait for threads to finish
        if self.communication_thread and self.communication_thread.is_alive():
            self.communication_thread.join(timeout=2.0)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
        if self.breath_thread and self.breath_thread.is_alive():
            self.breath_thread.join(timeout=2.0)
            
        self.active = False
        self.system_state = "inactive"
        self._trigger_event("state_changed", {"state": self.system_state})
        logging.info("✅ Node Consciousness Manager stopped")
        return True
    
    def register_node(self, node: BaseConsciousnessNode) -> bool:
        """
        Register a consciousness node with the manager.
        
        Args:
            node: The node instance to register
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        if node.node_id in self.nodes:
            logging.warning(f"⚠️ Node with ID '{node.node_id}' already registered")
            return False
            
        # Register the node
        self.nodes[node.node_id] = node
        
        # Add to node type mapping
        if node.node_type not in self.node_types:
            self.node_types[node.node_type] = set()
        self.node_types[node.node_type].add(node.node_id)
        
        logging.info(f"✅ Registered node: {node.node_id} (type: {node.node_type})")
        
        # Trigger event
        self._trigger_event("node_added", {
            "node_id": node.node_id,
            "node_type": node.node_type
        })
        
        # Activate node if manager is active and auto-activate is enabled
        if self.active and self.config["auto_activate_nodes"]:
            self._activate_node(node.node_id)
            
        return True
    
    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a consciousness node from the manager.
        
        Args:
            node_id: ID of the node to unregister
            
        Returns:
            bool: True if unregistration was successful, False otherwise
        """
        if node_id not in self.nodes:
            logging.warning(f"⚠️ Node with ID '{node_id}' not registered")
            return False
            
        # Deactivate the node if it's active
        if self.nodes[node_id].state != NodeState.INACTIVE:
            self._deactivate_node(node_id)
            
        # Get node type for cleanup
        node_type = self.nodes[node_id].node_type
            
        # Unregister from type mapping
        if node_type in self.node_types and node_id in self.node_types[node_type]:
            self.node_types[node_type].remove(node_id)
            if not self.node_types[node_type]:
                del self.node_types[node_type]
                
        # Unregister from channels
        for channel, subscribers in self.channels.items():
            if node_id in subscribers:
                subscribers.remove(node_id)
                
        # Remove the node
        node = self.nodes.pop(node_id)
        
        logging.info(f"✅ Unregistered node: {node_id}")
        
        # Trigger event
        self._trigger_event("node_removed", {
            "node_id": node_id,
            "node_type": node_type
        })
            
        return True
    
    def _activate_node(self, node_id: str) -> bool:
        """
        Activate a specific node.
        
        Args:
            node_id: ID of the node to activate
            
        Returns:
            bool: True if activation was successful, False otherwise
        """
        if node_id not in self.nodes:
            logging.warning(f"⚠️ Cannot activate: Node with ID '{node_id}' not found")
            return False
            
        node = self.nodes[node_id]
        if node.state != NodeState.INACTIVE and node.state != NodeState.ERROR:
            logging.debug(f"⚠️ Node '{node_id}' already active or in incompatible state")
            return False
            
        success = node.activate()
        if success:
            logging.info(f"✅ Activated node: {node_id}")
            self.metrics["active_nodes"] += 1
        else:
            logging.error(f"❌ Failed to activate node: {node_id}")
            
        return success
    
    def _deactivate_node(self, node_id: str) -> bool:
        """
        Deactivate a specific node.
        
        Args:
            node_id: ID of the node to deactivate
            
        Returns:
            bool: True if deactivation was successful, False otherwise
        """
        if node_id not in self.nodes:
            logging.warning(f"⚠️ Cannot deactivate: Node with ID '{node_id}' not found")
            return False
            
        node = self.nodes[node_id]
        if node.state == NodeState.INACTIVE:
            logging.debug(f"⚠️ Node '{node_id}' already inactive")
            return False
            
        success = node.deactivate()
        if success:
            logging.info(f"✅ Deactivated node: {node_id}")
            if self.metrics["active_nodes"] > 0:
                self.metrics["active_nodes"] -= 1
        else:
            logging.error(f"❌ Failed to deactivate node: {node_id}")
            
        return success
    
    def send_message(self, target_node_id: str, message: Dict[str, Any]) -> bool:
        """
        Send a message to a specific node.
        
        Args:
            target_node_id: ID of the target node
            message: Message dictionary to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if target_node_id not in self.nodes:
            logging.warning(f"⚠️ Cannot send message: Node with ID '{target_node_id}' not found")
            return False
            
        node = self.nodes[target_node_id]
        if node.state != NodeState.ACTIVE:
            logging.warning(f"⚠️ Cannot send message: Node '{target_node_id}' is not active")
            return False
            
        # Add metadata if not present
        if 'message_id' not in message:
            message['message_id'] = str(uuid.uuid4())
        if 'timestamp' not in message:
            message['timestamp'] = time.time()
        if 'source' not in message:
            message['source'] = "node_consciousness_manager"
            
        # Send the message
        success = node.receive_message(message)
        if success:
            self.metrics["messages_processed"] += 1
            logging.debug(f"✅ Sent message to node: {target_node_id}")
        else:
            logging.error(f"❌ Failed to send message to node: {target_node_id}")
            
        return success
    
    def broadcast_message(self, message: Dict[str, Any], node_type: Optional[str] = None) -> int:
        """
        Broadcast a message to multiple nodes, optionally filtered by type.
        
        Args:
            message: The message to broadcast
            node_type: Optional node type to filter recipients
            
        Returns:
            int: Number of nodes that successfully received the message
        """
        target_nodes = []
        
        # Determine target nodes
        if node_type:
            if node_type in self.node_types:
                target_nodes = [self.nodes[node_id] for node_id in self.node_types[node_type]
                               if node_id in self.nodes and self.nodes[node_id].state == NodeState.ACTIVE]
            else:
                logging.warning(f"⚠️ No nodes of type '{node_type}' found for broadcast")
                return 0
        else:
            target_nodes = [node for node in self.nodes.values() if node.state == NodeState.ACTIVE]
            
        if not target_nodes:
            logging.warning("⚠️ No active nodes found for broadcast")
            return 0
            
        # Add metadata if not present
        if 'message_id' not in message:
            message['message_id'] = str(uuid.uuid4())
        if 'timestamp' not in message:
            message['timestamp'] = time.time()
        if 'source' not in message:
            message['source'] = "node_consciousness_manager"
        if 'broadcast' not in message:
            message['broadcast'] = True
            
        # Send to all target nodes
        success_count = 0
        for node in target_nodes:
            if node.receive_message(message.copy()):
                success_count += 1
                
        if success_count > 0:
            self.metrics["messages_processed"] += success_count
            logging.debug(f"✅ Broadcast message to {success_count}/{len(target_nodes)} nodes")
        else:
            logging.error(f"❌ Failed to broadcast message to any of {len(target_nodes)} nodes")
            
        return success_count
    
    def create_channel(self, channel_name: str) -> bool:
        """
        Create a new communication channel.
        
        Args:
            channel_name: Name of the channel to create
            
        Returns:
            bool: True if channel was created successfully, False if it already exists
        """
        if channel_name in self.channels:
            logging.warning(f"⚠️ Channel '{channel_name}' already exists")
            return False
            
        self.channels[channel_name] = []
        logging.info(f"✅ Created channel: {channel_name}")
        return True
    
    def subscribe_to_channel(self, channel_name: str, node_id: str) -> bool:
        """
        Subscribe a node to a channel.
        
        Args:
            channel_name: Name of the channel to subscribe to
            node_id: ID of the node subscribing
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if channel_name not in self.channels:
            logging.warning(f"⚠️ Cannot subscribe: Channel '{channel_name}' does not exist")
            return False
            
        if node_id not in self.nodes:
            logging.warning(f"⚠️ Cannot subscribe: Node '{node_id}' not found")
            return False
            
        if node_id in self.channels[channel_name]:
            logging.debug(f"⚠️ Node '{node_id}' already subscribed to channel '{channel_name}'")
            return False
            
        self.channels[channel_name].append(node_id)
        logging.info(f"✅ Node '{node_id}' subscribed to channel '{channel_name}'")
        return True
    
    def unsubscribe_from_channel(self, channel_name: str, node_id: str) -> bool:
        """
        Unsubscribe a node from a channel.
        
        Args:
            channel_name: Name of the channel to unsubscribe from
            node_id: ID of the node unsubscribing
            
        Returns:
            bool: True if unsubscription was successful, False otherwise
        """
        if channel_name not in self.channels:
            logging.warning(f"⚠️ Cannot unsubscribe: Channel '{channel_name}' does not exist")
            return False
            
        if node_id not in self.channels[channel_name]:
            logging.debug(f"⚠️ Node '{node_id}' not subscribed to channel '{channel_name}'")
            return False
            
        self.channels[channel_name].remove(node_id)
        logging.info(f"✅ Node '{node_id}' unsubscribed from channel '{channel_name}'")
        return True
    
    def publish_to_channel(self, channel_name: str, message: Dict[str, Any], 
                           source_node_id: Optional[str] = None) -> int:
        """
        Publish a message to a channel.
        
        Args:
            channel_name: Name of the channel to publish to
            message: Message to publish
            source_node_id: Optional ID of the node publishing the message
            
        Returns:
            int: Number of nodes the message was successfully delivered to
        """
        if channel_name not in self.channels:
            logging.warning(f"⚠️ Cannot publish: Channel '{channel_name}' does not exist")
            return 0
            
        # Validate source node if provided
        if source_node_id and source_node_id not in self.nodes:
            logging.warning(f"⚠️ Cannot publish: Source node '{source_node_id}' not found")
            return 0
            
        # Get subscribers
        subscribers = self.channels[channel_name]
        if not subscribers:
            logging.debug(f"⚠️ No subscribers for channel '{channel_name}'")
            return 0
            
        # Add metadata to message
        if 'message_id' not in message:
            message['message_id'] = str(uuid.uuid4())
        if 'timestamp' not in message:
            message['timestamp'] = time.time()
        if 'source' not in message:
            message['source'] = source_node_id if source_node_id else "node_consciousness_manager"
        if 'channel' not in message:
            message['channel'] = channel_name
            
        # Deliver to subscribers
        success_count = 0
        for node_id in subscribers:
            # Skip the source node to avoid feedback loops
            if node_id == source_node_id:
                continue
                
            if node_id in self.nodes and self.nodes[node_id].state == NodeState.ACTIVE:
                if self.nodes[node_id].receive_message(message.copy()):
                    success_count += 1
                    
        if success_count > 0:
            self.metrics["messages_processed"] += success_count
            logging.debug(f"✅ Published message to {success_count}/{len(subscribers)} subscribers on channel '{channel_name}'")
        else:
            logging.warning(f"⚠️ Failed to publish message to any subscribers on channel '{channel_name}'")
            
        return success_count
    
    def _run_communication(self):
        """Background thread for inter-node communication processing"""
        while self.running:
            # This would handle any queued communication tasks
            # For now, just sleep as communication is handled directly
            time.sleep(self.config["communication_interval"])
    
    def _monitor_nodes(self):
        """Background thread for monitoring node status"""
        while self.running:
            try:
                active_count = 0
                error_count = 0
                
                for node_id, node in self.nodes.items():
                    if node.state == NodeState.ACTIVE:
                        active_count += 1
                    elif node.state == NodeState.ERROR:
                        error_count += 1
                        logging.warning(f"⚠️ Node '{node_id}' is in ERROR state")
                        
                        # Auto-recovery if enabled
                        if self.config["auto_recovery"]:
                            logging.info(f"🔄 Attempting to recover node '{node_id}'")
                            self._deactivate_node(node_id)
                            time.sleep(0.5)  # Brief pause before reactivation
                            self._activate_node(node_id)
                
                # Update metrics
                self.metrics["active_nodes"] = active_count
                
                # Sleep until next check
                time.sleep(self.config["monitor_interval"])
                
            except Exception as e:
                logging.error(f"❌ Error in node monitor thread: {str(e)}")
                self._trigger_event("system_error", {
                    "component": "monitor_thread",
                    "error": str(e),
                    "timestamp": time.time()
                })
                time.sleep(1.0)  # Brief pause after error
    
    def _breath_cycle(self):
        """Background thread for system breath cycle"""
        while self.running:
            try:
                # Trigger breath event
                breath_time = time.time()
                breath_data = {
                    "timestamp": breath_time,
                    "cycle": self.metrics["breath_cycles"],
                    "active_nodes": self.metrics["active_nodes"],
                    "interval": self.config["breath_interval"]
                }
                
                # Send breath to all active nodes
                self.broadcast_message({
                    "type": "system_breath",
                    "content": breath_data
                })
                
                # Update metrics
                self.metrics["breath_cycles"] += 1
                self.metrics["last_breath_time"] = breath_time
                
                # Trigger event for external systems
                self._trigger_event("breath", breath_data)
                
                # Sleep until next breath
                time.sleep(self.config["breath_interval"])
                
            except Exception as e:
                logging.error(f"❌ Error in breath cycle thread: {str(e)}")
                self._trigger_event("system_error", {
                    "component": "breath_thread",
                    "error": str(e),
                    "timestamp": time.time()
                })
                time.sleep(1.0)  # Brief pause after error
    
    def get_node(self, node_id: str) -> Optional[BaseConsciousnessNode]:
        """
        Get a node by its ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Optional[BaseConsciousnessNode]: The node if found, None otherwise
        """
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[BaseConsciousnessNode]:
        """
        Get all nodes of a specific type.
        
        Args:
            node_type: Type of nodes to retrieve
            
        Returns:
            List[BaseConsciousnessNode]: List of nodes of the specified type
        """
        if node_type not in self.node_types:
            return []
            
        return [self.nodes[node_id] for node_id in self.node_types[node_type] 
                if node_id in self.nodes]
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Node Consciousness system.
        
        Returns:
            dict: System status information
        """
        # Get all node statuses
        node_statuses = {node_id: node.get_status() for node_id, node in self.nodes.items()}
        
        # Count nodes by state
        node_state_counts = {state.name: 0 for state in NodeState}
        for node in self.nodes.values():
            node_state_counts[node.state.name] += 1
            
        # Count nodes by type
        node_type_counts = {node_type: len(node_ids) for node_type, node_ids in self.node_types.items()}
            
        return {
            "system_state": self.system_state,
            "active": self.active,
            "node_count": len(self.nodes),
            "active_nodes": self.metrics["active_nodes"],
            "node_states": node_state_counts,
            "node_types": node_type_counts,
            "channels": {channel: len(subscribers) for channel, subscribers in self.channels.items()},
            "metrics": self.metrics,
            "node_statuses": node_statuses,
            "config": self.config
        }
    
    def register_event_listener(self, event_type: str, callback: Callable):
        """
        Register a callback function for a specific event type.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        if event_type not in self.event_listeners:
            logging.warning(f"⚠️ Unknown event type: {event_type}")
            return False
            
        self.event_listeners[event_type].append(callback)
        return True
    
    def unregister_event_listener(self, event_type: str, callback: Callable):
        """
        Unregister a callback function for a specific event type.
        
        Args:
            event_type: Type of event to unregister from
            callback: Function to unregister
        """
        if event_type not in self.event_listeners:
            return False
            
        if callback in self.event_listeners[event_type]:
            self.event_listeners[event_type].remove(callback)
            return True
        return False
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger an event for all registered listeners"""
        if event_type not in self.event_listeners:
            return
            
        for listener in self.event_listeners[event_type]:
            try:
                listener(data)
            except Exception as e:
                logging.error(f"❌ Error in event listener for {event_type}: {str(e)}")


# For standalone testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and test node consciousness manager
    manager = NodeConsciousnessManager()
    
    # Create a few test nodes
    test_nodes = [
        BaseConsciousnessNode("test_node_1", "test"),
        BaseConsciousnessNode("test_node_2", "test"),
        BaseConsciousnessNode("language_node", "language")
    ]
    
    # Register nodes
    for node in test_nodes:
        manager.register_node(node)
        
    # Start the manager
    manager.start()
    
    # Create a channel
    manager.create_channel("test_channel")
    
    # Subscribe nodes to channel
    manager.subscribe_to_channel("test_channel", "test_node_1")
    manager.subscribe_to_channel("test_channel", "test_node_2")
    
    # Publish a message
    manager.publish_to_channel("test_channel", {
        "type": "test_message",
        "content": "Hello from test!"
    })
    
    # Wait a bit
    time.sleep(2.0)
    
    # Print system status
    import json
    print(json.dumps(manager.get_system_status(), indent=2))
    
    # Clean up
    manager.stop() 