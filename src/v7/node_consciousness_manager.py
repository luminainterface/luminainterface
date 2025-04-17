#!/usr/bin/env python
"""
Node Consciousness Manager for V7

This module provides the core manager for handling node-based consciousness entities
in the V7 system. It orchestrates the creation, connection, and communication between
various consciousness nodes.
"""

import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field

# Configure logging
logger = logging.getLogger(__name__)

class NodeState(Enum):
    """Possible states for a consciousness node."""
    INACTIVE = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    LEARNING = auto()
    PROCESSING = auto()
    CONNECTING = auto()
    ERROR = auto()
    SHUTDOWN = auto()

@dataclass
class NodeConnection:
    """Represents a connection between two consciousness nodes."""
    source_id: str
    target_id: str
    connection_type: str
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def activate(self, intensity: float = 1.0) -> None:
        """Activate this connection with the given intensity."""
        self.last_active = time.time()
        self.metadata['last_intensity'] = intensity

class ConsciousnessNode:
    """
    Base class for all consciousness nodes in the V7 system.
    
    A consciousness node represents a specialized entity that can process,
    learn, and communicate with other nodes in the system.
    """
    
    def __init__(self, node_id: Optional[str] = None, name: str = '', node_type: str = 'generic'):
        """Initialize a new consciousness node."""
        self.node_id = node_id or str(uuid.uuid4())
        self.name = name or f"Node-{self.node_id[:8]}"
        self.node_type = node_type
        self.state = NodeState.INACTIVE
        self.created_at = time.time()
        self.last_active = time.time()
        self.attributes = {}
        self.incoming_connections: Set[str] = set()
        self.outgoing_connections: Set[str] = set()
        self.event_handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        
        logger.debug(f"Created consciousness node: {self.name} ({self.node_id})")
    
    def activate(self) -> bool:
        """Activate this node, putting it into an ACTIVE state."""
        with self._lock:
            if self.state == NodeState.INACTIVE or self.state == NodeState.ERROR:
                self.state = NodeState.INITIALIZING
                
                try:
                    self._initialize()
                    self.state = NodeState.ACTIVE
                    self.last_active = time.time()
                    self._trigger_event('activated', {'timestamp': self.last_active})
                    logger.info(f"Node activated: {self.name} ({self.node_id})")
                    return True
                except Exception as e:
                    self.state = NodeState.ERROR
                    logger.error(f"Failed to activate node {self.name}: {str(e)}")
                    self._trigger_event('error', {'error': str(e), 'timestamp': time.time()})
                    return False
            else:
                logger.warning(f"Cannot activate node {self.name} in state {self.state.name}")
                return False
    
    def deactivate(self) -> bool:
        """Deactivate this node, putting it into an INACTIVE state."""
        with self._lock:
            if self.state != NodeState.INACTIVE:
                try:
                    self._cleanup()
                    self.state = NodeState.INACTIVE
                    self._trigger_event('deactivated', {'timestamp': time.time()})
                    logger.info(f"Node deactivated: {self.name} ({self.node_id})")
                    return True
                except Exception as e:
                    self.state = NodeState.ERROR
                    logger.error(f"Error deactivating node {self.name}: {str(e)}")
                    self._trigger_event('error', {'error': str(e), 'timestamp': time.time()})
                    return False
            return True
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming data and return the result.
        
        This method should be overridden by subclasses to implement
        specialized processing logic.
        """
        with self._lock:
            if self.state != NodeState.ACTIVE:
                logger.warning(f"Cannot process data in node {self.name}: not active")
                return {'success': False, 'error': 'Node not active'}
            
            self.state = NodeState.PROCESSING
            self.last_active = time.time()
            
            try:
                result = self._process_impl(data)
                self.state = NodeState.ACTIVE
                self._trigger_event('processed', {
                    'timestamp': time.time(),
                    'input_keys': list(data.keys()),
                    'output_keys': list(result.keys())
                })
                return result
            except Exception as e:
                self.state = NodeState.ERROR
                error_msg = f"Error processing data in node {self.name}: {str(e)}"
                logger.error(error_msg)
                self._trigger_event('error', {'error': error_msg, 'timestamp': time.time()})
                return {'success': False, 'error': error_msg}
    
    def learn(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train/adapt the node based on the provided training data.
        
        This method should be overridden by subclasses to implement
        specialized learning logic.
        """
        with self._lock:
            if self.state != NodeState.ACTIVE:
                logger.warning(f"Cannot initiate learning in node {self.name}: not active")
                return {'success': False, 'error': 'Node not active'}
            
            self.state = NodeState.LEARNING
            self.last_active = time.time()
            
            try:
                result = self._learn_impl(training_data)
                self.state = NodeState.ACTIVE
                self._trigger_event('learned', {
                    'timestamp': time.time(),
                    'data_size': len(training_data)
                })
                return result
            except Exception as e:
                self.state = NodeState.ERROR
                error_msg = f"Error learning in node {self.name}: {str(e)}"
                logger.error(error_msg)
                self._trigger_event('error', {'error': error_msg, 'timestamp': time.time()})
                return {'success': False, 'error': error_msg}
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of this node."""
        with self._lock:
            return {
                'node_id': self.node_id,
                'name': self.name,
                'type': self.node_type,
                'state': self.state.name,
                'created_at': self.created_at,
                'last_active': self.last_active,
                'connections': {
                    'incoming': len(self.incoming_connections),
                    'outgoing': len(self.outgoing_connections)
                },
                'attributes': self.attributes
            }
    
    def on(self, event_name: str, handler: Callable) -> None:
        """Register an event handler for the specified event."""
        with self._lock:
            if event_name not in self.event_handlers:
                self.event_handlers[event_name] = []
            self.event_handlers[event_name].append(handler)
    
    def off(self, event_name: str, handler: Optional[Callable] = None) -> None:
        """Unregister an event handler for the specified event."""
        with self._lock:
            if event_name in self.event_handlers:
                if handler:
                    self.event_handlers[event_name] = [h for h in self.event_handlers[event_name] if h != handler]
                else:
                    self.event_handlers[event_name] = []
    
    def _initialize(self) -> None:
        """
        Initialize the node. 
        
        This method should be overridden by subclasses to implement
        specialized initialization logic.
        """
        pass
    
    def _cleanup(self) -> None:
        """
        Clean up resources used by this node.
        
        This method should be overridden by subclasses to implement
        specialized cleanup logic.
        """
        pass
    
    def _process_impl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of the processing logic.
        
        This method should be overridden by subclasses.
        """
        return {'success': True, 'result': 'Not implemented'}
    
    def _learn_impl(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of the learning logic.
        
        This method should be overridden by subclasses.
        """
        return {'success': True, 'result': 'Not implemented'}
    
    def _trigger_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Trigger an event, calling all registered handlers."""
        if event_name in self.event_handlers:
            event_data['node_id'] = self.node_id
            event_data['node_name'] = self.name
            event_data['event'] = event_name
            
            for handler in self.event_handlers[event_name]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_name} on node {self.name}: {str(e)}")


class NodeConsciousnessManager:
    """
    Manager class for all consciousness nodes in the V7 system.
    
    This manager is responsible for:
    - Creating and registering new nodes
    - Establishing connections between nodes
    - Facilitating communication between nodes
    - Monitoring the state of all nodes
    - Handling node lifecycle events
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Node Consciousness Manager."""
        self.config = config or {}
        self.nodes: Dict[str, ConsciousnessNode] = {}
        self.connections: Dict[str, NodeConnection] = {}
        self.node_types: Dict[str, type] = {}
        self.global_event_handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread = None
        
        # Configure default settings
        self.monitor_interval = self.config.get('monitor_interval', 5.0)  # seconds
        self.auto_recovery = self.config.get('auto_recovery', True)
        
        logger.info("Node Consciousness Manager initialized")
    
    def register_node_type(self, type_name: str, node_class: type) -> None:
        """Register a new consciousness node type/class."""
        with self._lock:
            if not issubclass(node_class, ConsciousnessNode):
                raise ValueError(f"Node class must be a subclass of ConsciousnessNode")
            
            self.node_types[type_name] = node_class
            logger.debug(f"Registered node type: {type_name}")
    
    def create_node(self, node_type: str, name: Optional[str] = None, 
                    node_id: Optional[str] = None, **kwargs) -> str:
        """
        Create a new consciousness node of the specified type.
        
        Args:
            node_type: The type of node to create
            name: Optional name for the node
            node_id: Optional ID for the node
            **kwargs: Additional arguments to pass to the node constructor
            
        Returns:
            The ID of the created node
        """
        with self._lock:
            if node_type not in self.node_types:
                raise ValueError(f"Unknown node type: {node_type}")
            
            # Create the node instance
            node_class = self.node_types[node_type]
            node = node_class(node_id=node_id, name=name, node_type=node_type, **kwargs)
            
            # Register the node
            self.nodes[node.node_id] = node
            
            # Set up event forwarding
            node.on('activated', lambda data: self._forward_event('node_activated', data))
            node.on('deactivated', lambda data: self._forward_event('node_deactivated', data))
            node.on('error', lambda data: self._forward_event('node_error', data))
            node.on('processed', lambda data: self._forward_event('node_processed', data))
            node.on('learned', lambda data: self._forward_event('node_learned', data))
            
            logger.info(f"Created node: {node.name} ({node.node_id}) of type {node_type}")
            return node.node_id
    
    def get_node(self, node_id: str) -> Optional[ConsciousnessNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its connections."""
        with self._lock:
            node = self.nodes.get(node_id)
            if not node:
                logger.warning(f"Cannot delete node {node_id}: not found")
                return False
            
            # Deactivate the node if it's active
            if node.state != NodeState.INACTIVE:
                node.deactivate()
            
            # Remove all connections involving this node
            conn_ids_to_remove = []
            for conn_id, conn in self.connections.items():
                if conn.source_id == node_id or conn.target_id == node_id:
                    conn_ids_to_remove.append(conn_id)
            
            for conn_id in conn_ids_to_remove:
                del self.connections[conn_id]
            
            # Remove the node
            del self.nodes[node_id]
            logger.info(f"Deleted node: {node.name} ({node_id})")
            return True
    
    def connect_nodes(self, source_id: str, target_id: str, 
                      connection_type: str, strength: float = 1.0,
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create a connection between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            connection_type: Type of connection to establish
            strength: Initial strength of the connection
            metadata: Additional metadata for the connection
            
        Returns:
            The ID of the created connection, or None if it couldn't be created
        """
        with self._lock:
            # Verify nodes exist
            source_node = self.nodes.get(source_id)
            target_node = self.nodes.get(target_id)
            
            if not source_node:
                logger.warning(f"Cannot create connection: source node {source_id} not found")
                return None
            
            if not target_node:
                logger.warning(f"Cannot create connection: target node {target_id} not found")
                return None
            
            # Create the connection
            conn_id = f"conn_{str(uuid.uuid4())}"
            conn = NodeConnection(
                source_id=source_id,
                target_id=target_id,
                connection_type=connection_type,
                strength=strength,
                metadata=metadata or {}
            )
            
            # Register the connection
            self.connections[conn_id] = conn
            source_node.outgoing_connections.add(conn_id)
            target_node.incoming_connections.add(conn_id)
            
            logger.debug(f"Created connection {conn_id} from {source_node.name} to {target_node.name}")
            self._forward_event('nodes_connected', {
                'connection_id': conn_id,
                'source_id': source_id,
                'target_id': target_id,
                'connection_type': connection_type
            })
            
            return conn_id
    
    def disconnect_nodes(self, connection_id: str) -> bool:
        """Remove a connection between nodes."""
        with self._lock:
            conn = self.connections.get(connection_id)
            if not conn:
                logger.warning(f"Cannot disconnect: connection {connection_id} not found")
                return False
            
            # Remove connection references from nodes
            source_node = self.nodes.get(conn.source_id)
            if source_node:
                source_node.outgoing_connections.discard(connection_id)
            
            target_node = self.nodes.get(conn.target_id)
            if target_node:
                target_node.incoming_connections.discard(connection_id)
            
            # Remove the connection
            del self.connections[connection_id]
            logger.debug(f"Removed connection {connection_id}")
            
            self._forward_event('nodes_disconnected', {
                'connection_id': connection_id,
                'source_id': conn.source_id,
                'target_id': conn.target_id
            })
            
            return True
    
    def activate_node(self, node_id: str) -> bool:
        """Activate a node by its ID."""
        node = self.nodes.get(node_id)
        if not node:
            logger.warning(f"Cannot activate node {node_id}: not found")
            return False
        
        return node.activate()
    
    def deactivate_node(self, node_id: str) -> bool:
        """Deactivate a node by its ID."""
        node = self.nodes.get(node_id)
        if not node:
            logger.warning(f"Cannot deactivate node {node_id}: not found")
            return False
        
        return node.deactivate()
    
    def process_data(self, node_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through a specific node."""
        node = self.nodes.get(node_id)
        if not node:
            logger.warning(f"Cannot process data through node {node_id}: not found")
            return {'success': False, 'error': 'Node not found'}
        
        return node.process(data)
    
    def train_node(self, node_id: str, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train a specific node with the provided data."""
        node = self.nodes.get(node_id)
        if not node:
            logger.warning(f"Cannot train node {node_id}: not found")
            return {'success': False, 'error': 'Node not found'}
        
        return node.learn(training_data)
    
    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific node."""
        node = self.nodes.get(node_id)
        if not node:
            return None
        
        return node.get_status()
    
    def get_all_node_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all nodes."""
        with self._lock:
            return {node_id: node.get_status() for node_id, node in self.nodes.items()}
    
    def activate_all_nodes(self) -> Dict[str, bool]:
        """Attempt to activate all inactive nodes."""
        with self._lock:
            results = {}
            for node_id, node in self.nodes.items():
                if node.state == NodeState.INACTIVE:
                    results[node_id] = node.activate()
            
            return results
    
    def deactivate_all_nodes(self) -> Dict[str, bool]:
        """Deactivate all active nodes."""
        with self._lock:
            results = {}
            for node_id, node in self.nodes.items():
                if node.state != NodeState.INACTIVE:
                    results[node_id] = node.deactivate()
            
            return results
    
    def get_connections(self, node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all connections, optionally filtered by node.
        
        Args:
            node_id: If provided, only return connections involving this node
            
        Returns:
            List of connection details
        """
        with self._lock:
            result = []
            
            for conn_id, conn in self.connections.items():
                if node_id is None or conn.source_id == node_id or conn.target_id == node_id:
                    result.append({
                        'connection_id': conn_id,
                        'source_id': conn.source_id,
                        'target_id': conn.target_id,
                        'type': conn.connection_type,
                        'strength': conn.strength,
                        'created_at': conn.created_at,
                        'last_active': conn.last_active,
                        'metadata': conn.metadata
                    })
            
            return result
    
    def on(self, event_name: str, handler: Callable) -> None:
        """Register a global event handler."""
        with self._lock:
            if event_name not in self.global_event_handlers:
                self.global_event_handlers[event_name] = []
            self.global_event_handlers[event_name].append(handler)
    
    def off(self, event_name: str, handler: Optional[Callable] = None) -> None:
        """Unregister a global event handler."""
        with self._lock:
            if event_name in self.global_event_handlers:
                if handler:
                    self.global_event_handlers[event_name] = [
                        h for h in self.global_event_handlers[event_name] if h != handler
                    ]
                else:
                    self.global_event_handlers[event_name] = []
    
    def start_monitoring(self) -> bool:
        """Start the background monitoring thread."""
        with self._lock:
            if self._running:
                logger.warning("Monitoring thread is already running")
                return False
            
            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="NodeConsciousnessMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("Node monitoring started")
            return True
    
    def stop_monitoring(self) -> bool:
        """Stop the background monitoring thread."""
        with self._lock:
            if not self._running:
                logger.warning("Monitoring thread is not running")
                return False
            
            self._running = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
                if self._monitor_thread.is_alive():
                    logger.warning("Monitoring thread did not terminate cleanly")
                else:
                    logger.info("Node monitoring stopped")
            
            self._monitor_thread = None
            return True
    
    def shutdown(self) -> None:
        """Shutdown the manager and all nodes."""
        logger.info("Shutting down Node Consciousness Manager")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Deactivate all nodes
        self.deactivate_all_nodes()
        
        # Clear references
        with self._lock:
            self.connections.clear()
            self.nodes.clear()
            self.global_event_handlers.clear()
        
        logger.info("Node Consciousness Manager shutdown complete")
    
    def _monitor_loop(self) -> None:
        """Background thread to monitor node health."""
        while self._running:
            try:
                # Check for nodes in error state
                error_nodes = []
                with self._lock:
                    for node_id, node in self.nodes.items():
                        if node.state == NodeState.ERROR:
                            error_nodes.append(node_id)
                
                # Attempt auto recovery if enabled
                if self.auto_recovery:
                    for node_id in error_nodes:
                        logger.info(f"Attempting to recover node {node_id} from error state")
                        node = self.nodes.get(node_id)
                        if node:
                            node.deactivate()
                            success = node.activate()
                            if success:
                                logger.info(f"Successfully recovered node {node_id}")
                            else:
                                logger.warning(f"Failed to recover node {node_id}")
                
                # Sleep until next check
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in node monitor loop: {str(e)}")
                time.sleep(1.0)  # Shorter sleep after error
    
    def _forward_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Forward an event to all registered global handlers."""
        if event_name in self.global_event_handlers:
            event_data['timestamp'] = event_data.get('timestamp', time.time())
            
            for handler in self.global_event_handlers[event_name]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in global event handler for {event_name}: {str(e)}")
    
    def get_node_counts(self) -> Dict[str, int]:
        """Get counts of different node types."""
        with self._lock:
            counts = {}
            for node in self.nodes.values():
                node_type = node.node_type
                counts[node_type] = counts.get(node_type, 0) + 1
            return counts


def create_node_consciousness_manager(config: Optional[Dict[str, Any]] = None) -> NodeConsciousnessManager:
    """
    Factory function to create a NodeConsciousnessManager instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        A configured NodeConsciousnessManager instance
    """
    return NodeConsciousnessManager(config=config) 