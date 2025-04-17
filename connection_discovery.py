import logging
import threading
import time
import json
import os
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, Union
from pathlib import Path
import importlib.util
import inspect
import socket
import uuid
import queue
from dataclasses import dataclass, field
from threading import RLock

logger = logging.getLogger(__name__)

@dataclass
class NodeInfo:
    """Information about a registered node"""
    node_id: str
    node_type: str
    name: str
    node: Any = None  # The actual node object
    registration_time: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConnectionDiscovery:
    """
    Service for node discovery and connection management
    
    This singleton class provides:
    - Node registration and deregistration
    - Node discovery by ID, type, or name
    - Automatic heartbeat tracking
    - Connection suggestions based on compatibility
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ConnectionDiscovery':
        """Get or create the singleton instance"""
        if cls._instance is None:
            cls._instance = ConnectionDiscovery()
        return cls._instance
    
    def __init__(self):
        """Initialize the discovery service"""
        if ConnectionDiscovery._instance is not None:
            raise RuntimeError("Use ConnectionDiscovery.get_instance() to get the singleton instance")
        
        self._lock = RLock()
        self._nodes: Dict[str, NodeInfo] = {}  # node_id -> NodeInfo
        self._nodes_by_type: Dict[str, Set[str]] = {}  # node_type -> set of node_ids
        self._nodes_by_name: Dict[str, str] = {}  # node_name -> node_id
        
        self._heartbeat_timeout = 60.0  # seconds
        self._cleanup_interval = 300.0  # seconds
        self._last_cleanup = time.time()
        
        # Callbacks
        self._node_registered_callbacks: List[Callable] = []
        self._node_deregistered_callbacks: List[Callable] = []
        
        logger.info("Connection Discovery service initialized")
    
    def register_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        node: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> NodeInfo:
        """
        Register a node with the discovery service
        
        Args:
            node_id: Unique ID of the node
            node_type: Type of the node
            name: Human-readable name
            node: Actual node object (optional)
            metadata: Additional node metadata
            
        Returns:
            NodeInfo: The registered node info
        """
        with self._lock:
            if node_id in self._nodes:
                # Node already registered, update info
                node_info = self._nodes[node_id]
                old_type = node_info.node_type
                old_name = node_info.name
                
                # Update fields
                node_info.node_type = node_type
                node_info.name = name
                node_info.node = node if node is not None else node_info.node
                node_info.last_heartbeat = time.time()
                node_info.is_active = True
                if metadata is not None:
                    node_info.metadata.update(metadata)
                
                # Update indices if type or name changed
                if old_type != node_type:
                    self._nodes_by_type.setdefault(old_type, set()).discard(node_id)
                    self._nodes_by_type.setdefault(node_type, set()).add(node_id)
                
                if old_name != name:
                    if old_name in self._nodes_by_name and self._nodes_by_name[old_name] == node_id:
                        del self._nodes_by_name[old_name]
                    self._nodes_by_name[name] = node_id
                
                logger.debug(f"Updated registration for node {name} (ID: {node_id}, Type: {node_type})")
                
            else:
                # New node
                node_info = NodeInfo(
                    node_id=node_id,
                    node_type=node_type,
                    name=name,
                    node=node,
                    registration_time=time.time(),
                    last_heartbeat=time.time(),
                    is_active=True,
                    metadata=metadata or {}
                )
                
                # Add to indices
                self._nodes[node_id] = node_info
                self._nodes_by_type.setdefault(node_type, set()).add(node_id)
                self._nodes_by_name[name] = node_id
                
                logger.info(f"Registered new node {name} (ID: {node_id}, Type: {node_type})")
                
                # Call callbacks
                for callback in self._node_registered_callbacks:
                    try:
                        callback(node_info)
                    except Exception as e:
                        logger.error(f"Error in node registration callback: {str(e)}")
            
            # Perform periodic cleanup
            self._maybe_cleanup()
            
            return node_info
    
    def deregister_node(self, node_id: str) -> bool:
        """
        Deregister a node from the discovery service
        
        Args:
            node_id: ID of the node to deregister
            
        Returns:
            bool: True if deregistered, False if not found
        """
        with self._lock:
            if node_id not in self._nodes:
                return False
            
            node_info = self._nodes[node_id]
            
            # Remove from indices
            del self._nodes[node_id]
            if node_info.node_type in self._nodes_by_type:
                self._nodes_by_type[node_info.node_type].discard(node_id)
            if node_info.name in self._nodes_by_name and self._nodes_by_name[node_info.name] == node_id:
                del self._nodes_by_name[node_info.name]
            
            logger.info(f"Deregistered node {node_info.name} (ID: {node_id}, Type: {node_info.node_type})")
            
            # Call callbacks
            for callback in self._node_deregistered_callbacks:
                try:
                    callback(node_info)
                except Exception as e:
                    logger.error(f"Error in node deregistration callback: {str(e)}")
            
            return True
    
    def heartbeat(self, node_id: str) -> bool:
        """
        Update heartbeat for a node
        
        Args:
            node_id: ID of the node
            
        Returns:
            bool: True if updated, False if node not found
        """
        with self._lock:
            if node_id not in self._nodes:
                return False
            
            node_info = self._nodes[node_id]
            node_info.last_heartbeat = time.time()
            node_info.is_active = True
            
            return True
    
    def get_node_by_id(self, node_id: str) -> Optional[NodeInfo]:
        """
        Get node info by ID
        
        Args:
            node_id: ID of the node
            
        Returns:
            Optional[NodeInfo]: Node info if found, None otherwise
        """
        with self._lock:
            return self._nodes.get(node_id)
    
    def get_node_by_name(self, name: str) -> Optional[NodeInfo]:
        """
        Get node info by name
        
        Args:
            name: Name of the node
            
        Returns:
            Optional[NodeInfo]: Node info if found, None otherwise
        """
        with self._lock:
            node_id = self._nodes_by_name.get(name)
            if node_id is None:
                return None
            return self._nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[NodeInfo]:
        """
        Get all nodes of a specific type
        
        Args:
            node_type: Type of nodes to find
            
        Returns:
            List[NodeInfo]: List of node infos
        """
        with self._lock:
            node_ids = self._nodes_by_type.get(node_type, set())
            return [self._nodes[node_id] for node_id in node_ids if node_id in self._nodes]
    
    def get_all_nodes(self) -> List[NodeInfo]:
        """
        Get all registered nodes
        
        Returns:
            List[NodeInfo]: List of all node infos
        """
        with self._lock:
            return list(self._nodes.values())
    
    def get_active_nodes(self) -> List[NodeInfo]:
        """
        Get all active nodes (those with recent heartbeats)
        
        Returns:
            List[NodeInfo]: List of active node infos
        """
        with self._lock:
            now = time.time()
            return [
                node_info for node_info in self._nodes.values()
                if node_info.is_active and now - node_info.last_heartbeat <= self._heartbeat_timeout
            ]
    
    def suggest_connections(self, node_id: str, max_suggestions: int = 5) -> List[NodeInfo]:
        """
        Suggest potential connections for a node based on compatibility
        
        Args:
            node_id: ID of the node to find connections for
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List[NodeInfo]: List of suggested nodes
        """
        # This is a placeholder implementation
        # In a real system, this would use more sophisticated compatibility rules
        with self._lock:
            if node_id not in self._nodes:
                return []
            
            node_info = self._nodes[node_id]
            
            # Simple suggestion algorithm: suggest active nodes of different types
            suggestions = []
            for other_node_id, other_info in self._nodes.items():
                if other_node_id == node_id:
                    continue
                    
                if not other_info.is_active:
                    continue
                    
                # Prefer nodes of different types for diversity
                if other_info.node_type != node_info.node_type:
                    suggestions.append(other_info)
                    
                if len(suggestions) >= max_suggestions:
                    break
            
            # If we don't have enough, add any active nodes
            if len(suggestions) < max_suggestions:
                for other_node_id, other_info in self._nodes.items():
                    if other_node_id == node_id or other_info in suggestions:
                        continue
                        
                    if other_info.is_active:
                        suggestions.append(other_info)
                        
                    if len(suggestions) >= max_suggestions:
                        break
            
            return suggestions
    
    def _maybe_cleanup(self) -> None:
        """Periodically clean up inactive nodes"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
            
        self._last_cleanup = now
        inactive_ids = []
        
        # Find inactive nodes
        for node_id, node_info in self._nodes.items():
            if now - node_info.last_heartbeat > self._heartbeat_timeout:
                if node_info.is_active:
                    logger.warning(f"Node {node_info.name} (ID: {node_id}) is inactive (last heartbeat: {now - node_info.last_heartbeat:.1f}s ago)")
                    node_info.is_active = False
                
                # If node has been inactive for a very long time, mark for removal
                if now - node_info.last_heartbeat > self._heartbeat_timeout * 5:
                    inactive_ids.append(node_id)
        
        # Remove very inactive nodes
        for node_id in inactive_ids:
            logger.info(f"Removing long-inactive node {self._nodes[node_id].name} (ID: {node_id})")
            self.deregister_node(node_id)
    
    def register_node_registered_callback(self, callback: Callable[[NodeInfo], None]) -> None:
        """Register a callback for node registration events"""
        self._node_registered_callbacks.append(callback)
    
    def register_node_deregistered_callback(self, callback: Callable[[NodeInfo], None]) -> None:
        """Register a callback for node deregistration events"""
        self._node_deregistered_callbacks.append(callback)


class NodeRegistrationClient:
    """
    Client for registering with the discovery service and maintaining registration
    
    This class handles:
    - Initial registration
    - Heartbeat sending
    - Clean deregistration
    """
    
    def __init__(
        self,
        node_id: str,
        node_type: str,
        name: str,
        node: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        heartbeat_interval: float = 15.0
    ):
        """
        Initialize the registration client
        
        Args:
            node_id: Unique ID of the node
            node_type: Type of the node
            name: Human-readable name
            node: Actual node object
            metadata: Additional node metadata
            heartbeat_interval: Interval between heartbeats in seconds
        """
        self.node_id = node_id
        self.node_type = node_type
        self.name = name
        self.node = node
        self.metadata = metadata or {}
        self.heartbeat_interval = heartbeat_interval
        
        self._discovery = ConnectionDiscovery.get_instance()
        self._registered = False
        self._heartbeat_thread = None
        self._running = False
    
    def register(self) -> bool:
        """
        Register with the discovery service
        
        Returns:
            bool: True if registered, False if already registered
        """
        if self._registered:
            return False
        
        try:
            self._discovery.register_node(
                node_id=self.node_id,
                node_type=self.node_type,
                name=self.name,
                node=self.node,
                metadata=self.metadata
            )
            self._registered = True
            return True
        except Exception as e:
            logger.error(f"Error registering node {self.name}: {str(e)}")
            return False
    
    def deregister(self) -> bool:
        """
        Deregister from the discovery service
        
        Returns:
            bool: True if deregistered, False if not registered
        """
        if not self._registered:
            return False
        
        try:
            self.stop_heartbeat()
            success = self._discovery.deregister_node(self.node_id)
            if success:
                self._registered = False
            return success
        except Exception as e:
            logger.error(f"Error deregistering node {self.name}: {str(e)}")
            return False
    
    def start_heartbeat(self) -> bool:
        """
        Start sending periodic heartbeats
        
        Returns:
            bool: True if started, False if already running or not registered
        """
        if not self._registered or self._running:
            return False
        
        import threading
        
        def heartbeat_loop():
            while self._running:
                try:
                    self._discovery.heartbeat(self.node_id)
                except Exception as e:
                    logger.error(f"Error sending heartbeat for node {self.name}: {str(e)}")
                
                # Sleep for the heartbeat interval
                for _ in range(int(self.heartbeat_interval * 2)):  # Check twice per interval
                    if not self._running:
                        break
                    time.sleep(0.5)
        
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name=f"Heartbeat-{self.name}",
            daemon=True
        )
        self._heartbeat_thread.start()
        
        return True
    
    def stop_heartbeat(self) -> bool:
        """
        Stop sending heartbeats
        
        Returns:
            bool: True if stopped, False if not running
        """
        if not self._running:
            return False
        
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        
        return True
    
    def is_registered(self) -> bool:
        """Check if the node is registered"""
        return self._registered
    
    def update_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Update node metadata
        
        Args:
            metadata: New metadata to merge with existing
            
        Returns:
            bool: True if updated, False if not registered
        """
        if not self._registered:
            return False
        
        try:
            self.metadata.update(metadata)
            self._discovery.register_node(
                node_id=self.node_id,
                node_type=self.node_type,
                name=self.name,
                node=self.node,
                metadata=self.metadata
            )
            return True
        except Exception as e:
            logger.error(f"Error updating metadata for node {self.name}: {str(e)}")
            return False
    
    def find_nodes_by_type(self, node_type: str) -> List[NodeInfo]:
        """
        Find nodes by type
        
        Args:
            node_type: Type of nodes to find
            
        Returns:
            List[NodeInfo]: List of matching node infos
        """
        return self._discovery.get_nodes_by_type(node_type)
    
    def find_node_by_name(self, name: str) -> Optional[NodeInfo]:
        """
        Find a node by name
        
        Args:
            name: Name of the node to find
            
        Returns:
            Optional[NodeInfo]: Node info if found, None otherwise
        """
        return self._discovery.get_node_by_name(name)
    
    def find_node_by_id(self, node_id: str) -> Optional[NodeInfo]:
        """
        Find a node by ID
        
        Args:
            node_id: ID of the node to find
            
        Returns:
            Optional[NodeInfo]: Node info if found, None otherwise
        """
        return self._discovery.get_node_by_id(node_id)
    
    def get_suggested_connections(self, max_suggestions: int = 5) -> List[NodeInfo]:
        """
        Get connection suggestions for this node
        
        Args:
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List[NodeInfo]: List of suggested node infos
        """
        if not self._registered:
            return []
        
        return self._discovery.suggest_connections(self.node_id, max_suggestions)


def register_node(node) -> NodeRegistrationClient:
    """
    Helper function to register a node with the discovery service
    
    Args:
        node: Node to register (must have node_id, node_type, and name attributes)
        
    Returns:
        NodeRegistrationClient: Registration client for the node
    """
    if not hasattr(node, 'node_id') or not hasattr(node, 'node_type') or not hasattr(node, 'name'):
        raise ValueError("Node must have node_id, node_type, and name attributes")
    
    # Extract metadata from node if available
    metadata = {}
    if hasattr(node, 'metadata'):
        metadata = node.metadata
    
    # Create and register client
    client = NodeRegistrationClient(
        node_id=node.node_id,
        node_type=str(node.node_type),
        name=node.name,
        node=node,
        metadata=metadata
    )
    
    client.register()
    client.start_heartbeat()
    
    return client 