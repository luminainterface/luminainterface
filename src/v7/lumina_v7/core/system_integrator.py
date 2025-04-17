"""
LUMINA V7 System Integrator

This module provides the SystemIntegrator class which manages the integration
of consciousness nodes according to configurable patterns and strategies.
"""

import logging
import importlib
import time
from typing import Dict, List, Any, Optional, Set, Tuple

from src.v7.lumina_v7.core.node_consciousness_manager import NodeConsciousnessManager, BaseConsciousnessNode
from src.v7.lumina_v7.core.node_integration_config import (
    get_connection_properties, 
    get_node_capabilities,
    get_compatible_node_types,
    get_integration_strategy,
    get_connection_pattern,
    NODE_TYPES, 
    CONNECTION_PATTERNS,
    INTEGRATION_STRATEGIES
)

logger = logging.getLogger("lumina_v7.system_integrator")

class SystemIntegrator:
    """
    System Integrator for LUMINA V7 that manages node connections and integration strategies.
    """
    
    def __init__(self, node_manager: Optional[NodeConsciousnessManager] = None):
        """
        Initialize the system integrator.
        
        Args:
            node_manager: Optional NodeConsciousnessManager instance to use
        """
        # Use provided manager or create a new one
        self.node_manager = node_manager or NodeConsciousnessManager()
        self.current_strategy = "balanced"
        self.integration_status = {
            "active_strategy": self.current_strategy,
            "connection_count": 0,
            "integration_level": 0.0,
            "last_integration": 0.0,
            "pattern_status": {}
        }
        
        # Nodes by type
        self.nodes_by_type: Dict[str, Set[str]] = {
            node_type: set() for node_type in NODE_TYPES
        }
        
        # Created connections
        self.connections: Dict[str, Dict[str, Any]] = {}
        
        logger.info("✅ System Integrator initialized")
    
    def register_node(self, node: BaseConsciousnessNode, node_type: Optional[str] = None) -> bool:
        """
        Register a node with the system integrator.
        
        Args:
            node: The node to register
            node_type: Optional node type override (defaults to node.node_type)
            
        Returns:
            bool: True if registration was successful
        """
        # Use provided type or get from node
        if not node_type:
            node_type = node.node_type
            
        # Validate node type
        if node_type not in NODE_TYPES:
            logger.warning(f"⚠️ Unknown node type: {node_type}, treating as generic")
            node_type = "generic"
            
        # Register with node manager
        success = self.node_manager.register_node(node)
        if not success:
            logger.error(f"❌ Failed to register node {node.node_id} with node manager")
            return False
            
        # Add to type mapping
        if node_type in self.nodes_by_type:
            self.nodes_by_type[node_type].add(node.node_id)
            
        logger.info(f"✅ Registered node {node.node_id} as type {node_type}")
        
        # Apply current integration strategy
        self._apply_node_activation(node.node_id, node_type)
        
        return True
    
    def apply_integration_strategy(self, strategy_name: str) -> bool:
        """
        Apply an integration strategy to the current nodes.
        
        Args:
            strategy_name: Name of the strategy to apply
            
        Returns:
            bool: True if strategy was applied successfully
        """
        strategy = get_integration_strategy(strategy_name)
        if not strategy:
            logger.error(f"❌ Unknown integration strategy: {strategy_name}")
            return False
            
        logger.info(f"🔄 Applying integration strategy: {strategy_name}")
        
        # Store current strategy
        self.current_strategy = strategy_name
        self.integration_status["active_strategy"] = strategy_name
        
        # Apply node activation levels
        for node_type, level in strategy["node_activation_levels"].items():
            for node_id in self.nodes_by_type.get(node_type, []):
                self._apply_node_activation(node_id, node_type, level)
                
        # Apply connection pattern
        pattern_name = strategy["pattern"]
        pattern = get_connection_pattern(pattern_name)
        
        if pattern_name == "monday_star":
            self._apply_star_pattern(pattern)
        elif pattern_name == "language_tree":
            self._apply_tree_pattern(pattern)
        elif pattern_name == "full_mesh":
            self._apply_mesh_pattern(pattern)
        elif pattern_name == "processing_pipeline":
            self._apply_pipeline_pattern(pattern)
        else:
            # Default to pairwise connections
            self._apply_default_connections()
            
        # Update integration status
        self.integration_status["last_integration"] = time.time()
        self.integration_status["connection_count"] = len(self.connections)
        
        # Calculate integration level (0.0-1.0)
        node_count = sum(len(nodes) for nodes in self.nodes_by_type.values())
        if node_count > 1:
            max_connections = node_count * (node_count - 1) / 2  # Maximum possible connections
            self.integration_status["integration_level"] = min(1.0, len(self.connections) / max_connections)
        else:
            self.integration_status["integration_level"] = 0.0
            
        logger.info(f"✅ Applied integration strategy {strategy_name}, connection count: {len(self.connections)}")
        return True
    
    def _apply_node_activation(self, node_id: str, node_type: str, level: float = 0.8) -> None:
        """Apply activation level to a node"""
        try:
            node = self.node_manager.get_node(node_id)
            if not node:
                logger.warning(f"⚠️ Node {node_id} not found for activation")
                return
                
            # Set activation level if node supports it
            if hasattr(node, 'set_activation_level'):
                node.set_activation_level(level)
                logger.debug(f"Set activation level {level} for node {node_id}")
        except Exception as e:
            logger.error(f"❌ Error setting activation level for node {node_id}: {str(e)}")
    
    def _apply_star_pattern(self, pattern: Dict[str, Any]) -> None:
        """Apply a star connection pattern"""
        center_type = pattern["center_node_type"]
        satellite_types = pattern["satellite_node_types"]
        conn_type = pattern["connection_type"]
        bidirectional = pattern["bidirectional"]
        strength = pattern["strength"]
        
        # Get center nodes
        center_nodes = list(self.nodes_by_type.get(center_type, []))
        if not center_nodes:
            logger.warning(f"⚠️ No center nodes of type {center_type} found for star pattern")
            return
            
        # Use first center node (or implement selection logic)
        center_node_id = center_nodes[0]
        
        # Connect satellites to center
        for sat_type in satellite_types:
            for sat_node_id in self.nodes_by_type.get(sat_type, []):
                if sat_node_id != center_node_id:
                    self._create_connection(
                        center_node_id, sat_node_id, 
                        conn_type, bidirectional, strength
                    )
        
        # Update status
        self.integration_status["pattern_status"]["star"] = {
            "center_node": center_node_id,
            "satellite_count": sum(len(self.nodes_by_type.get(t, [])) for t in satellite_types)
        }
    
    def _apply_tree_pattern(self, pattern: Dict[str, Any]) -> None:
        """Apply a tree connection pattern"""
        root_type = pattern["root_node_type"]
        child_types = pattern["child_types"]
        bidirectional = pattern["bidirectional"]
        strength = pattern["strength"]
        
        # Get root nodes
        root_nodes = list(self.nodes_by_type.get(root_type, []))
        if not root_nodes:
            logger.warning(f"⚠️ No root nodes of type {root_type} found for tree pattern")
            return
            
        # Use first root node (or implement selection logic)
        root_node_id = root_nodes[0]
        
        # Connect first level children
        for child_type, child_info in child_types.items():
            conn_type = child_info["connection_type"]
            
            for child_node_id in self.nodes_by_type.get(child_type, []):
                if child_node_id != root_node_id:
                    self._create_connection(
                        root_node_id, child_node_id, 
                        conn_type, bidirectional, strength
                    )
                    
                    # Connect second level children
                    for grandchild_type in child_info.get("children", []):
                        for grandchild_node_id in self.nodes_by_type.get(grandchild_type, []):
                            if grandchild_node_id != child_node_id and grandchild_node_id != root_node_id:
                                self._create_connection(
                                    child_node_id, grandchild_node_id, 
                                    conn_type, bidirectional, strength * 0.9  # Slightly weaker
                                )
        
        # Update status
        self.integration_status["pattern_status"]["tree"] = {
            "root_node": root_node_id,
            "child_count": sum(len(self.nodes_by_type.get(t, [])) for t in child_types.keys())
        }
    
    def _apply_mesh_pattern(self, pattern: Dict[str, Any]) -> None:
        """Apply a mesh connection pattern"""
        node_types = pattern["node_types"]
        conn_type = pattern["connection_type"]
        bidirectional = pattern["bidirectional"]
        strength = pattern["strength"]
        
        # Get all nodes of specified types
        all_nodes = []
        for node_type in node_types:
            all_nodes.extend(self.nodes_by_type.get(node_type, []))
            
        # Connect all nodes to all other nodes
        for i, source_id in enumerate(all_nodes):
            for target_id in all_nodes[i+1:]:  # Skip self and previously connected
                self._create_connection(
                    source_id, target_id,
                    conn_type, bidirectional, strength
                )
        
        # Update status
        self.integration_status["pattern_status"]["mesh"] = {
            "node_count": len(all_nodes),
            "connection_count": len(self.connections)
        }
    
    def _apply_pipeline_pattern(self, pattern: Dict[str, Any]) -> None:
        """Apply a pipeline connection pattern"""
        node_sequence = pattern["node_sequence"]
        forward_type = pattern["forward_connection_type"]
        backward_type = pattern["backward_connection_type"]
        
        # Build node sequence
        pipeline_nodes = []
        for node_type in node_sequence:
            # Get first available node of each type
            nodes = list(self.nodes_by_type.get(node_type, []))
            if nodes:
                pipeline_nodes.append(nodes[0])
                
        # Connect sequential nodes
        for i in range(len(pipeline_nodes) - 1):
            source_id = pipeline_nodes[i]
            target_id = pipeline_nodes[i+1]
            
            # Forward connection
            self._create_connection(
                source_id, target_id,
                forward_type, False, 0.8
            )
            
            # Backward connection
            self._create_connection(
                target_id, source_id,
                backward_type, False, 0.6
            )
        
        # Update status
        self.integration_status["pattern_status"]["pipeline"] = {
            "node_sequence": pipeline_nodes,
            "sequence_length": len(pipeline_nodes)
        }
    
    def _apply_default_connections(self) -> None:
        """Apply default connections based on node types"""
        # For each source type
        for source_type, source_nodes in self.nodes_by_type.items():
            # Get compatible target types
            compatible_types = get_compatible_node_types(source_type)
            
            # For each source node
            for source_id in source_nodes:
                # For each compatible target type
                for target_type in compatible_types:
                    # Get connection properties
                    props = get_connection_properties(source_type, target_type)
                    
                    # For each target node
                    for target_id in self.nodes_by_type.get(target_type, []):
                        if source_id != target_id:
                            self._create_connection(
                                source_id, target_id,
                                props["type"], props["bidirectional"], props["strength"]
                            )
    
    def _create_connection(self, source_id: str, target_id: str, 
                         conn_type: str, bidirectional: bool, strength: float) -> bool:
        """Create a connection between nodes"""
        # Generate a unique connection ID
        conn_id = f"{source_id}_{target_id}_{conn_type}"
        if conn_id in self.connections:
            # Connection already exists, update instead
            self.connections[conn_id]["strength"] = strength
            return True
            
        try:
            # Create channel if needed
            channel_name = f"{conn_type}_channel"
            self.node_manager.create_channel(channel_name)
            
            # Subscribe nodes to channel
            self.node_manager.subscribe_to_channel(channel_name, source_id)
            self.node_manager.subscribe_to_channel(channel_name, target_id)
            
            # Store connection info
            self.connections[conn_id] = {
                "source_id": source_id,
                "target_id": target_id,
                "type": conn_type,
                "channel": channel_name,
                "bidirectional": bidirectional,
                "strength": strength,
                "created_at": time.time()
            }
            
            logger.debug(f"Created connection: {source_id} -> {target_id} ({conn_type})")
            
            # Create reverse connection if bidirectional
            if bidirectional:
                reverse_id = f"{target_id}_{source_id}_{conn_type}"
                if reverse_id not in self.connections:
                    self.connections[reverse_id] = {
                        "source_id": target_id,
                        "target_id": source_id,
                        "type": conn_type,
                        "channel": channel_name,
                        "bidirectional": True,
                        "strength": strength,
                        "created_at": time.time()
                    }
            
            return True
        except Exception as e:
            logger.error(f"❌ Error creating connection {conn_id}: {str(e)}")
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get the current integration status"""
        # Update node counts
        node_counts = {
            node_type: len(nodes) for node_type, nodes in self.nodes_by_type.items() if nodes
        }
        
        return {
            **self.integration_status,
            "node_counts": node_counts,
            "total_nodes": sum(len(nodes) for nodes in self.nodes_by_type.values()),
            "timestamp": time.time()
        }
    
    def optimize_integration(self) -> Dict[str, Any]:
        """
        Optimize integration based on system state and performance metrics.
        
        Returns:
            Dict with optimization results
        """
        # Get system status
        system_status = self.node_manager.get_system_status()
        
        # Choose best strategy based on active nodes
        active_node_types = set()
        for node_type, nodes in self.nodes_by_type.items():
            for node_id in nodes:
                node_status = system_status["node_statuses"].get(node_id, {})
                if node_status.get("state") == "ACTIVE":
                    active_node_types.add(node_type)
        
        # Select optimal strategy based on active node types
        strategy_name = "balanced"  # Default
        
        if "monday" in active_node_types and len(active_node_types) >= 3:
            strategy_name = "monday_centered"
        elif "language" in active_node_types and "memory" in active_node_types:
            strategy_name = "language_focused"
        elif "breath" in active_node_types and "attention" in active_node_types:
            strategy_name = "processing_flow"
            
        # Apply the selected strategy if different from current
        if strategy_name != self.current_strategy:
            self.apply_integration_strategy(strategy_name)
            
        return {
            "optimized_strategy": strategy_name,
            "active_node_types": list(active_node_types),
            "changes_made": strategy_name != self.current_strategy,
            "timestamp": time.time()
        }
    
    def shutdown(self) -> None:
        """Shutdown the system integrator"""
        logger.info("Shutting down System Integrator")
        # Nothing specific to cleanup for now
        # The node manager will handle shutting down the nodes 