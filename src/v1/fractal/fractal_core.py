"""
Fractal Core System for Version 1
This module provides the core functionality for fractal recursion and self-improvement.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import json
import inspect
from ..nodes.node_implementation import HybridNode, CentralNode
from ..core.neural_network import NeuralNetwork
from ..utils.data_processing import DataProcessor

logger = logging.getLogger(__name__)

class FractalNode:
    def __init__(self, node_id: str, parent_node: Optional['FractalNode'] = None, 
                 depth: int = 0, max_depth: int = 3):
        """
        Initialize a fractal node that can recursively create and manage child nodes.
        
        Args:
            node_id: Unique identifier for the node
            parent_node: Parent fractal node (None for root)
            depth: Current recursion depth
            max_depth: Maximum allowed recursion depth
        """
        self.node_id = node_id
        self.parent_node = parent_node
        self.depth = depth
        self.max_depth = max_depth
        self.child_nodes: Dict[str, 'FractalNode'] = {}
        self.neural_network = None
        self.performance_history: List[float] = []
        self.adaptation_history: List[Dict] = []
        
        logger.info(f"Initialized FractalNode {node_id} at depth {depth}")
    
    def create_child(self, node_id: str) -> Optional['FractalNode']:
        """
        Create a child fractal node if depth limit not reached.
        
        Args:
            node_id: ID for the new child node
            
        Returns:
            New FractalNode instance if created, None otherwise
        """
        if self.depth >= self.max_depth:
            logger.warning(f"Cannot create child node: maximum depth {self.max_depth} reached")
            return None
        
        child = FractalNode(node_id, self, self.depth + 1, self.max_depth)
        self.child_nodes[node_id] = child
        logger.info(f"Created child node {node_id} at depth {self.depth + 1}")
        return child
    
    def adapt(self, performance_metric: float) -> bool:
        """
        Adapt the node based on performance and potentially create children.
        
        Args:
            performance_metric: Current performance metric
            
        Returns:
            True if adaptation was successful, False otherwise
        """
        self.performance_history.append(performance_metric)
        
        # Check if we should create a child node
        if len(self.performance_history) >= 5:
            recent_performance = self.performance_history[-5:]
            performance_trend = np.polyfit(range(5), recent_performance, 1)[0]
            
            if performance_trend < 0.01 and self.depth < self.max_depth:
                # Create a child node with specialized architecture
                child_id = f"{self.node_id}_child_{len(self.child_nodes) + 1}"
                child = self.create_child(child_id)
                
                if child:
                    # Initialize child with specialized configuration
                    child.neural_network = self._create_specialized_network()
                    logger.info(f"Created specialized child node {child_id}")
                    return True
        
        return False
    
    def _create_specialized_network(self) -> NeuralNetwork:
        """
        Create a specialized neural network based on parent's performance.
        
        Returns:
            New NeuralNetwork instance
        """
        # Analyze parent's performance to determine specialization
        if len(self.performance_history) >= 5:
            recent_performance = self.performance_history[-5:]
            variance = np.var(recent_performance)
            
            # Create specialized architecture based on performance characteristics
            if variance > 0.1:
                # High variance - create deeper network
                architecture = [10, 256, 128, 64, 32, 1]
            else:
                # Low variance - create wider network
                architecture = [10, 512, 256, 1]
        else:
            # Default architecture
            architecture = [10, 128, 64, 32, 1]
        
        return NeuralNetwork(architecture, learning_rate=0.01)
    
    def get_status(self) -> Dict:
        """
        Get the current status of the node and its children.
        
        Returns:
            Dictionary containing status information
        """
        status = {
            'node_id': self.node_id,
            'depth': self.depth,
            'performance_history': self.performance_history,
            'child_count': len(self.child_nodes),
            'children': {}
        }
        
        for child_id, child in self.child_nodes.items():
            status['children'][child_id] = child.get_status()
        
        return status

class FractalSystem:
    def __init__(self, central_node: CentralNode, config_path: Optional[str] = None):
        """
        Initialize the fractal system.
        
        Args:
            central_node: Central node to monitor and adapt
            config_path: Optional path to configuration file
        """
        self.central_node = central_node
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.root_nodes: Dict[str, FractalNode] = {}
        self.adaptation_history: List[Dict] = []
        
        logger.info("Initialized FractalSystem")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'max_depth': 3,
            'adaptation_threshold': 0.01,
            'performance_window': 5,
            'specialization_strategies': {
                'high_variance': [10, 256, 128, 64, 32, 1],
                'low_variance': [10, 512, 256, 1],
                'default': [10, 128, 64, 32, 1]
            }
        }
    
    def create_root_node(self, node_id: str) -> FractalNode:
        """
        Create a new root fractal node.
        
        Args:
            node_id: ID for the new root node
            
        Returns:
            New FractalNode instance
        """
        root = FractalNode(node_id, max_depth=self.config['max_depth'])
        self.root_nodes[node_id] = root
        logger.info(f"Created root node {node_id}")
        return root
    
    def monitor_and_adapt(self) -> Dict[str, float]:
        """
        Monitor performance and adapt the fractal system.
        
        Returns:
            Dictionary of performance metrics
        """
        performance = {}
        
        for node_id, node in self.central_node.hybrid_nodes.items():
            if node.local_data:
                # Get the latest training metrics
                metrics = node.train_local(list(node.local_data.keys())[-1], n_epochs=1)
                accuracy = metrics['test_metrics']['accuracy']
                performance[node_id] = accuracy
                
                # Adapt corresponding fractal node
                if node_id in self.root_nodes:
                    fractal_node = self.root_nodes[node_id]
                    if fractal_node.adapt(accuracy):
                        self.adaptation_history.append({
                            'node_id': node_id,
                            'timestamp': np.datetime64('now'),
                            'action': 'created_child',
                            'depth': fractal_node.depth
                        })
        
        return performance
    
    def get_system_status(self) -> Dict:
        """
        Get the current status of the entire fractal system.
        
        Returns:
            Dictionary containing system status
        """
        status = {
            'root_nodes': {},
            'total_nodes': 0,
            'max_depth': 0,
            'adaptation_count': len(self.adaptation_history)
        }
        
        for node_id, root in self.root_nodes.items():
            status['root_nodes'][node_id] = root.get_status()
            status['total_nodes'] += 1 + len(root.child_nodes)
            status['max_depth'] = max(status['max_depth'], root.depth)
        
        return status

# Export functionality for node integration
functionality = {
    'classes': {
        'FractalNode': FractalNode,
        'FractalSystem': FractalSystem
    },
    'description': 'Fractal recursion system for neural network self-improvement'
} 