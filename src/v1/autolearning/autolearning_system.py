"""
AutoLearning System for Version 1
This module provides automatic learning and adaptation capabilities for the neural network system.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from ..nodes.node_implementation import HybridNode, CentralNode
from ..core.neural_network import NeuralNetwork
from ..utils.data_processing import DataProcessor

logger = logging.getLogger(__name__)

class AutoLearningSystem:
    def __init__(self, central_node: CentralNode, config_path: Optional[str] = None):
        """
        Initialize the auto-learning system.
        
        Args:
            central_node: Central node to monitor and adapt
            config_path: Optional path to configuration file
        """
        self.central_node = central_node
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.performance_history: Dict[str, List[float]] = {}
        self.adaptation_history: List[Dict] = []
        
        logger.info("Initialized AutoLearningSystem")
    
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
            'learning_rate_range': [0.001, 0.1],
            'batch_size_range': [16, 128],
            'architecture_options': [
                [10, 64, 32, 1],
                [10, 128, 64, 32, 1],
                [10, 256, 128, 64, 32, 1]
            ],
            'adaptation_threshold': 0.05,
            'max_adaptations': 10,
            'performance_window': 5
        }
    
    def monitor_performance(self) -> Dict[str, float]:
        """
        Monitor performance of all nodes.
        
        Returns:
            Dictionary of performance metrics for each node
        """
        performance = {}
        for node_id, node in self.central_node.hybrid_nodes.items():
            if node.local_data:
                # Get the latest training metrics
                metrics = node.train_local(list(node.local_data.keys())[-1], n_epochs=1)
                accuracy = metrics['test_metrics']['accuracy']
                performance[node_id] = accuracy
                
                # Update performance history
                if node_id not in self.performance_history:
                    self.performance_history[node_id] = []
                self.performance_history[node_id].append(accuracy)
                
                logger.info(f"Node {node_id} performance: {accuracy:.4f}")
        
        return performance
    
    def should_adapt(self, node_id: str) -> bool:
        """
        Determine if a node should be adapted.
        
        Args:
            node_id: ID of the node to check
            
        Returns:
            True if adaptation is needed, False otherwise
        """
        if node_id not in self.performance_history:
            return False
        
        history = self.performance_history[node_id]
        if len(history) < self.config['performance_window']:
            return False
        
        # Calculate performance trend
        recent_performance = history[-self.config['performance_window']:]
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        return performance_trend < self.config['adaptation_threshold']
    
    def adapt_node(self, node_id: str) -> bool:
        """
        Adapt a node's configuration.
        
        Args:
            node_id: ID of the node to adapt
            
        Returns:
            True if adaptation was successful, False otherwise
        """
        node = self.central_node.hybrid_nodes.get(node_id)
        if not node:
            return False
        
        # Generate new configuration
        new_config = self._generate_new_config(node)
        
        # Apply new configuration
        self._apply_config(node, new_config)
        
        # Record adaptation
        self.adaptation_history.append({
            'node_id': node_id,
            'timestamp': np.datetime64('now'),
            'config': new_config
        })
        
        logger.info(f"Adapted node {node_id} with new configuration")
        return True
    
    def _generate_new_config(self, node: HybridNode) -> Dict:
        """
        Generate a new configuration for a node.
        
        Args:
            node: Node to generate configuration for
            
        Returns:
            New configuration dictionary
        """
        # Randomly select new architecture
        architecture = np.random.choice(self.config['architecture_options'])
        
        # Randomly select new learning rate
        lr_min, lr_max = self.config['learning_rate_range']
        learning_rate = np.random.uniform(lr_min, lr_max)
        
        # Randomly select new batch size
        bs_min, bs_max = self.config['batch_size_range']
        batch_size = int(np.random.uniform(bs_min, bs_max))
        
        return {
            'architecture': architecture,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
    
    def _apply_config(self, node: HybridNode, config: Dict) -> None:
        """
        Apply new configuration to a node.
        
        Args:
            node: Node to apply configuration to
            config: Configuration to apply
        """
        # Create new neural network with new architecture
        new_network = NeuralNetwork(config['architecture'], config['learning_rate'])
        
        # Transfer weights if possible
        if len(node.neural_network.weights) == len(new_network.weights):
            for i in range(len(node.neural_network.weights)):
                if node.neural_network.weights[i].shape == new_network.weights[i].shape:
                    new_network.weights[i] = node.neural_network.weights[i]
                    new_network.biases[i] = node.neural_network.biases[i]
        
        # Update node's neural network
        node.neural_network = new_network
        node.batch_size = config['batch_size']
    
    def run(self, n_iterations: int = 100) -> Dict:
        """
        Run the auto-learning system for a number of iterations.
        
        Args:
            n_iterations: Number of iterations to run
            
        Returns:
            Dictionary of final performance metrics
        """
        logger.info(f"Starting auto-learning system for {n_iterations} iterations")
        
        for iteration in range(n_iterations):
            logger.info(f"Iteration {iteration + 1}/{n_iterations}")
            
            # Monitor performance
            performance = self.monitor_performance()
            
            # Adapt nodes if needed
            for node_id in self.central_node.hybrid_nodes:
                if self.should_adapt(node_id):
                    self.adapt_node(node_id)
            
            # Coordinate training
            self.central_node.coordinate_training(n_rounds=1)
        
        return self.monitor_performance()

# Export functionality for node integration
functionality = {
    'classes': {
        'AutoLearningSystem': AutoLearningSystem
    },
    'description': 'Auto-learning system for neural network adaptation'
} 