"""
Node Implementation for Version 1
This module contains the HybridNode and CentralNode classes for the neural network system.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from ..core.neural_network import NeuralNetwork
from ..utils.data_processing import DataProcessor, ModelEvaluator

logger = logging.getLogger(__name__)

class HybridNode:
    def __init__(self, node_id: str, layer_sizes: List[int], learning_rate: float = 0.01):
        """
        Initialize a hybrid node that combines local processing with neural network capabilities.
        
        Args:
            node_id: Unique identifier for the node
            layer_sizes: List of integers representing the neural network architecture
            learning_rate: Learning rate for the neural network
        """
        self.node_id = node_id
        self.neural_network = NeuralNetwork(layer_sizes, learning_rate)
        self.data_processor = DataProcessor()
        self.model_evaluator = ModelEvaluator(self.neural_network)
        self.local_data: Dict[str, np.ndarray] = {}
        self.connected_nodes: List[str] = []
        self.is_training = False
        self.module_functionality: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized HybridNode {node_id} with architecture {layer_sizes}")
    
    def connect_to_node(self, node_id: str) -> None:
        """
        Connect this node to another node.
        
        Args:
            node_id: ID of the node to connect to
        """
        if node_id not in self.connected_nodes:
            self.connected_nodes.append(node_id)
            logger.info(f"Node {self.node_id} connected to node {node_id}")
    
    def store_local_data(self, data_id: str, X: np.ndarray, y: np.ndarray) -> None:
        """
        Store data locally in the node.
        
        Args:
            data_id: Identifier for the data
            X: Input features
            y: Target values
        """
        self.local_data[data_id] = (X, y)
        logger.info(f"Stored data {data_id} in node {self.node_id}")
    
    def train_local(self, data_id: str, n_epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        """
        Train the node's neural network on local data.
        
        Args:
            data_id: Identifier of the data to train on
            n_epochs: Number of training epochs
            batch_size: Size of training batches
            
        Returns:
            Dictionary of training metrics
        """
        if data_id not in self.local_data:
            raise ValueError(f"Data {data_id} not found in node {self.node_id}")
        
        X, y = self.local_data[data_id]
        (X_train, y_train), (X_test, y_test) = self.data_processor.preprocess_data(X, y)
        
        self.is_training = True
        training_metrics = []
        
        # Apply any module functionality before training
        self._apply_module_functionality('pre_train', X_train, y_train)
        
        for epoch in range(n_epochs):
            # Shuffle the training data
            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Train in mini-batches
            epoch_loss = 0
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i+batch_size].T
                y_batch = y_train_shuffled[i:i+batch_size].T
                
                # Apply any module functionality before batch training
                X_batch, y_batch = self._apply_module_functionality('pre_batch', X_batch, y_batch)
                
                loss = self.neural_network.train(X_batch, y_batch)
                epoch_loss += loss
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Node {self.node_id} - Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / (X_train.shape[0] / batch_size):.4f}")
                training_metrics.append({
                    'epoch': epoch + 1,
                    'loss': epoch_loss / (X_train.shape[0] / batch_size)
                })
        
        self.is_training = False
        
        # Apply any module functionality after training
        self._apply_module_functionality('post_train', X_test, y_test)
        
        # Evaluate on test set
        test_metrics = self.model_evaluator.evaluate(X_test.T, y_test.T)
        return {
            'training_metrics': training_metrics,
            'test_metrics': test_metrics
        }
    
    def get_weights(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get the current weights and biases of the node's neural network.
        
        Returns:
            Tuple of (weights, biases)
        """
        return self.neural_network.weights, self.neural_network.biases
    
    def set_weights(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        """
        Set the weights and biases of the node's neural network.
        
        Args:
            weights: List of weight matrices
            biases: List of bias vectors
        """
        self.neural_network.weights = weights
        self.neural_network.biases = biases
        logger.info(f"Updated weights in node {self.node_id}")
    
    def add_module_functionality(self, file_path: str, functionality: Optional[Dict[str, Any]] = None) -> None:
        """
        Add functionality from an infected module to this node.
        
        Args:
            file_path: Path to the infected module
            functionality: Optional pre-extracted functionality
        """
        if functionality is None:
            from ..infection.infection_module import InfectedModule, get_node_integration
            node_integration = get_node_integration()
            if node_integration:
                infected_module = InfectedModule(file_path, node_integration)
                functionality = infected_module.functionality
        
        if functionality:
            self.module_functionality[file_path] = functionality
            logger.info(f"Added functionality from {file_path} to node {self.node_id}")
    
    def _apply_module_functionality(self, hook_name: str, *args, **kwargs) -> Any:
        """
        Apply any registered module functionality at the specified hook point.
        
        Args:
            hook_name: Name of the hook point (e.g., 'pre_train', 'pre_batch', 'post_train')
            *args: Positional arguments to pass to the hook
            **kwargs: Keyword arguments to pass to the hook
            
        Returns:
            Result of applying the functionality, or original arguments if no functionality
        """
        result = args[0] if args else None
        
        for file_path, functionality in self.module_functionality.items():
            if 'functions' in functionality:
                hook_func = functionality['functions'].get(f'hook_{hook_name}')
                if hook_func:
                    try:
                        result = hook_func(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error applying hook {hook_name} from {file_path}: {str(e)}")
        
        return result

class CentralNode:
    def __init__(self, node_id: str = "central"):
        """
        Initialize the central node that coordinates the hybrid nodes.
        
        Args:
            node_id: Unique identifier for the central node
        """
        self.node_id = node_id
        self.hybrid_nodes: Dict[str, HybridNode] = {}
        self.global_model: Optional[NeuralNetwork] = None
        self.is_coordinating = False
        
        logger.info(f"Initialized CentralNode {node_id}")
    
    def register_hybrid_node(self, node: HybridNode) -> None:
        """
        Register a hybrid node with the central node.
        
        Args:
            node: HybridNode instance to register
        """
        if node.node_id in self.hybrid_nodes:
            raise ValueError(f"Node {node.node_id} already registered")
        
        self.hybrid_nodes[node.node_id] = node
        logger.info(f"Registered hybrid node {node.node_id} with central node")
    
    def coordinate_training(self, n_rounds: int = 10) -> Dict[str, List[float]]:
        """
        Coordinate training across all registered hybrid nodes.
        
        Args:
            n_rounds: Number of coordination rounds
            
        Returns:
            Dictionary of training metrics for each round
        """
        self.is_coordinating = True
        round_metrics = {}
        
        for round_num in range(n_rounds):
            logger.info(f"Starting coordination round {round_num + 1}/{n_rounds}")
            
            # Collect weights from all nodes
            all_weights = []
            all_biases = []
            
            for node_id, node in self.hybrid_nodes.items():
                weights, biases = node.get_weights()
                all_weights.append(weights)
                all_biases.append(biases)
            
            # Average the weights and biases
            avg_weights = [np.mean(weights, axis=0) for weights in zip(*all_weights)]
            avg_biases = [np.mean(biases, axis=0) for biases in zip(*all_biases)]
            
            # Distribute averaged weights back to nodes
            for node_id, node in self.hybrid_nodes.items():
                node.set_weights(avg_weights, avg_biases)
            
            # Train each node locally
            round_metrics[round_num] = {}
            for node_id, node in self.hybrid_nodes.items():
                metrics = node.train_local(f"round_{round_num}")
                round_metrics[round_num][node_id] = metrics
        
        self.is_coordinating = False
        return round_metrics
    
    def get_node_status(self) -> Dict[str, Dict]:
        """
        Get the status of all registered hybrid nodes.
        
        Returns:
            Dictionary containing status information for each node
        """
        status = {}
        for node_id, node in self.hybrid_nodes.items():
            status[node_id] = {
                'is_training': node.is_training,
                'connected_nodes': node.connected_nodes,
                'local_data_count': len(node.local_data),
                'module_count': len(node.module_functionality)
            }
        return status 