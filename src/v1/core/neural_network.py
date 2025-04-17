"""
Core Neural Network Implementation for Version 1
This module contains the main neural network class and its components.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        """
        Initialize a neural network with the given layer sizes.
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # He initialization
            weight_matrix = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            bias_vector = np.zeros((layer_sizes[i + 1], 1))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
            
        logger.info(f"Initialized neural network with layers: {layer_sizes}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.
        
        Args:
            x: Input data (numpy array)
            
        Returns:
            Network output
        """
        activation = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = self._relu(z)
        return activation
    
    def backward(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backpropagation to compute gradients.
        
        Args:
            x: Input data
            y: Target output
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        # Forward pass
        activations = [x]
        zs = []
        activation = x
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self._relu(z)
            activations.append(activation)
        
        # Backward pass
        delta = self._compute_output_delta(activations[-1], y)
        weight_gradients = []
        bias_gradients = []
        
        for i in reversed(range(len(self.weights))):
            weight_gradients.insert(0, np.dot(delta, activations[i].T))
            bias_gradients.insert(0, delta)
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self._relu_derivative(zs[i-1])
        
        return weight_gradients, bias_gradients
    
    def train(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Train the network on a single batch.
        
        Args:
            x: Input data
            y: Target output
            
        Returns:
            Loss value
        """
        weight_gradients, bias_gradients = self.backward(x, y)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
        
        # Compute and return loss
        output = self.forward(x)
        loss = self._compute_loss(output, y)
        return loss
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU activation function"""
        return (x > 0).astype(float)
    
    def _compute_output_delta(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute output layer delta"""
        return output - target
    
    def _compute_loss(self, output: np.ndarray, target: np.ndarray) -> float:
        """Compute mean squared error loss"""
        return np.mean((output - target) ** 2)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained network.
        
        Args:
            x: Input data
            
        Returns:
            Network predictions
        """
        return self.forward(x)
    
    def save_weights(self, filepath: str) -> None:
        """
        Save network weights and biases to a file.
        
        Args:
            filepath: Path to save the weights
        """
        np.savez(filepath, weights=self.weights, biases=self.biases)
        logger.info(f"Saved weights to {filepath}")
    
    def load_weights(self, filepath: str) -> None:
        """
        Load network weights and biases from a file.
        
        Args:
            filepath: Path to load the weights from
        """
        data = np.load(filepath)
        self.weights = data['weights']
        self.biases = data['biases']
        logger.info(f"Loaded weights from {filepath}") 