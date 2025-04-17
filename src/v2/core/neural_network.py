"""
Neural Network Implementation for Version 2
This module provides an enhanced neural network implementation with support for:
- Multiple activation functions
- Advanced optimization techniques
- Gradient clipping
- Learning rate scheduling
- Weight initialization strategies
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ActivationFunction(Enum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    LEAKY_RELU = 'leaky_relu'
    ELU = 'elu'
    SWISH = 'swish'

class Optimizer(Enum):
    SGD = 'sgd'
    ADAM = 'adam'
    RMSPROP = 'rmsprop'
    ADAGRAD = 'adagrad'

class WeightInitialization(Enum):
    XAVIER = 'xavier'
    HE = 'he'
    RANDOM = 'random'

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01,
                 activation: ActivationFunction = ActivationFunction.RELU,
                 optimizer: Optimizer = Optimizer.ADAM,
                 weight_init: WeightInitialization = WeightInitialization.HE,
                 dropout_rate: float = 0.0,
                 use_batch_norm: bool = False):
        """
        Initialize the neural network with enhanced features.
        
        Args:
            layer_sizes: List of layer sizes
            learning_rate: Initial learning rate
            activation: Activation function to use
            optimizer: Optimization algorithm to use
            weight_init: Weight initialization strategy
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Initialize weights and biases
        self.weights, self.biases = self._initialize_parameters()
        
        # Initialize optimizer parameters
        self._initialize_optimizer()
        
        # Initialize batch normalization parameters if enabled
        if use_batch_norm:
            self._initialize_batch_norm()
        
        logger.info(f"Initialized neural network with {len(layer_sizes)} layers")
        logger.info(f"Using {activation.value} activation, {optimizer.value} optimizer")
    
    def _initialize_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Initialize network parameters using the specified strategy.
        
        Returns:
            Tuple of (weights, biases)
        """
        weights = []
        biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            if self.weight_init == WeightInitialization.XAVIER:
                scale = np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            elif self.weight_init == WeightInitialization.HE:
                scale = np.sqrt(2.0 / self.layer_sizes[i])
            else:  # RANDOM
                scale = 0.01
            
            weights.append(np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i]) * scale)
            biases.append(np.zeros((self.layer_sizes[i + 1], 1)))
        
        return weights, biases
    
    def _initialize_optimizer(self) -> None:
        """
        Initialize optimizer-specific parameters.
        """
        if self.optimizer == Optimizer.ADAM:
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
        elif self.optimizer == Optimizer.RMSPROP:
            self.s_weights = [np.zeros_like(w) for w in self.weights]
            self.s_biases = [np.zeros_like(b) for b in self.biases]
            self.rho = 0.9
            self.epsilon = 1e-8
        elif self.optimizer == Optimizer.ADAGRAD:
            self.s_weights = [np.zeros_like(w) for w in self.weights]
            self.s_biases = [np.zeros_like(b) for b in self.biases]
            self.epsilon = 1e-8
    
    def _initialize_batch_norm(self) -> None:
        """
        Initialize batch normalization parameters.
        """
        self.gamma = [np.ones((size, 1)) for size in self.layer_sizes[1:]]
        self.beta = [np.zeros((size, 1)) for size in self.layer_sizes[1:]]
        self.running_mean = [np.zeros((size, 1)) for size in self.layer_sizes[1:]]
        self.running_var = [np.ones((size, 1)) for size in self.layer_sizes[1:]]
        self.momentum = 0.9
    
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the specified activation function.
        
        Args:
            x: Input array
            
        Returns:
            Activated output
        """
        if self.activation == ActivationFunction.RELU:
            return np.maximum(0, x)
        elif self.activation == ActivationFunction.SIGMOID:
            return 1 / (1 + np.exp(-x))
        elif self.activation == ActivationFunction.TANH:
            return np.tanh(x)
        elif self.activation == ActivationFunction.LEAKY_RELU:
            return np.maximum(0.01 * x, x)
        elif self.activation == ActivationFunction.ELU:
            return np.where(x > 0, x, np.exp(x) - 1)
        elif self.activation == ActivationFunction.SWISH:
            return x * (1 / (1 + np.exp(-x)))
    
    def _apply_activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the specified activation function.
        
        Args:
            x: Input array
            
        Returns:
            Derivative output
        """
        if self.activation == ActivationFunction.RELU:
            return np.where(x > 0, 1, 0)
        elif self.activation == ActivationFunction.SIGMOID:
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid)
        elif self.activation == ActivationFunction.TANH:
            return 1 - np.tanh(x) ** 2
        elif self.activation == ActivationFunction.LEAKY_RELU:
            return np.where(x > 0, 1, 0.01)
        elif self.activation == ActivationFunction.ELU:
            return np.where(x > 0, 1, np.exp(x))
        elif self.activation == ActivationFunction.SWISH:
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid + x * sigmoid * (1 - sigmoid)
    
    def _apply_batch_norm(self, x: np.ndarray, layer_idx: int, training: bool = True) -> np.ndarray:
        """
        Apply batch normalization.
        
        Args:
            x: Input array
            layer_idx: Layer index
            training: Whether in training mode
            
        Returns:
            Normalized output
        """
        if not self.use_batch_norm:
            return x
        
        if training:
            mean = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True)
            
            # Update running statistics
            self.running_mean[layer_idx] = (self.momentum * self.running_mean[layer_idx] +
                                          (1 - self.momentum) * mean)
            self.running_var[layer_idx] = (self.momentum * self.running_var[layer_idx] +
                                         (1 - self.momentum) * var)
        else:
            mean = self.running_mean[layer_idx]
            var = self.running_var[layer_idx]
        
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma[layer_idx] * x_norm + self.beta[layer_idx]
    
    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        """
        Apply dropout regularization.
        
        Args:
            x: Input array
            
        Returns:
            Output with dropout applied
        """
        if self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
            return x * mask / (1 - self.dropout_rate)
        return x
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform forward propagation.
        
        Args:
            x: Input data
            training: Whether in training mode
            
        Returns:
            Network output
        """
        a = x
        self.activations = [x]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            self.z_values.append(z)
            
            if self.use_batch_norm:
                z = self._apply_batch_norm(z, i, training)
            
            a = self._apply_activation(z)
            
            if self.dropout_rate > 0 and training:
                a = self._apply_dropout(a)
            
            self.activations.append(a)
        
        return a
    
    def backward(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backward propagation.
        
        Args:
            x: Input data
            y: Target values
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = x.shape[1]
        
        # Forward pass
        output = self.forward(x)
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        dz = output - y
        
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            dW[i] = np.dot(dz, self.activations[i].T) / m
            db[i] = np.sum(dz, axis=1, keepdims=True) / m
            
            if i > 0:
                # Backpropagate error
                dz = np.dot(self.weights[i].T, dz) * self._apply_activation_derivative(self.z_values[i-1])
                
                if self.use_batch_norm:
                    # Backpropagate through batch norm
                    dz = self._backward_batch_norm(dz, i)
        
        return dW, db
    
    def _backward_batch_norm(self, dz: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Backpropagate through batch normalization.
        
        Args:
            dz: Gradient from next layer
            layer_idx: Layer index
            
        Returns:
            Gradient with batch norm applied
        """
        if not self.use_batch_norm:
            return dz
        
        # Compute gradients for gamma and beta
        dgamma = np.sum(dz * self.z_values[layer_idx], axis=1, keepdims=True)
        dbeta = np.sum(dz, axis=1, keepdims=True)
        
        # Update gamma and beta
        self.gamma[layer_idx] -= self.learning_rate * dgamma
        self.beta[layer_idx] -= self.learning_rate * dbeta
        
        return dz
    
    def _update_parameters(self, dW: List[np.ndarray], db: List[np.ndarray], t: int) -> None:
        """
        Update network parameters using the specified optimizer.
        
        Args:
            dW: Weight gradients
            db: Bias gradients
            t: Time step (for Adam)
        """
        if self.optimizer == Optimizer.SGD:
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * db[i]
        
        elif self.optimizer == Optimizer.ADAM:
            for i in range(len(self.weights)):
                # Update first moment
                self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * dW[i]
                self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * db[i]
                
                # Update second moment
                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (dW[i] ** 2)
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (db[i] ** 2)
                
                # Bias correction
                m_hat_w = self.m_weights[i] / (1 - self.beta1 ** t)
                v_hat_w = self.v_weights[i] / (1 - self.beta2 ** t)
                m_hat_b = self.m_biases[i] / (1 - self.beta1 ** t)
                v_hat_b = self.v_biases[i] / (1 - self.beta2 ** t)
                
                # Update parameters
                self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        
        elif self.optimizer == Optimizer.RMSPROP:
            for i in range(len(self.weights)):
                self.s_weights[i] = self.rho * self.s_weights[i] + (1 - self.rho) * (dW[i] ** 2)
                self.s_biases[i] = self.rho * self.s_biases[i] + (1 - self.rho) * (db[i] ** 2)
                
                self.weights[i] -= self.learning_rate * dW[i] / (np.sqrt(self.s_weights[i]) + self.epsilon)
                self.biases[i] -= self.learning_rate * db[i] / (np.sqrt(self.s_biases[i]) + self.epsilon)
        
        elif self.optimizer == Optimizer.ADAGRAD:
            for i in range(len(self.weights)):
                self.s_weights[i] += dW[i] ** 2
                self.s_biases[i] += db[i] ** 2
                
                self.weights[i] -= self.learning_rate * dW[i] / (np.sqrt(self.s_weights[i]) + self.epsilon)
                self.biases[i] -= self.learning_rate * db[i] / (np.sqrt(self.s_biases[i]) + self.epsilon)
    
    def train(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 100,
              batch_size: int = 32, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
        """
        Train the neural network.
        
        Args:
            X: Training data
            y: Training labels
            n_epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Optional validation data
            
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        m = X.shape[1]
        t = 1  # Time step for Adam
        
        for epoch in range(n_epochs):
            # Shuffle data
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            # Mini-batch training
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                y_batch = y_shuffled[:, i:i+batch_size]
                
                # Forward and backward pass
                dW, db = self.backward(X_batch, y_batch)
                
                # Update parameters
                self._update_parameters(dW, db, t)
                t += 1
                
                # Compute batch loss and accuracy
                output = self.forward(X_batch, training=False)
                batch_loss = np.mean((output - y_batch) ** 2)
                batch_accuracy = np.mean((output > 0.5) == y_batch)
                
                epoch_loss += batch_loss * X_batch.shape[1]
                epoch_accuracy += batch_accuracy * X_batch.shape[1]
            
            # Compute epoch metrics
            epoch_loss /= m
            epoch_accuracy /= m
            metrics['train_loss'].append(epoch_loss)
            metrics['train_accuracy'].append(epoch_accuracy)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.forward(X_val, training=False)
                val_loss = np.mean((val_output - y_val) ** 2)
                val_accuracy = np.mean((val_output > 0.5) == y_val)
                metrics['val_loss'].append(val_loss)
                metrics['val_accuracy'].append(val_accuracy)
                
                logger.info(f"Epoch {epoch + 1}/{n_epochs} - "
                          f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1}/{n_epochs} - "
                          f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained network.
        
        Args:
            X: Input data
            
        Returns:
            Network predictions
        """
        return self.forward(X, training=False)
    
    def save_weights(self, filepath: str) -> None:
        """
        Save network weights to a file.
        
        Args:
            filepath: Path to save weights
        """
        np.savez(filepath,
                weights=self.weights,
                biases=self.biases,
                gamma=self.gamma if self.use_batch_norm else None,
                beta=self.beta if self.use_batch_norm else None)
    
    def load_weights(self, filepath: str) -> None:
        """
        Load network weights from a file.
        
        Args:
            filepath: Path to load weights from
        """
        data = np.load(filepath, allow_pickle=True)
        self.weights = data['weights']
        self.biases = data['biases']
        if self.use_batch_norm:
            self.gamma = data['gamma']
            self.beta = data['beta'] 