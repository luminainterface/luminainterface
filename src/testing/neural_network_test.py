"""
Neural Network Test Module

This module demonstrates how to test neural network components
using the Lumina testing framework, including model training,
inference performance, and layer-wise testing.
"""

import time
import numpy as np
import unittest
from typing import List, Dict, Any, Tuple, Optional

# Import our testing framework
from test_framework import (
    TestCase, PerformanceTestCase, IntegrationTestCase, RegressionTestCase,
    TestCategory, TestPriority, test_case, create_test_suite, run_tests
)

# Mock neural network classes for demonstration
class NeuralLayer:
    """A simple neural network layer"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        """Initialize a neural layer with random weights"""
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the layer"""
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        
        # Apply activation function
        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-self.z))
        elif self.activation == 'tanh':
            self.output = np.tanh(self.z)
        else:
            self.output = self.z  # Linear activation
            
        return self.output
    
    def backward(self, d_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """Backward pass through the layer"""
        # Compute derivative of activation function
        if self.activation == 'relu':
            d_z = d_output.copy()
            d_z[self.z <= 0] = 0
        elif self.activation == 'sigmoid':
            d_z = d_output * self.output * (1 - self.output)
        elif self.activation == 'tanh':
            d_z = d_output * (1 - np.power(self.output, 2))
        else:
            d_z = d_output  # Linear activation derivative is 1
        
        # Compute gradients
        d_weights = np.dot(self.inputs.T, d_z)
        d_biases = np.sum(d_z, axis=0, keepdims=True)
        d_inputs = np.dot(d_z, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_inputs


class SimpleNeuralNetwork:
    """A simple neural network model"""
    
    def __init__(self, layer_sizes: List[int], activations: List[str]):
        """Initialize a neural network with the given layer sizes"""
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                NeuralLayer(layer_sizes[i], layer_sizes[i+1], activations[i])
            )
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self, d_output: np.ndarray, learning_rate: float = 0.01) -> None:
        """Backward pass through the network"""
        current_gradient = d_output
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient, learning_rate)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              learning_rate: float = 0.01, batch_size: int = 32) -> List[float]:
        """Train the network on the given data"""
        num_samples = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Process mini-batches
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss (Mean Squared Error)
                loss = np.mean(np.square(y_pred - y_batch))
                epoch_loss += loss * len(X_batch) / num_samples
                
                # Compute gradient of loss with respect to output
                d_output = 2 * (y_pred - y_batch) / batch_size
                
                # Backward pass
                self.backward(d_output, learning_rate)
            
            losses.append(epoch_loss)
            
            # Early stopping (simplified)
            if epoch > 5 and abs(losses[-1] - losses[-2]) < 1e-6:
                break
                
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained network"""
        return self.forward(X)


# Helper function to generate synthetic data
def generate_synthetic_data(
    num_samples: int = 1000, 
    input_dim: int = 5,
    output_dim: int = 1,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for neural network testing"""
    # Create random input features
    X = np.random.randn(num_samples, input_dim)
    
    # Create target values with some non-linear relationship
    W = np.random.randn(input_dim, output_dim)
    y = np.dot(X, W) + np.sin(X[:, 0:1]) + np.random.randn(num_samples, output_dim) * noise_level
    
    return X, y


# Unit Tests for Neural Networks
class NeuralNetworkUnitTests(TestCase):
    """Unit tests for neural network components"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        # Create a small network for testing
        self.network = SimpleNeuralNetwork([3, 4, 1], ['relu', 'sigmoid'])
        # Generate some test data
        self.X, self.y = generate_synthetic_data(100, 3, 1)
    
    @test_case(category=TestCategory.UNIT, priority=TestPriority.CRITICAL, component="NeuralLayer")
    def test_layer_forward(self):
        """Test forward pass through a neural layer"""
        layer = NeuralLayer(3, 2, 'relu')
        inputs = np.random.randn(5, 3)  # 5 samples, 3 features
        
        # Test forward pass
        outputs = layer.forward(inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (5, 2))
        
        # Check ReLU activation (all values should be >= 0)
        self.assertTrue(np.all(outputs >= 0))
    
    @test_case(category=TestCategory.UNIT, priority=TestPriority.HIGH, component="NeuralNetwork")
    def test_network_forward(self):
        """Test forward pass through the entire network"""
        inputs = np.random.randn(10, 3)  # 10 samples, 3 features
        
        # Test forward pass
        outputs = self.network.forward(inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (10, 1))
        
        # Sigmoid output should be between 0 and 1
        self.assertTrue(np.all(outputs >= 0) and np.all(outputs <= 1))
    
    @test_case(category=TestCategory.UNIT, priority=TestPriority.MEDIUM, component="NeuralNetwork")
    def test_network_backward(self):
        """Test backward pass through the network"""
        # Forward pass to initialize layer states
        inputs = np.random.randn(5, 3)
        outputs = self.network.forward(inputs)
        
        # Create gradient for output
        d_outputs = np.random.randn(*outputs.shape)
        
        # Test backward pass doesn't raise exceptions
        try:
            self.network.backward(d_outputs, learning_rate=0.01)
            successful = True
        except Exception as e:
            successful = False
            
        self.assertTrue(successful, "Backward pass failed")


# Performance Tests for Neural Networks
class NeuralNetworkPerformanceTests(PerformanceTestCase):
    """Performance tests for neural network operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        # Create a network with more layers for performance testing
        self.network = SimpleNeuralNetwork([10, 20, 15, 5, 1], 
                                           ['relu', 'relu', 'relu', 'sigmoid'])
        
        # Generate test data
        self.X, self.y = generate_synthetic_data(1000, 10, 1)
        
        # Set performance thresholds
        self.set_thresholds({
            "training_time_ms": 5000,      # 5 seconds for training
            "inference_time_ms": 100,      # 100ms for inference
            "memory_delta_mb": 50.0,       # 50MB memory usage
            "model_size_mb": 1.0           # 1MB model size
        })
    
    @test_case(category=TestCategory.PERFORMANCE, priority=TestPriority.HIGH, component="NeuralNetwork")
    def test_training_performance(self):
        """Test the performance of model training"""
        # Create small batch for quick testing
        X_batch = self.X[:100]
        y_batch = self.y[:100]
        
        # Measure training performance
        metrics = self.measure_performance(
            self.network.train,
            X_batch, y_batch, epochs=10, learning_rate=0.01
        )
        
        # Assert performance meets thresholds
        self.assert_performance(metrics)
        
        # Additional specific assertions
        self.assertLess(metrics["execution_time_ms"], 1000, 
                        "Training took too long for a small batch")
    
    @test_case(category=TestCategory.PERFORMANCE, priority=TestPriority.CRITICAL, component="NeuralNetwork")
    def test_inference_performance(self):
        """Test the performance of model inference"""
        # Measure inference performance
        metrics = self.measure_performance(
            self.network.predict,
            self.X
        )
        
        # Assert performance meets thresholds
        self.assert_performance(metrics)
        
        # Additional specific assertions
        self.assertLess(metrics["execution_time_ms"], 50, 
                        "Inference is too slow for production use")


# Integration Tests for Neural Networks
class NeuralNetworkIntegrationTests(IntegrationTestCase):
    """Integration tests for neural networks with other components"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.network = SimpleNeuralNetwork([5, 8, 3, 1], 
                                           ['relu', 'relu', 'sigmoid'])
        self.X, self.y = generate_synthetic_data(200, 5, 1)
        
        # Check for dependencies (just for demonstration)
        try:
            self.require_dependency("numpy")
        except ImportError:
            self.skipTest("NumPy not available")
    
    @test_case(category=TestCategory.INTEGRATION, priority=TestPriority.MEDIUM, 
              component="NeuralNetwork", tags=["data_pipeline"])
    def test_network_with_data_pipeline(self):
        """Test neural network with a data pipeline"""
        # Mock data pipeline (in real code, this would be an actual component)
        def data_pipeline(X, y, batch_size=32):
            """Simple data pipeline that yields batches"""
            num_samples = X.shape[0]
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, num_samples, batch_size):
                yield X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
        
        # Test integration between network and data pipeline
        for X_batch, y_batch in data_pipeline(self.X, self.y):
            # Train for one batch
            self.network.train(X_batch, y_batch, epochs=1)
            
            # Make predictions
            predictions = self.network.predict(X_batch)
            
            # Validate predictions shape
            self.assertEqual(predictions.shape, y_batch.shape)
        
        # Check component health
        self.assertTrue(self.check_component_health("NeuralNetwork"))


# Regression Tests for Neural Networks
class NeuralNetworkRegressionTests(RegressionTestCase):
    """Regression tests for neural network behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        # Fixed random seed for reproducibility
        np.random.seed(42)
        
        # Create network with fixed initialization
        self.network = SimpleNeuralNetwork([3, 4, 1], ['relu', 'sigmoid'])
        
        # Fixed test data
        self.X = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        self.y = np.array([[0.2], [0.5], [0.8]])
        
        # Set up baseline data (in real tests, this would be loaded from a file)
        # For this example, we'll compute it once and save it
        predictions = self.network.predict(self.X)
        train_loss = np.mean(np.square(predictions - self.y))
        
        # Train for a fixed number of steps
        losses = self.network.train(self.X, self.y, epochs=100, learning_rate=0.1)
        
        # New predictions after training
        trained_predictions = self.network.predict(self.X)
        
        self.save_baseline({
            "initial_prediction_sample": predictions[0, 0],
            "initial_loss": train_loss,
            "final_loss": losses[-1],
            "trained_prediction_sample": trained_predictions[0, 0]
        })
    
    @test_case(category=TestCategory.REGRESSION, priority=TestPriority.CRITICAL, 
              component="NeuralNetwork", tags=["regression", "training"])
    def test_training_regression(self):
        """Test that model training behaves consistently"""
        # Reset random seed
        np.random.seed(42)
        
        # Create a new network with same initialization
        network = SimpleNeuralNetwork([3, 4, 1], ['relu', 'sigmoid'])
        
        # Get initial predictions and loss
        predictions = network.predict(self.X)
        initial_loss = np.mean(np.square(predictions - self.y))
        
        # Check that initial predictions match baseline
        self.assert_matches_baseline(
            predictions[0, 0], 
            "initial_prediction_sample",
            tolerance=1e-6
        )
        
        # Check that initial loss matches baseline
        self.assert_matches_baseline(
            initial_loss, 
            "initial_loss",
            tolerance=1e-6
        )
        
        # Train for fixed number of steps
        losses = network.train(self.X, self.y, epochs=100, learning_rate=0.1)
        
        # Check that final loss matches baseline
        self.assert_matches_baseline(
            losses[-1], 
            "final_loss",
            tolerance=1e-6
        )
        
        # Check that trained predictions match baseline
        trained_predictions = network.predict(self.X)
        self.assert_matches_baseline(
            trained_predictions[0, 0], 
            "trained_prediction_sample",
            tolerance=1e-6
        )


def run_neural_network_tests():
    """Run the neural network tests"""
    # Create a test suite using unittest's TestSuite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests([
        unittest.TestLoader().loadTestsFromTestCase(NeuralNetworkUnitTests),
        unittest.TestLoader().loadTestsFromTestCase(NeuralNetworkPerformanceTests),
        unittest.TestLoader().loadTestsFromTestCase(NeuralNetworkIntegrationTests),
        unittest.TestLoader().loadTestsFromTestCase(NeuralNetworkRegressionTests)
    ])
    
    # Run the tests with unittest's TextTestRunner
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    
    print("Tests completed!")

if __name__ == "__main__":
    run_neural_network_tests() 