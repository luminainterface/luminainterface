"""
Example script demonstrating the usage of the Version 1 neural network implementation.
"""

import numpy as np
from core.neural_network import NeuralNetwork
from utils.data_processing import DataProcessor, ModelEvaluator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples: int = 1000, n_features: int = 10) -> tuple:
    """
    Generate sample data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        
    Returns:
        Tuple of (X, y) where X is the input data and y is the target
    """
    # Generate random input data
    X = np.random.randn(n_samples, n_features)
    
    # Generate target values (binary classification)
    # Using a simple rule: if sum of features > 0, class 1, else class 0
    y = (np.sum(X, axis=1) > 0).astype(int).reshape(-1, 1)
    
    return X, y

def main():
    # Generate sample data
    X, y = generate_sample_data()
    logger.info(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    
    # Preprocess the data
    processor = DataProcessor()
    (X_train, y_train), (X_test, y_test) = processor.preprocess_data(X, y)
    
    # Initialize the neural network
    layer_sizes = [X.shape[1], 64, 32, 1]  # Input -> Hidden -> Hidden -> Output
    model = NeuralNetwork(layer_sizes, learning_rate=0.01)
    
    # Train the model
    n_epochs = 100
    batch_size = 32
    
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
            loss = model.train(X_batch, y_batch)
            epoch_loss += loss
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {epoch_loss / (X_train.shape[0] / batch_size):.4f}")
    
    # Evaluate the model
    evaluator = ModelEvaluator(model)
    test_metrics = evaluator.evaluate(X_test.T, y_test.T)
    
    # Perform cross-validation
    cv_metrics = evaluator.cross_validate(X.T, y.T)
    
    # Save the trained model
    model.save_weights("trained_model_weights.npz")
    logger.info("Saved trained model weights")

if __name__ == "__main__":
    main() 