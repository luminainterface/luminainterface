"""
Example Module for Infection
This module demonstrates how to create a module that can be infected and integrated with the node system.
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def hook_pre_train(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hook function called before training starts.
    Adds random noise to the training data.
    
    Args:
        X: Input features
        y: Target values
        
    Returns:
        Modified (X, y) tuple
    """
    logger.info("Applying pre-train hook: Adding noise to training data")
    noise = np.random.normal(0, 0.01, X.shape)
    return X + noise, y

def hook_pre_batch(X_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hook function called before each mini-batch training.
    Normalizes the batch data.
    
    Args:
        X_batch: Batch input features
        y_batch: Batch target values
        
    Returns:
        Modified (X_batch, y_batch) tuple
    """
    logger.info("Applying pre-batch hook: Normalizing batch data")
    X_normalized = (X_batch - np.mean(X_batch, axis=0)) / (np.std(X_batch, axis=0) + 1e-8)
    return X_normalized, y_batch

def hook_post_train(X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hook function called after training completes.
    Logs test data statistics.
    
    Args:
        X_test: Test input features
        y_test: Test target values
        
    Returns:
        Original (X_test, y_test) tuple
    """
    logger.info(f"Applying post-train hook: Test data shape - {X_test.shape}")
    logger.info(f"Test target distribution: {np.bincount(y_test.astype(int).flatten())}")
    return X_test, y_test

# Export the functionality
functionality = {
    'functions': {
        'hook_pre_train': hook_pre_train,
        'hook_pre_batch': hook_pre_batch,
        'hook_post_train': hook_post_train
    },
    'description': 'Example module with data preprocessing hooks'
} 