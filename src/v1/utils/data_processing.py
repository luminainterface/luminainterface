"""
Utility classes for data processing and model evaluation in Version 1
"""

import numpy as np
from typing import Tuple, List, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the data processor.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Preprocess the data by scaling and splitting into train/test sets.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Tuple of (X_train, y_train), (X_test, y_test)
        """
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"Preprocessed data: {X.shape[0]} samples split into {X_train.shape[0]} train and {X_test.shape[0]} test samples")
        return (X_train, y_train), (X_test, y_test)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using the fitted scaler.
        
        Args:
            X: Input features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming data")
        return self.scaler.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            X: Scaled features to transform back
            
        Returns:
            Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming data")
        return self.scaler.inverse_transform(X)

class ModelEvaluator:
    def __init__(self, model):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained neural network model
        """
        self.model = model
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model on the given data.
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Convert predictions to binary (0 or 1) for classification metrics
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred_binary),
            'precision': precision_score(y, y_pred_binary, average='binary'),
            'recall': recall_score(y, y_pred_binary, average='binary'),
            'f1': f1_score(y, y_pred_binary, average='binary'),
            'mse': np.mean((y_pred - y) ** 2)  # Mean squared error
        }
        
        logger.info("Model evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Input features
            y: True labels
            n_splits: Number of cross-validation folds
            
        Returns:
            Dictionary of average evaluation metrics across folds
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        metrics_list = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train the model on this fold
            self.model.train(X_train, y_train)
            
            # Evaluate on test set
            metrics = self.evaluate(X_test, y_test)
            metrics_list.append(metrics)
        
        # Calculate average metrics across folds
        avg_metrics = {
            metric: np.mean([m[metric] for m in metrics_list])
            for metric in metrics_list[0].keys()
        }
        
        logger.info("Cross-validation results:")
        for metric, value in avg_metrics.items():
            logger.info(f"Average {metric}: {value:.4f}")
            
        return avg_metrics 