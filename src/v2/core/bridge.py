"""
Bridge Module for Version 2
This module provides functionality to migrate from v1 to v2, ensuring compatibility and smooth transition.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import importlib.util
import inspect
from .neural_network import NeuralNetwork, ActivationFunction, Optimizer, WeightInitialization

logger = logging.getLogger(__name__)

class V1ToV2Bridge:
    def __init__(self, v1_module_path: str):
        """
        Initialize the bridge between v1 and v2.
        
        Args:
            v1_module_path: Path to the v1 module to migrate
        """
        self.v1_module_path = Path(v1_module_path)
        self.v1_module = self._load_v1_module()
        self.migration_status: Dict[str, Any] = {}
        
        logger.info(f"Initialized V1ToV2Bridge for {v1_module_path}")
    
    def _load_v1_module(self) -> Any:
        """
        Load the v1 module dynamically.
        
        Returns:
            Loaded v1 module
        """
        try:
            spec = importlib.util.spec_from_file_location("v1_module", str(self.v1_module_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.error(f"Failed to load v1 module: {str(e)}")
            return None
    
    def migrate_neural_network(self, v1_network: Any) -> NeuralNetwork:
        """
        Migrate a v1 neural network to v2.
        
        Args:
            v1_network: v1 neural network instance
            
        Returns:
            New v2 neural network
        """
        # Extract v1 network parameters
        layer_sizes = v1_network.layer_sizes
        weights = v1_network.weights
        biases = v1_network.biases
        
        # Create v2 network with default settings
        v2_network = NeuralNetwork(
            layer_sizes=layer_sizes,
            activation=ActivationFunction.RELU,
            optimizer=Optimizer.ADAM,
            weight_init=WeightInitialization.HE
        )
        
        # Transfer weights and biases
        for i in range(len(weights)):
            v2_network.weights[i] = weights[i]
            v2_network.biases[i] = biases[i]
        
        logger.info(f"Migrated neural network with {len(layer_sizes)} layers")
        return v2_network
    
    def migrate_config(self, v1_config: Dict) -> Dict:
        """
        Migrate v1 configuration to v2 format.
        
        Args:
            v1_config: v1 configuration dictionary
            
        Returns:
            New v2 configuration dictionary
        """
        v2_config = {
            'network': {
                'layer_sizes': v1_config.get('layer_sizes', [10, 64, 32, 1]),
                'learning_rate': v1_config.get('learning_rate', 0.01),
                'activation': 'relu',
                'optimizer': 'adam',
                'weight_init': 'he',
                'dropout_rate': 0.0,
                'use_batch_norm': False
            },
            'training': {
                'batch_size': v1_config.get('batch_size', 32),
                'n_epochs': v1_config.get('n_epochs', 100),
                'validation_split': 0.2
            },
            'monitoring': {
                'metrics': ['loss', 'accuracy'],
                'log_interval': 10
            }
        }
        
        logger.info("Migrated configuration to v2 format")
        return v2_config
    
    def migrate_data(self, v1_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Migrate v1 data format to v2.
        
        Args:
            v1_data: v1 data dictionary
            
        Returns:
            New v2 data dictionary
        """
        v2_data = {
            'X_train': v1_data.get('X_train'),
            'y_train': v1_data.get('y_train'),
            'X_test': v1_data.get('X_test'),
            'y_test': v1_data.get('y_test')
        }
        
        logger.info("Migrated data to v2 format")
        return v2_data
    
    def check_compatibility(self) -> bool:
        """
        Check if the v1 module is compatible with v2.
        
        Returns:
            True if compatible, False otherwise
        """
        if not self.v1_module:
            return False
        
        required_attributes = ['layer_sizes', 'weights', 'biases', 'train']
        for attr in required_attributes:
            if not hasattr(self.v1_module, attr):
                logger.error(f"Missing required attribute: {attr}")
                return False
        
        return True
    
    def get_migration_status(self) -> Dict:
        """
        Get the current migration status.
        
        Returns:
            Dictionary containing migration status
        """
        return {
            'v1_module_path': str(self.v1_module_path),
            'compatibility_check': self.check_compatibility(),
            'migration_status': self.migration_status
        }

# Export functionality for node integration
functionality = {
    'classes': {
        'V1ToV2Bridge': V1ToV2Bridge
    },
    'description': 'Bridge module for migrating from v1 to v2'
} 