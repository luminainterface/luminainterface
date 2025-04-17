import logging
from typing import Dict, Any

class Config:
    """Configuration singleton for the neural network system"""
    _instance = None
    
    def __init__(self):
        if Config._instance is not None:
            raise Exception("Config is a singleton class")
            
        self.logger = logging.getLogger(__name__)
        self.logger.info("Using default configuration")
        
        # Default configuration
        self.config = {
            'model': {
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100
            },
            'system': {
                'log_level': 'INFO',
                'data_dir': 'data',
                'model_dir': 'models'
            }
        }
        
        Config._instance = self
        self.logger.info("Config singleton initialized")
        
    @staticmethod
    def get_instance():
        """Get or create the singleton instance"""
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']
        
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config['training']
        
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.config['system']
        
    def update_config(self, section: str, key: str, value: Any):
        """Update a configuration value"""
        if section not in self.config:
            raise KeyError(f"Configuration section {section} not found")
        if key not in self.config[section]:
            raise KeyError(f"Configuration key {key} not found in section {section}")
            
        self.config[section][key] = value
        self.logger.info(f"Updated config: {section}.{key} = {value}")
        
    def get_config(self) -> Dict[str, Any]:
        """Get the entire configuration"""
        return self.config 
 
 
from typing import Dict, Any

class Config:
    """Configuration singleton for the neural network system"""
    _instance = None
    
    def __init__(self):
        if Config._instance is not None:
            raise Exception("Config is a singleton class")
            
        self.logger = logging.getLogger(__name__)
        self.logger.info("Using default configuration")
        
        # Default configuration
        self.config = {
            'model': {
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100
            },
            'system': {
                'log_level': 'INFO',
                'data_dir': 'data',
                'model_dir': 'models'
            }
        }
        
        Config._instance = self
        self.logger.info("Config singleton initialized")
        
    @staticmethod
    def get_instance():
        """Get or create the singleton instance"""
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']
        
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config['training']
        
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.config['system']
        
    def update_config(self, section: str, key: str, value: Any):
        """Update a configuration value"""
        if section not in self.config:
            raise KeyError(f"Configuration section {section} not found")
        if key not in self.config[section]:
            raise KeyError(f"Configuration key {key} not found in section {section}")
            
        self.config[section][key] = value
        self.logger.info(f"Updated config: {section}.{key} = {value}")
        
    def get_config(self) -> Dict[str, Any]:
        """Get the entire configuration"""
        return self.config 
 