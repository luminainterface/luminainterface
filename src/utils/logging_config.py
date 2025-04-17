"""
Centralized Logging Configuration for Lumina Neural Network Project

This module provides a unified logging configuration for all components
of the system, including log rotation, formatting, and level management.
"""

import os
import logging
import logging.handlers
from pathlib import Path
import json
from datetime import datetime
from pythonjsonlogger import jsonlogger

class LuminaLogger:
    """Centralized logging configuration for Lumina system"""
    
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.loggers = {}
        self._setup_logging()
    
    def _load_config(self, config_path):
        """Load logging configuration from file or use defaults"""
        default_config = {
            "log_level": "INFO",
            "log_dir": "logs",
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "date_format": "%Y-%m-%d %H:%M:%S"
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return {**default_config, **config}
            except Exception as e:
                print(f"Error loading logging config: {e}")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create log directory if it doesn't exist
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config["log_level"])
        
        # Clear any existing handlers
        root_logger.handlers = []
        
        # Create formatters
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create and configure file handler
        log_file = log_dir / f"lumina_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(
            log_file,
            mode='a',
            encoding='utf-8'
        )
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(self.config["log_level"])
        
        # Create and configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(self.config["log_level"])
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Set up specific loggers
        loggers = [
            'lumina',
            'memory',
            'neural_network',
            'version_bridge',
            'visualization'
        ]
        
        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(self.config["log_level"])
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
    
    def get_logger(self, name):
        """Get or create a logger with the given name"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        return self.loggers[name]
    
    def set_level(self, level):
        """Set logging level for all loggers"""
        for logger in self.loggers.values():
            logger.setLevel(level)
    
    def add_handler(self, handler):
        """Add a handler to all loggers"""
        for logger in self.loggers.values():
            logger.addHandler(handler)

# Create global instance
lumina_logger = LuminaLogger()

def get_logger(name):
    """Get a logger instance for the given name"""
    return lumina_logger.get_logger(name) 