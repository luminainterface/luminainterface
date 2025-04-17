"""
Logging Setup for Lumina Frontend
================================

This module provides logging configuration for the Lumina Frontend system.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

def setup_logging(config) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        config: Configuration manager instance
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("lumina")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Get logging configuration
    log_level = getattr(logging, config.get("logging.level", "INFO"))
    log_file = Path(config.get("logging.file", "lumina.log"))
    max_size = config.get("logging.max_size", 10485760)  # 10MB
    backup_count = config.get("logging.backup_count", 5)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 