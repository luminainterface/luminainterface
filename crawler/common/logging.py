"""Logging configuration for the crawler service."""
import logging
import sys
from typing import Optional, Union

from .config import LOG_LEVEL, LOG_FORMAT

def setup_logging(name: Optional[str] = None, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        name: Optional logger name. If None, returns the root logger.
        level: Optional logging level. If None, uses LOG_LEVEL from config.
        
    Returns:
        A configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    log_level = level if level is not None else getattr(logging, LOG_LEVEL.upper())
    logger.setLevel(log_level)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: The name of the logger.
        
    Returns:
        A configured logger instance.
    """
    return setup_logging(name) 