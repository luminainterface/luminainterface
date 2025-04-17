#!/usr/bin/env python3
"""
Filtering Utilities

This module provides common filtering functions for use across
the neural network system.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global configuration
_config = None

def load_config():
    """Load filtering configuration"""
    global _config
    
    config_path = os.path.join(project_root, "data", "filter_config.json")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                _config = json.load(f)
        else:
            # Use default configuration
            _config = {
                "default_thresholds": {
                    "pattern_confidence": 0.7,
                    "association_strength": 0.5,
                    "concept_importance": 0.6,
                    "learning_value": 0.5
                },
                "enabled_filters": {
                    "pattern_filtering": True,
                    "association_filtering": True,
                    "concept_filtering": True,
                    "memory_filtering": True
                }
            }
    except Exception as e:
        logger.error(f"Error loading filtering configuration: {e}")
        # Use a minimal default configuration
        _config = {
            "default_thresholds": {
                "pattern_confidence": 0.7
            },
            "enabled_filters": {
                "pattern_filtering": True
            }
        }

def get_config():
    """Get the filtering configuration"""
    global _config
    
    if _config is None:
        load_config()
        
    return _config

def filter_patterns_by_confidence(patterns: List[Dict], threshold: Optional[float] = None) -> List[Dict]:
    """
    Filter patterns by confidence score
    
    Args:
        patterns: List of pattern dictionaries
        threshold: Confidence threshold, uses default if None
        
    Returns:
        List[Dict]: Filtered patterns
    """
    config = get_config()
    
    if threshold is None:
        threshold = config["default_thresholds"]["pattern_confidence"]
    
    # Check if filtering is enabled
    if not config["enabled_filters"].get("pattern_filtering", True):
        return patterns
    
    # Apply filtering
    return [p for p in patterns if p.get("confidence", 0) >= threshold]

def filter_associations_by_strength(associations: Dict[str, float], threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Filter associations by strength
    
    Args:
        associations: Dictionary of associations and strengths
        threshold: Strength threshold, uses default if None
        
    Returns:
        Dict[str, float]: Filtered associations
    """
    config = get_config()
    
    if threshold is None:
        threshold = config["default_thresholds"]["association_strength"]
    
    # Check if filtering is enabled
    if not config["enabled_filters"].get("association_filtering", True):
        return associations
    
    # Apply filtering
    return {word: strength for word, strength in associations.items() if strength >= threshold}

def filter_concepts_by_importance(concepts: Dict[str, Dict], threshold: Optional[float] = None) -> Dict[str, Dict]:
    """
    Filter concepts by importance
    
    Args:
        concepts: Dictionary of concepts
        threshold: Importance threshold, uses default if None
        
    Returns:
        Dict[str, Dict]: Filtered concepts
    """
    config = get_config()
    
    if threshold is None:
        threshold = config["default_thresholds"]["concept_importance"]
    
    # Check if filtering is enabled
    if not config["enabled_filters"].get("concept_filtering", True):
        return concepts
    
    # Apply filtering
    return {
        name: data for name, data in concepts.items() 
        if data.get("importance", 0) >= threshold
    }

def apply_custom_filter(items: List[Any], filter_func: Callable[[Any], bool]) -> List[Any]:
    """
    Apply a custom filter function to a list of items
    
    Args:
        items: List of items to filter
        filter_func: Filter function returning True for items to keep
        
    Returns:
        List[Any]: Filtered items
    """
    return [item for item in items if filter_func(item)]

def save_config(config: Dict[str, Any]) -> bool:
    """
    Save filtering configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    global _config
    
    config_path = os.path.join(project_root, "data", "filter_config.json")
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update global configuration
        _config = config
        
        return True
    except Exception as e:
        logger.error(f"Error saving filtering configuration: {e}")
        return False

# Initialize configuration
load_config()
