"""
Configuration Manager for Lumina Frontend
========================================

This module provides configuration management functionality for the
Lumina Frontend system.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file. If not provided,
                        will use default location.
        """
        # Set default paths
        self.config_dir = Path.home() / ".lumina"
        self.default_config_path = self.config_dir / "config.json"
        
        # Use provided path or default
        self.config_path = Path(config_path) if config_path else self.default_config_path
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        # Return default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "ui": {
                "theme": "dark",
                "font_size": 12,
                "window_size": [1280, 720],
                "maximized": False
            },
            "logging": {
                "level": "INFO",
                "file": str(self.config_dir / "lumina.log"),
                "max_size": 10485760,  # 10MB
                "backup_count": 5
            },
            "visualization": {
                "update_interval": 100,  # ms
                "max_points": 1000,
                "color_scheme": "viridis"
            },
            "neural": {
                "model_path": str(self.config_dir / "models"),
                "cache_size": 1000,
                "batch_size": 32
            }
        }
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key in dot notation (e.g., "ui.theme")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value.
        
        Args:
            key: Configuration key in dot notation (e.g., "ui.theme")
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Save changes
        self.save_config() 