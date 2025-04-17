"""
Configuration for the V5 Fractal Echo Visualization System.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# V5 specific paths
V5_DATA_DIR = DATA_DIR / "v5"
V5_DATA_DIR.mkdir(exist_ok=True)

# Plugin configuration
PLUGIN_CONFIG = {
    "neural_state": {
        "enabled": True,
        "update_interval_ms": 50,
        "cache_size": 1000
    },
    "pattern_processor": {
        "enabled": True,
        "fractal_depth": 5,
        "pattern_cache_size": 500
    },
    "consciousness_analytics": {
        "enabled": True,
        "metrics_update_interval_ms": 100,
        "history_length": 1000
    },
    "api_service": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8765,
        "debug": False
    }
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "fractal_pattern": {
        "max_depth": 8,
        "color_scheme": "neural",
        "animation_speed": 1.0
    },
    "node_consciousness": {
        "visualization_mode": "real_time",
        "node_size_scale": 1.0,
        "edge_thickness_scale": 1.0
    },
    "network_display": {
        "layout": "force_directed",
        "node_spacing": 100,
        "edge_length": 150
    }
}

# UI settings
UI_CONFIG = {
    "theme": "dark",
    "window_size": (1920, 1080),
    "refresh_rate": 60,
    "enable_animations": True
}

# Memory integration settings
MEMORY_CONFIG = {
    "storage_path": str(V5_DATA_DIR / "memory"),
    "max_cache_size_mb": 512,
    "auto_save_interval_sec": 300
}

# Language integration settings
LANGUAGE_CONFIG = {
    "synthesis_batch_size": 32,
    "max_context_length": 2048,
    "enable_streaming": True
}

# Socket configuration
SOCKET_CONFIG = {
    "max_message_size": 1024 * 1024,  # 1MB
    "heartbeat_interval": 30,
    "reconnect_attempts": 3
}

# Plugin paths
PLUGIN_PATHS = {
    "neural_state": "neural_state_plugin.py",
    "pattern_processor": "pattern_processor_plugin.py",
    "consciousness_analytics": "consciousness_analytics_plugin.py",
    "api_service": "api_service_plugin.py"
}

# Export all configurations
__all__ = [
    'PLUGIN_CONFIG',
    'VISUALIZATION_CONFIG',
    'UI_CONFIG',
    'MEMORY_CONFIG',
    'LANGUAGE_CONFIG',
    'SOCKET_CONFIG',
    'PLUGIN_PATHS'
] 