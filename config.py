#!/usr/bin/env python
"""
Configuration module for Lumina Neural Network system

Contains configuration settings and environment variables for the application.
"""

import os
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
DB_DIR = ROOT_DIR / "db"
LOG_DIR = ROOT_DIR / "logs"
MODEL_OUTPUT_DIR = ROOT_DIR / "model_output"
TRAINING_DATA_DIR = ROOT_DIR / "training_data"
ICONS_DIR = ROOT_DIR / "icons"

# Ensure directories exist
for directory in [DATA_DIR, DB_DIR, LOG_DIR, MODEL_OUTPUT_DIR, TRAINING_DATA_DIR]:
    directory.mkdir(exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "embedding_dim": 256,
        "hidden_size": 512,
        "num_layers": 2,
        "dropout": 0.1,
        "learning_rate": 0.001
    },
    "training": {
        "batch_size": 32,
        "epochs": 10,
        "validation_split": 0.2,
        "early_stopping_patience": 3
    },
    "system": {
        "log_level": "INFO",
        "save_checkpoints": True,
        "checkpoint_interval": 5,
        "gpu_enabled": True
    },
    "ui": {
        "theme": "dark",
        "font_size": 12,
        "default_width": 1200,
        "default_height": 800,
        "refresh_rate": 60
    }
}

# Memory system settings
MEMORY_SETTINGS = {
    "memories_dir": str(DATA_DIR / "memories"),
    "max_session_memories": 100,
    "default_emotion": "neutral",
    "enabled": True
}

# Language system settings
LANGUAGE_SETTINGS = {
    "vocabulary_file": str(DATA_DIR / "vocabulary.json"),
    "embeddings_dir": str(MODEL_OUTPUT_DIR / "embeddings"),
    "tokenization_level": "subword",  # Options: 'char', 'word', 'subword'
    "embedding_dim": 300,
    "context_window": 5,
    "max_vocab_size": 10000
}

# Neural network settings
NEURAL_SETTINGS = {
    "model_dir": str(MODEL_OUTPUT_DIR / "models"),
    "hidden_dim": 512,
    "num_layers": 4,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
}

# Component paths
COMPONENT_PATHS = {
    "nodes": [
        "RSEN_node.py",
        "hybrid_node.py",
        "node_zero.py",
        "portal_node.py",
        "wormhole_node.py",
        "zpe_node.py",
        "neutrino_node.py",
        "game_theory_node.py",
        "consciousness_node.py",
        "gauge_theory_node.py",
        "fractal_nodes.py",
        "infinite_minds_node.py",
        "void_infinity_node.py"
    ],
    "processors": [
        "neural_processor.py",
        "language_processor.py",
        "lumina_processor.py",
        "mood_processor.py",
        "node_manager.py",
        "wiki_learner.py",
        "wiki_vocabulary.py",
        "wikipedia_training_module.py",
        "wikipedia_trainer.py",
        "lumina_neural.py",
        "physics_engine.py",
        "calculus_engine.py",
        "physics_metaphysics_framework.py",
        "hyperdimensional_thought.py",
        "quantum_infection.py",
        "node_integration.py"
    ],
    "trainers": [
        "english_language_trainer.py",
        "knowledge_source.py",
        "internal_language.py"
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOG_DIR / "system.log"),
    "console_level": logging.INFO
}

# UI settings
UI_SETTINGS = {
    "theme": "dark",  # Options: 'light', 'dark', 'gold_black'
    "memory_visualization_enabled": True,
    "glyph_panel_enabled": True,
    "breath_tracker_enabled": True,
    "node_mode_menu_enabled": True,
    "command_interface_enabled": True,
    "debug_mode": False
}

# Node mode settings
NODE_MODES = {
    "resonance": {
        "description": "Active listening mode",
        "color": "#3498db",
        "animation": "pulse",
        "default": True
    },
    "fractal": {
        "description": "Deep recursion mode",
        "color": "#9b59b6",
        "animation": "spiral",
        "default": False
    },
    "mirror": {
        "description": "Contradiction glitch mode",
        "color": "#e74c3c",
        "animation": "flicker",
        "default": False
    },
    "echo": {
        "description": "Past memory field mode",
        "color": "#2ecc71",
        "animation": "ripple",
        "default": False
    },
    "glyph": {
        "description": "Symbolic activation mode",
        "color": "#f1c40f",
        "animation": "glow",
        "default": False
    }
}

# System status
SYSTEM_VERSION = "3.0"
DEBUG_MODE = os.environ.get("LUMINA_DEBUG", "False").lower() in ("true", "1", "yes")

# Export all configuration as a dictionary
CONFIG = {
    "paths": {
        "root": str(ROOT_DIR),
        "data": str(DATA_DIR),
        "training_data": str(TRAINING_DATA_DIR),
        "model_output": str(MODEL_OUTPUT_DIR),
        "logs": str(LOG_DIR)
    },
    "memory": MEMORY_SETTINGS,
    "language": LANGUAGE_SETTINGS,
    "neural": NEURAL_SETTINGS,
    "components": COMPONENT_PATHS,
    "logging": LOGGING_CONFIG,
    "ui": UI_SETTINGS,
    "node_modes": NODE_MODES,
    "version": SYSTEM_VERSION,
    "debug": DEBUG_MODE
}

# Singleton Config class implementation
class Config:
    """
    Singleton configuration class that provides access to system configuration
    settings from anywhere in the application.
    """
    _instance = None
    
    def __init__(self):
        """Private constructor - use get_instance() instead"""
        if Config._instance is not None:
            raise Exception("This is a singleton class. Use Config.get_instance() instead.")
        
        self.config = DEFAULT_CONFIG
        self.user_config = {}
        
        # Load any existing configuration
        self._load_config()
        
        # Add additional sections from the module-level configuration
        self.config.update({
            "memory": MEMORY_SETTINGS,
            "language": LANGUAGE_SETTINGS,
            "neural": NEURAL_SETTINGS,
            "components": COMPONENT_PATHS,
            "ui": UI_SETTINGS,
            "node_modes": NODE_MODES,
            "system": {
                "version": SYSTEM_VERSION,
                "debug": DEBUG_MODE,
                **self.config.get("system", {})
            }
        })
        
        logger.info("Config singleton initialized")
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the Config class"""
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance
    
    def _load_config(self, config_path="lumina_config.json"):
        """Load configuration from file"""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    self.user_config = user_config
                    
                    # Update default config with user config
                    for section, values in user_config.items():
                        if section in self.config and isinstance(self.config[section], dict):
                            self.config[section].update(values)
                        else:
                            self.config[section] = values
                            
                    logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_config(self, config_path="lumina_config.json"):
        """Save current configuration to file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.user_config, f, indent=2)
            logger.info(f"Saved configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_model_config(self):
        """Get the model configuration section"""
        return self.config.get("model", {})
    
    def get_training_config(self):
        """Get the training configuration section"""
        return self.config.get("training", {})
    
    def get_system_config(self):
        """Get the system configuration section"""
        return self.config.get("system", {})
    
    def get_ui_config(self):
        """Get the UI configuration section"""
        return self.config.get("ui", {})
    
    def get_memory_config(self):
        """Get the memory configuration section"""
        return self.config.get("memory", {})
    
    def get_language_config(self):
        """Get the language configuration section"""
        return self.config.get("language", {})
    
    def get_config_section(self, section_name):
        """Get a specific configuration section"""
        return self.config.get(section_name, {})
    
    def update_config_section(self, section_name, values):
        """Update a specific configuration section"""
        if section_name not in self.config:
            self.config[section_name] = {}
            
        self.config[section_name].update(values)
        
        # Also update user config for persistence
        if section_name not in self.user_config:
            self.user_config[section_name] = {}
            
        self.user_config[section_name].update(values)
        
        return True

def load_config(config_path: str = "lumina_config.json") -> dict:
    """Load configuration from file or use default"""
    try:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
    
    logger.info("Using default configuration")
    return DEFAULT_CONFIG

def save_config(config: dict, config_path: str = "lumina_config.json") -> bool:
    """Save configuration to file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

# Load active configuration
ACTIVE_CONFIG = load_config()

# Environment variables (with defaults)
LOG_LEVEL = os.environ.get("LUMINA_LOG_LEVEL", ACTIVE_CONFIG["system"]["log_level"])
GPU_ENABLED = os.environ.get("LUMINA_GPU_ENABLED", str(ACTIVE_CONFIG["system"]["gpu_enabled"])).lower() in ("true", "1", "yes")
UI_THEME = os.environ.get("LUMINA_UI_THEME", ACTIVE_CONFIG["ui"]["theme"])

# Configure log level based on configuration
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL))

if DEBUG_MODE:
    logger.info("Debug mode enabled")
    
if __name__ == "__main__":
    # Display the current configuration when run directly
    print("Current Configuration:")
    print(json.dumps(ACTIVE_CONFIG, indent=2))
    print("\nDirectories:")
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Output Directory: {MODEL_OUTPUT_DIR}")
    print(f"Training Data Directory: {TRAINING_DATA_DIR}")
    
    print("\nMemory System Settings:")
    print(json.dumps(MEMORY_SETTINGS, indent=2))
    print("\nLanguage System Settings:")
    print(json.dumps(LANGUAGE_SETTINGS, indent=2))
    print("\nNeural Network Settings:")
    print(json.dumps(NEURAL_SETTINGS, indent=2))
    print("\nComponent Paths:")
    print(json.dumps(COMPONENT_PATHS, indent=2))
    print("\nLogging Configuration:")
    print(json.dumps(LOGGING_CONFIG, indent=2))
    print("\nUI Settings:")
    print(json.dumps(UI_SETTINGS, indent=2))
    print("\nNode Modes:")
    print(json.dumps(NODE_MODES, indent=2))
    print("\nSystem Version:")
    print(SYSTEM_VERSION)
    print("\nConfiguration:")
    print(json.dumps(CONFIG, indent=2)) 