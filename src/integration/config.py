"""
Configuration Module

Centralizes all configuration settings for the backend system.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "signals.db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Database configuration
DATABASE_CONFIG = {
    "path": str(DB_PATH),
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 1800
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "fmt": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "backend.log"),
            "formatter": "standard"
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO"
        }
    }
}

# Logic Gate configuration
LOGIC_GATE_CONFIG = {
    "types": {
        "AND": {"color": "orange", "path_type": "LITERAL"},
        "OR": {"color": "blue", "path_type": "SEMANTIC"},
        "XOR": {"color": "purple", "path_type": "HYBRID"},
        "NOT": {"color": "red", "path_type": "HYBRID"},
        "NAND": {"color": "yellow", "path_type": "LITERAL"},
        "NOR": {"color": "green", "path_type": "SEMANTIC"}
    },
    "creation_interval": 1.0,
    "max_gates": 5,
    "connection_probability": 0.5,
    "state_threshold": 0.8
}

# Ping System configuration
PING_CONFIG = {
    "ping_interval": 0.1,
    "timeout": 1.0,
    "max_retries": 2,
    "health_threshold": 0.7,
    "sync_window": 10,
    "batch_size": 16,
    "adaptive_timing": True,
    "min_interval": 0.05,
    "max_interval": 1.0,
    "allow_all_data": True,
    "data_sorting": True,
    "self_writing": True
}

# AutoWiki configuration
AUTOWIKI_CONFIG = {
    "startup": {
        "mode": "automatic",
        "priority": "high",
        "dependencies": ["neural_seed", "version_bridge"],
        "initialization": {
            "timeout": 20,
            "retry_attempts": 3,
            "retry_delay": 5
        }
    },
    "operation": {
        "mode": "background",
        "visibility": "hidden",
        "persistence": True,
        "monitoring_interval": 200
    },
    "resources": {
        "cpu_priority": "normal",
        "memory_limit": "1GB",
        "thread_count": 2
    },
    "learning": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "validation_split": 0.2,
        "early_stopping_patience": 3
    }
}

# Machine Learning configuration
ML_CONFIG = {
    "model": {
        "type": "transformer",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "dropout": 0.1
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 0.0001,
        "epochs": 100,
        "warmup_steps": 1000,
        "gradient_clip": 1.0
    },
    "optimization": {
        "mixed_precision": True,
        "distributed_training": True,
        "cache_dir": str(DATA_DIR / "model_cache")
    }
}

# Monitoring configuration
MONITORING_CONFIG = {
    "metrics_port": 9090,
    "metrics_path": "/metrics",
    "collection_interval": 15,
    "export_interval": 60,
    "retention_days": 30
}

# Security configuration
SECURITY_CONFIG = {
    "encryption_key": os.getenv("ENCRYPTION_KEY", "default_key"),
    "token_expiration": 3600,
    "max_login_attempts": 3,
    "password_min_length": 12,
    "require_2fa": False
}

# Backend connection settings
BACKEND_CONFIG: Dict[str, Any] = {
    'host': 'localhost',
    'port': 8000,
    'ping_interval': 1.0,  # seconds
    'reconnect_interval': 5.0,  # seconds
    'timeout': 10.0,  # seconds
}

# Metrics collection settings
METRICS_CONFIG: Dict[str, Any] = {
    'update_interval': 1.0,  # seconds
    'history_length': 100,  # number of samples
    'thresholds': {
        'cpu_usage': 80.0,  # percentage
        'memory_usage': 80.0,  # percentage
        'network_traffic': 1000.0,  # MB/s
        'disk_io': 100.0,  # MB/s
    }
}

# Health monitoring settings
HEALTH_CONFIG: Dict[str, Any] = {
    'check_interval': 5.0,  # seconds
    'components': [
        'ml_model',
        'monitoring',
        'autowiki',
        'database'
    ],
    'thresholds': {
        'response_time': 1.0,  # seconds
        'error_rate': 0.1,  # percentage
    }
}

# Signal processing settings
SIGNAL_CONFIG: Dict[str, Any] = {
    'buffer_size': 1000,
    'processing_interval': 0.1,  # seconds
    'max_retries': 3,
    'retry_delay': 1.0,  # seconds
}

def get_full_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary."""
    return {
        "database": DATABASE_CONFIG,
        "logging": LOGGING_CONFIG,
        "logic_gates": LOGIC_GATE_CONFIG,
        "ping": PING_CONFIG,
        "autowiki": AUTOWIKI_CONFIG,
        "ml": ML_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "security": SECURITY_CONFIG,
        "backend": BACKEND_CONFIG,
        "metrics": METRICS_CONFIG,
        "health": HEALTH_CONFIG,
        "signal": SIGNAL_CONFIG
    }

def update_config(section: str, updates: Dict[str, Any]) -> None:
    """Update configuration settings for a specific section."""
    config = globals().get(f"{section.upper()}_CONFIG")
    if config is None:
        raise ValueError(f"Unknown configuration section: {section}")
    
    config.update(updates) 