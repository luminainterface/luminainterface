#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUMINA V7 Dashboard Configuration Generator
===========================================

Creates default configuration files for the LUMINA V7 Dashboard.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConfigGenerator")

# Default configuration files
DEFAULT_CONFIG = {
    "dashboard_config.json": {
        "dashboard": {
            "version": "7.0.0.3",
            "title": "LUMINA V7 Unified Dashboard",
            "refresh_rate": 1000,
            "max_history": 3600,
            "mock_mode": {
                "enabled": True,
                "randomization_factor": 0.2,
                "data_generation_mode": "realistic"
            }
        },
        "connection": {
            "v7_host": "localhost",
            "v7_port": 5678,
            "reconnect_attempts": 3,
            "reconnect_delay": 5000,
            "socket_timeout": 10000,
            "auto_reconnect": True,
            "fallback_to_mock": True
        },
        "persistence": {
            "storage_type": "sqlite",
            "db_path": "data/neural_metrics.db",
            "retention_days": 30,
            "auto_cleanup": True,
            "backup_enabled": False,
            "backup_interval_hours": 24,
            "backup_location": "data/backups/"
        },
        "ui": {
            "color_theme": "dark",
            "font_family": "Segoe UI",
            "layout_type": "tabbed",
            "startup_panel": "overview",
            "window_size": {
                "width": 1200,
                "height": 800,
                "min_width": 800,
                "min_height": 600
            },
            "custom_colors": {
                "neural_activity": "#3498db",
                "language_processing": "#2ecc71",
                "system_metrics": "#e74c3c",
                "dream_patterns": "#9b59b6"
            },
            "plot_styles": {
                "line_width": 1.5,
                "symbol_size": 5,
                "grid_alpha": 0.3
            },
            "fonts": {
                "title": "Segoe UI, 12pt, bold",
                "axis": "Segoe UI, 10pt",
                "legend": "Segoe UI, 9pt"
            }
        },
        "panels": {
            "overview": {
                "enabled": True,
                "position": "top-left",
                "refresh_rate": 5000,
                "auto_refresh": True
            },
            "neural_activity": {
                "enabled": True,
                "position": "top-right",
                "refresh_rate": 1000,
                "auto_refresh": True,
                "visualization": {
                    "renderer": "pyqtgraph",
                    "fallback_renderer": "matplotlib",
                    "display_mode": "realtime",
                    "max_points": 1000,
                    "auto_scale": True
                },
                "metrics": {
                    "learning_rate": True,
                    "connection_strength": True,
                    "pattern_formation": True,
                    "consciousness_level": True
                }
            },
            "language_processing": {
                "enabled": True,
                "position": "bottom-left",
                "refresh_rate": 2000,
                "auto_refresh": True,
                "visualization": {
                    "renderer": "pyqtgraph",
                    "fallback_renderer": "matplotlib",
                    "display_mode": "realtime",
                    "max_points": 500
                },
                "metrics": {
                    "semantic_understanding": True,
                    "vocabulary_growth": True,
                    "conversation_flow": True,
                    "integration_level": True
                }
            },
            "system_metrics": {
                "enabled": True,
                "position": "bottom-right",
                "refresh_rate": 2000,
                "auto_refresh": True,
                "visualization": {
                    "renderer": "pyqtgraph",
                    "fallback_renderer": "custom",
                    "display_mode": "realtime",
                    "time_window": 60,
                    "max_points": 300
                },
                "metrics": {
                    "cpu_usage": True,
                    "memory_usage": True,
                    "gpu_usage": True,
                    "disk_usage": True
                },
                "thresholds": {
                    "cpu_warning": 70,
                    "cpu_critical": 90,
                    "memory_warning": 70,
                    "memory_critical": 90,
                    "gpu_warning": 70,
                    "gpu_critical": 90
                }
            }
        },
        "neural_parameters": {
            "nn_weight": 0.6,
            "llm_weight": 0.7,
            "creativity_factor": 0.5,
            "randomness": 0.2,
            "learning_rate": 0.01,
            "decay_rate": 0.001,
            "activation_threshold": 0.3
        },
        "api": {
            "enabled": False,
            "port": 5679,
            "host": "localhost",
            "rate_limit": 100,
            "authentication": {
                "required": True,
                "method": "token",
                "tokens": ["dashboard_api_token_12345"],
                "expire_days": 30
            },
            "cors": {
                "enabled": True,
                "allowed_origins": ["http://localhost:3000"]
            }
        },
        "logging": {
            "level": "INFO",
            "file_path": "logs/dashboard.log",
            "max_size_mb": 10,
            "backup_count": 5,
            "log_startup": True,
            "log_shutdown": True,
            "log_connections": True,
            "log_errors": True
        },
        "advanced": {
            "gpu_acceleration": True,
            "multi_threading": True,
            "thread_count": 4,
            "memory_limit_mb": 512,
            "development_mode": False,
            "debug_mode": False
        }
    },
    
    "visualization_config.json": {
        "global": {
            "preferred_backend": "pyqtgraph",
            "fallback_backend": "matplotlib",
            "last_resort_backend": "custom",
            "use_gpu_acceleration": True,
            "anti_aliasing": True,
            "render_quality": "high"
        },
        "themes": {
            "dark": {
                "background": "#2d3436",
                "foreground": "#ecf0f1",
                "axis": "#95a5a6",
                "grid": "#7f8c8d",
                "accent": "#3498db",
                "plot_colors": [
                    "#3498db", "#2ecc71", "#e74c3c", "#9b59b6", 
                    "#f1c40f", "#1abc9c", "#e67e22", "#34495e"
                ]
            },
            "light": {
                "background": "#ecf0f1",
                "foreground": "#2d3436",
                "axis": "#7f8c8d",
                "grid": "#bdc3c7",
                "accent": "#2980b9",
                "plot_colors": [
                    "#2980b9", "#27ae60", "#c0392b", "#8e44ad", 
                    "#f39c12", "#16a085", "#d35400", "#2c3e50"
                ]
            }
        },
        "pyqtgraph": {
            "antialias": True,
            "background": "themed",
            "foreground": "themed",
            "line_width": 1.5,
            "symbol_size": 5,
            "advanced": {
                "gl_enabled": True,
                "decimation_threshold": 1000,
                "downsampling_method": "peak-detection"
            }
        },
        "matplotlib": {
            "style": "themed",
            "dpi": 100,
            "figure_inches": [6, 4]
        },
        "panel_specific": {
            "system_metrics": {
                "cpu_color": "#e74c3c",
                "memory_color": "#3498db",
                "gpu_color": "#f39c12",
                "disk_color": "#2ecc71"
            }
        }
    },
    
    "panel_plugins.json": {
        "enabled": True,
        "auto_discover": True,
        "discovery_paths": [
            "src/visualization/panels",
            "plugins/dashboard"
        ],
        "plugins": [
            {
                "name": "NeuralActivityPanel",
                "module": "src.visualization.panels.neural_activity_panel",
                "class": "NeuralActivityPanel",
                "enabled": True,
                "tab_order": 0,
                "tab_title": "Neural Activity"
            },
            {
                "name": "LanguageProcessingPanel",
                "module": "src.visualization.panels.language_processing_panel",
                "class": "LanguageProcessingPanel", 
                "enabled": True,
                "tab_order": 1,
                "tab_title": "Language"
            },
            {
                "name": "SystemMetricsPanel",
                "module": "src.visualization.panels.system_metrics_panel",
                "class": "SystemMetricsPanel",
                "enabled": True,
                "tab_order": 2,
                "tab_title": "System"
            }
        ],
        "compatibility": {
            "minimum_version": "7.0.0.0",
            "maximum_version": "7.9.9.9",
            "check_compatibility": True
        }
    },
    
    "auth_config.json": {
        "auth_enabled": False,
        "auth_method": "basic",
        "session_timeout_minutes": 60,
        "users": [
            {
                "username": "admin",
                "password_hash": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8", 
                "role": "admin"
            },
            {
                "username": "viewer",
                "password_hash": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3", 
                "role": "viewer"
            }
        ],
        "roles": {
            "admin": {
                "permissions": ["read:all", "write:all", "configure:all"]
            },
            "viewer": {
                "permissions": ["read:all"]
            }
        }
    }
}

def create_config_files(config_dir, force=False):
    """
    Create default configuration files
    
    Args:
        config_dir: Directory to create the config files in
        force: Whether to overwrite existing files
    """
    # Create config directory if it doesn't exist
    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)
    
    # Create each config file
    for filename, config in DEFAULT_CONFIG.items():
        file_path = config_path / filename
        
        if file_path.exists() and not force:
            logger.info(f"Config file {filename} already exists, skipping...")
            continue
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created config file: {file_path}")
        except Exception as e:
            logger.error(f"Error creating config file {filename}: {e}")
            continue

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LUMINA V7 Dashboard Configuration Generator")
    
    parser.add_argument(
        "--config-dir", 
        dest="config_dir",
        default="config",
        help="Directory to create configuration files in"
    )
    
    parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Overwrite existing configuration files"
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    logger.info(f"Generating configuration files in {args.config_dir}")
    create_config_files(args.config_dir, args.force)
    
    logger.info(f"Configuration file generation complete.")
    if args.force:
        logger.info("Existing configuration files were overwritten.")
    
    # Create directories if they don't exist
    directories = [
        "data",
        "data/neural_metrics",
        "data/backups",
        "logs",
        "output/exports"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

if __name__ == "__main__":
    main() 