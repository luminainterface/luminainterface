"""
Lumina Neural Network Bridge System

This package provides bridging functionality between different versions
of the Lumina Neural Network system, enabling seamless integration,
data migration, and compatibility between versions.

Components:
- v8_v9_bridge: Bridge between v8 and v9 neural systems
  - Database migration between versions
  - Neural state conversion
  - Instance compatibility layers
"""

# Version information
__version__ = "1.0.0"
__author__ = "Lumina Neural Network Project"

# Core components
from .v8_v9_bridge import V8V9Bridge, import_v8_components, import_v9_components

# Migration utilities
from .v8_v9_bridge import migrate_database, migrate_neural_states 