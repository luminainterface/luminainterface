"""
LUMINA V7.5 UI Module
This module provides UI components for the LUMINA V7.5 system,
including the holographic frontend and visualization tools.
"""

# Import basic visualization components
from pathlib import Path
import sys
import os
import importlib.util

# Define the UI package
__all__ = [
    'holographic_frontend',
    'consciousness_node',
    'metrics_display',
    'neural_visualizer',
    'hologram_effects',
    'ui_manager'
]

# Dynamically import any available modules from v7 UI if needed
def import_from_v7(module_name):
    """Import a module from v7 UI if it exists and create a reference in v7_5"""
    try:
        # Check if we can find the module in v7
        v7_module_path = Path(__file__).parent.parent.parent / 'v7' / 'ui' / f"{module_name}.py"
        if v7_module_path.exists():
            spec = importlib.util.spec_from_file_location(f"src.v7_5.ui.{module_name}", v7_module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return None
    except Exception as e:
        print(f"Could not import {module_name} from v7: {e}")
        return None 