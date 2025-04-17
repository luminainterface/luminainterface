"""
V5 Integration Package

This package provides integration components between the Language Memory System
and the V5 Fractal Echo Visualization system.
"""

from pathlib import Path
import sys

# Make imports from parent directory work
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Try to import the visualization bridge
try:
    from src.v5_integration.visualization_bridge import get_visualization_bridge, VisualizationBridge
except ImportError:
    pass

__all__ = ['get_visualization_bridge', 'VisualizationBridge'] 