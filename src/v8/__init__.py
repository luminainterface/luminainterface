"""
Lumina V8 - Seed Dispersal System

This package implements the fruit-bearing and seed dispersal capabilities of the Lumina system.
Like fruits in nature that attract animals to spread seeds, this system creates attractive
interfaces and interaction points to spread neural patterns and knowledge.

Components:
- seed_dispersal_system.py: Main implementation with PySide6 interface
- spatial_temple_mapper.py: Spatial organization of knowledge patterns

The V8 system connects to the central seed.py and provides mechanisms for:
1. Knowledge packaging (fruits)
2. Pattern visualization
3. Interactive exchange
4. Cross-system pollination
"""

from .seed_dispersal_system import (
    KnowledgeFruit,
    ChatPanel,
    SeedDispersalWindow,
    run_dispersal_system
)

__version__ = "8.0.0"
__all__ = [
    'KnowledgeFruit',
    'ChatPanel',
    'SeedDispersalWindow',
    'run_dispersal_system'
] 