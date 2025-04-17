"""
Seed package for the Lumina Neural Network system.
Contains components for seeding, growing, and evolving neural patterns.
"""

from .neural_seed import NeuralSeed
from .seed_engine import SeedEngine

def get_neural_seed() -> NeuralSeed:
    """Get a new instance of NeuralSeed"""
    return NeuralSeed()

__all__ = ['NeuralSeed', 'SeedEngine', 'get_neural_seed'] 