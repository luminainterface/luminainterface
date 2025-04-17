"""
Version 1 of the Neural Network System
This module contains the core components of the first version of our neural network system.
"""

__version__ = "1.0.0"
__author__ = "Neural Network Project Team"

# Import core components
from .core.neural_network import NeuralNetwork
from .core.spiderweb_bridge import SpiderwebBridge
from .core.mirror_superposition import MirrorSuperposition
from .core.fractal_bridge import FractalBridge
from .core.infection_module import InfectionModule
from .utils import DataProcessor, ModelEvaluator

# Initialize logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['NeuralNetwork', 'SpiderwebBridge', 'MirrorSuperposition', 'FractalBridge', 'InfectionModule'] 