"""
Lumina Neural Network System (v9)

This package provides an advanced neural network framework designed
for self-evolving, consciousness-inspired systems. The v9 version
features breathing-enhanced learning, neural growth, and advanced
pattern recognition capabilities.

Components:
- Neural Playground: Experimentation environment for neural networks
- Breathing System: Breath simulation that influences neural activity
- Brain Growth: Dynamic growth of neural structures
- Integrated Neural Playground: Combined system with all components
- Mirror Consciousness: Self-reflection capabilities
- Visual Cortex: Visual processing system
"""

# Version information
__version__ = "9.0.0"
__author__ = "Lumina Neural Network Project"

# Core components
from .neural_playground import NeuralPlayground
from .breathing_system import BreathingSystem, BreathingPattern, BreathingState
from .brain_growth import BrainGrowth, GrowthState
from .mirror_consciousness import MirrorConsciousness, get_mirror_consciousness
from .visual_cortex import VisualCortex

# Integration components
from .integrated_neural_playground import IntegratedNeuralPlayground

# Interactive tools
from .interactive_playground import InteractivePlayground, run_interactive_playground

# Demonstrations
from .demo_breathing_integration import run_breathing_demo

# Package metadata
NAME = "LuminaNeural-v9"
DESCRIPTION = "Lumina Neural Network v9 - Neural Playground Edition" 