"""
V7 Node Consciousness Package

This package implements the V7 Node Consciousness system for the Lumina Neural Network,
providing self-awareness capabilities for individual nodes, advanced knowledge representation,
and a sophisticated learning system that enables autonomous knowledge acquisition and organization.

Key features:
- Self-aware processing units with personality and communication capabilities
- Advanced graph-based knowledge organization
- Autonomous learning pathway management
- AutoWiki learning system for self-directed knowledge acquisition
- Monday integration with enhanced emotional intelligence
- Breath detection system with LLM/NN weight integration
- Dream Mode system for memory consolidation and pattern synthesis
- Enhanced Language Integration with Mistral AI models
"""

__version__ = "7.0.0.2"

# Core nodes
from .node_consciousness import (
    LanguageConsciousnessNode,
    get_language_consciousness_node
)

# Node consciousness manager
from .node_consciousness_manager import (
    NodeConsciousnessManager,
    create_node_consciousness_manager
)

# Bridge components
from .v6_v7_bridge import V6V7Bridge
from .v6_v7_connector import V6V7Connector

# Breath systems
from .breath_detector import (
    BreathDetector,
    BreathPhase,
    BreathPattern
)
from .breath_contradiction_bridge import BreathContradictionBridge

# Node integration
from .v7_integration import V7Integration
from .v7_v6_integration import main as v7_v6_integration_main

# Dream Mode
from .lumina_v7.core.dream_controller import (
    DreamController,
    get_dream_controller,
    DreamState
)
from .lumina_v7.core.memory_consolidator import MemoryConsolidator
from .lumina_v7.core.pattern_synthesizer import PatternSynthesizer
from .lumina_v7.core.dream_archive import DreamArchive
from .lumina_v7.core.dream_integration import (
    DreamIntegration,
    get_dream_integration
)

# Language Integration
from .enhanced_language_integration import EnhancedLanguageIntegration
from .mistral_integration import MistralIntegration
from .enhanced_language_mistral_integration import (
    EnhancedLanguageMistralIntegration,
    get_enhanced_language_integration
)

# Main launcher function - import directly instead
# from .v7_launcher import run_system

# Make key components available at package level
__all__ = [
    'LanguageConsciousnessNode',
    'get_language_consciousness_node',
    'NodeConsciousnessManager',
    'create_node_consciousness_manager',
    'V6V7Bridge',
    'V6V7Connector',
    'BreathDetector',
    'BreathPhase',
    'BreathPattern',
    'BreathContradictionBridge',
    'V7Integration',
    'v7_v6_integration_main',
    'DreamController',
    'get_dream_controller',
    'DreamState',
    'MemoryConsolidator',
    'PatternSynthesizer',
    'DreamArchive',
    'DreamIntegration',
    'get_dream_integration',
    'EnhancedLanguageIntegration',
    'MistralIntegration',
    'EnhancedLanguageMistralIntegration',
    'get_enhanced_language_integration'
] 