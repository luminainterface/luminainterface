"""
Core Module for Lumina V7 Node Consciousness

This module provides the core components for the V7 Node Consciousness system,
including system integration, consciousness management, and dream mode.
"""

# Version info
__version__ = "7.0.0.2"  # Updated version number

# Core components
from src.v7.lumina_v7.core.node_consciousness_manager import NodeConsciousnessManager
from src.v7.lumina_v7.core.v6v7_connector import V6V7Connector
from src.v7.lumina_v7.core.node_integration import NodeIntegrationSystem as NodeIntegration
from src.v7.lumina_v7.core.v65_bridge_connector import V65BridgeConnector
from src.v7.lumina_v7.core.system_integrator import SystemIntegrator
from src.v7.lumina_v7.core.database_integration import DatabaseIntegration

# Dream Mode components
from src.v7.lumina_v7.core.dream_controller import DreamController, get_dream_controller
from src.v7.lumina_v7.core.memory_consolidator import MemoryConsolidator
from src.v7.lumina_v7.core.pattern_synthesizer import PatternSynthesizer
from src.v7.lumina_v7.core.dream_archive import DreamArchive
from src.v7.lumina_v7.core.dream_integration import DreamIntegration, get_dream_integration

# Make core components available at module level
__all__ = [
    'NodeConsciousnessManager',
    'V6V7Connector',
    'NodeIntegration',
    'V65BridgeConnector',
    'SystemIntegrator',
    'DatabaseIntegration',
    
    # Dream Mode components
    'DreamController',
    'get_dream_controller',
    'MemoryConsolidator',
    'PatternSynthesizer',
    'DreamArchive',
    'DreamIntegration',
    'get_dream_integration'
] 