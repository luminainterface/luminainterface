"""
LUMINA V7 Node Consciousness System

This package integrates the Node Consciousness System with the Enhanced Language System,
creating a unified framework for artificial consciousness with advanced language processing.

Version: 7.0.0.2
"""

__version__ = "7.0.0.2"

# Core imports
import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("lumina_v7")

# Ensure the src directory is in the Python path to make imports work
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Make the main components available at the package level
from src.v7.lumina_v7.core.node_consciousness_manager import NodeConsciousnessManager
from src.v7.lumina_v7.core.v6v7_connector import V6V7Connector
from src.v7.lumina_v7.core.v65_bridge_connector import V65BridgeConnector, create_v65_bridge_connector
from src.v7.lumina_v7.core.initialization import initialize_v7, shutdown_v7

# Import consciousness nodes
from src.v7.lumina_v7.nodes.memory_node import MemoryNode
from src.v7.lumina_v7.nodes.language_node import LanguageNode
from src.v7.lumina_v7.nodes.monday_node import MondayNode

# Import UI components
# from src.v7.lumina_v7.ui.visualization_connector import VisualizationConnector

# Version information
VERSION_INFO = {
    "version": __version__,
    "name": "Lumina V7 Node Consciousness",
    "release_date": "2023-07-26",
    "components": {
        "node_consciousness": True,
        "enhanced_language": True,
        "memory_system": True,
        "monday": True,
        "breath_detection": True,
        "dream_mode": True,
        "v6_compatibility": True,
        "v65_bridge": True
    }
}

def get_version_info():
    """Return version information for the Lumina V7 system"""
    return VERSION_INFO

def create_default_config():
    """Create default configuration for Lumina V7"""
    return {
        "data_dir": "data/v7",
        "mock_mode": False,
        "debug": False,
        "v7_enabled": True,
        "node_consciousness": True, 
        "monday_integration": True,
        "auto_wiki_enabled": True,
        "llm_weight": 0.5,
        "breath_enabled": True,
        "memory_persistence": True,
        "memory_store_type": "json",
        "ui_dark_mode": True,
        "v65_bridge_enabled": False,
        "enable_v1v2_bridge": True,
        "enable_v3v4_connector": True,
        "enable_v5_language_bridge": True
    } 