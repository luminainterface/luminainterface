"""
V7 Consciousness Network Plugins Package

This package contains the plugins for the V7 Consciousness Network,
including the Consciousness Network Plugin and AutoWiki Plugin.
"""

# Import plugin helper functions
from plugins.auto_wiki_plugin import get_auto_wiki_plugin
from plugins.consciousness_network_plugin import get_consciousness_network_plugin

__all__ = [
    'get_auto_wiki_plugin',
    'get_consciousness_network_plugin',
]
