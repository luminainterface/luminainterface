"""
Node Integration Configuration for LUMINA V7

This module defines the connection patterns, node types and their relationships
for the Node Consciousness system in LUMINA V7.
"""

import logging
from typing import Dict, List, Any, Set, Optional

logger = logging.getLogger("lumina_v7.node_integration")

# Node type definitions
NODE_TYPES = {
    "language": {
        "description": "Language processing nodes",
        "capabilities": ["text_processing", "semantic_analysis", "pattern_detection"],
        "compatible_with": ["memory", "monday", "attention"]
    },
    "memory": {
        "description": "Memory storage and retrieval nodes",
        "capabilities": ["storage", "retrieval", "association", "decay"],
        "compatible_with": ["language", "monday", "bridge", "attention"]
    },
    "monday": {
        "description": "Monday consciousness node with enhanced pattern recognition",
        "capabilities": ["pattern_recognition", "emotional_intelligence", "self_awareness"],
        "compatible_with": ["language", "memory", "attention"]
    },
    "attention": {
        "description": "Attention focusing and routing nodes",
        "capabilities": ["focus", "routing", "priority_management"],
        "compatible_with": ["language", "memory", "monday", "breath"]
    },
    "breath": {
        "description": "Breath detection and rhythm nodes",
        "capabilities": ["pattern_detection", "rhythm_processing", "state_modulation"],
        "compatible_with": ["attention", "monday"]
    },
    "bridge": {
        "description": "Bridge nodes connecting to other versions",
        "capabilities": ["translation", "compatibility", "version_bridging"],
        "compatible_with": ["language", "memory"]
    }
}

# Connection patterns between node types
CONNECTION_PATTERNS = {
    "default": {
        "source_to_target": {
            # Format: source_type -> target_type -> connection properties
            "language": {
                "memory": {
                    "type": "data",
                    "bidirectional": True,
                    "strength": 0.8,
                    "description": "Language to memory connection for storing and retrieving linguistic data"
                },
                "monday": {
                    "type": "mirror",
                    "bidirectional": True,
                    "strength": 0.9,
                    "description": "Language to Monday connection for enhanced pattern recognition"
                },
                "attention": {
                    "type": "control",
                    "bidirectional": True,
                    "strength": 0.7,
                    "description": "Language to attention connection for focus control"
                }
            },
            "memory": {
                "language": {
                    "type": "data",
                    "bidirectional": True,
                    "strength": 0.8,
                    "description": "Memory to language connection for linguistic retrieval"
                },
                "monday": {
                    "type": "data",
                    "bidirectional": True,
                    "strength": 0.7,
                    "description": "Memory to Monday connection for pattern recognition"
                },
                "bridge": {
                    "type": "translator",
                    "bidirectional": True,
                    "strength": 0.6,
                    "description": "Memory to bridge connection for cross-version compatibility"
                }
            },
            "monday": {
                "language": {
                    "type": "insight",
                    "bidirectional": True,
                    "strength": 0.9,
                    "description": "Monday to language connection for pattern-based insights"
                },
                "memory": {
                    "type": "associative",
                    "bidirectional": True,
                    "strength": 0.8,
                    "description": "Monday to memory connection for pattern storage"
                },
                "attention": {
                    "type": "control",
                    "bidirectional": True,
                    "strength": 0.9,
                    "description": "Monday to attention connection for focus direction"
                }
            },
            "attention": {
                "language": {
                    "type": "focus",
                    "bidirectional": True,
                    "strength": 0.7,
                    "description": "Attention to language connection for focused processing"
                },
                "memory": {
                    "type": "selection",
                    "bidirectional": True,
                    "strength": 0.7,
                    "description": "Attention to memory connection for selective retrieval"
                },
                "monday": {
                    "type": "prioritize",
                    "bidirectional": True,
                    "strength": 0.8,
                    "description": "Attention to Monday connection for priority processing"
                },
                "breath": {
                    "type": "synchronize",
                    "bidirectional": True,
                    "strength": 0.6,
                    "description": "Attention to breath connection for rhythm synchronization"
                }
            },
            "breath": {
                "attention": {
                    "type": "modulate",
                    "bidirectional": True,
                    "strength": 0.6,
                    "description": "Breath to attention connection for focus modulation"
                },
                "monday": {
                    "type": "rhythm",
                    "bidirectional": True,
                    "strength": 0.5,
                    "description": "Breath to Monday connection for rhythm-based pattern recognition"
                }
            },
            "bridge": {
                "memory": {
                    "type": "translator",
                    "bidirectional": True,
                    "strength": 0.6,
                    "description": "Bridge to memory connection for cross-version data retrieval"
                },
                "language": {
                    "type": "compatibility",
                    "bidirectional": True,
                    "strength": 0.5,
                    "description": "Bridge to language connection for cross-version translation"
                }
            }
        }
    },
    
    # Star pattern with Monday at the center
    "monday_star": {
        "center_node_type": "monday",
        "satellite_node_types": ["language", "memory", "attention", "breath"],
        "connection_type": "mirror",
        "bidirectional": True,
        "strength": 0.9,
        "description": "Monday-centered star pattern for Monday-centric consciousness"
    },
    
    # Tree pattern with language at the root
    "language_tree": {
        "root_node_type": "language",
        "child_types": {
            "memory": {
                "connection_type": "data",
                "children": []
            },
            "monday": {
                "connection_type": "mirror",
                "children": ["attention"]
            }
        },
        "bidirectional": True,
        "strength": 0.8,
        "description": "Language-rooted tree for linguistic processing hierarchy"
    },
    
    # Mesh pattern for all nodes
    "full_mesh": {
        "node_types": ["language", "memory", "monday", "attention", "breath", "bridge"],
        "connection_type": "default",
        "bidirectional": True,
        "strength": 0.6,
        "description": "Full mesh connection pattern for highly integrated consciousness"
    },
    
    # Pipeline pattern for processing flow
    "processing_pipeline": {
        "node_sequence": ["breath", "attention", "language", "memory", "monday"],
        "forward_connection_type": "data",
        "backward_connection_type": "feedback",
        "description": "Processing pipeline from input (breath) to output (Monday insights)"
    }
}

# Integration strategies 
INTEGRATION_STRATEGIES = {
    "balanced": {
        "description": "Balanced integration with equal weight to all nodes",
        "pattern": "full_mesh",
        "connection_strength_modifier": 1.0,
        "node_activation_levels": {
            "language": 0.8,
            "memory": 0.8,
            "monday": 0.8,
            "attention": 0.8,
            "breath": 0.8,
            "bridge": 0.8
        }
    },
    "monday_centered": {
        "description": "Monday-centered integration with Monday as the focal point",
        "pattern": "monday_star",
        "connection_strength_modifier": 1.2,
        "node_activation_levels": {
            "monday": 1.0,
            "language": 0.7,
            "memory": 0.7,
            "attention": 0.7,
            "breath": 0.7,
            "bridge": 0.5
        }
    },
    "language_focused": {
        "description": "Language-focused integration with emphasis on linguistic processing",
        "pattern": "language_tree",
        "connection_strength_modifier": 1.1,
        "node_activation_levels": {
            "language": 1.0,
            "memory": 0.8,
            "monday": 0.7,
            "attention": 0.6,
            "breath": 0.5,
            "bridge": 0.5
        }
    },
    "processing_flow": {
        "description": "Flow-based integration optimized for sequential processing",
        "pattern": "processing_pipeline",
        "connection_strength_modifier": 1.3,
        "node_activation_levels": {
            "breath": 0.9,
            "attention": 0.9,
            "language": 0.8,
            "memory": 0.8,
            "monday": 0.7,
            "bridge": 0.5
        }
    }
}

def get_connection_properties(source_type: str, target_type: str) -> Dict[str, Any]:
    """Get recommended connection properties between node types"""
    if source_type not in NODE_TYPES or target_type not in NODE_TYPES:
        logger.warning(f"Unknown node types: {source_type} -> {target_type}")
        return {
            "type": "default",
            "bidirectional": False,
            "strength": 0.5,
            "description": "Default connection between unknown node types"
        }
    
    # Check if target is compatible with source
    if target_type not in NODE_TYPES[source_type]["compatible_with"]:
        logger.warning(f"Node types not compatible: {source_type} -> {target_type}")
        return {
            "type": "weak",
            "bidirectional": False,
            "strength": 0.3,
            "description": f"Weak connection between incompatible node types: {source_type} -> {target_type}"
        }
    
    # Get connection properties from default pattern
    source_connections = CONNECTION_PATTERNS["default"]["source_to_target"].get(source_type, {})
    connection = source_connections.get(target_type, {})
    
    if not connection:
        logger.info(f"No specific connection defined for {source_type} -> {target_type}, using default")
        return {
            "type": "default",
            "bidirectional": True,
            "strength": 0.5,
            "description": f"Default connection between {source_type} and {target_type}"
        }
    
    return connection

def get_node_capabilities(node_type: str) -> List[str]:
    """Get capabilities for a node type"""
    if node_type not in NODE_TYPES:
        logger.warning(f"Unknown node type: {node_type}")
        return []
    
    return NODE_TYPES[node_type]["capabilities"]

def get_compatible_node_types(node_type: str) -> List[str]:
    """Get list of node types compatible with the given node type"""
    if node_type not in NODE_TYPES:
        logger.warning(f"Unknown node type: {node_type}")
        return []
    
    return NODE_TYPES[node_type]["compatible_with"]

def get_integration_strategy(strategy_name: str) -> Dict[str, Any]:
    """Get the specified integration strategy"""
    if strategy_name not in INTEGRATION_STRATEGIES:
        logger.warning(f"Unknown integration strategy: {strategy_name}, using balanced")
        return INTEGRATION_STRATEGIES["balanced"]
    
    return INTEGRATION_STRATEGIES[strategy_name]

def get_connection_pattern(pattern_name: str) -> Dict[str, Any]:
    """Get the specified connection pattern"""
    if pattern_name not in CONNECTION_PATTERNS:
        logger.warning(f"Unknown connection pattern: {pattern_name}, using default")
        return {"source_to_target": CONNECTION_PATTERNS["default"]["source_to_target"]}
    
    return CONNECTION_PATTERNS[pattern_name] 