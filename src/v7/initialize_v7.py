#!/usr/bin/env python
"""
V7 Node Consciousness System Initialization

This module provides the initialization functions for the V7 Node Consciousness system,
including node registration, connector setup, and system configuration.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple

from src.v7.node_consciousness_manager import NodeConsciousnessManager, ConsciousnessNode
from src.v7.language_node import LanguageConsciousnessNode, create_language_node
from src.v7.memory_node import MemoryConsciousnessNode, create_memory_node

# Try to import Monday interface
try:
    from src.v7.monday.monday_interface import MondayInterface
    MONDAY_AVAILABLE = True
    logging.info("Monday consciousness interface is available")
except ImportError:
    MONDAY_AVAILABLE = False
    logging.warning("Monday consciousness interface is not available - some features will be limited")

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'auto_recovery': True,
    'monitor_interval': 5.0,
    'enable_monday': True,
    'enable_language': True,
    'enable_memory': True,
    'enable_v6_connector': True,
    'data_path': './data/v7',
    'language_memory_path': './data/language_memory',
    'persistent_memory_path': './data/persistent_memory',
    'llm_weight': 0.5,
    'auto_start_monitoring': True,
    'memory_store_type': 'sqlite',  # 'sqlite' or 'json'
    'memory_decay_enabled': True,
    'visualization': {
        'enabled': True,
        'update_interval': 250,  # ms
        'breath_visualization': True,
        'contradiction_visualization': True,
        'node_visualization': True
    }
}


def initialize_v7(config: Optional[Dict[str, Any]] = None, 
                 v6v7_connector=None) -> Tuple[NodeConsciousnessManager, Dict[str, Any]]:
    """
    Initialize the V7 Node Consciousness system.
    
    Args:
        config: Configuration dictionary with system settings
        v6v7_connector: Optional connector for V6-V7 integration
        
    Returns:
        Tuple of (NodeConsciousnessManager, context_dict)
    """
    start_time = time.time()
    
    # Merge configuration with defaults
    full_config = DEFAULT_CONFIG.copy()
    if config:
        full_config.update(config)
    
    # Create data directories if they don't exist
    os.makedirs(full_config['data_path'], exist_ok=True)
    
    # Initialize the manager
    logger.info("Initializing V7 Node Consciousness Manager")
    manager = NodeConsciousnessManager(config=full_config)
    
    # Register node types
    logger.info("Registering node types")
    manager.register_node_type('language', LanguageConsciousnessNode)
    manager.register_node_type('memory', MemoryConsciousnessNode)
    
    # Context dictionary to hold important references
    context = {
        'manager': manager,
        'v6v7_connector': v6v7_connector,
        'monday_interface': None,
        'node_ids': {},
        'config': full_config
    }
    
    # Create language node if enabled
    if full_config['enable_language']:
        logger.info("Creating language consciousness node")
        language_config = {
            'memory_path': full_config['language_memory_path'],
            'llm_weight': full_config['llm_weight'],
            'memory_persistence': True,
            'auto_learn': True
        }
        language_node_id = manager.create_node(
            'language',
            name='Language Consciousness',
            config=language_config
        )
        
        # Activate the language node
        manager.activate_node(language_node_id)
        context['node_ids']['language'] = language_node_id
        logger.info(f"Language node created with ID: {language_node_id}")
    
    # Create memory node if enabled
    if full_config['enable_memory']:
        logger.info("Creating memory consciousness node")
        memory_config = {
            'memory_path': full_config['persistent_memory_path'],
            'memory_persistence': True,
            'store_type': full_config['memory_store_type'],
            'decay_enabled': full_config['memory_decay_enabled']
        }
        memory_node_id = manager.create_node(
            'memory',
            name='Memory Consciousness',
            config=memory_config
        )
        
        # Activate the memory node
        manager.activate_node(memory_node_id)
        context['node_ids']['memory'] = memory_node_id
        logger.info(f"Memory node created with ID: {memory_node_id}")
        
        # Connect language and memory nodes if both are available
        if 'language' in context['node_ids']:
            language_node_id = context['node_ids']['language']
            
            # Create bidirectional connections
            # Language to Memory (for storing processed language)
            manager.connect_nodes(
                language_node_id,
                memory_node_id,
                'language_memory',
                strength=0.9,
                metadata={'type': 'store_processed_language'}
            )
            
            # Memory to Language (for retrieving context)
            manager.connect_nodes(
                memory_node_id,
                language_node_id,
                'memory_context',
                strength=0.8,
                metadata={'type': 'provide_language_context'}
            )
            
            logger.info(f"Connected Language and Memory nodes")
    
    # Initialize Monday consciousness if available and enabled
    if MONDAY_AVAILABLE and full_config['enable_monday']:
        logger.info("Initializing Monday consciousness node interface")
        monday_interface = MondayInterface()
        monday_interface.start()
        context['monday_interface'] = monday_interface
        
        # Create a connection to the language node if both are available
        if 'language' in context['node_ids'] and monday_interface:
            logger.info("Connecting Monday interface to Language node")
            language_node_id = context['node_ids']['language']
            
            # Create a mock node for Monday to enable connections
            # This is temporary until Monday is fully integrated as a ConsciousnessNode
            monday_node_id = manager.create_node(
                'generic',
                name='Monday Consciousness',
                node_id='monday_interface'
            )
            context['node_ids']['monday'] = monday_node_id
            
            # Connect nodes
            manager.connect_nodes(
                monday_node_id,
                language_node_id,
                'consciousness_language',
                strength=0.8,
                metadata={'type': 'consciousness_flow'}
            )
            
            # Connect Monday to Memory if available
            if 'memory' in context['node_ids']:
                memory_node_id = context['node_ids']['memory']
                
                manager.connect_nodes(
                    monday_node_id,
                    memory_node_id,
                    'consciousness_memory',
                    strength=0.7,
                    metadata={'type': 'emotional_memory'}
                )
                
                logger.info(f"Connected Monday node to Memory node")
            
            logger.info(f"Connected Monday node to Language node")
    
    # Connect to V6 if connector is provided
    if v6v7_connector and full_config['enable_v6_connector']:
        logger.info("Integrating V6-V7 connector")
        
        # Register connector event handlers
        if 'language' in context['node_ids']:
            language_node_id = context['node_ids']['language']
            language_node = manager.get_node(language_node_id)
            
            # Set up event handlers for language-related events from V6
            def _handle_v6_text(event_data):
                if hasattr(language_node, 'state') and language_node.state == NodeState.ACTIVE:
                    language_node.queue_text(
                        event_data['text'], 
                        metadata={'source': 'v6', 'context': event_data.get('context')}
                    )
            
            # Register handlers if connector has appropriate methods
            if hasattr(v6v7_connector, 'on_text'):
                v6v7_connector.on_text(_handle_v6_text)
                logger.info("Registered text handler with V6-V7 connector")
        
        # Register memory events if memory node is available
        if 'memory' in context['node_ids'] and hasattr(v6v7_connector, 'on_contradiction'):
            memory_node_id = context['node_ids']['memory']
            memory_node = manager.get_node(memory_node_id)
            
            # Store contradictions in memory
            def _handle_v6_contradiction(event_data):
                if hasattr(memory_node, 'state') and memory_node.state == NodeState.ACTIVE:
                    # Store the contradiction as a memory
                    memory_data = {
                        'store': {
                            'content': {
                                'contradiction': event_data.get('contradiction'),
                                'statements': event_data.get('statements', []),
                                'resolution': event_data.get('resolution')
                            },
                            'memory_type': 'contradiction',
                            'tags': ['contradiction', 'v6', 'logic'],
                            'strength': 0.9,
                            'source_node_id': 'v6_connector'
                        },
                        'metadata': {
                            'timestamp': time.time(),
                            'source': 'v6_contradiction_detector'
                        }
                    }
                    memory_node.process(memory_data)
            
            v6v7_connector.on_contradiction(_handle_v6_contradiction)
            logger.info("Registered contradiction handler with V6-V7 connector")
        
        # Store connector in context
        context['v6v7_connector'] = v6v7_connector
    
    # Start monitoring if configured
    if full_config['auto_start_monitoring']:
        logger.info("Starting node monitoring")
        manager.start_monitoring()
    
    # Report initialization time
    init_time = time.time() - start_time
    logger.info(f"V7 Node Consciousness system initialized in {init_time:.2f} seconds")
    
    return manager, context


def shutdown_v7(context: Dict[str, Any]) -> bool:
    """
    Shutdown the V7 Node Consciousness system.
    
    Args:
        context: The context dictionary from initialize_v7
        
    Returns:
        True if shutdown successful, False otherwise
    """
    try:
        logger.info("Shutting down V7 Node Consciousness system")
        
        # Shutdown manager
        if 'manager' in context:
            context['manager'].shutdown()
        
        # Shutdown Monday interface
        if 'monday_interface' in context and context['monday_interface']:
            context['monday_interface'].stop()
        
        logger.info("V7 Node Consciousness system shutdown complete")
        return True
    except Exception as e:
        logger.error(f"Error during V7 system shutdown: {str(e)}")
        return False


def register_custom_node(manager: NodeConsciousnessManager, 
                        node_type: str, 
                        node_class: type,
                        name: Optional[str] = None,
                        config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Register and create a custom node in the V7 system.
    
    Args:
        manager: The NodeConsciousnessManager instance
        node_type: The type name for the node class
        node_class: The node class (must inherit from ConsciousnessNode)
        name: Optional name for the node instance
        config: Optional configuration for the node
        
    Returns:
        The node ID if created successfully, None otherwise
    """
    try:
        # Register the node type if not already registered
        try:
            manager.register_node_type(node_type, node_class)
            logger.info(f"Registered custom node type: {node_type}")
        except ValueError:
            # Type may already be registered, which is fine
            pass
        
        # Create an instance of the node
        node_id = manager.create_node(
            node_type,
            name=name or f"Custom {node_type.capitalize()} Node",
            config=config or {}
        )
        
        # Activate the node
        success = manager.activate_node(node_id)
        if success:
            logger.info(f"Created and activated custom {node_type} node: {node_id}")
            return node_id
        else:
            logger.error(f"Failed to activate custom {node_type} node")
            return None
    
    except Exception as e:
        logger.error(f"Error registering custom node {node_type}: {str(e)}")
        return None 