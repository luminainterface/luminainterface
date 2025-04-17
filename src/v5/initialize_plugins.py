"""
Plugin Initializer for V5 Visualization

This script initializes the Pattern Processor and Consciousness Analytics plugins
and registers them with the FrontendSocketManager.
"""

import logging
from .pattern_processor_plugin import PatternProcessorPlugin
from .consciousness_analytics_plugin import ConsciousnessAnalyticsPlugin
from .v5_plugin import V5Plugin
from .frontend_socket_manager import FrontendSocketManager
from .node_socket import NodeSocket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_plugins(socket_manager):
    """
    Initialize all necessary plugins for the V5 visualization system
    
    Args:
        socket_manager: The FrontendSocketManager instance
        
    Returns:
        Dictionary of initialized plugins
    """
    plugins = {}
    
    # Initialize Pattern Processor Plugin
    pattern_processor = PatternProcessorPlugin("pattern_processor_1")
    socket_descriptor = socket_manager.register_plugin(pattern_processor)
    if socket_descriptor:
        logger.info(f"Registered Pattern Processor Plugin: {socket_descriptor['plugin_id']}")
        logger.info(f"UI Components: {socket_descriptor.get('ui_components', [])}")
        plugins['pattern_processor'] = pattern_processor
    else:
        logger.error("Failed to register Pattern Processor Plugin")
    
    # Initialize Consciousness Analytics Plugin
    consciousness_analytics = ConsciousnessAnalyticsPlugin("consciousness_analytics_1")
    socket_descriptor = socket_manager.register_plugin(consciousness_analytics)
    if socket_descriptor:
        logger.info(f"Registered Consciousness Analytics Plugin: {socket_descriptor['plugin_id']}")
        logger.info(f"UI Components: {socket_descriptor.get('ui_components', [])}")
        plugins['consciousness_analytics'] = consciousness_analytics
        
        # Generate initial data and send to socket manager
        try:
            # Force generation of initial consciousness data
            initial_data = consciousness_analytics._generate_consciousness_data()
            consciousness_analytics.socket.send_message({
                "type": "consciousness_data_updated",
                "data": initial_data
            })
            logger.info("Sent initial consciousness data")
        except Exception as e:
            logger.error(f"Error sending initial consciousness data: {str(e)}")
    else:
        logger.error("Failed to register Consciousness Analytics Plugin")
    
    # Verify plugin connections
    logger.info(f"Available UI component providers:")
    for component, providers in socket_manager.ui_component_map.items():
        logger.info(f"  {component}: {providers}")
    
    return plugins

def fix_component_mapping(socket_manager):
    """
    Fix the UI component mapping for specific plugins
    
    Args:
        socket_manager: The FrontendSocketManager instance
    """
    # Map UI components to plugin IDs
    pattern_processor_id = "pattern_processor_1"
    consciousness_analytics_id = "consciousness_analytics_1"
    
    # Make sure fractal_view is mapped to pattern processor
    if "fractal_view" not in socket_manager.ui_component_map:
        socket_manager.ui_component_map["fractal_view"] = []
    if pattern_processor_id not in socket_manager.ui_component_map["fractal_view"]:
        socket_manager.ui_component_map["fractal_view"].append(pattern_processor_id)
    
    # Make sure consciousness_view is mapped to consciousness analytics
    if "consciousness_view" not in socket_manager.ui_component_map:
        socket_manager.ui_component_map["consciousness_view"] = []
    if consciousness_analytics_id not in socket_manager.ui_component_map["consciousness_view"]:
        socket_manager.ui_component_map["consciousness_view"].append(consciousness_analytics_id)
    
    # Make sure consciousness_meter is mapped (an alternative name used in some panels)
    if "consciousness_meter" not in socket_manager.ui_component_map:
        socket_manager.ui_component_map["consciousness_meter"] = []
    if consciousness_analytics_id not in socket_manager.ui_component_map["consciousness_meter"]:
        socket_manager.ui_component_map["consciousness_meter"].append(consciousness_analytics_id)
    
    logger.info("UI component mapping fixed")
    return socket_manager

def establish_forced_connections(socket_manager, plugins):
    """
    Force direct socket connections between plugins and the frontend manager
    
    Args:
        socket_manager: The FrontendSocketManager instance
        plugins: Dictionary of initialized plugins
    """
    logger.info("Establishing forced connections between plugins and frontend...")
    
    # Connect pattern processor directly to the manager socket
    if 'pattern_processor' in plugins:
        pattern_processor = plugins['pattern_processor']
        logger.info(f"Forcibly connecting pattern processor socket to manager")
        # Bidirectional connection
        pattern_processor.socket.connect_to(socket_manager.manager_socket)
        
        # Add message handlers for pattern processor
        socket_manager.manager_socket.register_message_handler(
            "request_pattern_data", 
            lambda msg: pattern_processor.socket.receive_message(msg)
        )
        
        # Send notification to ensure connection is working
        socket_manager.manager_socket.send_message({
            "type": "plugin_connection_check",
            "plugin_id": pattern_processor.node_id,
            "content": {"component": "fractal_view"}
        })
        
        logger.info(f"Forced connection established for pattern processor")
    
    # Connect consciousness analytics directly to the manager socket
    if 'consciousness_analytics' in plugins:
        consciousness_analytics = plugins['consciousness_analytics']
        logger.info(f"Forcibly connecting consciousness analytics socket to manager")
        # Bidirectional connection
        consciousness_analytics.socket.connect_to(socket_manager.manager_socket)
        
        # Add message handlers for consciousness analytics
        socket_manager.manager_socket.register_message_handler(
            "request_consciousness_data", 
            lambda msg: consciousness_analytics.socket.receive_message(msg)
        )
        
        # Add handler for node state updates to ensure real-time updates
        socket_manager.manager_socket.register_message_handler(
            "update_node_states", 
            lambda msg: consciousness_analytics.socket.receive_message(msg)
        )
        
        # Ensure we initialize with some data for the UI
        consciousness_analytics._update_metrics_from_nodes()
        
        # Send initial data to any listening components
        initial_data = consciousness_analytics._generate_consciousness_data()
        socket_manager.manager_socket.send_message({
            "type": "consciousness_data_updated",
            "data": initial_data
        })
        
        # Send notification to ensure connection is working
        socket_manager.manager_socket.send_message({
            "type": "plugin_connection_check",
            "plugin_id": consciousness_analytics.node_id,
            "content": {"component": "consciousness_view"}
        })
        
        logger.info(f"Forced connection established for consciousness analytics")
    
    return socket_manager 