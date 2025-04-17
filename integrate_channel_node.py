#!/usr/bin/env python
"""
Channel Node Integration Script
============================

This script integrates the ChannelNode with the FlexNode and connects them to the
existing neural network system to optimize data flow between components.
"""

import os
import sys
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
import importlib.util
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("channel_node_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChannelNodeIntegration")

# Ensure required modules are available
try:
    from channel_node import ChannelNode, ChannelType, MessagePriority
    logger.info("ChannelNode module imported successfully")
except ImportError:
    logger.error("Failed to import ChannelNode. Make sure channel_node.py is in the current directory.")
    sys.exit(1)

try:
    from flex_node import FlexNode
    logger.info("FlexNode module imported successfully")
except ImportError:
    logger.error("Failed to import FlexNode. Make sure flex_node.py is in the current directory.")
    sys.exit(1)

def import_module_from_path(file_path: str, module_name: Optional[str] = None) -> Any:
    """
    Import a module from a file path.
    
    Args:
        file_path: Path to the Python file
        module_name: Optional name for the module
        
    Returns:
        Imported module or None on failure
    """
    try:
        if module_name is None:
            module_name = Path(file_path).stem
            
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            logger.error(f"Could not create module spec for {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error importing module from {file_path}: {str(e)}")
        return None

def discover_central_node() -> Optional[Any]:
    """
    Find and instantiate the central node.
    
    Returns:
        Central node instance or None if not found
    """
    try:
        # Try to import the central node module
        try:
            from central_node import CentralNode
            logger.info("Imported CentralNode directly")
            return CentralNode()
        except ImportError:
            logger.warning("Failed to import CentralNode directly, trying file path approach")
            
        # Try file path approach
        central_node_path = "central_node.py"
        if not os.path.exists(central_node_path):
            central_node_path = os.path.join("src", "central_node.py")
            if not os.path.exists(central_node_path):
                logger.error("Could not find central_node.py")
                return None
                
        central_module = import_module_from_path(central_node_path)
        if not central_module:
            return None
            
        central_node_class = getattr(central_module, "CentralNode", None)
        if not central_node_class:
            logger.error("CentralNode class not found in module")
            return None
            
        central_node = central_node_class()
        logger.info("Created CentralNode instance")
        return central_node
        
    except Exception as e:
        logger.error(f"Error discovering central node: {str(e)}")
        return None

def find_node_modules() -> List[str]:
    """
    Find all Python files that might contain node implementations.
    
    Returns:
        List of file paths
    """
    node_files = []
    
    # Check current directory
    for filename in os.listdir('.'):
        if filename.endswith(".py") and any(term in filename.lower() for term in ["node", "neural", "processor"]):
            # Skip our own modules
            if filename not in ["channel_node.py", "flex_node.py", "integrate_channel_node.py", "integrate_flex_node.py"]:
                node_files.append(filename)
    
    # Check src directory if it exists
    src_dir = Path("src")
    if src_dir.exists() and src_dir.is_dir():
        for filename in os.listdir(src_dir):
            if filename.endswith(".py") and any(term in filename.lower() for term in ["node", "neural", "processor"]):
                # Skip our own modules
                if filename not in ["channel_node.py", "flex_node.py", "integrate_channel_node.py", "integrate_flex_node.py"]:
                    node_files.append(str(src_dir / filename))
    
    logger.info(f"Found {len(node_files)} potential node modules")
    return node_files

def discover_node_instances(node_files: List[str]) -> Dict[str, Tuple[Any, str]]:
    """
    Discover and instantiate nodes from the given files.
    
    Args:
        node_files: List of Python files
        
    Returns:
        Dictionary of node_id -> (node_instance, node_type)
    """
    nodes = {}
    
    for file_path in node_files:
        try:
            module = import_module_from_path(file_path)
            if not module:
                continue
                
            # Check for node classes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Check if it's a class with node-like name
                if isinstance(attr, type) and any(term in attr_name for term in ["Node", "RSEN", "Processor"]):
                    try:
                        # Create an instance
                        instance = attr()
                        node_id = f"{Path(file_path).stem}.{attr_name}"
                        nodes[node_id] = (instance, attr_name)
                        logger.info(f"Discovered node: {node_id}")
                    except Exception as e:
                        logger.warning(f"Could not instantiate {attr_name} from {file_path}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    return nodes

def create_channel_node() -> Optional[ChannelNode]:
    """
    Create and initialize a ChannelNode.
    
    Returns:
        Initialized ChannelNode or None on failure
    """
    try:
        channel_node = ChannelNode()
        logger.info(f"Created ChannelNode with ID: {channel_node.node_id}")
        return channel_node
    except Exception as e:
        logger.error(f"Failed to create ChannelNode: {str(e)}")
        return None

def create_flex_node() -> Optional[FlexNode]:
    """
    Create and initialize a FlexNode.
    
    Returns:
        Initialized FlexNode or None on failure
    """
    try:
        flex_node = FlexNode(embedding_dim=256, hidden_dims=[512, 256])
        logger.info(f"Created FlexNode with ID: {flex_node.node_id}")
        return flex_node
    except Exception as e:
        logger.error(f"Failed to create FlexNode: {str(e)}")
        return None

def connect_flex_to_channel(flex_node: FlexNode, channel_node: ChannelNode) -> bool:
    """
    Connect the FlexNode to the ChannelNode.
    
    Args:
        flex_node: FlexNode instance
        channel_node: ChannelNode instance
        
    Returns:
        True if connection was successful, False otherwise
    """
    try:
        # Register FlexNode with ChannelNode
        channel_node.register_node(
            flex_node.node_id,
            flex_node,
            [ChannelType.DEFAULT, ChannelType.TENSOR, ChannelType.EMBEDDING]
        )
        
        # Connect ChannelNode to FlexNode
        flex_node.connect_to_node(
            channel_node.node_id,
            channel_node,
            connection_type="channel",
            weight=0.9,
            bidirectional=True
        )
        
        logger.info(f"Connected FlexNode and ChannelNode")
        return True
    except Exception as e:
        logger.error(f"Failed to connect FlexNode to ChannelNode: {str(e)}")
        return False

def register_nodes_with_channel(channel_node: ChannelNode, nodes: Dict[str, Tuple[Any, str]]) -> int:
    """
    Register discovered nodes with the ChannelNode.
    
    Args:
        channel_node: The ChannelNode instance
        nodes: Dictionary of node_id -> (node_instance, node_type)
        
    Returns:
        Number of successfully registered nodes
    """
    registered_count = 0
    
    # Define channel mappings based on node types
    channel_mapping = {
        "RSEN": [ChannelType.DEFAULT, ChannelType.TENSOR, ChannelType.EMBEDDING],
        "Hybrid": [ChannelType.DEFAULT, ChannelType.TENSOR, ChannelType.EMBEDDING],
        "Zero": [ChannelType.DEFAULT, ChannelType.CONTROL],
        "Portal": [ChannelType.DEFAULT, ChannelType.QUANTUM],
        "Wormhole": [ChannelType.DEFAULT, ChannelType.QUANTUM],
        "ZPE": [ChannelType.DEFAULT, ChannelType.QUANTUM],
        "Processor": [ChannelType.DEFAULT, ChannelType.TEXT],
        "Neural": [ChannelType.DEFAULT, ChannelType.TENSOR],
        "Language": [ChannelType.DEFAULT, ChannelType.TEXT],
        "Lumina": [ChannelType.DEFAULT, ChannelType.TEXT, ChannelType.CONTROL],
        "Physics": [ChannelType.DEFAULT, ChannelType.TENSOR],
        "Game": [ChannelType.DEFAULT],
        "Consciousness": [ChannelType.DEFAULT, ChannelType.FEEDBACK],
        "Fractal": [ChannelType.DEFAULT, ChannelType.TENSOR],
        "Void": [ChannelType.DEFAULT, ChannelType.QUANTUM]
    }
    
    for node_id, (node_instance, node_type) in nodes.items():
        try:
            # Determine appropriate channels for this node
            channels = [ChannelType.DEFAULT]  # Default channel for all nodes
            
            for type_key, channel_list in channel_mapping.items():
                if type_key in node_type:
                    channels = channel_list
                    break
            
            # Register with ChannelNode
            if channel_node.register_node(node_id, node_instance, channels):
                registered_count += 1
                logger.info(f"Registered {node_id} with channels: {[ch.value for ch in channels]}")
                
        except Exception as e:
            logger.error(f"Failed to register {node_id} with ChannelNode: {str(e)}")
    
    return registered_count

def connect_flex_to_nodes(flex_node: FlexNode, nodes: Dict[str, Tuple[Any, str]]) -> int:
    """
    Connect the FlexNode directly to important nodes.
    
    Args:
        flex_node: The FlexNode instance
        nodes: Dictionary of node_id -> (node_instance, node_type)
        
    Returns:
        Number of successful connections
    """
    # Define connection priorities - only connect directly to these important nodes
    priority_nodes = [
        "RSEN", "Hybrid", "Zero", "Neural", "Lumina", 
        "Consciousness", "Processor", "Quantum", "Portal"
    ]
    
    # Connection type mapping
    connection_types = {
        "RSEN": "resonance",
        "Hybrid": "data",
        "Zero": "control",
        "Neural": "neural",
        "Lumina": "data",
        "Consciousness": "mirror",
        "Processor": "processing",
        "Quantum": "quantum",
        "Portal": "portal",
        "Fractal": "fractal"
    }
    
    connected_count = 0
    
    for node_id, (node_instance, node_type) in nodes.items():
        # Check if this is a priority node
        is_priority = any(term in node_type for term in priority_nodes)
        
        if is_priority:
            try:
                # Determine connection type
                conn_type = "default"
                for type_key, type_value in connection_types.items():
                    if type_key in node_type:
                        conn_type = type_value
                        break
                
                # Connect FlexNode to this node
                if flex_node.connect_to_node(
                    node_id=node_id,
                    node_instance=node_instance,
                    connection_type=conn_type,
                    weight=0.7,
                    bidirectional=True
                ):
                    connected_count += 1
                    logger.info(f"Connected FlexNode to {node_id} with {conn_type} connection")
                    
            except Exception as e:
                logger.error(f"Failed to connect FlexNode to {node_id}: {str(e)}")
    
    return connected_count

def integrate_with_central_node(
    central_node: Any, 
    channel_node: ChannelNode, 
    flex_node: FlexNode
) -> bool:
    """
    Integrate the ChannelNode and FlexNode with the central node.
    
    Args:
        central_node: The central node instance
        channel_node: The ChannelNode instance
        flex_node: The FlexNode instance
        
    Returns:
        True if integration was successful, False otherwise
    """
    try:
        # Set central node references
        channel_node.set_central_node(central_node)
        flex_node.set_central_node(central_node)
        
        # Register nodes with central node if possible
        if hasattr(central_node, '_register_component'):
            central_node._register_component("ChannelNode", channel_node)
            central_node._register_component("FlexNode", flex_node)
            
            # Also add to nodes dictionary if present
            if hasattr(central_node, 'nodes'):
                central_node.nodes["ChannelNode"] = channel_node
                central_node.nodes["FlexNode"] = flex_node
                
            logger.info("Registered nodes with central node")
            return True
        else:
            logger.warning("Central node does not have _register_component method")
            return False
            
    except Exception as e:
        logger.error(f"Failed to integrate with central node: {str(e)}")
        return False

def add_data_transformers(channel_node: ChannelNode) -> None:
    """
    Add specialized data transformers to the ChannelNode for common transformations.
    """
    # Neural tensor transformer (normalizes tensors)
    def normalize_tensor(data):
        if isinstance(data, torch.Tensor):
            # Normalize to mean 0, std 1
            return (data - data.mean()) / (data.std() + 1e-8)
        return data
    
    # Text data cleaner
    def clean_text(data):
        if isinstance(data, str):
            # Simple cleaning: lowercase and remove extra whitespace
            return " ".join(data.lower().split())
        return data
    
    # Add transformers to appropriate channels
    try:
        import torch
        channel_node.add_transformer(ChannelType.TENSOR, "default", normalize_tensor)
        logger.info("Added tensor normalizer transformer")
    except ImportError:
        logger.warning("Could not add tensor transformer - torch not available")
    
    channel_node.add_transformer(ChannelType.TEXT, "default", clean_text)
    logger.info("Added text cleaner transformer")

def save_integration_state(channel_node: ChannelNode, flex_node: FlexNode) -> str:
    """
    Save the integration state to a file.
    
    Args:
        channel_node: The ChannelNode instance
        flex_node: The FlexNode instance
        
    Returns:
        Path to the state file
    """
    state_file = "channel_flex_integration_state.json"
    
    try:
        # Save FlexNode state separately
        flex_state_path = flex_node.save_state()
        
        # Create integration state
        state = {
            "timestamp": time.time(),
            "channel_node_id": channel_node.node_id,
            "flex_node_id": flex_node.node_id,
            "flex_node_state_path": flex_state_path,
            "registered_nodes": len(channel_node.registered_nodes),
            "channel_metrics": channel_node.get_channel_metrics(),
            "routing_rules": len(channel_node.routing_rules)
        }
        
        # Save to file
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved integration state to {state_file}")
        return state_file
        
    except Exception as e:
        logger.error(f"Failed to save integration state: {str(e)}")
        return None

def main():
    """Main integration function"""
    logger.info("Starting ChannelNode and FlexNode integration")
    
    # Create the nodes
    channel_node = create_channel_node()
    if not channel_node:
        logger.error("Failed to create ChannelNode. Exiting.")
        return
        
    flex_node = create_flex_node()
    if not flex_node:
        logger.error("Failed to create FlexNode. Exiting.")
        return
    
    # Start the FlexNode
    flex_node.start()
    
    # Connect FlexNode to ChannelNode
    if not connect_flex_to_channel(flex_node, channel_node):
        logger.warning("Failed to connect FlexNode to ChannelNode")
    
    # Discover and integrate with central node
    central_node = discover_central_node()
    if central_node:
        if integrate_with_central_node(central_node, channel_node, flex_node):
            logger.info("Successfully integrated with central node")
        else:
            logger.warning("Failed to integrate with central node")
    else:
        logger.warning("Could not discover central node. Integration will be limited.")
    
    # Find and register nodes
    node_files = find_node_modules()
    discovered_nodes = discover_node_instances(node_files)
    
    # Register nodes with ChannelNode
    registered_count = register_nodes_with_channel(channel_node, discovered_nodes)
    logger.info(f"Registered {registered_count}/{len(discovered_nodes)} nodes with ChannelNode")
    
    # Connect FlexNode directly to important nodes
    connected_count = connect_flex_to_nodes(flex_node, discovered_nodes)
    logger.info(f"Connected FlexNode directly to {connected_count} important nodes")
    
    # Add data transformers
    add_data_transformers(channel_node)
    
    # Save integration state
    state_file = save_integration_state(channel_node, flex_node)
    
    # Setup complete
    logger.info("ChannelNode and FlexNode integration completed successfully")
    logger.info(f"ChannelNode is running with {len(channel_node.registered_nodes)} registered nodes")
    logger.info(f"FlexNode has {len(flex_node.connections)} active connections")
    
    # Keep running to maintain connections
    try:
        logger.info("Press Ctrl+C to exit")
        
        while True:
            # Periodically optimize FlexNode connections
            if hasattr(flex_node, 'optimize_connections'):
                optimization_result = flex_node.optimize_connections()
                if optimization_result.get("changes"):
                    logger.info(f"Optimized FlexNode connections: {len(optimization_result['changes'])} changes")
            
            # Log channel metrics occasionally
            if int(time.time()) % 300 == 0:  # Every 5 minutes
                metrics = channel_node.get_channel_metrics()
                logger.info(f"Channel metrics: {len(metrics)} active channels")
                
                # Log the busiest channel
                busiest_channel = None
                max_messages = 0
                for ch_type, ch_metrics in metrics.items():
                    msg_count = ch_metrics.get("metrics", {}).get("message_count", 0)
                    if msg_count > max_messages:
                        max_messages = msg_count
                        busiest_channel = ch_type
                
                if busiest_channel:
                    logger.info(f"Busiest channel: {busiest_channel} with {max_messages} messages")
            
            # Sleep to prevent CPU spinning
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        # Shutdown nodes
        logger.info("Shutting down nodes...")
        flex_node.stop()
        channel_node.shutdown()
        logger.info("Nodes stopped")

if __name__ == "__main__":
    main() 