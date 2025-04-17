#!/usr/bin/env python
"""
Flex Node Integration Script
============================

This script integrates the FlexNode into the existing neural network system.
It discovers existing nodes, establishes connections, and registers the FlexNode
with the central node architecture.
"""

import os
import sys
import logging
import time
import json
from typing import Dict, List, Any, Optional
import traceback
from pathlib import Path
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("flex_node_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FlexNodeIntegration")

# Ensure FlexNode is available
try:
    from flex_node import FlexNode, NodeConnection
    logger.info("FlexNode module imported successfully")
except ImportError:
    logger.error("Failed to import FlexNode. Make sure flex_node.py is in the current directory.")
    sys.exit(1)

def load_module_from_file(file_path: str, module_name: Optional[str] = None) -> Any:
    """
    Dynamically load a Python module from a file path.
    
    Args:
        file_path: Path to the Python file
        module_name: Name to assign to the module (defaults to filename without extension)
        
    Returns:
        The loaded module or None if loading failed
    """
    try:
        if module_name is None:
            module_name = Path(file_path).stem
            
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            logger.error(f"Failed to create module spec for {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Failed to load module from {file_path}: {str(e)}")
        return None

def discover_central_node() -> Any:
    """
    Discover and load the central node.
    
    Returns:
        Central node instance or None if not found
    """
    try:
        # Try to import directly first
        try:
            from central_node import CentralNode
            logger.info("Imported CentralNode directly")
            return CentralNode()
        except ImportError:
            logger.warning("Failed to import CentralNode directly, trying file-based import")
        
        # Try file-based import
        central_node_path = "central_node.py"
        if not os.path.exists(central_node_path):
            central_node_path = os.path.join("src", "central_node.py")
            if not os.path.exists(central_node_path):
                logger.error("Could not find central_node.py")
                return None
                
        central_module = load_module_from_file(central_node_path)
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

def find_node_files() -> List[str]:
    """
    Find all Python files that likely contain node implementations.
    
    Returns:
        List of file paths
    """
    node_files = []
    
    # Look in current directory
    for filename in os.listdir("."):
        if filename.endswith(".py") and ("node" in filename.lower() or "neural" in filename.lower()):
            if filename not in ["flex_node.py", "integrate_flex_node.py", "central_node.py"]:
                node_files.append(filename)
    
    # Also check src directory if it exists
    src_dir = Path("src")
    if src_dir.exists() and src_dir.is_dir():
        for filename in os.listdir(src_dir):
            if filename.endswith(".py") and ("node" in filename.lower() or "neural" in filename.lower()):
                if filename not in ["flex_node.py", "integrate_flex_node.py", "central_node.py"]:
                    node_files.append(os.path.join("src", filename))
    
    logger.info(f"Found {len(node_files)} potential node files")
    return node_files

def discover_nodes(node_files: List[str]) -> Dict[str, Any]:
    """
    Discover and load node instances from the given files.
    
    Args:
        node_files: List of Python files to check for node classes
        
    Returns:
        Dictionary mapping node IDs to node instances
    """
    nodes = {}
    
    for file_path in node_files:
        try:
            module = load_module_from_file(file_path)
            if not module:
                continue
                
            # Look for node classes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Check if it's a class with 'Node' in the name
                if isinstance(attr, type) and (
                    "Node" in attr_name or 
                    "RSEN" in attr_name or 
                    "Processor" in attr_name
                ):
                    try:
                        # Create instance
                        node_instance = attr()
                        node_id = f"{Path(file_path).stem}.{attr_name}"
                        nodes[node_id] = node_instance
                        logger.info(f"Discovered node: {node_id}")
                    except Exception as e:
                        logger.warning(f"Could not instantiate {attr_name} from {file_path}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    return nodes

def create_flex_node(embedding_dim: int = 256, hidden_dims: List[int] = [512, 256]) -> FlexNode:
    """
    Create and initialize a FlexNode instance.
    
    Args:
        embedding_dim: Dimension of input embeddings
        hidden_dims: Dimensions of hidden layers
        
    Returns:
        Initialized FlexNode instance
    """
    try:
        flex_node = FlexNode(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            adaptation_rate=0.01,
            max_connections=15
        )
        
        logger.info(f"Created FlexNode with ID: {flex_node.node_id}")
        return flex_node
    except Exception as e:
        logger.error(f"Failed to create FlexNode: {str(e)}")
        traceback.print_exc()
        return None

def integrate_with_central_node(flex_node: FlexNode, central_node: Any) -> bool:
    """
    Integrate the FlexNode with the central node.
    
    Args:
        flex_node: The FlexNode instance to integrate
        central_node: The central node instance
        
    Returns:
        True if integration was successful, False otherwise
    """
    try:
        # Set central node reference in FlexNode
        flex_node.set_central_node(central_node)
        
        # Register FlexNode with central node
        if hasattr(central_node, '_register_component'):
            central_node._register_component("FlexNode", flex_node)
            logger.info("Registered FlexNode with central node")
            
            # Also add to nodes dict if present
            if hasattr(central_node, 'nodes'):
                central_node.nodes["FlexNode"] = flex_node
                logger.info("Added FlexNode to central node's nodes dictionary")
            
            return True
        else:
            logger.error("Central node does not have _register_component method")
            return False
            
    except Exception as e:
        logger.error(f"Failed to integrate with central node: {str(e)}")
        return False

def connect_to_discovered_nodes(flex_node: FlexNode, nodes: Dict[str, Any]) -> int:
    """
    Connect the FlexNode to discovered nodes.
    
    Args:
        flex_node: The FlexNode instance
        nodes: Dictionary of discovered nodes
        
    Returns:
        Number of successful connections
    """
    successful_connections = 0
    
    # Define connection mapping for different node types
    connection_config = {
        "RSEN": {"type": "resonance", "weight": 0.8, "bidirectional": True},
        "Hybrid": {"type": "data", "weight": 0.7, "bidirectional": True},
        "Zero": {"type": "control", "weight": 0.6, "bidirectional": False},
        "Portal": {"type": "data", "weight": 0.9, "bidirectional": True},
        "Wormhole": {"type": "data", "weight": 0.9, "bidirectional": True},
        "Processor": {"type": "processing", "weight": 0.6, "bidirectional": False},
        "Neural": {"type": "neural", "weight": 0.7, "bidirectional": True},
        "Language": {"type": "language", "weight": 0.7, "bidirectional": False},
        "Lumina": {"type": "data", "weight": 0.8, "bidirectional": True},
        "Quantum": {"type": "quantum", "weight": 0.9, "bidirectional": True},
        "Gauge": {"type": "physics", "weight": 0.5, "bidirectional": False},
        "Fractal": {"type": "fractal", "weight": 0.8, "bidirectional": True},
        "Consciousness": {"type": "mirror", "weight": 0.9, "bidirectional": True},
        "Void": {"type": "void", "weight": 0.7, "bidirectional": True}
    }
    
    for node_id, node_instance in nodes.items():
        try:
            # Determine connection parameters based on node type
            connection_params = {"type": "default", "weight": 0.5, "bidirectional": False}
            
            for node_type, params in connection_config.items():
                if node_type in node_id:
                    connection_params = params
                    break
            
            # Connect to the node
            if flex_node.connect_to_node(
                node_id=node_id,
                node_instance=node_instance,
                connection_type=connection_params["type"],
                weight=connection_params["weight"],
                bidirectional=connection_params["bidirectional"]
            ):
                successful_connections += 1
                logger.info(f"Connected to {node_id} with {connection_params['type']} connection")
                
        except Exception as e:
            logger.error(f"Failed to connect to {node_id}: {str(e)}")
    
    return successful_connections

def save_integration_state(flex_node: FlexNode, integrated_nodes: Dict[str, Any]) -> str:
    """
    Save the integration state to a file.
    
    Args:
        flex_node: The FlexNode instance
        integrated_nodes: Dictionary of integrated nodes
        
    Returns:
        Path to the saved state file
    """
    state_file = "flex_node_integration_state.json"
    
    try:
        # Save FlexNode state
        flex_node_state_path = flex_node.save_state()
        
        # Create integration state
        integration_state = {
            "flex_node_id": flex_node.node_id,
            "flex_node_state_path": flex_node_state_path,
            "timestamp": time.time(),
            "integrated_nodes": list(integrated_nodes.keys()),
            "connection_count": len(flex_node.connections)
        }
        
        # Save to file
        with open(state_file, 'w') as f:
            json.dump(integration_state, f, indent=2)
            
        logger.info(f"Saved integration state to {state_file}")
        return state_file
        
    except Exception as e:
        logger.error(f"Failed to save integration state: {str(e)}")
        return None

def main():
    """Main integration function"""
    logger.info("Starting FlexNode integration")
    
    # Discover the central node
    central_node = discover_central_node()
    if not central_node:
        logger.error("Could not discover central node. Integration will be limited.")
    
    # Find node files
    node_files = find_node_files()
    
    # Discover nodes
    discovered_nodes = discover_nodes(node_files)
    logger.info(f"Discovered {len(discovered_nodes)} nodes")
    
    # Create FlexNode
    flex_node = create_flex_node()
    if not flex_node:
        logger.error("Failed to create FlexNode. Exiting.")
        return
    
    # Start the FlexNode
    flex_node.start()
    
    # Integrate with central node if available
    if central_node:
        if integrate_with_central_node(flex_node, central_node):
            logger.info("Successfully integrated with central node")
        else:
            logger.warning("Failed to integrate with central node")
    
    # Connect to discovered nodes
    connection_count = connect_to_discovered_nodes(flex_node, discovered_nodes)
    logger.info(f"Connected to {connection_count}/{len(discovered_nodes)} nodes")
    
    # Save integration state
    state_file = save_integration_state(flex_node, discovered_nodes)
    if state_file:
        logger.info(f"Integration state saved to {state_file}")
    
    logger.info("FlexNode integration completed successfully")
    logger.info(f"FlexNode is running with ID: {flex_node.node_id}")
    logger.info(f"FlexNode has {len(flex_node.connections)} active connections")
    
    # Keep the script running to maintain the FlexNode
    try:
        logger.info("Press Ctrl+C to exit")
        while True:
            # Periodically optimize connections
            optimization_result = flex_node.optimize_connections()
            if optimization_result["changes"]:
                logger.info(f"Optimized connections: {len(optimization_result['changes'])} changes")
                
            # Print metrics every 5 minutes
            if int(time.time()) % 300 == 0:  # Every 5 minutes
                metrics = flex_node.get_metrics()
                logger.info(f"FlexNode metrics: {metrics['processed_messages']} messages processed, "
                           f"{metrics['avg_processing_time']:.4f}s avg latency")
            
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Stopping FlexNode...")
        flex_node.stop()
        logger.info("FlexNode stopped")

if __name__ == "__main__":
    main() 