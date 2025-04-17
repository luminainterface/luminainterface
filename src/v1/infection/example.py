"""
Example script demonstrating the usage of the infection module in Version 1.
"""

import os
import logging
from pathlib import Path
from node_implementation import HybridNode, CentralNode
from infection_module import FileInfector, initialize_node_integration
import numpy as np

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
    
    # Print infection status
    logger.info("\nInfection Status:")
    for file_path, infections in infector.infected_files.items():
        logger.info(f"\nFile: {file_path}")
        logger.info(f"Number of infections: {len(infections)}")
    
    # Perform coordinated training with infected functionality
    logger.info("\nStarting coordinated training with infected functionality...")
    training_metrics = central_node.coordinate_training(n_rounds=5)
    
    # Print final status
    status = central_node.get_node_status()
    logger.info("\nFinal Node Status:")
    for node_id, node_status in status.items():
        logger.info(f"\nNode {node_id}:")
        logger.info(f"  Training: {node_status['is_training']}")
        logger.info(f"  Connected Nodes: {node_status['connected_nodes']}")
        logger.info(f"  Local Data Count: {node_status['local_data_count']}")
    
    # Print training metrics for the last round
    last_round = max(training_metrics.keys())
    logger.info("\nTraining Metrics for Last Round:")
    for node_id, metrics in training_metrics[last_round].items():
        logger.info(f"\nNode {node_id}:")
        logger.info(f"  Test Accuracy: {metrics['test_metrics']['accuracy']:.4f}")
        logger.info(f"  Test F1 Score: {metrics['test_metrics']['f1']:.4f}")

if __name__ == "__main__":
    main() 