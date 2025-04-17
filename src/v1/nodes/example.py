"""
Example script demonstrating the usage of HybridNode and CentralNode in Version 1.
"""

import numpy as np
from node_implementation import HybridNode, CentralNode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_node_data(n_samples: int = 1000, n_features: int = 10, noise_level: float = 0.1) -> tuple:
    """
    Generate sample data for a node with some noise to simulate different data distributions.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        noise_level: Level of noise to add to the data
        
    Returns:
        Tuple of (X, y) where X is the input data and y is the target
    """
    # Generate random input data
    X = np.random.randn(n_samples, n_features)
    
    # Add some noise to create different distributions
    noise = np.random.normal(0, noise_level, size=X.shape)
    X = X + noise
    
    # Generate target values (binary classification)
    # Using a simple rule: if sum of features > 0, class 1, else class 0
    y = (np.sum(X, axis=1) > 0).astype(int).reshape(-1, 1)
    
    return X, y

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
        
        # Generate and store unique data for each node
        X, y = generate_node_data(noise_level=0.1 * (i + 1))
        node.store_local_data("initial_data", X, y)
        
        # Connect nodes in a ring topology
        if i > 0:
            node.connect_to_node(nodes[i-1].node_id)
    
    # Connect the last node to the first to complete the ring
    nodes[-1].connect_to_node(nodes[0].node_id)
    
    # Perform coordinated training
    logger.info("Starting coordinated training...")
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