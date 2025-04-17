"""
Spiderweb V3 Example Usage
Demonstrates the key features of the V3 system including state management,
caching, and metrics tracking.
"""

import logging
import time
from typing import Dict
from ..core.spiderweb_manager_v3 import SpiderwebManagerV3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_quantum_state(phase: float, amplitudes: list) -> Dict:
    """Create a quantum state with given parameters."""
    return {
        'quantum_state': amplitudes,
        'phase': phase,
        'timestamp': time.time()
    }

def create_cosmic_state(dimensions: list) -> Dict:
    """Create a cosmic state with given dimensional signature."""
    return {
        'cosmic_state': 'active',
        'dimensional_signature': ','.join(dimensions),
        'timestamp': time.time()
    }

def main():
    """Main example function demonstrating Spiderweb V3 usage."""
    try:
        # Initialize the Spiderweb V3 manager
        with SpiderwebManagerV3(db_path="example_v3.db", max_cache_mb=50) as manager:
            # Create a quantum node state
            quantum_state_data = create_quantum_state(
                phase=0.5,
                amplitudes=[1/2**0.5, 1/2**0.5]
            )
            
            quantum_state_id = manager.create_node_state(
                node_id="quantum_node_1",
                state_type="quantum",
                state_data=quantum_state_data
            )
            logger.info(f"Created quantum state with ID: {quantum_state_id}")

            # Create a cosmic node state
            cosmic_state_data = create_cosmic_state(
                dimensions=['x', 'y', 'z', 'time']
            )
            
            cosmic_state_id = manager.create_node_state(
                node_id="cosmic_node_1",
                state_type="cosmic",
                state_data=cosmic_state_data
            )
            logger.info(f"Created cosmic state with ID: {cosmic_state_id}")

            # Perform quantum state transition
            new_quantum_state = create_quantum_state(
                phase=0.7,
                amplitudes=[0.8660254037844386, 0.5]  # cos(30°), sin(30°)
            )
            
            transition_id = manager.transition_node_state(
                node_id="quantum_node_1",
                source_state_id=quantum_state_id,
                target_state_data=new_quantum_state,
                transition_type="quantum"
            )
            logger.info(f"Performed quantum transition with ID: {transition_id}")

            # Perform cosmic state transition
            new_cosmic_state = create_cosmic_state(
                dimensions=['x', 'y', 'z', 'time', 'consciousness']
            )
            
            transition_id = manager.transition_node_state(
                node_id="cosmic_node_1",
                source_state_id=cosmic_state_id,
                target_state_data=new_cosmic_state,
                transition_type="cosmic"
            )
            logger.info(f"Performed cosmic transition with ID: {transition_id}")

            # Get state history for quantum node
            quantum_history = manager.get_node_state_history("quantum_node_1")
            logger.info("Quantum node state history:")
            for state in quantum_history:
                logger.info(f"  State ID: {state['id']}")
                logger.info(f"  Type: {state['state_type']}")
                logger.info(f"  Data: {state['state_data']}")
                if 'transition_from' in state:
                    logger.info(f"  Transition: {state['transition_from']}")
                logger.info("---")

            # Get system metrics
            metrics = manager.get_system_metrics()
            logger.info("\nSystem Metrics:")
            logger.info(f"Cache utilization: {metrics['cache']['utilization_percent']:.2f}%")
            logger.info(f"Cache hit rate: {metrics['cache']['hit_rate']:.2f}%")
            
            if 'avg_state_creation' in metrics:
                logger.info(f"Average state creation rate: {metrics['avg_state_creation']:.2f}")
            if 'avg_state_transition' in metrics:
                logger.info(f"Average state transition energy: {metrics['avg_state_transition']:.2f}")

    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise

if __name__ == "__main__":
    main() 