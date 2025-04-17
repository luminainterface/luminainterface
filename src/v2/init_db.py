"""
Enhanced initialization script for the V2 Spiderweb Bridge database.
"""

import os
import logging
import asyncio
from pathlib import Path
from datetime import datetime
import numpy as np

from .database.database_manager import DatabaseManager
from .bridge_connector import BridgeConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/v2_spiderweb.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def init_database(db_path: str = "v2_spiderweb.db") -> bool:
    """
    Initialize the V2 database.
    
    Args:
        db_path: Path to the database file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        # Initialize database manager
        db = DatabaseManager(db_path)
        logger.info("Database initialized successfully")
        
        # Create bridge connector
        connector = BridgeConnector(db_path)
        logger.info("Bridge connector created successfully")
        
        # Add test quantum node
        quantum_node = {
            'node_id': 'quantum_test_1',
            'name': 'Quantum Test Node 1',
            'type': 'quantum',
            'status': 'ACTIVE',
            'version': 'v2',
            'quantum_enabled': True,
            'config': {
                'quantum_channels': 2,
                'decoherence_threshold': 0.1
            },
            'metadata': {
                'created_by': 'init_script',
                'purpose': 'quantum_testing'
            }
        }
        
        if connector.handle_node_creation(quantum_node):
            logger.info("Quantum test node created successfully")
            
            # Add quantum sync data
            quantum_sync = {
                'node_id': 'quantum_test_1',
                'state_vector': [1/np.sqrt(2), 1/np.sqrt(2)],  # Superposition state
                'entanglement_map': {},
                'coherence_level': 0.95,
                'decoherence_rate': 0.01,
                'measurement_basis': 'computational',
                'collapse_probability': 0.1
            }
            
            if connector.handle_quantum_sync(quantum_sync):
                logger.info("Quantum sync data added successfully")
        
        # Add test cosmic node
        cosmic_node = {
            'node_id': 'cosmic_test_1',
            'name': 'Cosmic Test Node 1',
            'type': 'cosmic',
            'status': 'ACTIVE',
            'version': 'v2',
            'cosmic_enabled': True,
            'config': {
                'dimensional_channels': 4,
                'resonance_threshold': 0.2
            },
            'metadata': {
                'created_by': 'init_script',
                'purpose': 'cosmic_testing'
            }
        }
        
        if connector.handle_node_creation(cosmic_node):
            logger.info("Cosmic test node created successfully")
            
            # Add cosmic sync data
            cosmic_sync = {
                'node_id': 'cosmic_test_1',
                'dimensional_signature': [1.0, 1.0, 1.0, 1.0],
                'resonance_pattern': {},
                'universal_phase': 0.5,
                'cosmic_frequency': 1.2,
                'stability_matrix': [[0.9, 0.1], [0.1, 0.9]],
                'harmonic_index': 0.95
            }
            
            if connector.handle_cosmic_sync(cosmic_sync):
                logger.info("Cosmic sync data added successfully")
        
        # Create quantum-cosmic relationship
        relationship = {
            'source_node_id': 'quantum_test_1',
            'target_node_id': 'cosmic_test_1',
            'relationship_type': 'quantum_cosmic_bridge',
            'quantum_entangled': True,
            'cosmic_resonant': True,
            'entanglement_strength': 0.8,
            'resonance_strength': 0.9,
            'phase_difference': 0.1,
            'metadata': {
                'bridge_type': 'bidirectional',
                'stability_score': 0.85
            }
        }
        
        if connector.handle_node_relationship(relationship):
            logger.info("Quantum-cosmic relationship created successfully")
        
        # Add some performance metrics
        performance_metrics = [
            {
                'metric_name': 'quantum_coherence',
                'value': 0.95,
                'component_id': 'quantum_test_1',
                'component_type': 'quantum_node',
                'aggregation_window': '1m',
                'threshold_value': 0.5,
                'alert_level': 'normal',
                'metadata': {'decoherence_rate': 0.01}
            },
            {
                'metric_name': 'cosmic_resonance',
                'value': 0.92,
                'component_id': 'cosmic_test_1',
                'component_type': 'cosmic_node',
                'aggregation_window': '1m',
                'threshold_value': 0.5,
                'alert_level': 'normal',
                'metadata': {'phase_stability': 0.98}
            }
        ]
        
        for metric in performance_metrics:
            if db.add_performance_metric(metric):
                logger.info(f"Added performance metric: {metric['metric_name']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

async def main():
    """Main entry point for database initialization."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Initialize database
        if await init_database():
            logger.info("V2 Spiderweb Bridge database initialization completed successfully")
        else:
            logger.error("Failed to initialize V2 Spiderweb Bridge database")
            
    except Exception as e:
        logger.error(f"Error in main initialization: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 