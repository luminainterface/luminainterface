"""
Initialization script for the V1 Spiderweb Bridge database.
"""

import os
import logging
import asyncio
from pathlib import Path
from datetime import datetime

from .database.database_manager import DatabaseManager
from .bridge_connector import BridgeConnector
from .data_flow_manager import DataFlowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/v1_spiderweb.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def init_database(db_path: str = "v1_spiderweb.db") -> bool:
    """
    Initialize the V1 database.
    
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
        
        # Create and start data flow manager
        flow_manager = DataFlowManager(connector)
        await flow_manager.start()
        logger.info("Data flow manager started successfully")
        
        # Add some initial test data
        test_node = {
            'id': 'test_node_1',
            'name': 'Test Node 1',
            'type': 'test',
            'status': 'ACTIVE',
            'version': 'v1',
            'config': {'test': True},
            'metadata': {'created_by': 'init_script'}
        }
        
        await flow_manager.handle_event('node_created', test_node)
        
        test_metric = {
            'metric_type': 'initialization',
            'value': 1.0,
            'node_id': 'test_node_1',
            'metadata': {'test': True}
        }
        
        await flow_manager.handle_event('metric_update', test_metric)
        
        # Test quantum sync
        test_quantum = {
            'node_id': 'test_node_1',
            'field_strength': 0.85,
            'entangled_nodes_count': 3,
            'phase': 0.5,
            'frequency': 1.2
        }
        
        await flow_manager.handle_event('quantum_sync', test_quantum)
        
        # Test cosmic sync
        test_cosmic = {
            'node_id': 'test_node_1',
            'field_strength': 0.92,
            'resonance': 0.78,
            'phase': 0.65,
            'frequency': 1.5
        }
        
        await flow_manager.handle_event('cosmic_sync', test_cosmic)
        
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
            logger.info("V1 Spiderweb Bridge database initialization completed successfully")
        else:
            logger.error("Failed to initialize V1 Spiderweb Bridge database")
            
    except Exception as e:
        logger.error(f"Error in main initialization: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 