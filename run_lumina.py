#!/usr/bin/env python3
"""
LUMINA v7.5 Launcher
Sets up the environment and launches the GUI
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from src.central_node import CentralNode

def setup_logging():
    """Setup logging configuration for LUMINA"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"lumina_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("LUMINA")

def main():
    """Main entry point for LUMINA"""
    logger = setup_logging()
    logger.info("Starting LUMINA v7.5 in debug mode...")
    
    try:
        # Create and initialize central node
        central_node = CentralNode()
        
        # Print system information
        logger.info("\nSystem Information:")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
        
        # Print available components
        logger.info("\nAvailable Components:")
        for category, components in central_node.list_available_components().items():
            logger.info(f"\n{category.capitalize()}:")
            for component in components:
                logger.info(f"  - {component}")
        
        # Print system status
        logger.info("\nSystem Status:")
        for key, value in central_node.get_system_status().items():
            logger.info(f"{key}: {value}")
        
        # Test the flow pipeline with sample data
        logger.info("\nTesting Flow Pipeline:")
        input_data = {
            'symbol': 'infinity',
            'emotion': 'wonder',
            'breath': 'deep',
            'paradox': 'existence'
        }
        output = central_node.process_complete_flow(input_data)
        
        logger.info("\nOutput:")
        for key, value in output.items():
            logger.info(f"  - {key}: {value}")
            
        logger.info("\nLUMINA initialization complete.")
        return 0
        
    except Exception as e:
        logger.error(f"Error during LUMINA initialization: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 