#!/usr/bin/env python3
"""
Central Node Monitor Launcher
Launches the PySide6 version of the central node monitor
"""

import sys
import logging
from PySide6.QtWidgets import QApplication
from src.central_node_monitor import CentralNodeMonitor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Central Node Monitor")
    
    try:
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle('Fusion')
        
        # Create and show the monitor
        monitor = CentralNodeMonitor()
        monitor.show()
        
        # Start the application event loop
        exit_code = app.exec()
        
        logger.info("Central Node Monitor closed")
        return exit_code
        
    except Exception as e:
        logger.error(f"Error running monitor: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 