#!/usr/bin/env python
"""
V6 Language Module Minimal Test Script

This script loads just the V6MainWidget with a Language module to test it works.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_v6_language.log")
    ]
)
logger = logging.getLogger("V6Test")

def main():
    """Main test function"""
    try:
        # Import PySide6
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt
        
        # Import socket manager
        from src.v6.socket_manager import V6SocketManager
        
        # Ensure directory structure exists
        directories = [
            "data/memory/language_memory",
            "data/neural_linguistic",
            "data/v10",
            "data/central_language"
        ]
        
        for directory in directories:
            path = Path(directory)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
        
        # Create application
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("V6 Language Module Test")
        
        # Create socket manager
        socket_manager = V6SocketManager(mock_mode=True)
        logger.info("Socket manager created in mock mode")
        
        try:
            # Try to import V6 main widget
            from src.v6.ui.main_widget import V6MainWidget
            logger.info("Successfully imported V6MainWidget")
            
            # Create main window
            main_window = QtWidgets.QMainWindow()
            main_window.setWindowTitle("V6 Language Module Test")
            
            # Create the main widget
            try:
                main_widget = V6MainWidget(socket_manager)
                logger.info("Successfully created V6MainWidget")
                
                # Create central widget
                main_window.setCentralWidget(main_widget)
                main_window.resize(1200, 800)
                main_window.show()
                
                logger.info("Application window shown")
                return app.exec()
            except Exception as e:
                logger.error(f"Error creating V6MainWidget: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        except ImportError:
            logger.error("Could not import V6MainWidget")
            
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    return 1

if __name__ == "__main__":
    sys.exit(main()) 