"""
Application Entry Point

This module provides the entry point for the application, initializing
the PySide6 integration and visualization system.
"""

import sys
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .components.pyside6_integration import pyside6_integration
from .components.visualization_system import visualization_system
from .main_window import create_main_window

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    try:
        # Create application
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("Neural Network Visualization")
        app.setApplicationVersion("1.0.0")
        
        # Create main window
        main_window = create_main_window()
        main_window.show()
        
        # Start application event loop
        sys.exit(app.exec())
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 