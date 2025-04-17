"""
Main Application Class for Lumina Frontend
=========================================

This module contains the main application class that initializes and manages
the Lumina Frontend system.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Signal

from ..ui.main_window import MainWindow
from ..utils.config import ConfigManager
from ..utils.logger import setup_logging

class LuminaApplication(QObject):
    """Main application class for Lumina Frontend."""
    
    # Signals
    initialization_complete = Signal()
    shutdown_requested = Signal()
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        
        # Initialize configuration
        self.config = ConfigManager(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config)
        
        # Initialize Qt application
        self.qt_app = QApplication(sys.argv)
        
        # Create main window
        self.main_window = MainWindow(self.config)
        
        # Connect signals
        self.main_window.shutdown_requested.connect(self.shutdown)
        
    def initialize(self) -> bool:
        """Initialize the application components."""
        try:
            self.logger.info("Initializing Lumina Frontend...")
            
            # Initialize components
            self.main_window.initialize()
            
            self.logger.info("Initialization complete")
            self.initialization_complete.emit()
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def run(self) -> int:
        """Run the application main loop."""
        try:
            self.logger.info("Starting Lumina Frontend...")
            
            # Show main window
            self.main_window.show()
            
            # Start event loop
            return self.qt_app.exec()
            
        except Exception as e:
            self.logger.error(f"Runtime error: {str(e)}")
            return 1
    
    def shutdown(self):
        """Shutdown the application."""
        self.logger.info("Shutting down Lumina Frontend...")
        self.shutdown_requested.emit()
        self.qt_app.quit() 