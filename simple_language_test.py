#!/usr/bin/env python
"""
Simplified V6 Language Module Test

This script loads just the language module with minimal dependencies to test functionality.
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
        logging.FileHandler("simple_language_test.log")
    ]
)
logger = logging.getLogger("SimpleLangTest")

def ensure_directories():
    """Ensure all required directories exist"""
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
    
    return True

def main():
    """Main entry point for the simple test"""
    print("Starting Simplified Language Module Test...")
    
    # Ensure directories exist
    ensure_directories()
    
    try:
        # Import PySide6
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt
        
        # Create socket manager (minimal mock version)
        class MockSocketManager:
            def __init__(self):
                self.handlers = {}
                logger.info("Created mock socket manager")
            
            def register_handler(self, event, handler):
                if event not in self.handlers:
                    self.handlers[event] = []
                self.handlers[event].append(handler)
                logger.info(f"Registered handler for {event}")
                return True
            
            def emit(self, event, data=None):
                logger.info(f"Emitting event: {event}")
                if not data:
                    data = {}
                return True
        
        # Create application
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("Simple Language Module Test")
        
        # Set application style
        app.setStyle("Fusion")
        
        # Create dark palette
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(15, 25, 35))
        dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
        dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 35, 45))
        dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(35, 45, 55))
        dark_palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
        dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(35, 45, 55))
        dark_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220, 220, 220))
        app.setPalette(dark_palette)
        
        # Create socket manager
        socket_manager = MockSocketManager()
        
        try:
            # Import the language module panel
            from src.v6.ui.panels.language_module_panel import LanguageModulePanel
            logger.info("Successfully imported LanguageModulePanel")
            
            # Create main window
            main_window = QtWidgets.QMainWindow()
            main_window.setWindowTitle("Simple Language Module Test")
            
            # Create the language panel with socket manager
            panel = LanguageModulePanel(socket_manager)
            logger.info("Successfully created LanguageModulePanel")
            
            # Set as central widget
            main_window.setCentralWidget(panel)
            main_window.resize(1200, 800)
            main_window.show()
            
            logger.info("Application window shown")
            return app.exec()
            
        except ImportError as e:
            logger.error(f"Error importing LanguageModulePanel: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    return 1

if __name__ == "__main__":
    sys.exit(main()) 