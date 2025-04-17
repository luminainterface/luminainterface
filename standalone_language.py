#!/usr/bin/env python
"""
Standalone Language Module Panel

This script runs only the Language Module Panel without any bridge components.
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
        logging.FileHandler("standalone_language.log")
    ]
)
logger = logging.getLogger("StandaloneLang")

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
    """Main entry point for the standalone language module"""
    print("Starting Standalone Language Module...")
    
    # Ensure directories exist
    ensure_directories()
    
    # Check for PySide6
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt
        logger.info("PySide6 is available")
    except ImportError:
        logger.error("PySide6 is required but not installed!")
        print("Error: PySide6 is required but not installed.")
        print("Please install with: pip install PySide6")
        return 1
    
    try:
        # Create a socket manager
        class StandaloneSocketManager:
            def __init__(self):
                self.handlers = {}
                logger.info("Created standalone socket manager")
            
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
                
                # If there are handlers for this event, call them
                if event in self.handlers:
                    for handler in self.handlers[event]:
                        try:
                            handler(data)
                        except Exception as e:
                            logger.error(f"Error in handler for {event}: {e}")
                
                return True
        
        # Create a basic app
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("Standalone Language Module")
        
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
        dark_palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(52, 152, 219))
        dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(35, 35, 35))
        app.setPalette(dark_palette)
        
        # Create socket manager
        socket_manager = StandaloneSocketManager()
        
        # Import language module panel
        try:
            from src.v6.ui.panels.language_module_panel import LanguageModulePanel
            logger.info("Successfully imported Language Module Panel")
            
            # Create main window
            main_window = QtWidgets.QMainWindow()
            main_window.setWindowTitle("Standalone Language Module")
            
            # Create language panel
            language_panel = LanguageModulePanel(socket_manager)
            logger.info("Successfully created Language Module Panel")
            
            # Set window properties
            main_window.setCentralWidget(language_panel)
            main_window.resize(1200, 800)
            
            # Center the window
            screen_geometry = QtGui.QGuiApplication.primaryScreen().availableGeometry()
            window_geometry = main_window.frameGeometry()
            window_geometry.moveCenter(screen_geometry.center())
            main_window.move(window_geometry.topLeft())
            
            # Show window
            main_window.show()
            logger.info("Window displayed")
            
            # Run application
            return app.exec()
            
        except ImportError as e:
            logger.error(f"Error importing Language Module Panel: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        logger.error(f"General error: {e}")
        import traceback
        traceback.print_exc()
    
    return 1

if __name__ == "__main__":
    sys.exit(main()) 