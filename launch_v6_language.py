#!/usr/bin/env python
"""
V6 Portal of Contradiction Launch Script with Language Module

This script launches the V6 Portal with language module integration.
It ensures all necessary directories are created and dependencies are available.
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("V6LanguageLauncher")

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

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import PySide6
        logger.info("PySide6 is available")
    except ImportError:
        logger.error("PySide6 is required but not installed!")
        print("Error: PySide6 is required but not installed.")
        print("Please install with: pip install PySide6")
        return False
        
    return True

def main():
    """Main entry point for the launcher"""
    print("Starting V6 Portal of Contradiction with Language Module...")
    
    # Ensure directories exist
    if not ensure_directories():
        return 1
        
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Import necessary modules
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt
        
        # Import V6 components
        from src.v6.socket_manager import V6SocketManager
        from src.v6.ui.main_widget import V6MainWidget
        from src.v6.symbolic_state_manager import V6SymbolicStateManager
        from src.v6.version_bridge_manager import VersionBridgeManager
        
        # Create application
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("V6 Portal of Contradiction - Language Edition")
        
        # Set application style
        app.setStyle("Fusion")
        
        # Create dark palette for the entire application
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(15, 25, 35))
        dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
        dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 35, 45))
        dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(35, 45, 55))
        dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(25, 35, 45))
        dark_palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(220, 220, 220))
        dark_palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
        dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(35, 45, 55))
        dark_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220, 220, 220))
        dark_palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(52, 152, 219))
        dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(35, 35, 35))
        
        # Apply the palette
        app.setPalette(dark_palette)
        
        # Create socket manager
        socket_manager = V6SocketManager(mock_mode=True)
        
        # Create symbolic state manager
        symbolic_state_manager = V6SymbolicStateManager(socket_manager)
        
        # Create bridge manager with proper config
        bridge_config = {
            "mock_mode": True,
            "enable_language_memory_v5_bridge": True,
            "debug": True
        }
        bridge_manager = VersionBridgeManager(bridge_config)
        
        # Set socket manager for bridge components to use
        for component_id, component in bridge_manager.bridge_components.items():
            if hasattr(component, "set_socket_manager"):
                component.set_socket_manager(socket_manager)
        
        # Start bridge manager
        try:
            bridge_manager.start()
            logger.info("Started bridge manager")
            
            # Special handling for language memory bridge
            language_memory_bridge = bridge_manager.get_component("language_memory_v5_bridge")
            if language_memory_bridge:
                logger.info("Language Memory V5 Bridge is available")
                if hasattr(language_memory_bridge, "set_socket_manager"):
                    language_memory_bridge.set_socket_manager(socket_manager)
                    logger.info("Set socket manager for Language Memory V5 Bridge")
            else:
                logger.warning("Language Memory V5 Bridge is not available")
        except Exception as e:
            logger.error(f"Error starting bridge manager: {e}")
            # Continue anyway - we can function without bridges
        
        # Create main window
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle("V6 Portal of Contradiction - Language Edition")
        
        # Set minimum size for the window
        main_window.setMinimumSize(1600, 1000)
        
        # Create central widget with layout
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create and add the main UI widget - only pass the socket_manager
        v6_widget = V6MainWidget(socket_manager)
        main_layout.addWidget(v6_widget)
        
        # Set central widget
        main_window.setCentralWidget(central_widget)
        
        # Center the window on the screen
        screen_geometry = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        window_geometry = main_window.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        main_window.move(window_geometry.topLeft())
        
        # Show the window
        main_window.show()
        
        # Log the initialization
        logger.info("V6 main window initialized with language module")
        
        # Start the application event loop
        return app.exec()
        
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error launching V6 Portal: {str(e)}")
        print(f"Error launching V6 Portal: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 