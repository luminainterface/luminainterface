#!/usr/bin/env python3
"""
V5 + Language Memory Integration Example

This script demonstrates how to integrate the Language Memory System with the
V5 Fractal Echo Visualization system.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/v5_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v5_integration")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Force PySide6 usage
os.environ["V5_QT_FRAMEWORK"] = "PySide6"

try:
    # Import required components
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtCompat
    from src.v5.frontend_socket_manager import FrontendSocketManager
    from src.language_memory_v5_bridge import LanguageMemoryV5Bridge
    from src.v5.ui.main_widget import V5MainWidget
    from src.memory_api_socket_provider import MemoryAPISocketProvider
    
    logger.info("Successfully imported all required components")
except ImportError as e:
    logger.error(f"Failed to import required components: {str(e)}")
    raise


def main():
    """Main integration function"""
    logger.info("Starting V5 + Language Memory Integration")
    
    # Initialize the Language Memory V5 Bridge
    bridge = LanguageMemoryV5Bridge()
    logger.info("Initialized Language Memory V5 Bridge")
    
    # Initialize the Memory API Socket Provider
    memory_provider = MemoryAPISocketProvider()
    logger.info("Initialized Memory API Socket Provider")
    
    # Initialize the Frontend Socket Manager
    socket_manager = FrontendSocketManager()
    logger.info("Initialized Frontend Socket Manager")
    
    # Register the Memory API Socket Provider with the Frontend Socket Manager
    if hasattr(socket_manager, 'register_plugin'):
        socket_manager.register_plugin(memory_provider)
        logger.info("Registered Memory API Socket Provider with Frontend Socket Manager")
    
    # Start plugin discovery (if available)
    if hasattr(socket_manager, 'start_plugin_discovery'):
        socket_manager.start_plugin_discovery()
        logger.info("Started plugin discovery")
    
    # Create Qt application
    app = QtCompat.get_application()
    app.setApplicationName("V5 + Language Memory Integration")
    
    # Create main window
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("V5 Fractal Echo Visualization with Language Memory")
    main_window.resize(1200, 800)
    
    # Create main widget
    main_widget = V5MainWidget(socket_manager)
    main_window.setCentralWidget(main_widget)
    
    # Show the window and run
    main_window.show()
    logger.info("Application window shown, starting event loop")
    
    # Start the Qt event loop
    return app.exec() if hasattr(app, 'exec') else app.exec_()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1) 