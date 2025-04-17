#!/usr/bin/env python
"""
V6 Portal of Contradiction Launcher

Main entry point for the V6 Portal of Contradiction system
with holographic UI and integrated backend systems.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("V6Launcher")

# Add language module-specific paths
language_module_path = Path(__file__).resolve().parent.parent / "language"
if language_module_path.exists() and str(language_module_path) not in sys.path:
    sys.path.append(str(language_module_path))
    logger.info(f"Added language module path: {language_module_path}")

try:
    # Import Qt compatibility layer from V5
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt
except ImportError:
    logger.warning("V5 Qt compatibility layer not found. Using direct PySide6 imports.")
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import Qt
    except ImportError:
        logger.error("PySide6 not found. Please install PySide6 or configure the V5 Qt compatibility layer.")
        sys.exit(1)

# Import backend components
try:
    from src.v6.socket_manager import V6SocketManager
except ImportError:
    logger.warning("V6 socket manager not found. Using mock socket manager.")
    # Create a mock socket manager for development/testing
    class V6SocketManager:
        def __init__(self, mock_mode=True):
            self.mock_mode = mock_mode
            self.connected = True
            self.handlers = {}
            logger.info("Created mock V6SocketManager")
        
        def send_message(self, message_type, data):
            logger.info(f"MOCK: Sending message: {message_type}, {data}")
            return True
        
        def register_handler(self, message_type, handler):
            if message_type not in self.handlers:
                self.handlers[message_type] = []
            self.handlers[message_type].append(handler)
            logger.info(f"MOCK: Registered handler for {message_type}")
            return True
            
        def emit(self, event, data=None):
            logger.info(f"MOCK: Emitting event: {event}")
            if not data:
                data = {}
            return True
            
        def is_connected(self):
            return self.connected

# Import V6 UI components
from src.v6.ui.main_widget import V6MainWidget
from src.v6.ui.panels.duality_processor_panel import DualityProcessorPanel
from src.v6.ui.panels.memory_reflection_panel import MemoryReflectionPanel

def center_window(window):
    """Center the window on the screen"""
    center_point = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()
    frame_geometry = window.frameGeometry()
    frame_geometry.moveCenter(center_point)
    window.move(frame_geometry.topLeft())

if __name__ == "__main__":
    # Create application
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("V6 Portal of Contradiction")
    
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
    socket_manager = V6SocketManager()
    
    # Create main window
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("V6 Portal of Contradiction")
    
    # Set minimum size for the window
    main_window.setMinimumSize(1600, 1000)
    
    # Create central widget with layout
    central_widget = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout(central_widget)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)
    
    # Create and add the main UI widget
    v6_widget = V6MainWidget(socket_manager)
    
    # Add our specialized panels to the tabs
    duality_panel = DualityProcessorPanel(socket_manager)
    memory_panel = MemoryReflectionPanel(socket_manager)
    
    # Access the panels in the main widget to add our new panels
    # These would be added to appropriate containers in a real implementation
    # For example, add to the Mirror Mode or Glyph Field tabs
    
    main_layout.addWidget(v6_widget)
    
    # Set central widget
    main_window.setCentralWidget(central_widget)
    
    # Center the window on the screen
    center_window(main_window)
    
    # Show the window
    main_window.show()
    
    # Log the initialization
    logger.info("V6 main window initialized")
    
    # Start the application event loop
    sys.exit(app.exec()) 