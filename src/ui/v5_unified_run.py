#!/usr/bin/env python3
"""
Unified Launcher for the V5 Fractal Echo Visualization System

This launcher uses the Qt compatibility layer to support both PyQt5 and PySide6.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the Qt compatibility layer
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtCompat, Qt, QtGui
from src.v5.frontend_socket_manager import FrontendSocketManager
from src.v5.ui.main_widget import V5MainWidget

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class V5VisualizerApp:
    """Main application for the V5 Fractal Echo Visualization System."""
    
    def __init__(self):
        """Initialize the application."""
        # Parse command line arguments
        self.args = self._parse_args()
        
        # Set environment variable for Qt framework if specified
        if self.args.framework:
            os.environ["V5_QT_FRAMEWORK"] = self.args.framework
            logger.info(f"Using Qt framework: {self.args.framework}")
        
        # Create Qt application
        self.app = QtCompat.get_application()
        self.app.setApplicationName("V5 Fractal Echo Visualizer")
        
        # Load stylesheets
        self._load_styles()
        
        # Initialize socket manager
        self.socket_manager = FrontendSocketManager()
        
        # Create main window
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setWindowTitle("V5 Fractal Echo Visualizer")
        self.main_window.resize(1200, 800)
        
        # Create main widget
        self.main_widget = V5MainWidget(self.socket_manager)
        self.main_window.setCentralWidget(self.main_widget)
        
        # Set up menu bar
        self._setup_menu()
        
        # Load plugins
        self._load_plugins()
        
        # Register signal handlers
        self._register_signals()
        
    def _parse_args(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="V5 Fractal Echo Visualization System")
        parser.add_argument(
            "--framework", 
            choices=["PySide6", "PyQt5"], 
            help="Qt framework to use (default: auto-detect)"
        )
        parser.add_argument(
            "--mock", 
            action="store_true", 
            help="Use mock data for visualization"
        )
        parser.add_argument(
            "--log-level", 
            choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
            default="INFO",
            help="Set logging level"
        )
        return parser.parse_args()
    
    def _load_styles(self):
        """Load application stylesheets."""
        # Try to load stylesheet file
        stylesheet_path = project_root / "src" / "ui" / "styles" / "v5_dark.qss"
        
        if stylesheet_path.exists():
            with open(stylesheet_path, "r") as f:
                self.app.setStyleSheet(f.read())
            logger.info(f"Loaded stylesheet: {stylesheet_path}")
        else:
            # Apply default dark theme
            palette = QtGui.QPalette() if QtCompat.is_pyside6() else QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
            palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
            palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142, 45, 197).lighter())
            palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
            self.app.setPalette(palette)
            logger.info("Applied default dark theme")
    
    def _setup_menu(self):
        """Set up the application menu."""
        menubar = self.main_window.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        export_action = QtWidgets.QAction("&Export Visualization", self.main_window)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_visualization)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction("&Exit", self.main_window)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.app.quit)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        views = {
            "Fractal Pattern": "fractal_pattern",
            "Node Consciousness": "node_consciousness",
            "Network Visualization": "network",
            "Memory Synthesis": "memory_synthesis",
            "Metrics": "metrics"
        }
        
        for view_name, view_id in views.items():
            action = QtWidgets.QAction(view_name, self.main_window)
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(lambda checked, view=view_id: self.main_widget.toggle_view(view))
            view_menu.addAction(action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        refresh_action = QtWidgets.QAction("&Refresh All", self.main_window)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.main_widget.update_all)
        tools_menu.addAction(refresh_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QtWidgets.QAction("&About", self.main_window)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _load_plugins(self):
        """Load and initialize plugins."""
        # If mock mode is enabled, set environment variable
        if self.args.mock:
            os.environ["V5_USE_MOCK_DATA"] = "1"
            logger.info("Mock mode enabled")
        
        # Initialize plugins in socket manager
        self.socket_manager.initialize_plugins()
        
        logger.info(f"Loaded {len(self.socket_manager.plugins)} plugins")
        for plugin in self.socket_manager.plugins:
            logger.info(f"  - {plugin.plugin_id} ({plugin.plugin_type})")
    
    def _register_signals(self):
        """Register signal handlers for clean shutdown."""
        # Handle application exit
        self.app.aboutToQuit.connect(self._cleanup)
    
    def _cleanup(self):
        """Clean up resources before exiting."""
        logger.info("Cleaning up resources...")
        self.main_widget.cleanup()
        self.socket_manager.cleanup()
    
    def _export_visualization(self):
        """Export the current visualization to an image file."""
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self.main_window, 
            "Export Visualization",
            os.path.expanduser("~/v5_visualization.png"),
            "Images (*.png *.jpg)"
        )
        
        if file_path:
            pixmap = QtGui.QPixmap(self.main_window.size())
            self.main_window.render(pixmap)
            pixmap.save(file_path)
            logger.info(f"Visualization exported to {file_path}")
    
    def _show_about(self):
        """Show the about dialog."""
        QtWidgets.QMessageBox.about(
            self.main_window,
            "About V5 Fractal Echo Visualizer",
            f"""
            <h3>V5 Fractal Echo Visualizer</h3>
            <p>Version: 5.0.0</p>
            <p>Framework: {QtCompat.get_framework_name()}</p>
            <p>A visualization system for neural network patterns and consciousness metrics.</p>
            """
        )
    
    def run(self):
        """Run the application."""
        self.main_window.show()
        return self.app.exec()

def main():
    """Application entry point."""
    # Set up logging based on arguments
    app = V5VisualizerApp()
    sys.exit(app.run())

if __name__ == "__main__":
    main() 