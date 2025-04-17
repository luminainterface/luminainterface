#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUMINA V7 Qt Dashboard
=====================

Main script to launch the LUMINA V7 Qt Dashboard panels.
"""

import os
import sys
import time
import logging
import argparse
import threading
from pathlib import Path

# Create a Qt compatibility layer
def import_qt_modules(gui_framework="PyQt5"):
    """Import Qt modules based on framework choice"""
    if gui_framework.lower() == "pyside6":
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QStatusBar, QToolBar, QSlider
        )
        from PySide6.QtCore import Qt, QTimer, Slot
        from PySide6.QtGui import QIcon, QFont, QAction
        
        # Screen geometry handling is different in PySide6
        from PySide6.QtWidgets import QApplication
        
        def center_window(window):
            """Center window on screen for PySide6"""
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - window.width()) // 2
            y = (screen.height() - window.height()) // 2
            window.move(x, y)
            
        return {
            "QApplication": QApplication,
            "QMainWindow": QMainWindow, 
            "QWidget": QWidget,
            "QVBoxLayout": QVBoxLayout,
            "QHBoxLayout": QHBoxLayout,
            "QLabel": QLabel,
            "QStatusBar": QStatusBar,
            "QToolBar": QToolBar,
            "QAction": QAction,
            "Qt": Qt,
            "QTimer": QTimer,
            "Slot": Slot,
            "QIcon": QIcon,
            "QFont": QFont,
            "QSlider": QSlider,
            "center_window": center_window
        }
    else:  # Default to PyQt5
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QStatusBar, QDesktopWidget, QToolBar, QSlider
        )
        from PyQt5.QtCore import Qt, QTimer, pyqtSlot
        from PyQt5.QtGui import QIcon, QFont, QAction
        
        def center_window(window):
            """Center window on screen for PyQt5"""
            screen = QDesktopWidget().screenGeometry()
            x = (screen.width() - window.width()) // 2
            y = (screen.height() - window.height()) // 2
            window.move(x, y)
            
        return {
            "QApplication": QApplication,
            "QMainWindow": QMainWindow, 
            "QWidget": QWidget,
            "QVBoxLayout": QVBoxLayout,
            "QHBoxLayout": QHBoxLayout,
            "QLabel": QLabel,
            "QStatusBar": QStatusBar,
            "QToolBar": QToolBar,
            "QAction": QAction,
            "Qt": Qt,
            "QTimer": QTimer,
            "Slot": pyqtSlot,
            "QIcon": QIcon,
            "QFont": QFont,
            "QSlider": QSlider,
            "center_window": center_window
        }

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/qt_dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuminaQtDashboard")

# Parse arguments early to get GUI framework
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LUMINA V7 Qt Dashboard")
    
    parser.add_argument(
        "--db-path", 
        dest="db_path",
        default="data/neural_metrics.db",
        help="Path to metrics database"
    )
    
    parser.add_argument(
        "--v7-port", 
        dest="v7_port",
        type=int,
        default=5678,
        help="Port number for V7 system connection"
    )
    
    parser.add_argument(
        "--mock",
        dest="mock_mode",
        action="store_true",
        help="Use mock data"
    )
    
    parser.add_argument(
        "--gui-framework",
        choices=["PyQt5", "PySide6"],
        default="PyQt5",
        help="GUI framework to use (PyQt5 or PySide6)"
    )
    
    parser.add_argument(
        "--nn-weight",
        type=float,
        default=0.5,
        help="Neural network weight (0.0-1.0)"
    )
    
    parser.add_argument(
        "--llm-weight",
        type=float,
        default=0.5,
        help="Language model weight (0.0-1.0)"
    )
    
    return parser.parse_args()

# Parse args early to set up the Qt framework
args = parse_args()

# Import Qt modules based on framework choice
qt = import_qt_modules(args.gui_framework)
QMainWindow = qt["QMainWindow"]
QWidget = qt["QWidget"]
QVBoxLayout = qt["QVBoxLayout"]
QHBoxLayout = qt["QHBoxLayout"]
QLabel = qt["QLabel"]
QStatusBar = qt["QStatusBar"]
QToolBar = qt["QToolBar"]
QAction = qt["QAction"]
Qt = qt["Qt"]
QTimer = qt["QTimer"]
Slot = qt["Slot"]
QIcon = qt["QIcon"]
QFont = qt["QFont"]
QSlider = qt["QSlider"]

logger.info(f"Using {args.gui_framework} framework")

# Now import dashboard components after Qt setup
from src.visualization.panels.dashboard_panel_factory import create_dashboard

# Try to import the Language Dashboard Bridge 
try:
    from src.language.language_dashboard_bridge import get_language_dashboard_bridge
    HAS_LANGUAGE_BRIDGE = True
    logger.info("Language Dashboard Bridge module loaded")
except ImportError:
    HAS_LANGUAGE_BRIDGE = False
    logger.warning("Language Dashboard Bridge module not found, some features will be disabled")

class LuminaQtDashboard(QMainWindow):
    """LUMINA V7 Qt Dashboard main window"""
    
    def __init__(self, db_path="data/neural_metrics.db", v7_port=5678, mock_mode=False, nn_weight=0.5, llm_weight=0.5, gui_framework="PyQt5"):
        """
        Initialize the LUMINA V7 Qt Dashboard
        
        Args:
            db_path: Path to metrics database
            v7_port: Port number for V7 system connection
            mock_mode: Whether to use mock data
            nn_weight: Neural network weight (0.0-1.0)
            llm_weight: Language model weight (0.0-1.0)
            gui_framework: GUI framework to use ("PyQt5" or "PySide6")
        """
        super().__init__()
        
        self.db_path = db_path
        self.v7_port = v7_port
        self.mock_mode = mock_mode
        self.nn_weight = nn_weight
        self.llm_weight = llm_weight
        self.gui_framework = gui_framework
        
        # Initialize Language Dashboard Bridge if available
        self.language_bridge = None
        if HAS_LANGUAGE_BRIDGE:
            try:
                # Get bridge instance
                self.language_bridge = get_language_dashboard_bridge({
                    "db_path": db_path,
                    "mock_mode": mock_mode,
                    "llm_weight": llm_weight,
                    "nn_weight": nn_weight
                })
                logger.info("Dashboard connected to Language Dashboard Bridge")
            except Exception as e:
                logger.error(f"Error connecting to Language Dashboard Bridge: {e}")
        
        # Set up the UI
        self.setup_ui()
        
        # Set initial weights
        self.set_weights(nn_weight, llm_weight)
        
        # Log startup
        logger.info(f"LUMINA V7 Qt Dashboard started")
        logger.info(f"Database path: {self.db_path}")
        logger.info(f"V7 port: {self.v7_port}")
        logger.info(f"Mock mode: {self.mock_mode}")
        logger.info(f"NN weight: {self.nn_weight}")
        logger.info(f"LLM weight: {self.llm_weight}")
        
    def setup_ui(self):
        """Set up the dashboard UI"""
        # Set window title and size
        self.setWindowTitle("LUMINA V7 Unified Dashboard")
        self.resize(1200, 800)
        
        # Center window on screen
        qt["center_window"](self)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header with title
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add LUMINA title
        title_label = QLabel("LUMINA V7 Unified Dashboard")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        header_layout.addWidget(title_label)
        
        # Add version and mode info
        version_label = QLabel("v7.0.0.3")
        version_label.setFont(QFont("Arial", 10))
        version_label.setStyleSheet("color: gray;")
        header_layout.addWidget(version_label)
        
        # Add spacer to push items to the sides
        header_layout.addStretch(1)
        
        # Add mock mode indicator if enabled
        if self.mock_mode:
            mock_label = QLabel("MOCK MODE")
            mock_label.setFont(QFont("Arial", 10, QFont.Bold))
            mock_label.setStyleSheet("color: orange;")
            header_layout.addWidget(mock_label)
        
        # Add header to main layout
        main_layout.addWidget(header_widget)
        
        # Create and add dashboard
        self.dashboard = create_dashboard(
            parent=central_widget, 
            db_path=self.db_path,
            dashboard_type="full",
            gui_framework=self.gui_framework
        )
        main_layout.addWidget(self.dashboard, 1)  # 1 = stretch factor
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Create toolbar
        self.create_toolbar()
        
    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Add refresh action
        refresh_action = QAction("Refresh All", self)
        refresh_action.triggered.connect(self.refresh_all)
        toolbar.addAction(refresh_action)
        
        # Add toggle mock mode action
        mock_action = QAction("Toggle Mock Mode", self)
        mock_action.triggered.connect(self.toggle_mock_mode)
        toolbar.addAction(mock_action)
        
        # Add reset action
        reset_action = QAction("Reset Dashboard", self)
        reset_action.triggered.connect(self.reset_dashboard)
        toolbar.addAction(reset_action)

        # Add weight controls if language bridge is available
        if HAS_LANGUAGE_BRIDGE and self.language_bridge:
            toolbar.addSeparator()
            
            # Add NN weight label
            nn_label = QLabel("NN Weight: ")
            toolbar.addWidget(nn_label)
            
            # Add NN weight slider
            self.nn_slider = QSlider(Qt.Horizontal)
            self.nn_slider.setRange(0, 100)
            self.nn_slider.setValue(int(self.nn_weight * 100))
            self.nn_slider.setFixedWidth(100)
            self.nn_slider.valueChanged.connect(self.on_nn_weight_changed)
            toolbar.addWidget(self.nn_slider)
            
            # Add NN weight value display
            self.nn_value = QLabel(f"{self.nn_weight:.2f}")
            self.nn_value.setFixedWidth(40)
            toolbar.addWidget(self.nn_value)
            
            toolbar.addSeparator()
            
            # Add LLM weight label
            llm_label = QLabel("LLM Weight: ")
            toolbar.addWidget(llm_label)
            
            # Add LLM weight slider
            self.llm_slider = QSlider(Qt.Horizontal)
            self.llm_slider.setRange(0, 100)
            self.llm_slider.setValue(int(self.llm_weight * 100))
            self.llm_slider.setFixedWidth(100)
            self.llm_slider.valueChanged.connect(self.on_llm_weight_changed)
            toolbar.addWidget(self.llm_slider)
            
            # Add LLM weight value display
            self.llm_value = QLabel(f"{self.llm_weight:.2f}")
            self.llm_value.setFixedWidth(40)
            toolbar.addWidget(self.llm_value)
        
    @qt["Slot"]()
    def refresh_all(self):
        """Refresh all panels"""
        # TODO: Implement refresh for all panels
        self.statusBar().showMessage("Refreshing all panels...")
        
        # Reset after 2 seconds
        QTimer.singleShot(2000, lambda: self.statusBar().showMessage("Ready"))
        
    @qt["Slot"]()
    def toggle_mock_mode(self):
        """Toggle mock mode"""
        self.mock_mode = not self.mock_mode
        
        # Update status bar
        mode_str = "enabled" if self.mock_mode else "disabled"
        self.statusBar().showMessage(f"Mock mode {mode_str}")
        
        # TODO: Implement actual mode switching
        
        # Recreate the dashboard to apply the change
        self.reset_dashboard()
        
    @qt["Slot"]()
    def reset_dashboard(self):
        """Reset the dashboard"""
        # Remove the current dashboard
        self.centralWidget().layout().removeWidget(self.dashboard)
        
        # Create a new dashboard
        self.dashboard = create_dashboard(
            parent=self.centralWidget(), 
            db_path=self.db_path,
            dashboard_type="full",
            gui_framework=self.gui_framework
        )
        
        # Add the new dashboard
        self.centralWidget().layout().addWidget(self.dashboard, 1)
        
        # Update status bar
        self.statusBar().showMessage("Dashboard reset")
        
        # Reset after 2 seconds
        QTimer.singleShot(2000, lambda: self.statusBar().showMessage("Ready"))
        
    @qt["Slot"]()
    def on_nn_weight_changed(self, value):
        """Handle NN weight slider change"""
        nn_weight = value / 100.0
        self.nn_weight = nn_weight
        self.nn_value.setText(f"{nn_weight:.2f}")
        
        # Update language bridge
        if HAS_LANGUAGE_BRIDGE and self.language_bridge:
            self.language_bridge.set_nn_weight(nn_weight)
            
        # Update status bar
        self.statusBar().showMessage(f"Neural Network weight set to {nn_weight:.2f}")
        
        # Reset after 2 seconds
        QTimer.singleShot(2000, lambda: self.statusBar().showMessage("Ready"))
    
    @qt["Slot"]()
    def on_llm_weight_changed(self, value):
        """Handle LLM weight slider change"""
        llm_weight = value / 100.0
        self.llm_weight = llm_weight
        self.llm_value.setText(f"{llm_weight:.2f}")
        
        # Update language bridge
        if HAS_LANGUAGE_BRIDGE and self.language_bridge:
            self.language_bridge.set_llm_weight(llm_weight)
            
        # Update status bar
        self.statusBar().showMessage(f"Language Model weight set to {llm_weight:.2f}")
        
        # Reset after 2 seconds
        QTimer.singleShot(2000, lambda: self.statusBar().showMessage("Ready"))
    
    def set_weights(self, nn_weight, llm_weight):
        """Set neural network and language model weights"""
        # Update internal values
        self.nn_weight = nn_weight
        self.llm_weight = llm_weight
        
        # Update UI if sliders exist
        if hasattr(self, 'nn_slider'):
            self.nn_slider.setValue(int(nn_weight * 100))
            self.nn_value.setText(f"{nn_weight:.2f}")
            
        if hasattr(self, 'llm_slider'):
            self.llm_slider.setValue(int(llm_weight * 100))
            self.llm_value.setText(f"{llm_weight:.2f}")
        
        # Update language bridge
        if HAS_LANGUAGE_BRIDGE and self.language_bridge:
            self.language_bridge.set_nn_weight(nn_weight)
            self.language_bridge.set_llm_weight(llm_weight)
        
    def closeEvent(self, event):
        """Handle window close event"""
        # Shutdown language bridge if active
        if HAS_LANGUAGE_BRIDGE and self.language_bridge:
            try:
                self.language_bridge.stop()
                logger.info("Language Dashboard Bridge stopped")
            except Exception as e:
                logger.error(f"Error stopping Language Dashboard Bridge: {e}")
                
        # Log shutdown
        logger.info("LUMINA V7 Qt Dashboard shutting down")
        
        # Allow the close event to proceed
        event.accept()

def main():
    """Main entry point"""
    global args
    
    # Create QApplication
    app = qt["QApplication"](sys.argv)
    
    # Create main window
    main_window = LuminaQtDashboard(
        db_path=args.db_path,
        v7_port=args.v7_port,
        mock_mode=args.mock_mode,
        nn_weight=args.nn_weight,
        llm_weight=args.llm_weight,
        gui_framework=args.gui_framework
    )
    
    # Show main window
    main_window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 