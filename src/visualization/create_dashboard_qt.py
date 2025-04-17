#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUMINA V7 Qt Dashboard
=====================

Main script to create and launch the LUMINA V7 Qt Dashboard.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Create Qt compatibility layer - handles both PyQt5 and PySide6
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QStatusBar, QTabWidget, QSplitter, QDockWidget, QToolBar,
        QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
        QFrame, QSlider
    )
    from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize
    from PySide6.QtGui import QFont, QIcon, QAction
    QT_FRAMEWORK = "PySide6"
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QStatusBar, QTabWidget, QSplitter, QDockWidget, QToolBar,
            QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
            QFrame, QSlider
        )
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, pyqtSlot as Slot, QSize
        from PyQt5.QtGui import QFont, QIcon, QAction
        QT_FRAMEWORK = "PyQt5"
    except ImportError:
        raise ImportError("Neither PySide6 nor PyQt5 is installed. Please install at least one of them.")

# Import dashboard panels
from src.visualization.panels.base_panel import BasePanel
from src.visualization.panels.neural_activity_panel import NeuralActivityPanel
from src.visualization.panels.language_processing_panel import LanguageProcessingPanel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuminaDashboard")

class LuminaDashboard(QMainWindow):
    """Main dashboard window for LUMINA V7"""
    
    def __init__(self, db_path=None, v7_port=5678, mock_mode=False, nn_weight=0.5, llm_weight=0.5, gui_framework=None):
        """
        Initialize the dashboard
        
        Args:
            db_path: Path to metrics database
            v7_port: Port for V7 connection
            mock_mode: Whether to run in mock mode
            nn_weight: Neural Network weight
            llm_weight: Language Model weight
            gui_framework: GUI framework to use (PyQt5 or PySide6)
        """
        super().__init__()
        
        # Store parameters
        self.db_path = db_path or "data/neural_metrics.db"
        self.v7_port = v7_port
        self.mock_mode = mock_mode
        self.nn_weight = nn_weight
        self.llm_weight = llm_weight
        self.gui_framework = gui_framework or QT_FRAMEWORK
        
        # Set up the UI
        self._setup_ui()
        
        # Set window properties
        self.setWindowTitle("LUMINA V7 Unified Dashboard")
        self.resize(1200, 800)
        self.setMinimumSize(800, 600)
        
        # Center window on screen
        self._center_window()
        
        # Create dashboard components
        self._create_panels()
        
        # Set up timer for status updates
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(5000)  # Update every 5 seconds
        
        # Log startup
        logger.info("LUMINA V7 Dashboard started")
        logger.info(f"DB Path: {self.db_path}")
        logger.info(f"V7 Port: {self.v7_port}")
        logger.info(f"Mock Mode: {self.mock_mode}")
        logger.info(f"NN Weight: {self.nn_weight}")
        logger.info(f"LLM Weight: {self.llm_weight}")
        logger.info(f"GUI Framework: {self.gui_framework}")
    
    def _center_window(self):
        """Center the window on the screen"""
        if hasattr(QApplication, "primaryScreen"):
            # PySide6 method
            screen_geometry = QApplication.primaryScreen().availableGeometry()
        else:
            # PyQt5 method
            from PyQt5.QtWidgets import QDesktopWidget
            screen_geometry = QDesktopWidget().availableGeometry()
            
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
    
    def _setup_ui(self):
        """Set up the main UI components"""
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Header with title and info
        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        
        # LUMINA title
        self.title_label = QLabel("LUMINA V7")
        self.title_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.header_layout.addWidget(self.title_label)
        
        # Version
        self.version_label = QLabel("v7.0.0.3")
        self.version_label.setFont(QFont("Arial", 10))
        self.version_label.setStyleSheet("color: gray;")
        self.header_layout.addWidget(self.version_label)
        
        # Add spacer to push connection status to the right
        self.header_layout.addStretch(1)
        
        # Connection status
        self.connection_frame = QFrame()
        self.connection_frame.setFrameShape(QFrame.StyledPanel)
        self.connection_frame.setFixedSize(16, 16)
        self.connection_frame.setStyleSheet("background-color: red; border-radius: 8px;")
        self.header_layout.addWidget(self.connection_frame)
        
        self.connection_label = QLabel("Disconnected")
        self.connection_label.setStyleSheet("color: red;")
        self.header_layout.addWidget(self.connection_label)
        
        # Add header to main layout
        self.main_layout.addWidget(self.header_widget)
        
        # Add horizontal line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(line)
        
        # Create tab widget for panels
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget, 1)  # 1 = stretch factor
        
        # Create status bar
        self.statusBar().showMessage("Initializing dashboard...")
        
        # Create toolbar
        self._create_toolbar()
    
    def _create_toolbar(self):
        """Create the main toolbar"""
        self.toolbar = QToolBar("Dashboard Controls")
        self.toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(self.toolbar)
        
        # Refresh action
        refresh_action = QAction("Refresh All", self)
        refresh_action.triggered.connect(self._refresh_all_panels)
        self.toolbar.addAction(refresh_action)
        
        # Mock mode toggle
        self.mock_action = QAction("Mock Mode", self)
        self.mock_action.setCheckable(True)
        self.mock_action.setChecked(self.mock_mode)
        self.mock_action.triggered.connect(self._toggle_mock_mode)
        self.toolbar.addAction(self.mock_action)
        
        self.toolbar.addSeparator()
        
        # Neural Network weight
        nn_label = QLabel("NN Weight: ")
        self.toolbar.addWidget(nn_label)
        
        self.nn_slider = QSlider(Qt.Horizontal)
        self.nn_slider.setRange(0, 100)
        self.nn_slider.setValue(int(self.nn_weight * 100))
        self.nn_slider.setFixedWidth(100)
        self.nn_slider.valueChanged.connect(self._on_nn_weight_changed)
        self.toolbar.addWidget(self.nn_slider)
        
        self.nn_value_label = QLabel(f"{self.nn_weight:.2f}")
        self.nn_value_label.setFixedWidth(40)
        self.toolbar.addWidget(self.nn_value_label)
        
        self.toolbar.addSeparator()
        
        # LLM weight
        llm_label = QLabel("LLM Weight: ")
        self.toolbar.addWidget(llm_label)
        
        self.llm_slider = QSlider(Qt.Horizontal)
        self.llm_slider.setRange(0, 100)
        self.llm_slider.setValue(int(self.llm_weight * 100))
        self.llm_slider.setFixedWidth(100)
        self.llm_slider.valueChanged.connect(self._on_llm_weight_changed)
        self.toolbar.addWidget(self.llm_slider)
        
        self.llm_value_label = QLabel(f"{self.llm_weight:.2f}")
        self.llm_value_label.setFixedWidth(40)
        self.toolbar.addWidget(self.llm_value_label)
    
    def _create_panels(self):
        """Create and add all dashboard panels"""
        # Dictionary to hold all panels for easier access
        self.panels = {}
        
        # Create Neural Activity panel
        self.panels['neural_activity'] = NeuralActivityPanel(
            parent=self,
            db_path=self.db_path,
            refresh_rate=1000,  # Update every 1 second
            active=True,
            gui_framework=self.gui_framework
        )
        self.tab_widget.addTab(self.panels['neural_activity'], "Neural Activity")
        
        # Create Language Processing panel
        self.panels['language_processing'] = LanguageProcessingPanel(
            parent=self,
            db_path=self.db_path,
            refresh_rate=2000,  # Update every 2 seconds
            active=True,
            gui_framework=self.gui_framework
        )
        self.tab_widget.addTab(self.panels['language_processing'], "Language Processing")
        
        # Set mock mode for all panels
        if self.mock_mode:
            self._set_mock_mode(True)
    
    def _refresh_all_panels(self):
        """Force refresh on all panels"""
        for panel in self.panels.values():
            if hasattr(panel, 'refresh_data'):
                panel.refresh_data()
        
        self.statusBar().showMessage("All panels refreshed")
        # Reset message after 2 seconds
        QTimer.singleShot(2000, lambda: self.statusBar().showMessage("Ready"))
    
    def _toggle_mock_mode(self, checked):
        """Toggle mock mode for all panels"""
        self.mock_mode = checked
        self._set_mock_mode(checked)
        
        status = "enabled" if checked else "disabled"
        self.statusBar().showMessage(f"Mock mode {status}")
        # Reset message after 2 seconds
        QTimer.singleShot(2000, lambda: self.statusBar().showMessage("Ready"))
    
    def _set_mock_mode(self, enabled):
        """Set mock mode for all panels"""
        for panel in self.panels.values():
            if hasattr(panel, 'set_mock_mode'):
                panel.set_mock_mode(enabled)
    
    def _on_nn_weight_changed(self, value):
        """Handle Neural Network weight slider changes"""
        self.nn_weight = value / 100.0
        self.nn_value_label.setText(f"{self.nn_weight:.2f}")
        
        # TODO: Send weight update to V7 system
        
        self.statusBar().showMessage(f"Neural Network weight set to {self.nn_weight:.2f}")
        # Reset message after 2 seconds
        QTimer.singleShot(2000, lambda: self.statusBar().showMessage("Ready"))
    
    def _on_llm_weight_changed(self, value):
        """Handle LLM weight slider changes"""
        self.llm_weight = value / 100.0
        self.llm_value_label.setText(f"{self.llm_weight:.2f}")
        
        # TODO: Send weight update to V7 system
        
        self.statusBar().showMessage(f"Language Model weight set to {self.llm_weight:.2f}")
        # Reset message after 2 seconds
        QTimer.singleShot(2000, lambda: self.statusBar().showMessage("Ready"))
    
    def _update_status(self):
        """Update connection status"""
        # TODO: Implement actual V7 connection check
        
        # For now just toggle between connected/disconnected for visual feedback
        # This will be replaced with actual connection status checks in the future
        if self.mock_mode:
            # In mock mode, always show as disconnected
            self.connection_frame.setStyleSheet("background-color: orange; border-radius: 8px;")
            self.connection_label.setText("Mock Mode")
            self.connection_label.setStyleSheet("color: orange;")
        else:
            # Toggle connection status (simulated for now)
            current_color = self.connection_frame.styleSheet()
            if "red" in current_color:
                self.connection_frame.setStyleSheet("background-color: green; border-radius: 8px;")
                self.connection_label.setText("Connected")
                self.connection_label.setStyleSheet("color: green;")
            else:
                self.connection_frame.setStyleSheet("background-color: red; border-radius: 8px;")
                self.connection_label.setText("Disconnected")
                self.connection_label.setStyleSheet("color: red;")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up panel resources
        for panel in self.panels.values():
            if hasattr(panel, 'cleanup'):
                panel.cleanup()
        
        # Stop timers
        if self.status_timer:
            self.status_timer.stop()
        
        # Log shutdown
        logger.info("LUMINA V7 Dashboard shutting down")
        
        # Accept the close event
        event.accept()

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
        help="Use mock data instead of real connections"
    )
    
    parser.add_argument(
        "--gui-framework",
        choices=["PyQt5", "PySide6"],
        default=QT_FRAMEWORK,
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

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Set application style
    if args.gui_framework == "PySide6":
        app.setStyle("Fusion")
    else:
        # PyQt5
    app.setStyle("Fusion")
    
    # Create and show dashboard
    dashboard = LuminaDashboard(
        db_path=args.db_path,
        v7_port=args.v7_port,
        mock_mode=args.mock_mode,
        nn_weight=args.nn_weight,
        llm_weight=args.llm_weight,
        gui_framework=args.gui_framework
    )
    dashboard.show()
    
    # Start the event loop
    sys.exit(app.exec_() if QT_FRAMEWORK == "PyQt5" else app.exec())

if __name__ == "__main__":
    main() 