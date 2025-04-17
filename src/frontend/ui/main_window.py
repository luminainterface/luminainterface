"""
Main Window

This module provides the main window for the application, integrating
the visualization system and PySide6 components.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QDockWidget, QStatusBar, QMenuBar, QMenu,
    QAction, QToolBar, QSplitter
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon

from .components.pyside6_integration import pyside6_integration
from .components.visualization_system import visualization_system

# Configure logging
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Neural Network Visualization")
        self.resize(1280, 720)
        
        # Initialize UI
        self._initialize_ui()
        
        # Connect signals
        self._connect_signals()
    
    def _initialize_ui(self):
        """Initialize the user interface"""
        # Create central widget
        self._central_widget = QWidget()
        self.setCentralWidget(self._central_widget)
        
        # Create main layout
        self._main_layout = QVBoxLayout(self._central_widget)
        
        # Create tab widget
        self._tab_widget = QTabWidget()
        self._main_layout.addWidget(self._tab_widget)
        
        # Add visualization tabs
        self._add_visualization_tabs()
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create tool bar
        self._create_tool_bar()
        
        # Create status bar
        self._create_status_bar()
        
        # Create dock widgets
        self._create_dock_widgets()
    
    def _add_visualization_tabs(self):
        """Add visualization tabs to the tab widget"""
        # Network visualization
        network_viz = visualization_system.get_visualization('network')
        if network_viz:
            self._tab_widget.addTab(network_viz, "Network")
        
        # Growth visualization
        growth_viz = visualization_system.get_visualization('growth')
        if growth_viz:
            self._tab_widget.addTab(growth_viz, "Growth")
        
        # Metrics visualization
        metrics_viz = visualization_system.get_visualization('metrics')
        if metrics_viz:
            self._tab_widget.addTab(metrics_viz, "Metrics")
        
        # System visualization
        system_viz = visualization_system.get_visualization('system')
        if system_viz:
            self._tab_widget.addTab(system_viz, "System")
    
    def _create_menu_bar(self):
        """Create the menu bar"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        
        # View menu
        view_menu = menu_bar.addMenu("View")
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
    
    def _create_tool_bar(self):
        """Create the tool bar"""
        tool_bar = QToolBar()
        tool_bar.setIconSize(QSize(16, 16))
        self.addToolBar(tool_bar)
        
        # Add tool bar actions
        # TODO: Add tool bar actions
    
    def _create_status_bar(self):
        """Create the status bar"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # Add status bar widgets
        # TODO: Add status bar widgets
    
    def _create_dock_widgets(self):
        """Create dock widgets"""
        # Properties dock
        properties_dock = QDockWidget("Properties", self)
        properties_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, properties_dock)
        
        # Console dock
        console_dock = QDockWidget("Console", self)
        console_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, console_dock)
    
    def _connect_signals(self):
        """Connect signals and slots"""
        # Tab changed signal
        self._tab_widget.currentChanged.connect(self._on_tab_changed)
    
    def _on_tab_changed(self, index: int):
        """Handle tab changes"""
        tab_name = self._tab_widget.tabText(index)
        visualization_system.set_active_visualization(tab_name.lower())
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up resources
        visualization_system.cleanup()
        super().closeEvent(event)

def create_main_window() -> MainWindow:
    """Create and return the main window"""
    return MainWindow() 