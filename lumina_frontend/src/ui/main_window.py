"""
Main Window for Lumina Frontend
==============================

This module contains the main window class that provides the primary
interface for the Lumina Frontend system.
"""

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QMessageBox, QMenuBar, QMenu, QStatusBar
)
from PySide6.QtCore import Signal, Qt

from .tabs.performance_tab import PerformanceTab
from .tabs.network_tab import NetworkTab
from .tabs.resources_tab import ResourcesTab
from .tabs.learning_tab import LearningTab
from .tabs.quantum_tab import QuantumTab
from .tabs.cosmic_tab import CosmicTab

class MainWindow(QMainWindow):
    """Main window for Lumina Frontend."""
    
    # Signals
    shutdown_requested = Signal()
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.tabs = {}
        
        # Set window properties
        self.setWindowTitle("Lumina Frontend")
        self.resize(*self.config.get("ui.window_size", [1280, 720]))
        
        if self.config.get("ui.maximized", False):
            self.showMaximized()
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create layout
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setMovable(True)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        
        # Add tabs
        self._setup_tabs()
        
        # Add tab widget to layout
        self.layout.addWidget(self.tab_widget)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
    def _setup_tabs(self):
        """Setup the main tabs."""
        # Performance Tab
        self.tabs["performance"] = PerformanceTab(self.config)
        self.tab_widget.addTab(self.tabs["performance"], "Performance")
        
        # Network Tab
        self.tabs["network"] = NetworkTab(self.config)
        self.tab_widget.addTab(self.tabs["network"], "Network")
        
        # Resources Tab
        self.tabs["resources"] = ResourcesTab(self.config)
        self.tab_widget.addTab(self.tabs["resources"], "Resources")
        
        # Learning Tab
        self.tabs["learning"] = LearningTab(self.config)
        self.tab_widget.addTab(self.tabs["learning"], "Learning")
        
        # Quantum Tab
        self.tabs["quantum"] = QuantumTab(self.config)
        self.tab_widget.addTab(self.tabs["quantum"], "Quantum")
        
        # Cosmic Tab
        self.tabs["cosmic"] = CosmicTab(self.config)
        self.tab_widget.addTab(self.tabs["cosmic"], "Cosmic")
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menu_bar = QMenuBar()
        
        # File menu
        file_menu = QMenu("File", self)
        file_menu.addAction("Settings", self._show_settings)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        menu_bar.addMenu(file_menu)
        
        # View menu
        view_menu = QMenu("View", self)
        view_menu.addAction("Toggle Fullscreen", self._toggle_fullscreen)
        menu_bar.addMenu(view_menu)
        
        # Help menu
        help_menu = QMenu("Help", self)
        help_menu.addAction("About", self._show_about)
        menu_bar.addMenu(help_menu)
        
        self.setMenuBar(menu_bar)
    
    def initialize(self):
        """Initialize the main window components."""
        # Initialize tabs
        for tab in self.tabs.values():
            tab.initialize()
        
        # Update status bar
        self.status_bar.showMessage("Ready")
    
    def _on_tab_close_requested(self, index):
        """Handle tab close request."""
        tab_name = self.tab_widget.tabText(index)
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Close Tab",
            f"Are you sure you want to close the {tab_name} tab?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.tab_widget.removeTab(index)
    
    def _show_settings(self):
        """Show settings dialog."""
        # TODO: Implement settings dialog
        pass
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Lumina Frontend",
            "Lumina Frontend v0.1.0\n\n"
            "Advanced Neural Network Visualization and Control Interface"
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Exit",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Save window state
            self.config.set("ui.window_size", [self.width(), self.height()])
            self.config.set("ui.maximized", self.isMaximized())
            
            # Emit shutdown signal
            self.shutdown_requested.emit()
            event.accept()
        else:
            event.ignore() 