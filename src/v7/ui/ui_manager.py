#!/usr/bin/env python3
"""
LUMINA V7 UI Manager

Manages the UI components and layout for the V7 system.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine which Qt library to use
try:
    from PySide6.QtCore import Qt, QSize, QTimer
    from PySide6.QtGui import QIcon, QAction, QFont
    from PySide6.QtWidgets import (
        QMainWindow, QApplication, QTabWidget, QDockWidget, 
        QToolBar, QStatusBar, QMenu, QMenuBar, QWidget,
        QVBoxLayout, QLabel, QMessageBox
    )
    USING_PYSIDE6 = True
except ImportError:
    try:
        from PyQt5.QtCore import Qt, QSize, QTimer
        from PyQt5.QtGui import QIcon, QAction, QFont
        from PyQt5.QtWidgets import (
            QMainWindow, QApplication, QTabWidget, QDockWidget, 
            QToolBar, QStatusBar, QMenu, QMenuBar, QWidget,
            QVBoxLayout, QLabel, QMessageBox
        )
        USING_PYSIDE6 = False
    except ImportError:
        logger.error("Neither PySide6 nor PyQt5 is available. UI cannot be initialized.")
        raise ImportError("Qt libraries not found. Please install PySide6 or PyQt5.")

# Import panel modules
try:
    from src.v7.ui.panels.dashboard_panel import DashboardPanel
    from src.v7.ui.panels.neural_network_panel import NeuralNetworkPanel
    from src.v7.ui.panels.node_explorer_panel import NodeExplorerPanel
    from src.v7.ui.panels.settings_panel import SettingsPanel
    from src.v7.ui.panels.language_chat_panel import LanguageChatPanel
    PANELS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing panel modules: {e}")
    PANELS_AVAILABLE = False

class UIManager(QMainWindow):
    """
    Main UI Manager for LUMINA V7
    
    Handles the creation and management of all UI components,
    including the main window, panels, and toolbars.
    """
    
    def __init__(self, system_manager=None):
        super().__init__()
        self.system_manager = system_manager
        self.panels = {}
        self.dock_widgets = {}
        
        # Set up main window properties
        self.setWindowTitle("LUMINA V7.0.0.2")
        self.setMinimumSize(1200, 800)
        
        # Initialize UI components
        self.setup_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Set up status update timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
        
        logger.info("UI Manager initialized successfully")
    
    def setup_ui(self):
        """Set up the main UI components"""
        # Create central tab widget
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setTabsClosable(False)
        self.tab_widget.setMovable(True)
        self.setCentralWidget(self.tab_widget)
        
        # Create status bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        
        # Create system status label
        self.system_status_label = QLabel("System Status: Initializing...")
        self.status_bar.addPermanentWidget(self.system_status_label)
        
        # Create memory usage label
        self.memory_label = QLabel("Memory: 0 MB")
        self.status_bar.addPermanentWidget(self.memory_label)
        
        # Create panels if available
        if PANELS_AVAILABLE:
            self.create_panels()
        else:
            # Create error panel if panels aren't available
            error_widget = QWidget()
            error_layout = QVBoxLayout(error_widget)
            error_label = QLabel("Error: Panel modules could not be loaded.")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            error_layout.addWidget(error_label)
            self.tab_widget.addTab(error_widget, "Error")
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbars
        self.create_toolbars()
    
    def create_panels(self):
        """Create and add all panels to the UI"""
        # Dashboard panel
        try:
            self.panels["dashboard"] = DashboardPanel(self.system_manager)
            self.tab_widget.addTab(self.panels["dashboard"], "Dashboard")
        except Exception as e:
            logger.error(f"Failed to create Dashboard panel: {e}")
        
        # Neural Network panel
        try:
            self.panels["neural_network"] = NeuralNetworkPanel(self.system_manager)
            self.tab_widget.addTab(self.panels["neural_network"], "Neural Network")
        except Exception as e:
            logger.error(f"Failed to create Neural Network panel: {e}")
        
        # Node Explorer panel
        try:
            self.panels["node_explorer"] = NodeExplorerPanel(self.system_manager)
            self.tab_widget.addTab(self.panels["node_explorer"], "Node Explorer")
        except Exception as e:
            logger.error(f"Failed to create Node Explorer panel: {e}")
        
        # Settings panel
        try:
            self.panels["settings"] = SettingsPanel(self.system_manager)
            self.tab_widget.addTab(self.panels["settings"], "Settings")
        except Exception as e:
            logger.error(f"Failed to create Settings panel: {e}")
        
        # Language Chat panel
        try:
            self.panels["language_chat"] = LanguageChatPanel()
            self.tab_widget.addTab(self.panels["language_chat"], "Language Chat")
        except Exception as e:
            logger.error(f"Failed to create Language Chat panel: {e}")
    
    def create_menu_bar(self):
        """Create the application menu bar"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        # New action
        new_action = QAction("&New Project", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        # Open action
        open_action = QAction("&Open Project", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Save action
        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        # Save As action
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menu_bar.addMenu("&View")
        
        # Panels submenu
        panels_menu = view_menu.addMenu("&Panels")
        
        # Toggle panel visibility actions
        panel_names = ["Dashboard", "Neural Network", "Node Explorer", "Settings", "Language Chat"]
        for panel_name in panel_names:
            action = QAction(panel_name, self, checkable=True)
            action.setChecked(True)
            panels_menu.addAction(action)
        
        view_menu.addSeparator()
        
        # Reset layout action
        reset_layout_action = QAction("&Reset Layout", self)
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)
        
        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")
        
        # Run Neural Network action
        run_nn_action = QAction("&Run Neural Network", self)
        run_nn_action.triggered.connect(self.run_neural_network)
        tools_menu.addAction(run_nn_action)
        
        # Pause Neural Network action
        pause_nn_action = QAction("&Pause Neural Network", self)
        pause_nn_action.triggered.connect(self.pause_neural_network)
        tools_menu.addAction(pause_nn_action)
        
        tools_menu.addSeparator()
        
        # Language Chat action
        language_chat_action = QAction("Open &Language Chat", self)
        language_chat_action.triggered.connect(self.open_language_chat)
        tools_menu.addAction(language_chat_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        # Documentation action
        docs_action = QAction("&Documentation", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbars(self):
        """Create application toolbars"""
        # Main toolbar
        main_toolbar = QToolBar("Main Toolbar", self)
        main_toolbar.setMovable(False)
        main_toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.TopToolBarArea, main_toolbar)
        
        # Add toolbar actions
        toolbar_actions = [
            ("New Project", self.new_project),
            ("Open Project", self.open_project),
            ("Save Project", self.save_project),
            ("Run Neural Network", self.run_neural_network),
            ("Pause Neural Network", self.pause_neural_network),
            ("Language Chat", self.open_language_chat)
        ]
        
        for name, callback in toolbar_actions:
            action = QAction(name, self)
            action.triggered.connect(callback)
            main_toolbar.addAction(action)
    
    def connect_signals(self):
        """Connect signals to slots"""
        # Example: Connect tab widget's tab changed signal
        self.tab_widget.currentChanged.connect(self.tab_changed)
    
    def update_status(self):
        """Update status bar information"""
        if self.system_manager:
            # Update from system manager if available
            try:
                status = self.system_manager.get_status()
                self.system_status_label.setText(f"System Status: {status.get('status', 'Unknown')}")
                self.memory_label.setText(f"Memory: {status.get('memory_usage', '0')} MB")
            except Exception as e:
                logger.error(f"Error updating status: {e}")
        else:
            # Default values if system manager is not available
            self.system_status_label.setText("System Status: No System Manager")
            self.memory_label.setText("Memory: N/A")
    
    def tab_changed(self, index):
        """Handle tab changed event"""
        if index >= 0:
            tab_name = self.tab_widget.tabText(index)
            logger.debug(f"Switched to tab: {tab_name}")
    
    def new_project(self):
        """Create a new project"""
        QMessageBox.information(self, "New Project", "Creating a new project...")
    
    def open_project(self):
        """Open an existing project"""
        QMessageBox.information(self, "Open Project", "Opening project...")
    
    def save_project(self):
        """Save the current project"""
        QMessageBox.information(self, "Save Project", "Saving project...")
    
    def save_project_as(self):
        """Save the current project with a new name"""
        QMessageBox.information(self, "Save As", "Save project as...")
    
    def run_neural_network(self):
        """Run the neural network"""
        if self.system_manager:
            # Implement actual neural network start command
            QMessageBox.information(self, "Run Neural Network", "Starting neural network...")
    
    def pause_neural_network(self):
        """Pause the neural network"""
        if self.system_manager:
            # Implement actual neural network pause command
            QMessageBox.information(self, "Pause Neural Network", "Pausing neural network...")
    
    def open_language_chat(self):
        """Open the language chat panel"""
        # Switch to the Language Chat tab if it exists
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Language Chat":
                self.tab_widget.setCurrentIndex(i)
                return
        
        # If no Language Chat tab exists, create one
        try:
            if "language_chat" not in self.panels:
                self.panels["language_chat"] = LanguageChatPanel()
                self.tab_widget.addTab(self.panels["language_chat"], "Language Chat")
            
            # Switch to the newly created tab
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "Language Chat":
                    self.tab_widget.setCurrentIndex(i)
                    break
        except Exception as e:
            logger.error(f"Failed to create Language Chat panel: {e}")
            QMessageBox.warning(self, "Error", f"Could not open Language Chat panel: {str(e)}")
    
    def reset_layout(self):
        """Reset the UI layout to default"""
        QMessageBox.information(self, "Reset Layout", "Resetting layout...")
    
    def show_documentation(self):
        """Show documentation"""
        QMessageBox.information(self, "Documentation", "Opening documentation...")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About LUMINA V7", 
                         """<b>LUMINA V7.0.0.2</b>
                         <p>Advanced Neural Network System with Enhanced Language Capabilities</p>
                         <p>© 2023 LUMINA Labs</p>""")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Ask for confirmation before closing
        reply = QMessageBox.question(self, "Exit Confirmation",
                                    "Are you sure you want to exit?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Clean up resources
            if self.system_manager:
                self.system_manager.shutdown()
            
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    # For testing the UI manager independently
    app = QApplication(sys.argv)
    window = UIManager()
    window.show()
    sys.exit(app.exec() if USING_PYSIDE6 else app.exec_()) 