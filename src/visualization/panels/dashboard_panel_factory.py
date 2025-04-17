#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Panel Factory for LUMINA V7 Dashboard
===============================================

Factory class to create and assemble all the different panel types 
into a complete dashboard.
"""

import os
import sys
import logging
import importlib

# Qt Compatibility Layer - Try to import PySide6 first, fall back to PyQt5
try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QSplitter
    from PySide6.QtCore import Qt
    QT_FRAMEWORK = "PySide6"
    logger_qt = logging.getLogger("QtCompat")
    logger_qt.info("Using PySide6 for dashboard panels")
except ImportError:
    try:
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QSplitter
        from PyQt5.QtCore import Qt
        QT_FRAMEWORK = "PyQt5"
        logger_qt = logging.getLogger("QtCompat")
        logger_qt.info("Using PyQt5 for dashboard panels")
    except ImportError:
        raise ImportError("Neither PySide6 nor PyQt5 is installed. Please install at least one of them.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DashboardPanelFactory")

# Import panel types
from src.visualization.panels.neural_activity_panel import NeuralActivityPanel
from src.visualization.panels.language_processing_panel import LanguageProcessingPanel

# Try to import the Language Dashboard Bridge
try:
    from src.language.language_dashboard_bridge import get_language_dashboard_bridge
    HAS_LANGUAGE_BRIDGE = True
    logger.info("Language Dashboard Bridge module loaded")
except ImportError:
    HAS_LANGUAGE_BRIDGE = False
    logger.warning("Language Dashboard Bridge module not found, some features will be disabled")

class DashboardPanelFactory:
    """Factory class to create dashboard panels"""
    
    def __init__(self, db_path="data/neural_metrics.db", gui_framework=None):
        """
        Initialize the dashboard panel factory
        
        Args:
            db_path: Path to the metrics database
            gui_framework: GUI framework to use ("PyQt5" or "PySide6")
        """
        self.db_path = db_path
        self.panels = {}
        
        # Use specified GUI framework if provided
        self.gui_framework = gui_framework or QT_FRAMEWORK
        logger.info(f"Dashboard Panel Factory initialized with {self.gui_framework}")
        
        # Initialize Language Dashboard Bridge if available
        self.language_bridge = None
        if HAS_LANGUAGE_BRIDGE:
            try:
                # Get bridge instance with dashboard configuration
                self.language_bridge = get_language_dashboard_bridge({
                    "db_path": db_path,
                    "mock_mode": False,  # Try to use real components
                    "llm_weight": 0.5,
                    "nn_weight": 0.5
                })
                # Connect to language components
                bridge_ready = self.language_bridge.connect_language_components()
                if bridge_ready:
                    # Start the bridge
                    self.language_bridge.start()
                    logger.info("Language Dashboard Bridge started")
                else:
                    logger.warning("Language Dashboard Bridge failed to connect to components")
            except Exception as e:
                logger.error(f"Error initializing Language Dashboard Bridge: {e}")
        
    def create_panel(self, panel_type, parent=None, **kwargs):
        """
        Create a panel of the specified type
        
        Args:
            panel_type: Type of panel to create
            parent: Parent widget
            **kwargs: Additional arguments for the panel
            
        Returns:
            The created panel instance
        """
        kwargs['db_path'] = kwargs.get('db_path', self.db_path)
        kwargs['gui_framework'] = kwargs.get('gui_framework', self.gui_framework)
        
        panel_map = {
            "neural_activity": NeuralActivityPanel,
            "language_processing": LanguageProcessingPanel,
            # Add other panel types as they are implemented
        }
        
        if panel_type not in panel_map:
            raise ValueError(f"Unknown panel type: {panel_type}")
            
        panel_class = panel_map[panel_type]
        panel = panel_class(parent=parent, **kwargs)
        
        # Store the panel instance
        self.panels[panel_type] = panel
        
        # Connect panel to Language Dashboard Bridge if available
        if self.language_bridge and HAS_LANGUAGE_BRIDGE:
            panel_dict = {}
            if panel_type == "neural_activity":
                panel_dict["neural_activity_panel"] = panel
            elif panel_type == "language_processing":
                panel_dict["language_processing_panel"] = panel
                
            if panel_dict:
                self.language_bridge.connect_dashboard_panels(panel_dict)
                logger.info(f"Connected {panel_type} panel to Language Dashboard Bridge")
        
        return panel
    
    def create_dashboard_tabs(self, parent=None):
        """
        Create a tabbed dashboard with all panel types
        
        Args:
            parent: Parent widget
            
        Returns:
            QTabWidget containing all panels
        """
        tab_widget = QTabWidget(parent)
        
        # Add neural activity panel
        neural_panel = self.create_panel("neural_activity", parent=tab_widget)
        tab_widget.addTab(neural_panel, "Neural Activity")
        
        # Add language processing panel
        language_panel = self.create_panel("language_processing", parent=tab_widget)
        tab_widget.addTab(language_panel, "Language Processing")
        
        # Add more panel types here as they are implemented
        
        return tab_widget
    
    def create_overview_dashboard(self, parent=None):
        """
        Create an overview dashboard with all panels in a grid layout
        
        Args:
            parent: Parent widget
            
        Returns:
            QWidget containing the overview dashboard
        """
        dashboard = QWidget(parent)
        layout = QVBoxLayout(dashboard)
        
        # Create horizontal splitter for the top panels
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Add neural activity panel to left side
        neural_panel = self.create_panel(
            "neural_activity", 
            parent=top_splitter,
            refresh_rate=5000  # Slower refresh rate for overview
        )
        top_splitter.addWidget(neural_panel)
        
        # Add language processing panel to right side
        language_panel = self.create_panel(
            "language_processing", 
            parent=top_splitter,
            refresh_rate=5000  # Slower refresh rate for overview
        )
        top_splitter.addWidget(language_panel)
        
        # Add more panels to the splitter as they are implemented
        
        # Set initial sizes for the splitter
        top_splitter.setSizes([500, 500])
        
        # Add the splitter to the layout
        layout.addWidget(top_splitter)
        
        # Add space for bottom panels in the future
        # ...
        
        return dashboard
    
    def create_full_dashboard(self, parent=None):
        """
        Create a complete dashboard with tabbed and overview modes
        
        Args:
            parent: Parent widget
            
        Returns:
            QTabWidget containing the full dashboard
        """
        dashboard = QTabWidget(parent)
        
        # Add overview dashboard
        overview = self.create_overview_dashboard(parent=dashboard)
        dashboard.addTab(overview, "Overview")
        
        # Add individual panel tabs
        dashboard.addTab(self.create_panel("neural_activity", parent=dashboard), "Neural Activity")
        dashboard.addTab(self.create_panel("language_processing", parent=dashboard), "Language Processing")
        
        # Add more individual panel tabs as they are implemented
        
        return dashboard
    
    def shutdown_all_panels(self):
        """Shutdown and clean up all panels"""
        for panel_type, panel in self.panels.items():
            try:
                panel.set_active(False)
                panel.cleanup()
                logger.info(f"Shut down {panel_type} panel")
            except Exception as e:
                logger.error(f"Error shutting down {panel_type} panel: {e}")


def create_dashboard(parent=None, db_path="data/neural_metrics.db", dashboard_type="full", gui_framework=None):
    """
    Helper function to create a dashboard
    
    Args:
        parent: Parent widget
        db_path: Path to metrics database
        dashboard_type: Type of dashboard to create ("full", "tabs", "overview")
        gui_framework: GUI framework to use ("PyQt5" or "PySide6")
        
    Returns:
        The created dashboard widget
    """
    factory = DashboardPanelFactory(db_path, gui_framework)
    
    dashboard = None
    if dashboard_type == "full":
        dashboard = factory.create_full_dashboard(parent)
    elif dashboard_type == "tabs":
        dashboard = factory.create_dashboard_tabs(parent)
    elif dashboard_type == "overview":
        dashboard = factory.create_overview_dashboard(parent)
    else:
        raise ValueError(f"Unknown dashboard type: {dashboard_type}")
    
    # Connect all panels to the Language Dashboard Bridge if available
    if factory.language_bridge and HAS_LANGUAGE_BRIDGE:
        factory.language_bridge.connect_dashboard_panels(factory.panels)
        logger.info("Connected all panels to Language Dashboard Bridge")
        
    return dashboard 