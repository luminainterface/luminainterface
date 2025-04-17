#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Panel for LUMINA V7 Dashboard
==================================

Base class for all dashboard panels to provide common functionality.
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Qt compatibility layer - handles both PyQt5 and PySide6
try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QFont
    QT_FRAMEWORK = "PySide6"
except ImportError:
    try:
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal
        from PyQt5.QtGui import QFont
        QT_FRAMEWORK = "PyQt5"
    except ImportError:
        raise ImportError("Neither PySide6 nor PyQt5 is installed. Please install at least one of them.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LuminaPanels")

class BasePanel(QWidget):
    """Base class for all dashboard panels"""
    
    # Signal that can be connected to update the UI
    update_signal = Signal(object) if QT_FRAMEWORK == "PySide6" else Signal(object)
    
    def __init__(self, parent=None, panel_name="Base Panel", db_path=None, 
                 refresh_rate=2000, active=True, gui_framework=None):
        """
        Initialize the base panel
        
        Args:
            parent: Parent widget
            panel_name: Display name of the panel
            db_path: Path to metrics database
            refresh_rate: Data refresh rate in milliseconds
            active: Whether the panel is active at startup
            gui_framework: GUI framework to use ("PyQt5" or "PySide6")
        """
        super().__init__(parent)
        
        # Store parameters
        self.panel_name = panel_name
        self.db_path = db_path or "data/neural_metrics.db"
        self.refresh_rate = refresh_rate
        self.active = active
        self.gui_framework = gui_framework or QT_FRAMEWORK
        
        # Internal state
        self.last_update = None
        self.metrics = {}
        self.refresh_timer = None
        self.is_mock_mode = False
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Log initialization
        logger.info(f"Initializing {self.panel_name} panel with {self.gui_framework} framework")
        
        # Set up UI components
        self._setup_ui()
        
        # Connect signals
        self.update_signal.connect(self._update_ui_from_data)
        
        # Start refresh timer if active
        if self.active:
            self.start_refresh()
    
    def _setup_ui(self):
        """Set up the UI components"""
        # Set up main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # Add header
        self.header_label = QLabel(self.panel_name)
        self.header_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.layout.addWidget(self.header_label)
        
        # Add status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setWordWrap(True)
        self.layout.addWidget(self.status_label)
        
        # Add timestamp label
        self.timestamp_label = QLabel("")
        self.timestamp_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.timestamp_label)
        
        # Add stretch to push everything to the top
        self.layout.addStretch(1)
    
    def start_refresh(self):
        """Start the refresh timer"""
        if self.refresh_timer is None:
            self.refresh_timer = QTimer(self)
            self.refresh_timer.timeout.connect(self.refresh_data)
            self.refresh_timer.start(self.refresh_rate)
            self.active = True
            logger.debug(f"{self.panel_name}: Started refresh timer at {self.refresh_rate}ms")
    
    def stop_refresh(self):
        """Stop the refresh timer"""
        if self.refresh_timer is not None:
            self.refresh_timer.stop()
            self.refresh_timer = None
            self.active = False
            logger.debug(f"{self.panel_name}: Stopped refresh timer")
    
    def set_active(self, active):
        """
        Set the active state of the panel
        
        Args:
            active: Whether the panel should be active
        """
        if active and not self.active:
            self.start_refresh()
        elif not active and self.active:
            self.stop_refresh()
    
    def refresh_data(self):
        """
        Refresh data from the data source
        This method should be overridden by subclasses
        """
        # This is a placeholder - subclasses should override this method
        # to fetch data from their respective sources
        
        # Update timestamp
        self.last_update = datetime.now()
        
        # Example data
        data = {
            "status": "Active",
            "value": 0.5,
            "timestamp": self.last_update.isoformat()
        }
        
        # Emit signal to update UI
        self.update_signal.emit(data)
    
    def set_mock_mode(self, enabled=True):
        """
        Set mock mode (generate fake data)
        
        Args:
            enabled: Whether mock mode should be enabled
        """
        self.is_mock_mode = enabled
        self.status_label.setText("MOCK MODE ENABLED" if enabled else "")
    
    def _update_ui_from_data(self, data):
        """
        Update UI components with new data
        
        Args:
            data: New data to display
        """
        # Update basic information
        if "status" in data:
            self.status_label.setText(data["status"])
        
        if "timestamp" in data:
            # Format timestamp for display
            try:
                if isinstance(data["timestamp"], str):
                    # Parse ISO format timestamp
                    timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # Use datetime object directly
                    formatted_time = data["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                
                self.timestamp_label.setText(f"Last update: {formatted_time}")
            except Exception as e:
                logger.error(f"Error formatting timestamp: {e}")
                self.timestamp_label.setText(f"Last update: {data['timestamp']}")
    
    def cleanup(self):
        """Clean up resources when panel is no longer needed"""
        self.stop_refresh()
        logger.debug(f"{self.panel_name}: Cleaned up resources")

    def set_refresh_rate(self, rate_ms):
        """
        Set the data refresh rate
        
        Args:
            rate_ms: Refresh rate in milliseconds
        """
        self.refresh_rate = rate_ms
        
        # Restart timer if active
        if self.active and self.refresh_timer is not None:
            self.refresh_timer.stop()
            self.refresh_timer.start(self.refresh_rate)
            logger.debug(f"{self.panel_name}: Changed refresh rate to {self.refresh_rate}ms") 