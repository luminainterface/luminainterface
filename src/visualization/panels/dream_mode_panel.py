#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dream Mode Panel for LUMINA V7 Dashboard
========================================

Visualizes dream pattern generation in the neural network.
"""

import os
import sys
import time
import random
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Import base panel
from src.visualization.panels.base_panel import BasePanel, QT_FRAMEWORK

# Qt compatibility layer
if QT_FRAMEWORK == "PySide6":
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, 
        QFrame, QSlider, QPushButton, QComboBox, QSpinBox
    )
    from PySide6.QtCore import Qt, Signal, Slot, QTimer
    from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush, QLinearGradient
else:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, 
        QFrame, QSlider, QPushButton, QComboBox, QSpinBox
    )
    from PyQt5.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot, QTimer
    from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush, QLinearGradient

# Try to import PyQtGraph for enhanced visualizations
try:
    import pyqtgraph as pg
    import numpy as np
    HAVE_PYQTGRAPH = True
    
    # Configure PyQtGraph for dark theme
    pg.setConfigOption('background', (40, 44, 52))
    pg.setConfigOption('foreground', (255, 255, 255))
except ImportError:
    HAVE_PYQTGRAPH = False

# Setup logging
logger = logging.getLogger("LuminaPanels.DreamMode")

class DreamModePanel(BasePanel):
    """Panel for visualizing dream pattern generation"""
    
    def __init__(self, parent=None, db_path=None, refresh_rate=1000, active=True, gui_framework=None):
        """
        Initialize the dream mode panel
        
        Args:
            parent: Parent widget
            db_path: Path to metrics database
            refresh_rate: Data refresh rate in milliseconds
            active: Whether panel is active at startup
            gui_framework: GUI framework to use
        """
        super().__init__(
            parent=parent, 
            panel_name="Dream Mode", 
            db_path=db_path, 
            refresh_rate=refresh_rate, 
            active=active, 
            gui_framework=gui_framework
        )
        
        # Dream state parameters
        self.dream_active = False
        self.creativity_index = 0.0
        self.pattern_count = 0
        self.pattern_history = []
        self.max_patterns = 100
        self.current_pattern = None
        
        # Colors and visualization parameters
        self.base_color = QColor(155, 89, 182)  # Purple
        self.creativity_color = QColor(243, 156, 18)  # Orange
        self.pattern_color = QColor(46, 204, 113)  # Green
        self.use_color_cycling = True
        self.pattern_trails = True
        self.trail_length = 20
        
        # Set up the UI components
        self._setup_dream_ui()
        
        # Connect signals
        self.start_button.clicked.connect(self._toggle_dream_mode)
        self.creativity_slider.valueChanged.connect(self._update_creativity)
        self.visualization_combo.currentIndexChanged.connect(self._update_visualization_mode)
        
        # Initial refresh
        self.refresh_data()
    
    def _setup_dream_ui(self):
        """Set up UI components specific to dream mode"""
        # Create control panel
        self.controls_frame = QFrame()
        self.controls_frame.setFrameShape(QFrame.StyledPanel)
        self.controls_frame.setStyleSheet("background-color: rgba(60, 65, 75, 150); border-radius: 4px;")
        self.controls_layout = QGridLayout(self.controls_frame)
        
        # Add dream controls
        self.start_button = QPushButton("Start Dream Mode")
        self.start_button.setCheckable(True)
        self.controls_layout.addWidget(self.start_button, 0, 0, 1, 2)
        
        # Creativity slider
        creativity_label = QLabel("Creativity:")
        self.controls_layout.addWidget(creativity_label, 1, 0)
        
        self.creativity_slider = QSlider(Qt.Horizontal)
        self.creativity_slider.setRange(0, 100)
        self.creativity_slider.setValue(50)
        self.creativity_slider.setTracking(True)
        self.controls_layout.addWidget(self.creativity_slider, 1, 1)
        
        # Visualization type combo
        viz_label = QLabel("Visualization:")
        self.controls_layout.addWidget(viz_label, 2, 0)
        
        self.visualization_combo = QComboBox()
        self.visualization_combo.addItems(["Wave", "Network", "Particles", "Heatmap"])
        self.controls_layout.addWidget(self.visualization_combo, 2, 1)
        
        # Trail length spinner
        trail_label = QLabel("Trail Length:")
        self.controls_layout.addWidget(trail_label, 3, 0)
        
        self.trail_spinner = QSpinBox()
        self.trail_spinner.setRange(1, 100)
        self.trail_spinner.setValue(self.trail_length)
        self.trail_spinner.valueChanged.connect(lambda value: setattr(self, 'trail_length', value))
        self.controls_layout.addWidget(self.trail_spinner, 3, 1)
        
        # Add status indicators
        self.status_frame = QFrame()
        self.status_frame.setFrameShape(QFrame.StyledPanel)
        self.status_frame.setStyleSheet("background-color: rgba(60, 65, 75, 150); border-radius: 4px;")
        self.status_layout = QGridLayout(self.status_frame)
        
        # Dream state indicator
        self.state_label = QLabel("Dream State:")
        self.status_layout.addWidget(self.state_label, 0, 0)
        
        self.state_value = QLabel("Inactive")
        self.state_value.setStyleSheet("color: gray;")
        self.status_layout.addWidget(self.state_value, 0, 1)
        
        # Creativity index indicator
        self.creativity_label = QLabel("Creativity Index:")
        self.status_layout.addWidget(self.creativity_label, 1, 0)
        
        self.creativity_value = QLabel("0.00")
        self.status_layout.addWidget(self.creativity_value, 1, 1)
        
        # Pattern count indicator
        self.pattern_label = QLabel("Patterns Generated:")
        self.status_layout.addWidget(self.pattern_label, 2, 0)
        
        self.pattern_value = QLabel("0")
        self.status_layout.addWidget(self.pattern_value, 2, 1)
        
        # Create visualization area
        self.viz_frame = QFrame()
        self.viz_frame.setFrameShape(QFrame.StyledPanel)
        self.viz_frame.setStyleSheet("background-color: rgba(40, 44, 52, 200); border-radius: 4px;")
        self.viz_frame.setMinimumHeight(300)
        
        if HAVE_PYQTGRAPH:
            # Create PyQtGraph plot widget
            self.viz_layout = QVBoxLayout(self.viz_frame)
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground((40, 44, 52))
            self.plot_widget.setLabel('left', 'Pattern Intensity')
            self.plot_widget.setLabel('bottom', 'Time')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # Create plot items
            self.pattern_curve = self.plot_widget.plot([], [], 
                pen=pg.mkPen(color=self.pattern_color.getRgb()[:3], width=2),
                name="Current Pattern")
            
            self.creativity_curve = self.plot_widget.plot([], [],
                pen=pg.mkPen(color=self.creativity_color.getRgb()[:3], width=1.5, style=Qt.DashLine),
                name="Creativity")
            
            # Add to layout
            self.viz_layout.addWidget(self.plot_widget)
        else:
            # Custom rendering will be used in paintEvent
            self.viz_layout = QVBoxLayout(self.viz_frame)
            self.custom_viz = DreamVisualizationWidget()
            self.viz_layout.addWidget(self.custom_viz)
            
            # Add note about missing PyQtGraph
            note = QLabel("Note: Install PyQtGraph for enhanced visualizations")
            note.setStyleSheet("color: orange;")
            self.viz_layout.addWidget(note)
        
        # Add components to main layout
        self.layout.insertWidget(3, self.controls_frame)
        self.layout.insertWidget(4, self.status_frame)
        self.layout.insertWidget(5, self.viz_frame)
    
    def refresh_data(self):
        """Refresh dream state data"""
        try:
            # Generate mock data when in mock mode or no real data available
            if self.is_mock_mode or not self.db_path:
                dream_data = self._generate_mock_dream_data()
            else:
                # Fetch real dream data from database
                dream_data = self._fetch_dream_data()
            
            # Update the UI
            self.update_signal.emit(dream_data)
            
        except Exception as e:
            logger.error(f"Error refreshing dream data: {e}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def _fetch_dream_data(self):
        """Fetch real dream data from database"""
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query dream state data
            cursor.execute("""
                SELECT dream_active, creativity_index, pattern_count, pattern_data, timestamp
                FROM dream_metrics
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            
            if row:
                dream_active, creativity_index, pattern_count, pattern_data, timestamp = row
                
                # Parse pattern data from JSON
                import json
                pattern = json.loads(pattern_data) if pattern_data else None
                
                # Query pattern history
                cursor.execute("""
                    SELECT pattern_data, timestamp
                    FROM dream_metrics
                    WHERE pattern_data IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (self.trail_length,))
                
                history = []
                for row in cursor.fetchall():
                    pattern_data, timestamp = row
                    history.append({
                        "pattern": json.loads(pattern_data),
                        "timestamp": timestamp
                    })
                
                conn.close()
                
                return {
                    "dream_active": bool(dream_active),
                    "creativity_index": float(creativity_index),
                    "pattern_count": int(pattern_count),
                    "current_pattern": pattern,
                    "pattern_history": history,
                    "timestamp": timestamp
                }
            else:
                conn.close()
                # No data found, generate mock data instead
                return self._generate_mock_dream_data()
                
        except Exception as e:
            logger.error(f"Error fetching dream data: {e}")
            # Fall back to mock data on error
            return self._generate_mock_dream_data()
    
    def _generate_mock_dream_data(self):
        """Generate mock dream data for testing"""
        # Update internal state
        if self.dream_active:
            # Update creativity based on slider
            creativity_factor = self.creativity_slider.value() / 100.0
            
            # Generate random changes in creativity index
            self.creativity_index = max(0.0, min(1.0, 
                self.creativity_index + (random.random() * 0.1 - 0.05) * creativity_factor))
            
            # Possibly generate new pattern
            if random.random() < 0.2 * creativity_factor:
                # Generate new pattern
                x = np.linspace(0, 10, 100)
                # Base wave pattern
                pattern = np.sin(x) * 0.5 + 0.5
                
                # Add complexity based on creativity
                for i in range(1, int(5 * creativity_factor) + 1):
                    pattern += np.sin(x * (i+1) * random.uniform(0.8, 1.2)) * (0.5 / (i+1)) * random.uniform(0.8, 1.2)
                
                # Normalize to 0-1 range
                pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern))
                
                # Convert to list for JSON serialization
                pattern = pattern.tolist()
                
                # Update pattern history
                if self.current_pattern:
                    self.pattern_history.insert(0, {
                        "pattern": self.current_pattern,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Limit history length
                    if len(self.pattern_history) > self.trail_length:
                        self.pattern_history = self.pattern_history[:self.trail_length]
                
                # Set current pattern
                self.current_pattern = pattern
                self.pattern_count += 1
        
        return {
            "dream_active": self.dream_active,
            "creativity_index": self.creativity_index,
            "pattern_count": self.pattern_count,
            "current_pattern": self.current_pattern,
            "pattern_history": self.pattern_history,
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_ui_from_data(self, data):
        """Update UI with dream data"""
        # Call parent implementation
        super()._update_ui_from_data(data)
        
        # Update dream state indicator
        dream_active = data.get("dream_active", False)
        if dream_active:
            self.state_value.setText("Active")
            self.state_value.setStyleSheet("color: #2ecc71;")  # Green
        else:
            self.state_value.setText("Inactive")
            self.state_value.setStyleSheet("color: gray;")
        
        # Update creativity index
        creativity_index = data.get("creativity_index", 0.0)
        self.creativity_value.setText(f"{creativity_index:.2f}")
        
        # Update pattern count
        pattern_count = data.get("pattern_count", 0)
        self.pattern_value.setText(str(pattern_count))
        
        # Update button state if needed
        if self.dream_active != dream_active:
            self.dream_active = dream_active
            self.start_button.setChecked(dream_active)
            self.start_button.setText("Stop Dream Mode" if dream_active else "Start Dream Mode")
        
        # Update visualization
        current_pattern = data.get("current_pattern")
        pattern_history = data.get("pattern_history", [])
        
        if current_pattern:
            if HAVE_PYQTGRAPH:
                # Update PyQtGraph plots
                x = np.linspace(0, 10, len(current_pattern))
                self.pattern_curve.setData(x, current_pattern)
                
                # Update creativity curve
                creativity_curve = np.ones_like(x) * creativity_index
                self.creativity_curve.setData(x, creativity_curve)
                
            else:
                # Update custom visualization
                self.custom_viz.update_data(current_pattern, pattern_history, creativity_index)
    
    def _toggle_dream_mode(self, checked):
        """Toggle dream mode on/off"""
        self.dream_active = checked
        self.start_button.setText("Stop Dream Mode" if checked else "Start Dream Mode")
        
        if checked:
            logger.info("Starting dream mode")
        else:
            logger.info("Stopping dream mode")
    
    def _update_creativity(self, value):
        """Update creativity level from slider"""
        self.creativity_index = value / 100.0
    
    def _update_visualization_mode(self, index):
        """Update visualization mode"""
        modes = ["Wave", "Network", "Particles", "Heatmap"]
        mode = modes[index]
        logger.debug(f"Changed visualization mode to {mode}")
        
        # Update visualization based on mode
        # This would be implemented by changing the visualization widget
        # but for simplicity we're just logging it now


class DreamVisualizationWidget(QWidget):
    """Custom widget for dream pattern visualization when PyQtGraph is not available"""
    
    def __init__(self, parent=None):
        """Initialize the visualization widget"""
        super().__init__(parent)
        self.pattern_data = None
        self.pattern_history = []
        self.creativity_index = 0.0
        
        # Set minimum size
        self.setMinimumSize(400, 200)
    
    def update_data(self, pattern_data, pattern_history, creativity_index):
        """Update with new pattern data"""
        self.pattern_data = pattern_data
        self.pattern_history = pattern_history
        self.creativity_index = creativity_index
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the visualization"""
        if not self.pattern_data:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Draw background
        painter.fillRect(0, 0, width, height, QColor(40, 44, 52))
        
        # Draw pattern history first (oldest to newest)
        for i, history_item in enumerate(reversed(self.pattern_history)):
            hist_pattern = history_item.get("pattern")
            if hist_pattern:
                # Calculate alpha based on position in history
                alpha = 100 - int(70 * (i / max(1, len(self.pattern_history))))
                
                # Draw historical pattern
                self._draw_pattern(painter, hist_pattern, QColor(155, 89, 182, alpha))
        
        # Draw creativity level line
        painter.setPen(QPen(QColor(243, 156, 18), 1, Qt.DashLine))
        y = height - int(height * self.creativity_index)
        painter.drawLine(0, y, width, y)
        
        # Draw current pattern
        self._draw_pattern(painter, self.pattern_data, QColor(46, 204, 113))
    
    def _draw_pattern(self, painter, pattern_data, color):
        """Draw a pattern with the given color"""
        if not pattern_data:
            return
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Set pen for pattern
        painter.setPen(QPen(color, 2))
        
        # Calculate points
        points = []
        for i, value in enumerate(pattern_data):
            x = int(width * i / (len(pattern_data) - 1))
            y = height - int(height * value)
            points.append((x, y))
        
        # Draw lines between points
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])


# For testing outside of the dashboard
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Create and show the panel
    panel = DreamModePanel(db_path=None, refresh_rate=500)
    panel.set_mock_mode(True)
    panel.show()
    
    sys.exit(app.exec_()) 