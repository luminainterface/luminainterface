#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Activity Panel for LUMINA V7 Dashboard
============================================

Panel for visualizing neural network activity and consciousness metrics.
"""

import os
import sys
import time
import random
import logging
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Import base panel
from src.visualization.panels.base_panel import BasePanel, QT_FRAMEWORK

# Qt compatibility layer
try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
        QGridLayout, QFrame, QSplitter
    )
    from PySide6.QtCore import Qt, Signal, Slot
    from PySide6.QtGui import QFont, QPainter, QColor, QPen
    HAVE_PYSIDE6 = True
except ImportError:
    HAVE_PYSIDE6 = False
    
if not HAVE_PYSIDE6:
    try:
    from PyQt5.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
            QGridLayout, QFrame, QSplitter
        )
        from PyQt5.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot
        from PyQt5.QtGui import QFont, QPainter, QColor, QPen
    except ImportError:
        raise ImportError("Neither PySide6 nor PyQt5 is installed. Please install at least one of them.")

# Try to import PyQtGraph for enhanced visualizations
try:
    import pyqtgraph as pg
    HAVE_PYQTGRAPH = True
    # Configure PyQtGraph for dark theme
    pg.setConfigOption('background', (40, 44, 52))
    pg.setConfigOption('foreground', (255, 255, 255))
except ImportError:
    HAVE_PYQTGRAPH = False

# Try to import Matplotlib as fallback
if not HAVE_PYQTGRAPH:
    try:
    import matplotlib
        matplotlib.use('Qt5Agg' if QT_FRAMEWORK == "PyQt5" else 'Qt6Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
        import matplotlib.pyplot as plt
        HAVE_MATPLOTLIB = True
        
        # Configure Matplotlib dark theme
        plt.style.use('dark_background')
    except ImportError:
        HAVE_MATPLOTLIB = False

# Setup logging
logger = logging.getLogger("LuminaPanels")

class NeuralActivityPanel(BasePanel):
    """Panel for visualizing neural network activity"""
    
    def __init__(self, parent=None, db_path=None, refresh_rate=1000, active=True, gui_framework=None):
        """
        Initialize the neural activity panel
        
        Args:
            parent: Parent widget
            db_path: Path to metrics database
            refresh_rate: Data refresh rate in milliseconds
            active: Whether the panel is active at startup
            gui_framework: GUI framework to use
        """
        super().__init__(
            parent=parent, 
            panel_name="Neural Activity", 
            db_path=db_path, 
            refresh_rate=refresh_rate, 
            active=active, 
            gui_framework=gui_framework
        )
        
        # Internal data structures
        self.time_window = 60  # 60 seconds of data
        self.activity_history = []
        self.consciousness_history = []
        self.timestamps = []
        
        # Get screen dimensions
        if parent:
            self.screen_width = parent.width()
            self.screen_height = parent.height()
        else:
            self.screen_width = 800
            self.screen_height = 600
        
        # Setup the specific UI components for neural activity
        self._setup_neural_ui()
        
        # Initial refresh
        self.refresh_data()
    
    def _setup_neural_ui(self):
        """Set up specialized UI components for neural activity visualization"""
        # Clear the base layout (except for header)
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget not in (self.header_label, self.status_label, self.timestamp_label):
                widget.deleteLater()
        
        # Create a split layout
        self.splitter = QSplitter(Qt.Vertical)
        self.layout.insertWidget(3, self.splitter)  # Insert after header, status, and timestamp
        
        # Create metrics panel (top)
        self.metrics_widget = QWidget()
        self.metrics_layout = QGridLayout(self.metrics_widget)
        
        # Add metrics
        metrics = [
            ("Consciousness Level", "0.00", "consciousness_level"),
            ("Neural Activity", "0.00", "neural_activity"),
            ("Neural-Linguistic Score", "0.00", "neural_linguistic_score"),
            ("Pattern Recognition", "0.00", "pattern_recognition"),
            ("Integration Index", "0.00", "integration_index"),
            ("Node Coherence", "0.00", "node_coherence")
        ]
        
        self.metric_labels = {}
        self.metric_values = {}
        
        for i, (label_text, value_text, metric_key) in enumerate(metrics):
            row, col = divmod(i, 3)
            
            # Create label container
            container = QFrame()
            container.setFrameShape(QFrame.StyledPanel)
            container.setFrameShadow(QFrame.Raised)
            container.setStyleSheet("background-color: rgba(60, 65, 75, 150); border-radius: 4px;")
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(8, 8, 8, 8)
            
            # Add label and value
            label = QLabel(label_text)
            label.setFont(QFont("Arial", 10, QFont.Bold))
            container_layout.addWidget(label)
            
            value = QLabel(value_text)
            value.setFont(QFont("Arial", 14))
            container_layout.addWidget(value)
            
            # Store references
            self.metric_labels[metric_key] = label
            self.metric_values[metric_key] = value
            
            # Add to grid
            self.metrics_layout.addWidget(container, row, col)
        
        # Add metrics widget to splitter
        self.splitter.addWidget(self.metrics_widget)
        
        # Create graph panel (bottom)
        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_widget)
        
        # Create graph title
        graph_title = QLabel("Neural Activity Over Time")
        graph_title.setFont(QFont("Arial", 12, QFont.Bold))
        graph_title.setAlignment(Qt.AlignCenter)
        self.graph_layout.addWidget(graph_title)
        
        # Add visualization - PyQtGraph preferred, fallback to Matplotlib
        if HAVE_PYQTGRAPH:
            self._setup_pyqtgraph()
        elif HAVE_MATPLOTLIB:
            self._setup_matplotlib()
        else:
            # Fallback to simple custom visualization if neither is available
            self._setup_custom_visualization()
        
        # Add graph widget to splitter
        self.splitter.addWidget(self.graph_widget)
        
        # Set splitter sizes
        self.splitter.setSizes([int(self.height() * 0.4), int(self.height() * 0.6)])
    
    def _setup_pyqtgraph(self):
        """Set up PyQtGraph visualization components"""
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground((40, 44, 52))
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Create plot items
        self.activity_curve = self.plot_widget.plot([], [], 
                                                   pen=pg.mkPen(color=(76, 175, 80), width=2),
                                                   name="Neural Activity")
        
        self.consciousness_curve = self.plot_widget.plot([], [], 
                                                       pen=pg.mkPen(color=(33, 150, 243), width=2),
                                                       name="Consciousness Level")
        
        # Add legend
        legend = self.plot_widget.addLegend()
        
        # Add to layout
        self.graph_layout.addWidget(self.plot_widget)

    def _setup_matplotlib(self):
        """Set up Matplotlib visualization components as fallback"""
        # Create figure and canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Create subplot
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title('Neural Activity Over Time')
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Value')
        self.axes.grid(True, alpha=0.3)
        
        # Create empty lines
        self.activity_line, = self.axes.plot([], [], label='Neural Activity', color='#4CAF50')
        self.consciousness_line, = self.axes.plot([], [], label='Consciousness Level', color='#2196F3')
        
        # Add legend
        self.axes.legend()
        
        # Add to layout
        self.graph_layout.addWidget(self.canvas)
    
    def _setup_custom_visualization(self):
        """Set up simple custom visualization components as final fallback"""
        # Create custom visualization widget
        self.custom_viz = SimpleVisualizationWidget()
        self.graph_layout.addWidget(self.custom_viz)
        
        # Add a note about missing visualization libraries
        note = QLabel("Note: Install PyQtGraph or Matplotlib for better visualizations")
        note.setStyleSheet("color: orange;")
        self.graph_layout.addWidget(note)
    
    def refresh_data(self):
        """Refresh neural activity data"""
        try:
            # Get current timestamp
            current_time = datetime.now()
            
            # Attempt to fetch data from database
            metrics = self._fetch_metrics_from_db()
            
            # If no data or in mock mode, generate mock data
            if not metrics or self.is_mock_mode:
                metrics = self._generate_mock_metrics()
                if not self.is_mock_mode:
                    self.set_mock_mode(True)
            else:
                if self.is_mock_mode:
                    self.set_mock_mode(False)
            
            # Update history (keep only last time_window seconds)
            self.timestamps.append(current_time)
            self.activity_history.append(metrics.get('neural_activity', 0.0))
            self.consciousness_history.append(metrics.get('consciousness_level', 0.0))
            
            # Trim history to time window
            cutoff_time = current_time - timedelta(seconds=self.time_window)
            while self.timestamps and self.timestamps[0] < cutoff_time:
                self.timestamps.pop(0)
                self.activity_history.pop(0)
                self.consciousness_history.pop(0)
            
            # Update the UI with new data
            self.update_signal.emit(metrics)
            
        except Exception as e:
            logger.error(f"Error refreshing neural activity data: {e}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def _fetch_metrics_from_db(self) -> Dict:
        """Fetch the latest neural metrics from the database"""
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query the latest neural activity metrics
            cursor.execute("""
                SELECT value, description FROM neural_activity
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            neural_activity_row = cursor.fetchone()
            
            # If no data, return empty dict
            if not neural_activity_row:
                conn.close()
                return {}
                
            neural_activity = neural_activity_row[0]
            
            # Query additional metrics
            metrics = {
                'neural_activity': neural_activity,
                'consciousness_level': 0.0,
                'neural_linguistic_score': 0.0,
                'pattern_recognition': 0.0,
                'integration_index': 0.0,
                'node_coherence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to get consciousness level
            cursor.execute("""
                SELECT value FROM consciousness_metrics
                WHERE metric_name = 'consciousness_level'
            ORDER BY timestamp DESC 
                LIMIT 1
            """)
            consciousness_row = cursor.fetchone()
            if consciousness_row:
                metrics['consciousness_level'] = consciousness_row[0]
            
            # Close the database connection
            conn.close()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            return {}
    
    def _generate_mock_metrics(self) -> Dict:
        """Generate mock metrics for testing"""
        # If we have history, make the new values follow a realistic pattern
        if self.activity_history:
            last_activity = self.activity_history[-1]
            last_consciousness = self.consciousness_history[-1]
            
            # Add some random walk with constraints
            activity = max(0.1, min(0.9, last_activity + random.uniform(-0.05, 0.05)))
            consciousness = max(0.1, min(0.9, last_consciousness + random.uniform(-0.03, 0.03)))
        else:
            # Initial values
            activity = random.uniform(0.4, 0.6)
            consciousness = random.uniform(0.4, 0.6)
        
        # Generate related mock metrics
        metrics = {
            'neural_activity': activity,
            'consciousness_level': consciousness,
            'neural_linguistic_score': activity * 0.8 + consciousness * 0.2 + random.uniform(-0.1, 0.1),
            'pattern_recognition': activity * 0.7 + random.uniform(-0.1, 0.1),
            'integration_index': consciousness * 0.9 + random.uniform(-0.05, 0.05),
            'node_coherence': activity * 0.5 + consciousness * 0.5 + random.uniform(-0.1, 0.1),
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure all values are in valid range
        for key in metrics:
            if key != 'timestamp' and isinstance(metrics[key], float):
                metrics[key] = max(0.0, min(1.0, metrics[key]))
                metrics[key] = round(metrics[key], 2)
        
        return metrics
    
    def _update_ui_from_data(self, data):
        """Update all UI components with new data"""
        # First, call the parent class method
        super()._update_ui_from_data(data)
        
        # Update metric values
        for key, value_label in self.metric_values.items():
            if key in data:
                value = data[key]
                if isinstance(value, float):
                    # Format as percentage with 2 decimal places
                    value_text = f"{value:.2f}"
                    
                    # Set color based on value
                    if value > 0.8:
                        color = "color: #4CAF50;"  # Green
                    elif value > 0.5:
                        color = "color: #2196F3;"  # Blue
                    elif value > 0.3:
                        color = "color: #FFC107;"  # Yellow
                    else:
                        color = "color: #F44336;"  # Red
                    
                    value_label.setStyleSheet(color)
                else:
                    value_text = str(value)
                
                value_label.setText(value_text)
        
        # Update graph if we have history
        if self.timestamps:
            # Create time axis (seconds from now)
            now = datetime.now()
            time_axis = [(t - now).total_seconds() for t in self.timestamps]
            
            # Update visualization based on available libraries
            if HAVE_PYQTGRAPH:
                self.activity_curve.setData(time_axis, self.activity_history)
                self.consciousness_curve.setData(time_axis, self.consciousness_history)
                
            elif HAVE_MATPLOTLIB:
                self.activity_line.set_data(time_axis, self.activity_history)
                self.consciousness_line.set_data(time_axis, self.consciousness_history)
                
                # Update axes limits
                self.axes.set_xlim(min(time_axis), max(time_axis))
                self.axes.set_ylim(0, 1)
                
                # Redraw
                self.figure.canvas.draw()
                
            else:
                # Update custom visualization
                self.custom_viz.update_data(self.activity_history, self.consciousness_history)

class SimpleVisualizationWidget(QWidget):
    """Simple custom visualization widget as fallback when no graphing libraries are available"""
    
    def __init__(self, parent=None):
        """Initialize the simple visualization widget"""
        super().__init__(parent)
        self.activity_data = []
        self.consciousness_data = []
        self.setMinimumHeight(200)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(40, 44, 52))
        self.setPalette(palette)
    
    def update_data(self, activity_data, consciousness_data):
        """Update the data to visualize"""
        self.activity_data = activity_data
        self.consciousness_data = consciousness_data
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the visualization"""
        if not self.activity_data or not self.consciousness_data:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw background
        painter.fillRect(0, 0, width, height, QColor(40, 44, 52))
        
        # Draw grid
        painter.setPen(QPen(QColor(100, 100, 100, 100), 1))
        # Vertical grid lines
        for i in range(1, 10):
            x = width * (i / 10)
            painter.drawLine(int(x), 0, int(x), height)
        # Horizontal grid lines    
        for i in range(1, 10):
            y = height * (i / 10)
            painter.drawLine(0, int(y), width, int(y))
        
        # Draw data
        if len(self.activity_data) > 1:
            self._draw_line(painter, self.activity_data, QColor(76, 175, 80), height)
        
        if len(self.consciousness_data) > 1:
            self._draw_line(painter, self.consciousness_data, QColor(33, 150, 243), height)
        
        # Draw legend
        painter.setPen(QColor(76, 175, 80))
        painter.drawText(10, 20, "Neural Activity")
        painter.setPen(QColor(33, 150, 243))
        painter.drawText(10, 40, "Consciousness Level")
    
    def _draw_line(self, painter, data, color, height):
        """Draw a line series on the widget"""
        points = len(data)
        width = self.width()
        
        painter.setPen(QPen(color, 2))
        
        for i in range(points - 1):
            x1 = width * (i / (points - 1))
            y1 = height - (data[i] * height)
            x2 = width * ((i + 1) / (points - 1))
            y2 = height - (data[i + 1] * height)
            
            painter.drawLine(int(x1), int(y1), int(x2), int(y2)) 