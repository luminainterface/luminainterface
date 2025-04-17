#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language Processing Panel for LUMINA V7 Dashboard
===============================================

Panel for visualizing language model activity and semantic understanding metrics.
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
        QGridLayout, QFrame, QSplitter, QTableWidget, 
        QTableWidgetItem, QHeaderView
    )
    from PySide6.QtCore import Qt, Signal, Slot
    from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush
    HAVE_PYSIDE6 = True
except ImportError:
    HAVE_PYSIDE6 = False
    
if not HAVE_PYSIDE6:
    try:
    from PyQt5.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
            QGridLayout, QFrame, QSplitter, QTableWidget, 
            QTableWidgetItem, QHeaderView
        )
        from PyQt5.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot
        from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush
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

class LanguageProcessingPanel(BasePanel):
    """Panel for visualizing language model activity and metrics"""
    
    def __init__(self, parent=None, db_path=None, refresh_rate=2000, active=True, gui_framework=None):
        """
        Initialize the language processing panel
        
        Args:
            parent: Parent widget
            db_path: Path to metrics database
            refresh_rate: Data refresh rate in milliseconds
            active: Whether the panel is active at startup
            gui_framework: GUI framework to use
        """
        super().__init__(
            parent=parent, 
            panel_name="Language Processing", 
            db_path=db_path, 
            refresh_rate=refresh_rate, 
            active=active, 
            gui_framework=gui_framework
        )
        
        # Internal data structures
        self.time_window = 60  # 60 seconds of data
        self.mistral_activity_history = []
        self.token_usage_history = []
        self.response_time_history = []
        self.timestamps = []
        self.conversation_stats = {}
        self.recent_conversations = []
        
        # Get screen dimensions
        if parent:
            self.screen_width = parent.width()
            self.screen_height = parent.height()
        else:
            self.screen_width = 800
            self.screen_height = 600
        
        # Setup the specific UI components for language processing
        self._setup_language_ui()
        
        # Initial refresh
        self.refresh_data()
    
    def _setup_language_ui(self):
        """Set up specialized UI components for language processing visualization"""
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
            ("Mistral Activity", "0.00", "mistral_activity"),
            ("Token Usage", "0", "token_usage"),
            ("Response Time", "0.00s", "response_time"),
            ("Semantic Score", "0.00", "semantic_score"),
            ("LLM Weight", "0.00", "llm_weight"),
            ("Memory Usage", "0.00", "memory_usage")
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
        
        # Create statistics tabs
        self.stats_widget = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_widget)
        
        # Statistics section header
        stats_header = QLabel("Language Processing Statistics")
        stats_header.setFont(QFont("Arial", 12, QFont.Bold))
        stats_header.setAlignment(Qt.AlignCenter)
        self.stats_layout.addWidget(stats_header)
        
        # Add split layout for graphs and conversation table
        self.stats_splitter = QSplitter(Qt.Horizontal)
        self.stats_layout.addWidget(self.stats_splitter)
        
        # Add visualization - PyQtGraph preferred, fallback to Matplotlib
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        
        # Graph title
        graph_title = QLabel("Language Model Activity")
        graph_title.setFont(QFont("Arial", 10, QFont.Bold))
        graph_title.setAlignment(Qt.AlignCenter)
        graph_layout.addWidget(graph_title)
        
        # Create graph
        if HAVE_PYQTGRAPH:
            self._setup_pyqtgraph(graph_layout)
        elif HAVE_MATPLOTLIB:
            self._setup_matplotlib(graph_layout)
        else:
            # Fallback to simple custom visualization
            self._setup_custom_visualization(graph_layout)
        
        # Add graph widget to stats splitter
        self.stats_splitter.addWidget(graph_widget)
        
        # Add recent conversations table
        self.conversations_widget = QWidget()
        conversations_layout = QVBoxLayout(self.conversations_widget)
        
        # Table header
        conversations_title = QLabel("Recent Conversations")
        conversations_title.setFont(QFont("Arial", 10, QFont.Bold))
        conversations_title.setAlignment(Qt.AlignCenter)
        conversations_layout.addWidget(conversations_title)
        
        # Create table
        self.conversations_table = QTableWidget(0, 4)
        self.conversations_table.setHorizontalHeaderLabels(["Time", "Length", "Response Time", "LLM Weight"])
        self.conversations_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.conversations_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.conversations_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.conversations_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        conversations_layout.addWidget(self.conversations_table)
        
        # Add conversations widget to stats splitter
        self.stats_splitter.addWidget(self.conversations_widget)
        
        # Add stats widget to main splitter
        self.splitter.addWidget(self.stats_widget)
        
        # Set splitter sizes
        self.splitter.setSizes([int(self.height() * 0.3), int(self.height() * 0.7)])
        self.stats_splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
    
    def _setup_pyqtgraph(self, layout):
        """Set up PyQtGraph visualization components"""
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground((40, 44, 52))
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Create plot items
        self.mistral_curve = self.plot_widget.plot([], [], 
                                                 pen=pg.mkPen(color=(216, 27, 96), width=2),
                                                 name="Mistral Activity")
        
        self.response_time_curve = self.plot_widget.plot([], [], 
                                                      pen=pg.mkPen(color=(255, 193, 7), width=2, style=Qt.DashLine),
                                                      name="Response Time")
        
        # Add legend
        legend = self.plot_widget.addLegend()
        
        # Add to layout
        layout.addWidget(self.plot_widget)

    def _setup_matplotlib(self, layout):
        """Set up Matplotlib visualization components as fallback"""
        # Create figure and canvas
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Create subplot
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title('Language Model Activity')
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Value')
        self.axes.grid(True, alpha=0.3)
        
        # Create empty lines
        self.mistral_line, = self.axes.plot([], [], label='Mistral Activity', color='#D81B60')
        self.response_time_line, = self.axes.plot([], [], label='Response Time', color='#FFC107', linestyle='--')
        
        # Add legend
        self.axes.legend()
        
        # Add to layout
        layout.addWidget(self.canvas)
    
    def _setup_custom_visualization(self, layout):
        """Set up simple custom visualization components as final fallback"""
        # Create custom visualization widget
        self.custom_viz = SimpleVisualizationWidget()
        layout.addWidget(self.custom_viz)
        
        # Add a note about missing visualization libraries
        note = QLabel("Note: Install PyQtGraph or Matplotlib for better visualizations")
        note.setStyleSheet("color: orange;")
        layout.addWidget(note)
    
    def refresh_data(self):
        """Refresh language processing data"""
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
            self.mistral_activity_history.append(metrics.get('mistral_activity', 0.0))
            self.response_time_history.append(metrics.get('response_time', 0.0))
            
            # Convert token usage to integer if needed
            token_usage = metrics.get('token_usage', 0)
            if isinstance(token_usage, str):
                try:
                    token_usage = int(token_usage)
                except ValueError:
                    token_usage = 0
            
            self.token_usage_history.append(token_usage)
            
            # Trim history to time window
            cutoff_time = current_time - timedelta(seconds=self.time_window)
            while self.timestamps and self.timestamps[0] < cutoff_time:
                self.timestamps.pop(0)
                self.mistral_activity_history.pop(0)
                self.response_time_history.pop(0)
                self.token_usage_history.pop(0)
            
            # Update the UI with new data
            self.update_signal.emit(metrics)
            
        except Exception as e:
            logger.error(f"Error refreshing language processing data: {e}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def _fetch_metrics_from_db(self) -> Dict:
        """Fetch the latest language metrics from the database"""
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query the latest language metrics
            cursor.execute("""
                SELECT value, description FROM language_metrics
            ORDER BY timestamp DESC 
                LIMIT 1
            """)
            language_row = cursor.fetchone()
            
            # If no data, return empty dict
            if not language_row:
            conn.close()
                return {}
                
            mistral_activity = language_row[0]
            
            # Query additional metrics
            metrics = {
                'mistral_activity': mistral_activity,
                'token_usage': 0,
                'response_time': 0.0,
                'semantic_score': 0.0,
                'llm_weight': 0.0,
                'memory_usage': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to get token usage
            cursor.execute("""
                SELECT value FROM token_usage
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            token_row = cursor.fetchone()
            if token_row:
                metrics['token_usage'] = token_row[0]
            
            # Try to get response time
            cursor.execute("""
                SELECT value FROM response_time
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            response_row = cursor.fetchone()
            if response_row:
                metrics['response_time'] = response_row[0]
            
            # Try to get LLM weight
            cursor.execute("""
                SELECT value FROM llm_weight
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            weight_row = cursor.fetchone()
            if weight_row:
                metrics['llm_weight'] = weight_row[0]
            
            # Get recent conversations
            cursor.execute("""
                SELECT timestamp, length, response_time, llm_weight
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            conversations = cursor.fetchall()
            if conversations:
                self.recent_conversations = conversations
            
            # Close the database connection
            conn.close()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            return {}
    
    def _generate_mock_metrics(self) -> Dict:
        """Generate mock metrics for testing"""
        # If we have history, make the new values follow a realistic pattern
        if self.mistral_activity_history:
            last_activity = self.mistral_activity_history[-1]
            last_response_time = self.response_time_history[-1]
            
            # Add some random walk with constraints
            activity = max(0.1, min(0.9, last_activity + random.uniform(-0.05, 0.05)))
            response_time = max(0.1, min(2.0, last_response_time + random.uniform(-0.1, 0.1)))
        else:
            # Initial values
            activity = random.uniform(0.4, 0.7)
            response_time = random.uniform(0.2, 1.0)
        
        # Generate token usage with occasional spikes
        if random.random() < 0.05:  # 5% chance of spike
            token_usage = random.randint(500, 1500)
        else:
            token_usage = random.randint(50, 300)
        
        # Generate related mock metrics
        metrics = {
            'mistral_activity': activity,
            'token_usage': token_usage,
            'response_time': response_time,
            'semantic_score': activity * 0.8 + random.uniform(-0.1, 0.1),
            'llm_weight': 0.5 + random.uniform(-0.1, 0.1),
            'memory_usage': 0.3 + activity * 0.3 + random.uniform(-0.05, 0.05),
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate mock conversation if needed
        if random.random() < 0.1:  # 10% chance to add new conversation
            # Generate a mock conversation
            time_ago = random.randint(1, 30)
            length = random.randint(5, 50)
            conv_response_time = random.uniform(0.2, 1.5)
            llm_weight = 0.5 + random.uniform(-0.2, 0.2)
            
            # Add to the front of the list
            timestamp = datetime.now() - timedelta(minutes=time_ago)
            new_conversation = (timestamp.strftime("%Y-%m-%d %H:%M:%S"), length, conv_response_time, llm_weight)
            
            if hasattr(self, 'recent_conversations'):
                self.recent_conversations.insert(0, new_conversation)
                # Trim to max 10 conversations
                if len(self.recent_conversations) > 10:
                    self.recent_conversations = self.recent_conversations[:10]
            else:
                self.recent_conversations = [new_conversation]
        
        # Ensure all values are in valid range and properly formatted
        for key in metrics:
            if key != 'timestamp' and key != 'token_usage' and isinstance(metrics[key], float):
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
                
                # Format value based on type
                if key == 'token_usage':
                    value_text = f"{value}"
                elif key == 'response_time':
                    value_text = f"{value:.2f}s"
                elif isinstance(value, float):
                    # Format as decimal with 2 decimal places
                    value_text = f"{value:.2f}"
                    
                    # Set color based on value (for float metrics)
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
        
        # Update conversations table
        if hasattr(self, 'recent_conversations') and self.recent_conversations:
            # Clear the table
            self.conversations_table.setRowCount(0)
            
            # Add new rows
            for i, (timestamp, length, response_time, llm_weight) in enumerate(self.recent_conversations):
                self.conversations_table.insertRow(i)
                
                # Add items
                self.conversations_table.setItem(i, 0, QTableWidgetItem(timestamp))
                self.conversations_table.setItem(i, 1, QTableWidgetItem(f"{length}"))
                self.conversations_table.setItem(i, 2, QTableWidgetItem(f"{response_time:.2f}s"))
                self.conversations_table.setItem(i, 3, QTableWidgetItem(f"{llm_weight:.2f}"))
                
                # Color coding based on response time
                if response_time < 0.5:
                    color = QColor(76, 175, 80, 100)  # Green
                elif response_time < 1.0:
                    color = QColor(33, 150, 243, 100)  # Blue
                else:
                    color = QColor(255, 152, 0, 100)  # Orange
                
                for col in range(4):
                    item = self.conversations_table.item(i, col)
                    item.setBackground(QBrush(color))
        
        # Update graph if we have history
        if self.timestamps:
            # Create time axis (seconds from now)
            now = datetime.now()
            time_axis = [(t - now).total_seconds() for t in self.timestamps]
            
            # Normalize response times to 0-1 scale for better visualization
            max_response_time = max(max(self.response_time_history, default=1.0), 1.0)
            normalized_response_times = [min(1.0, rt / max_response_time) for rt in self.response_time_history]
            
            # Update visualization based on available libraries
            if HAVE_PYQTGRAPH:
                self.mistral_curve.setData(time_axis, self.mistral_activity_history)
                self.response_time_curve.setData(time_axis, normalized_response_times)
                
            elif HAVE_MATPLOTLIB:
                self.mistral_line.set_data(time_axis, self.mistral_activity_history)
                self.response_time_line.set_data(time_axis, normalized_response_times)
                
                # Update axes limits
                self.axes.set_xlim(min(time_axis), max(time_axis))
                self.axes.set_ylim(0, 1)
                
                # Redraw
                self.figure.canvas.draw()
                
            else:
                # Update custom visualization
                self.custom_viz.update_data(self.mistral_activity_history, normalized_response_times)

class SimpleVisualizationWidget(QWidget):
    """Simple custom visualization widget as fallback when no graphing libraries are available"""
    
    def __init__(self, parent=None):
        """Initialize the simple visualization widget"""
        super().__init__(parent)
        self.activity_data = []
        self.response_time_data = []
        self.setMinimumHeight(200)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(40, 44, 52))
        self.setPalette(palette)
    
    def update_data(self, activity_data, response_time_data):
        """Update the data to visualize"""
        self.activity_data = activity_data
        self.response_time_data = response_time_data
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the visualization"""
        if not self.activity_data or not self.response_time_data:
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
            self._draw_line(painter, self.activity_data, QColor(216, 27, 96), height)
        
        if len(self.response_time_data) > 1:
            self._draw_line(painter, self.response_time_data, QColor(255, 193, 7), height, dash=True)
        
        # Draw legend
        painter.setPen(QColor(216, 27, 96))
        painter.drawText(10, 20, "Mistral Activity")
        painter.setPen(QColor(255, 193, 7))
        painter.drawText(10, 40, "Response Time")
    
    def _draw_line(self, painter, data, color, height, dash=False):
        """Draw a line series on the widget"""
        points = len(data)
        width = self.width()
        
        pen = QPen(color, 2)
        if dash:
            # Create dashed line
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        
        for i in range(points - 1):
            x1 = width * (i / (points - 1))
            y1 = height - (data[i] * height)
            x2 = width * ((i + 1) / (points - 1))
            y2 = height - (data[i + 1] * height)
            
            painter.drawLine(int(x1), int(y1), int(x2), int(y2)) 