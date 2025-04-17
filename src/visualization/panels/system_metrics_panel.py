#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Metrics Panel for LUMINA V7 Dashboard
===========================================

Panel for visualizing system resource usage and performance metrics.
"""

import os
import sys
import time
import random
import logging
import sqlite3
import platform
import psutil
try:
    import GPUtil
    HAVE_GPUTIL = True
except ImportError:
    HAVE_GPUTIL = False

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Import base panel
from src.visualization.panels.base_panel import BasePanel, QT_FRAMEWORK

# Qt compatibility layer
try:
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
        QGridLayout, QFrame, QSplitter, QProgressBar
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
            QGridLayout, QFrame, QSplitter, QProgressBar
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

class SystemMetricsPanel(BasePanel):
    """Panel for visualizing system resource usage and performance metrics"""
    
    def __init__(self, parent=None, db_path=None, refresh_rate=2000, active=True, gui_framework=None):
        """
        Initialize the system metrics panel
        
        Args:
            parent: Parent widget
            db_path: Path to metrics database
            refresh_rate: Data refresh rate in milliseconds
            active: Whether the panel is active at startup
            gui_framework: GUI framework to use
        """
        super().__init__(
            parent=parent, 
            panel_name="System Metrics", 
            db_path=db_path, 
            refresh_rate=refresh_rate, 
            active=active, 
            gui_framework=gui_framework
        )
        
        # Internal data structures
        self.time_window = 60  # 60 seconds of data
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []
        self.timestamps = []
        
        # Set up system info
        self.system_info = self._get_system_info()
        
        # Setup the specific UI components for system metrics
        self._setup_system_ui()
        
        # Initial refresh
        self.refresh_data()
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "hostname": platform.node(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total": psutil.virtual_memory().total
        }
        
        # Try to get GPU info if available
        if HAVE_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    info["gpu_name"] = gpus[0].name
                    info["gpu_memory"] = gpus[0].memoryTotal
                    info["gpu_count"] = len(gpus)
                else:
                    info["gpu_available"] = False
            except Exception as e:
                logger.warning(f"Error getting GPU info: {e}")
                info["gpu_available"] = False
        else:
            info["gpu_available"] = False
        
        return info
    
    def _setup_system_ui(self):
        """Set up specialized UI components for system metrics visualization"""
        # Clear the base layout (except for header)
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget not in (self.header_label, self.status_label, self.timestamp_label):
                widget.deleteLater()
        
        # Create a split layout
        self.splitter = QSplitter(Qt.Vertical)
        self.layout.insertWidget(3, self.splitter)  # Insert after header, status, and timestamp
        
        # Create system info panel (top)
        self.info_widget = QWidget()
        self.info_layout = QVBoxLayout(self.info_widget)
        
        # Add system info
        self.info_frame = QFrame()
        self.info_frame.setFrameShape(QFrame.StyledPanel)
        self.info_frame.setStyleSheet("background-color: rgba(60, 65, 75, 150); border-radius: 4px;")
        info_inner_layout = QGridLayout(self.info_frame)
        
        # Add system info fields
        labels = [
            ("System", self.system_info.get("os", "Unknown")),
            ("Version", self.system_info.get("os_version", "Unknown")),
            ("Hostname", self.system_info.get("hostname", "Unknown")),
            ("CPU Cores", f"{self.system_info.get('physical_cpu_count', 0)} physical / {self.system_info.get('cpu_count', 0)} logical"),
            ("Memory", f"{self.system_info.get('memory_total', 0) / (1024**3):.2f} GB")
        ]
        
        # Add GPU info if available
        if self.system_info.get("gpu_available", False):
            labels.append(("GPU", self.system_info.get("gpu_name", "Unknown")))
            labels.append(("GPU Memory", f"{self.system_info.get('gpu_memory', 0) / 1024:.2f} GB"))
        
        for i, (key, value) in enumerate(labels):
            row = i // 2
            col = (i % 2) * 2
            
            key_label = QLabel(f"{key}:")
            key_label.setFont(QFont("Arial", 9, QFont.Bold))
            info_inner_layout.addWidget(key_label, row, col)
            
            value_label = QLabel(str(value))
            info_inner_layout.addWidget(value_label, row, col + 1)
        
        self.info_layout.addWidget(self.info_frame)
        
        # Create resource usage panel (middle)
        self.resources_widget = QWidget()
        self.resources_layout = QGridLayout(self.resources_widget)
        
        # CPU usage
        cpu_label = QLabel("CPU Usage:")
        cpu_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.resources_layout.addWidget(cpu_label, 0, 0)
        
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        self.cpu_progress.setValue(0)
        self.cpu_progress.setTextVisible(True)
        self.cpu_progress.setFormat("%p%")
        self.resources_layout.addWidget(self.cpu_progress, 0, 1)
        
        # Memory usage
        memory_label = QLabel("Memory Usage:")
        memory_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.resources_layout.addWidget(memory_label, 1, 0)
        
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 100)
        self.memory_progress.setValue(0)
        self.memory_progress.setTextVisible(True)
        self.memory_progress.setFormat("%p%")
        self.resources_layout.addWidget(self.memory_progress, 1, 1)
        
        # GPU usage (if available)
        if self.system_info.get("gpu_available", False):
            gpu_label = QLabel("GPU Usage:")
            gpu_label.setFont(QFont("Arial", 10, QFont.Bold))
            self.resources_layout.addWidget(gpu_label, 2, 0)
            
            self.gpu_progress = QProgressBar()
            self.gpu_progress.setRange(0, 100)
            self.gpu_progress.setValue(0)
            self.gpu_progress.setTextVisible(True)
            self.gpu_progress.setFormat("%p%")
            self.resources_layout.addWidget(self.gpu_progress, 2, 1)
        
        # Create graph panel (bottom)
        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_widget)
        
        # Create graph title
        graph_title = QLabel("Resource Usage Over Time")
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
        
        # Add panels to splitter
        self.splitter.addWidget(self.info_widget)
        self.splitter.addWidget(self.resources_widget)
        self.splitter.addWidget(self.graph_widget)
        
        # Set splitter sizes
        self.splitter.setSizes([int(self.height() * 0.2), int(self.height() * 0.2), int(self.height() * 0.6)])
    
    def _setup_pyqtgraph(self):
        """Set up PyQtGraph visualization components"""
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground((40, 44, 52))
        self.plot_widget.setLabel('left', 'Usage %')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Create plot items
        self.cpu_curve = self.plot_widget.plot([], [], 
                                             pen=pg.mkPen(color=(76, 175, 80), width=2),
                                             name="CPU Usage")
        
        self.memory_curve = self.plot_widget.plot([], [], 
                                                pen=pg.mkPen(color=(33, 150, 243), width=2),
                                                name="Memory Usage")
        
        if self.system_info.get("gpu_available", False):
            self.gpu_curve = self.plot_widget.plot([], [], 
                                                 pen=pg.mkPen(color=(255, 87, 34), width=2),
                                                 name="GPU Usage")
        
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
        self.axes.set_title('Resource Usage Over Time')
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Usage %')
        self.axes.grid(True, alpha=0.3)
        
        # Create empty lines
        self.cpu_line, = self.axes.plot([], [], label='CPU Usage', color='#4CAF50')
        self.memory_line, = self.axes.plot([], [], label='Memory Usage', color='#2196F3')
        
        if self.system_info.get("gpu_available", False):
            self.gpu_line, = self.axes.plot([], [], label='GPU Usage', color='#FF5722')
        
        # Add legend
        self.axes.legend()
        
        # Add to layout
        self.graph_layout.addWidget(self.canvas)
    
    def _setup_custom_visualization(self):
        """Set up simple custom visualization components as final fallback"""
        # Create custom visualization widget
        self.custom_viz = SimpleVisualizationWidget(has_gpu=self.system_info.get("gpu_available", False))
        self.graph_layout.addWidget(self.custom_viz)
        
        # Add a note about missing visualization libraries
        note = QLabel("Note: Install PyQtGraph or Matplotlib for better visualizations")
        note.setStyleSheet("color: orange;")
        self.graph_layout.addWidget(note)
    
    def refresh_data(self):
        """Refresh system metrics data"""
        try:
            # Get current timestamp
            current_time = datetime.now()
            
            # Fetch real system metrics
            metrics = self._fetch_system_metrics()
            
            # If in mock mode, generate mock data
            if self.is_mock_mode:
                metrics = self._generate_mock_metrics()
            
            # Update history (keep only last time_window seconds)
            self.timestamps.append(current_time)
            self.cpu_history.append(metrics.get('cpu_percent', 0.0))
            self.memory_history.append(metrics.get('memory_percent', 0.0))
            
            if 'gpu_percent' in metrics:
                self.gpu_history.append(metrics.get('gpu_percent', 0.0))
            
            # Trim history to time window
            cutoff_time = current_time - timedelta(seconds=self.time_window)
            while self.timestamps and self.timestamps[0] < cutoff_time:
                self.timestamps.pop(0)
                self.cpu_history.pop(0)
                self.memory_history.pop(0)
                if self.gpu_history:
                    self.gpu_history.pop(0)
            
            # Update the UI with new data
            self.update_signal.emit(metrics)
            
        except Exception as e:
            logger.error(f"Error refreshing system metrics data: {e}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def _fetch_system_metrics(self) -> Dict:
        """Fetch the real system metrics"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat()
            }
            
            # Get CPU usage
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_used'] = memory.used
            metrics['memory_total'] = memory.total
            
            # Get GPU usage if available
            if HAVE_GPUTIL and self.system_info.get("gpu_available", False):
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        metrics['gpu_percent'] = gpus[0].load * 100
                        metrics['gpu_memory_used'] = gpus[0].memoryUsed
                        metrics['gpu_memory_total'] = gpus[0].memoryTotal
                except Exception as e:
                    logger.warning(f"Error getting GPU metrics: {e}")
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = disk.percent
            metrics['disk_used'] = disk.used
            metrics['disk_total'] = disk.total
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching system metrics: {e}")
            return {}
    
    def _generate_mock_metrics(self) -> Dict:
        """Generate mock system metrics for testing"""
        # If we have history, make the new values follow a realistic pattern
        if self.cpu_history:
            last_cpu = self.cpu_history[-1]
            last_memory = self.memory_history[-1]
            
            # Add some random walk with constraints
            cpu = max(5.0, min(95.0, last_cpu + random.uniform(-5.0, 5.0)))
            memory = max(10.0, min(90.0, last_memory + random.uniform(-2.0, 2.0)))
        else:
            # Initial values
            cpu = random.uniform(20.0, 60.0)
            memory = random.uniform(30.0, 70.0)
        
        # Generate mock metrics
        metrics = {
            'cpu_percent': cpu,
            'memory_percent': memory,
            'memory_used': (memory / 100.0) * self.system_info.get('memory_total', 8 * 1024**3),
            'memory_total': self.system_info.get('memory_total', 8 * 1024**3),
            'disk_percent': random.uniform(40.0, 80.0),
            'disk_used': random.uniform(50, 200) * 1024**3,
            'disk_total': 500 * 1024**3,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add GPU metrics if applicable
        if self.system_info.get("gpu_available", False):
            if self.gpu_history:
                last_gpu = self.gpu_history[-1]
                gpu = max(5.0, min(95.0, last_gpu + random.uniform(-8.0, 8.0)))
            else:
                gpu = random.uniform(10.0, 90.0)
                
            metrics['gpu_percent'] = gpu
            metrics['gpu_memory_used'] = (gpu / 100.0) * self.system_info.get('gpu_memory', 8 * 1024)
            metrics['gpu_memory_total'] = self.system_info.get('gpu_memory', 8 * 1024)
        
        return metrics
    
    def _update_ui_from_data(self, data):
        """Update all UI components with new data"""
        # First, call the parent class method
        super()._update_ui_from_data(data)
        
        # Update progress bars
        if 'cpu_percent' in data:
            self.cpu_progress.setValue(int(data['cpu_percent']))
            # Set color based on value
            self._set_progress_bar_color(self.cpu_progress, data['cpu_percent'])
            
        if 'memory_percent' in data:
            self.memory_progress.setValue(int(data['memory_percent']))
            # Set color based on value
            self._set_progress_bar_color(self.memory_progress, data['memory_percent'])
            
        if hasattr(self, 'gpu_progress') and 'gpu_percent' in data:
            self.gpu_progress.setValue(int(data['gpu_percent']))
            # Set color based on value
            self._set_progress_bar_color(self.gpu_progress, data['gpu_percent'])
        
        # Update graph if we have history
        if self.timestamps:
            # Create time axis (seconds from now)
            now = datetime.now()
            time_axis = [(t - now).total_seconds() for t in self.timestamps]
            
            # Update visualization based on available libraries
            if HAVE_PYQTGRAPH:
                self.cpu_curve.setData(time_axis, self.cpu_history)
                self.memory_curve.setData(time_axis, self.memory_history)
                
                if hasattr(self, 'gpu_curve') and self.gpu_history:
                    self.gpu_curve.setData(time_axis, self.gpu_history)
                
            elif HAVE_MATPLOTLIB:
                self.cpu_line.set_data(time_axis, self.cpu_history)
                self.memory_line.set_data(time_axis, self.memory_history)
                
                if hasattr(self, 'gpu_line') and self.gpu_history:
                    self.gpu_line.set_data(time_axis, self.gpu_history)
                
                # Update axes limits
                self.axes.set_xlim(min(time_axis), max(time_axis))
                self.axes.set_ylim(0, 100)
                
                # Redraw
                self.figure.canvas.draw()
                
            else:
                # Update custom visualization
                gpu_data = self.gpu_history if hasattr(self, 'gpu_history') else None
                self.custom_viz.update_data(self.cpu_history, self.memory_history, gpu_data)
    
    def _set_progress_bar_color(self, progress_bar, value):
        """Set progress bar color based on value"""
        if value < 60:
            # Green for low usage
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #555;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
            """)
        elif value < 80:
            # Yellow for medium usage
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #555;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #FFC107;
                }
            """)
        else:
            # Red for high usage
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #555;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #F44336;
                }
            """)

class SimpleVisualizationWidget(QWidget):
    """Simple custom visualization widget as fallback when no graphing libraries are available"""
    
    def __init__(self, parent=None, has_gpu=False):
        """Initialize the simple visualization widget"""
        super().__init__(parent)
        self.cpu_data = []
        self.memory_data = []
        self.gpu_data = []
        self.has_gpu = has_gpu
        self.setMinimumHeight(200)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(40, 44, 52))
        self.setPalette(palette)
    
    def update_data(self, cpu_data, memory_data, gpu_data=None):
        """Update the data to visualize"""
        self.cpu_data = cpu_data
        self.memory_data = memory_data
        if gpu_data is not None:
            self.gpu_data = gpu_data
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Paint the visualization"""
        if not self.cpu_data or not self.memory_data:
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
        if len(self.cpu_data) > 1:
            # Normalize CPU data to 0-1 for drawing
            normalized_cpu = [min(100, max(0, value)) / 100 for value in self.cpu_data]
            self._draw_line(painter, normalized_cpu, QColor(76, 175, 80), height)
        
        if len(self.memory_data) > 1:
            # Normalize memory data to 0-1 for drawing
            normalized_memory = [min(100, max(0, value)) / 100 for value in self.memory_data]
            self._draw_line(painter, normalized_memory, QColor(33, 150, 243), height)
        
        if self.has_gpu and len(self.gpu_data) > 1:
            # Normalize GPU data to 0-1 for drawing
            normalized_gpu = [min(100, max(0, value)) / 100 for value in self.gpu_data]
            self._draw_line(painter, normalized_gpu, QColor(255, 87, 34), height)
        
        # Draw legend
        painter.setPen(QColor(76, 175, 80))
        painter.drawText(10, 20, "CPU Usage")
        painter.setPen(QColor(33, 150, 243))
        painter.drawText(10, 40, "Memory Usage")
        if self.has_gpu:
            painter.setPen(QColor(255, 87, 34))
            painter.drawText(10, 60, "GPU Usage")
    
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