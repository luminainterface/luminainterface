"""
CPU graph visualization component for the Lumina Frontend System.
Displays real-time CPU usage metrics.
"""

from typing import List, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import psutil
import numpy as np
from .base_visualization import BaseVisualization

class CPUGraph(BaseVisualization):
    """Widget for displaying CPU usage graph."""
    
    def __init__(self):
        super().__init__("CPU Usage")
        self._data: Dict[str, List[float]] = {
            'total': [],
            'cores': {}
        }
        
        # Initialize core data
        for i in range(psutil.cpu_count()):
            self._data['cores'][f'core_{i}'] = []
        
        # Set minimum size
        self.setMinimumSize(200, 100)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(palette)
    
    def _update_data(self) -> None:
        """Update CPU usage data."""
        if self._paused:
            return
            
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Update total CPU data
        self._data['total'].append(cpu_percent)
        if len(self._data['total']) > self._max_points:
            self._data['total'].pop(0)
        
        # Update per-core data
        for i, core_percent in enumerate(per_cpu):
            core_key = f'core_{i}'
            self._data['cores'][core_key].append(core_percent)
            if len(self._data['cores'][core_key]) > self._max_points:
                self._data['cores'][core_key].pop(0)
        
        # Emit data update signal
        self.data_updated.emit({
            'total': cpu_percent,
            'cores': per_cpu,
            'core_count': len(per_cpu)
        })
        
        self.update()
    
    def _draw_visualization(self, painter: QPainter) -> None:
        """Draw the CPU usage graph."""
        if not self._data['total']:
            return
            
        # Draw total CPU usage (white)
        self._draw_single_graph(painter, self._data['total'], QColor(255, 255, 255))
        
        # Draw per-core usage with different colors
        colors = [
            QColor(255, 0, 0),    # Red
            QColor(0, 255, 0),    # Green
            QColor(0, 0, 255),    # Blue
            QColor(255, 255, 0),  # Yellow
            QColor(255, 0, 255),  # Magenta
            QColor(0, 255, 255),  # Cyan
            QColor(128, 0, 0),    # Dark Red
            QColor(0, 128, 0)     # Dark Green
        ]
        
        for i, (core_key, core_data) in enumerate(self._data['cores'].items()):
            color = colors[i % len(colors)]
            self._draw_single_graph(painter, core_data, color)
    
    def _draw_single_graph(self, painter: QPainter, data: List[float], color: QColor) -> None:
        """Draw a single graph line."""
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        
        width = self.width()
        height = self.height()
        points = []
        
        for i, value in enumerate(data):
            x = (i / (self._max_points - 1)) * width
            y = height - (value / 100 * height)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
    
    def get_data(self) -> Dict[str, Any]:
        """Get the current CPU data."""
        data = super().get_data()
        data['cpu_count'] = psutil.cpu_count()
        return data
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the CPU data."""
        super().set_data(data)
        if 'cpu_count' in data:
            # Reinitialize core data if CPU count changed
            current_count = psutil.cpu_count()
            if data['cpu_count'] != current_count:
                self._data['cores'].clear()
                for i in range(current_count):
                    self._data['cores'][f'core_{i}'] = []
    
    def paintEvent(self, event) -> None:
        """Paint the graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Draw grid
        self._draw_grid(painter)
        
        # Draw graph
        self._draw_visualization(painter)
        
        # Draw labels
        self._draw_labels(painter)
    
    def _draw_grid(self, painter: QPainter) -> None:
        """Draw the grid lines."""
        pen = QPen(QColor(60, 60, 60))
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        
        # Draw horizontal grid lines
        height = self.height()
        width = self.width()
        for i in range(0, 101, 20):
            y = height - (i / 100 * height)
            painter.drawLine(0, y, width, y)
    
    def _draw_labels(self, painter: QPainter) -> None:
        """Draw the graph labels."""
        pen = QPen(QColor(200, 200, 200))
        painter.setPen(pen)
        
        # Draw CPU usage percentage
        if self._data['total']:
            usage = self._data['total'][-1]
            painter.drawText(10, 20, f"CPU: {usage:.1f}%")
    
    def shutdown(self) -> None:
        """Shutdown the graph."""
        self._paused = True
        self._data['total'].clear()
        for core_data in self._data['cores'].values():
            core_data.clear()
        self.update() 
CPU graph visualization component for the Lumina Frontend System.
Displays real-time CPU usage metrics.
"""

from typing import List, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import psutil
import numpy as np
from .base_visualization import BaseVisualization

class CPUGraph(BaseVisualization):
    """Widget for displaying CPU usage graph."""
    
    def __init__(self):
        super().__init__("CPU Usage")
        self._data: Dict[str, List[float]] = {
            'total': [],
            'cores': {}
        }
        
        # Initialize core data
        for i in range(psutil.cpu_count()):
            self._data['cores'][f'core_{i}'] = []
        
        # Set minimum size
        self.setMinimumSize(200, 100)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(palette)
    
    def _update_data(self) -> None:
        """Update CPU usage data."""
        if self._paused:
            return
            
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Update total CPU data
        self._data['total'].append(cpu_percent)
        if len(self._data['total']) > self._max_points:
            self._data['total'].pop(0)
        
        # Update per-core data
        for i, core_percent in enumerate(per_cpu):
            core_key = f'core_{i}'
            self._data['cores'][core_key].append(core_percent)
            if len(self._data['cores'][core_key]) > self._max_points:
                self._data['cores'][core_key].pop(0)
        
        # Emit data update signal
        self.data_updated.emit({
            'total': cpu_percent,
            'cores': per_cpu,
            'core_count': len(per_cpu)
        })
        
        self.update()
    
    def _draw_visualization(self, painter: QPainter) -> None:
        """Draw the CPU usage graph."""
        if not self._data['total']:
            return
            
        # Draw total CPU usage (white)
        self._draw_single_graph(painter, self._data['total'], QColor(255, 255, 255))
        
        # Draw per-core usage with different colors
        colors = [
            QColor(255, 0, 0),    # Red
            QColor(0, 255, 0),    # Green
            QColor(0, 0, 255),    # Blue
            QColor(255, 255, 0),  # Yellow
            QColor(255, 0, 255),  # Magenta
            QColor(0, 255, 255),  # Cyan
            QColor(128, 0, 0),    # Dark Red
            QColor(0, 128, 0)     # Dark Green
        ]
        
        for i, (core_key, core_data) in enumerate(self._data['cores'].items()):
            color = colors[i % len(colors)]
            self._draw_single_graph(painter, core_data, color)
    
    def _draw_single_graph(self, painter: QPainter, data: List[float], color: QColor) -> None:
        """Draw a single graph line."""
        pen = QPen(color)
        pen.setWidth(2)
        painter.setPen(pen)
        
        width = self.width()
        height = self.height()
        points = []
        
        for i, value in enumerate(data):
            x = (i / (self._max_points - 1)) * width
            y = height - (value / 100 * height)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
    
    def get_data(self) -> Dict[str, Any]:
        """Get the current CPU data."""
        data = super().get_data()
        data['cpu_count'] = psutil.cpu_count()
        return data
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the CPU data."""
        super().set_data(data)
        if 'cpu_count' in data:
            # Reinitialize core data if CPU count changed
            current_count = psutil.cpu_count()
            if data['cpu_count'] != current_count:
                self._data['cores'].clear()
                for i in range(current_count):
                    self._data['cores'][f'core_{i}'] = []
    
    def paintEvent(self, event) -> None:
        """Paint the graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Draw grid
        self._draw_grid(painter)
        
        # Draw graph
        self._draw_visualization(painter)
        
        # Draw labels
        self._draw_labels(painter)
    
    def _draw_grid(self, painter: QPainter) -> None:
        """Draw the grid lines."""
        pen = QPen(QColor(60, 60, 60))
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        
        # Draw horizontal grid lines
        height = self.height()
        width = self.width()
        for i in range(0, 101, 20):
            y = height - (i / 100 * height)
            painter.drawLine(0, y, width, y)
    
    def _draw_labels(self, painter: QPainter) -> None:
        """Draw the graph labels."""
        pen = QPen(QColor(200, 200, 200))
        painter.setPen(pen)
        
        # Draw CPU usage percentage
        if self._data['total']:
            usage = self._data['total'][-1]
            painter.drawText(10, 20, f"CPU: {usage:.1f}%")
    
    def shutdown(self) -> None:
        """Shutdown the graph."""
        self._paused = True
        self._data['total'].clear()
        for core_data in self._data['cores'].values():
            core_data.clear()
        self.update() 
 