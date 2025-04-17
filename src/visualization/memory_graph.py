"""
Memory graph visualization component for the Lumina Frontend System.
Displays real-time system memory usage including physical and swap memory.
"""

from typing import List, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import psutil
import numpy as np
from .base_visualization import BaseVisualization

class MemoryGraph(BaseVisualization):
    """Widget for displaying system memory usage."""
    
    # Signal emitted when memory data is updated
    data_updated = Signal(dict)
    
    def __init__(self):
        super().__init__("Memory Usage")
        self._data: Dict[str, List[float]] = {
            'physical': [],
            'swap': []
        }
        self._max_points = 100
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_data)
        
        # Create labels for current values
        self._physical_label = QLabel("Physical: 0%")
        self._swap_label = QLabel("Swap: 0%")
        self.layout().addWidget(self._physical_label)
        self.layout().addWidget(self._swap_label)
        
        # Set minimum size
        self.setMinimumSize(200, 100)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(palette)
    
    def initialize(self) -> None:
        """Initialize the graph."""
        self._update_timer.start(1000)  # Update every second
    
    def _update_data(self) -> None:
        """Update memory usage data."""
        if self._paused:
            return
            
        # Get memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Update physical memory data
        physical_percent = memory.percent
        self._data['physical'].append(physical_percent)
        if len(self._data['physical']) > self._max_points:
            self._data['physical'].pop(0)
        
        # Update swap memory data
        swap_percent = swap.percent
        self._data['swap'].append(swap_percent)
        if len(self._data['swap']) > self._max_points:
            self._data['swap'].pop(0)
        
        # Update labels
        self._physical_label.setText(f"Physical: {physical_percent:.1f}%")
        self._swap_label.setText(f"Swap: {swap_percent:.1f}%")
        
        # Emit data update signal
        self.data_updated.emit({
            'physical': physical_percent,
            'swap': swap_percent,
            'total_physical': memory.total,
            'used_physical': memory.used,
            'total_swap': swap.total,
            'used_swap': swap.used
        })
        
        self.update()
    
    def paintEvent(self, event) -> None:
        """Paint the graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Draw grid
        self._draw_grid(painter)
        
        # Draw graphs
        self._draw_visualization(painter)
    
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
    
    def _draw_visualization(self, painter: QPainter) -> None:
        """Draw the physical and swap memory graphs."""
        # Draw physical memory graph (green)
        self._draw_single_graph(painter, self._data['physical'], QColor(0, 255, 0))
        
        # Draw swap memory graph (red)
        self._draw_single_graph(painter, self._data['swap'], QColor(255, 0, 0))
    
    def _draw_single_graph(self, painter: QPainter, data: List[float], color: QColor) -> None:
        """Draw a single graph line."""
        if not data:
            return
            
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
        """Get the current memory data."""
        data = super().get_data()
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        data.update({
            'total_physical': memory.total,
            'total_swap': swap.total
        })
        return data
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the memory data."""
        super().set_data(data)
        if 'total_physical' in data:
            self._data['physical'].clear()
        if 'total_swap' in data:
            self._data['swap'].clear()
    
    def shutdown(self) -> None:
        """Shutdown the graph."""
        self._update_timer.stop()
        self._data['physical'].clear()
        self._data['swap'].clear() 
Memory graph visualization component for the Lumina Frontend System.
Displays real-time system memory usage including physical and swap memory.
"""

from typing import List, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import psutil
import numpy as np
from .base_visualization import BaseVisualization

class MemoryGraph(BaseVisualization):
    """Widget for displaying system memory usage."""
    
    # Signal emitted when memory data is updated
    data_updated = Signal(dict)
    
    def __init__(self):
        super().__init__("Memory Usage")
        self._data: Dict[str, List[float]] = {
            'physical': [],
            'swap': []
        }
        self._max_points = 100
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_data)
        
        # Create labels for current values
        self._physical_label = QLabel("Physical: 0%")
        self._swap_label = QLabel("Swap: 0%")
        self.layout().addWidget(self._physical_label)
        self.layout().addWidget(self._swap_label)
        
        # Set minimum size
        self.setMinimumSize(200, 100)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(palette)
    
    def initialize(self) -> None:
        """Initialize the graph."""
        self._update_timer.start(1000)  # Update every second
    
    def _update_data(self) -> None:
        """Update memory usage data."""
        if self._paused:
            return
            
        # Get memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Update physical memory data
        physical_percent = memory.percent
        self._data['physical'].append(physical_percent)
        if len(self._data['physical']) > self._max_points:
            self._data['physical'].pop(0)
        
        # Update swap memory data
        swap_percent = swap.percent
        self._data['swap'].append(swap_percent)
        if len(self._data['swap']) > self._max_points:
            self._data['swap'].pop(0)
        
        # Update labels
        self._physical_label.setText(f"Physical: {physical_percent:.1f}%")
        self._swap_label.setText(f"Swap: {swap_percent:.1f}%")
        
        # Emit data update signal
        self.data_updated.emit({
            'physical': physical_percent,
            'swap': swap_percent,
            'total_physical': memory.total,
            'used_physical': memory.used,
            'total_swap': swap.total,
            'used_swap': swap.used
        })
        
        self.update()
    
    def paintEvent(self, event) -> None:
        """Paint the graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Draw grid
        self._draw_grid(painter)
        
        # Draw graphs
        self._draw_visualization(painter)
    
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
    
    def _draw_visualization(self, painter: QPainter) -> None:
        """Draw the physical and swap memory graphs."""
        # Draw physical memory graph (green)
        self._draw_single_graph(painter, self._data['physical'], QColor(0, 255, 0))
        
        # Draw swap memory graph (red)
        self._draw_single_graph(painter, self._data['swap'], QColor(255, 0, 0))
    
    def _draw_single_graph(self, painter: QPainter, data: List[float], color: QColor) -> None:
        """Draw a single graph line."""
        if not data:
            return
            
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
        """Get the current memory data."""
        data = super().get_data()
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        data.update({
            'total_physical': memory.total,
            'total_swap': swap.total
        })
        return data
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the memory data."""
        super().set_data(data)
        if 'total_physical' in data:
            self._data['physical'].clear()
        if 'total_swap' in data:
            self._data['swap'].clear()
    
    def shutdown(self) -> None:
        """Shutdown the graph."""
        self._update_timer.stop()
        self._data['physical'].clear()
        self._data['swap'].clear() 
 