"""
Base visualization component for the Lumina Frontend System.
Provides common functionality for all visualization components.
"""

from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import numpy as np

class BaseVisualization(QWidget):
    """Base class for all visualization components."""
    
    # Signal emitted when visualization data is updated
    data_updated = Signal(dict)
    
    def __init__(self, title: str = ""):
        super().__init__()
        self._version = "v7.5"
        self._title = title
        self._data: Dict[str, List[float]] = {}
        self._max_points = 100
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_data)
        self._paused = False
        self._auto_scale = True
        self._time_range = 60  # seconds
        
        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create title label
        if title:
            self._title_label = QLabel(title)
            self._title_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self._title_label)
        
        # Set minimum size
        self.setMinimumSize(200, 100)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(palette)
    
    def initialize(self) -> None:
        """Initialize the visualization."""
        self._update_timer.start(1000)  # Update every second
    
    def _update_data(self) -> None:
        """Update visualization data. Override in subclasses."""
        if not self._paused:
            self.update()
    
    def paintEvent(self, event) -> None:
        """Paint the visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Draw grid
        self._draw_grid(painter)
        
        # Draw visualization
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
        """Draw the visualization. Override in subclasses."""
        pass
    
    def set_version(self, version: str) -> None:
        """Set the version for the visualization."""
        self._version = version
        self.update()  # Redraw with version-specific styling
    
    def set_paused(self, paused: bool) -> None:
        """Set whether the visualization is paused."""
        self._paused = paused
    
    def set_auto_scale(self, auto_scale: bool) -> None:
        """Set whether the visualization should auto-scale."""
        self._auto_scale = auto_scale
        self.update()
    
    def set_time_range(self, time_range: int) -> None:
        """Set the time range for the visualization in seconds."""
        self._time_range = time_range
        self._max_points = time_range  # One point per second
        self.update()
    
    def get_data(self) -> Dict[str, Any]:
        """Get the current visualization data."""
        return {
            'version': self._version,
            'paused': self._paused,
            'auto_scale': self._auto_scale,
            'time_range': self._time_range,
            'data': self._data.copy()
        }
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the visualization data."""
        if 'version' in data:
            self._version = data['version']
        if 'paused' in data:
            self._paused = data['paused']
        if 'auto_scale' in data:
            self._auto_scale = data['auto_scale']
        if 'time_range' in data:
            self._time_range = data['time_range']
        if 'data' in data:
            self._data = data['data'].copy()
        self.update()
    
    def shutdown(self) -> None:
        """Shutdown the visualization."""
        self._update_timer.stop()
        self._data.clear() 
Base visualization component for the Lumina Frontend System.
Provides common functionality for all visualization components.
"""

from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import numpy as np

class BaseVisualization(QWidget):
    """Base class for all visualization components."""
    
    # Signal emitted when visualization data is updated
    data_updated = Signal(dict)
    
    def __init__(self, title: str = ""):
        super().__init__()
        self._version = "v7.5"
        self._title = title
        self._data: Dict[str, List[float]] = {}
        self._max_points = 100
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_data)
        self._paused = False
        self._auto_scale = True
        self._time_range = 60  # seconds
        
        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create title label
        if title:
            self._title_label = QLabel(title)
            self._title_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self._title_label)
        
        # Set minimum size
        self.setMinimumSize(200, 100)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(palette)
    
    def initialize(self) -> None:
        """Initialize the visualization."""
        self._update_timer.start(1000)  # Update every second
    
    def _update_data(self) -> None:
        """Update visualization data. Override in subclasses."""
        if not self._paused:
            self.update()
    
    def paintEvent(self, event) -> None:
        """Paint the visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Draw grid
        self._draw_grid(painter)
        
        # Draw visualization
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
        """Draw the visualization. Override in subclasses."""
        pass
    
    def set_version(self, version: str) -> None:
        """Set the version for the visualization."""
        self._version = version
        self.update()  # Redraw with version-specific styling
    
    def set_paused(self, paused: bool) -> None:
        """Set whether the visualization is paused."""
        self._paused = paused
    
    def set_auto_scale(self, auto_scale: bool) -> None:
        """Set whether the visualization should auto-scale."""
        self._auto_scale = auto_scale
        self.update()
    
    def set_time_range(self, time_range: int) -> None:
        """Set the time range for the visualization in seconds."""
        self._time_range = time_range
        self._max_points = time_range  # One point per second
        self.update()
    
    def get_data(self) -> Dict[str, Any]:
        """Get the current visualization data."""
        return {
            'version': self._version,
            'paused': self._paused,
            'auto_scale': self._auto_scale,
            'time_range': self._time_range,
            'data': self._data.copy()
        }
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the visualization data."""
        if 'version' in data:
            self._version = data['version']
        if 'paused' in data:
            self._paused = data['paused']
        if 'auto_scale' in data:
            self._auto_scale = data['auto_scale']
        if 'time_range' in data:
            self._time_range = data['time_range']
        if 'data' in data:
            self._data = data['data'].copy()
        self.update()
    
    def shutdown(self) -> None:
        """Shutdown the visualization."""
        self._update_timer.stop()
        self._data.clear() 
 