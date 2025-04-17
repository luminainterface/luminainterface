"""
Process graph visualization component for the Lumina Frontend System.
Displays real-time resource usage for selected processes.
"""

from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QProgressBar
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import psutil
import numpy as np
from .base_visualization import BaseVisualization

class ProcessGraph(BaseVisualization):
    """Widget for displaying process resource usage."""
    
    # Signal emitted when process data is updated
    data_updated = Signal(dict)
    
    def __init__(self):
        super().__init__("Process Usage")
        self._data: Dict[str, List[float]] = {
            'cpu': [],
            'memory': []
        }
        self._max_points = 100
        self._selected_pid: Optional[int] = None
        self._process_list: Dict[int, str] = {}
        
        # Create process selector
        self._process_selector = QComboBox()
        self._process_selector.currentTextChanged.connect(self._on_process_changed)
        self.layout().addWidget(self._process_selector)
        
        # Create progress bars for current values
        self._cpu_bar = QProgressBar()
        self._cpu_bar.setRange(0, 100)
        self._cpu_bar.setTextVisible(True)
        self._cpu_bar.setFormat("CPU: %p%")
        self.layout().addWidget(self._cpu_bar)
        
        self._memory_bar = QProgressBar()
        self._memory_bar.setRange(0, 100)
        self._memory_bar.setTextVisible(True)
        self._memory_bar.setFormat("Memory: %p%")
        self.layout().addWidget(self._memory_bar)
        
        # Set minimum size
        self.setMinimumSize(200, 100)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(palette)
        
        # Populate process list
        self._populate_processes()
    
    def _populate_processes(self) -> None:
        """Populate the process selector with running processes."""
        self._process_selector.clear()
        self._process_list.clear()
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = proc.info
                if info['cpu_percent'] is not None and info['memory_percent'] is not None:
                    display_name = f"{info['name']} (PID: {info['pid']})"
                    self._process_selector.addItem(display_name)
                    self._process_list[info['pid']] = display_name
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if self._process_selector.count() > 0:
            self._selected_pid = int(self._process_selector.currentText().split('(PID: ')[1].rstrip(')'))
    
    def _on_process_changed(self, process_text: str) -> None:
        """Handle process selection change."""
        if not process_text:
            return
            
        try:
            self._selected_pid = int(process_text.split('(PID: ')[1].rstrip(')'))
            self._data['cpu'].clear()
            self._data['memory'].clear()
        except (IndexError, ValueError):
            self._selected_pid = None
    
    def _update_data(self) -> None:
        """Update process data."""
        if self._paused or not self._selected_pid:
            return
            
        try:
            process = psutil.Process(self._selected_pid)
            cpu_percent = process.cpu_percent()
            memory_percent = process.memory_percent()
            
            # Update data arrays
            self._data['cpu'].append(cpu_percent)
            self._data['memory'].append(memory_percent)
            if len(self._data['cpu']) > self._max_points:
                self._data['cpu'].pop(0)
            if len(self._data['memory']) > self._max_points:
                self._data['memory'].pop(0)
            
            # Update progress bars
            self._cpu_bar.setValue(int(cpu_percent))
            self._memory_bar.setValue(int(memory_percent))
            
            # Emit data update signal
            self.data_updated.emit({
                'pid': self._selected_pid,
                'name': process.name(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used': process.memory_info().rss / (1024 * 1024)  # MB
            })
            
            self.update()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process no longer exists, refresh process list
            self._populate_processes()
    
    def _draw_visualization(self, painter: QPainter) -> None:
        """Draw the CPU and memory graphs."""
        # Draw CPU graph (blue)
        self._draw_single_graph(painter, self._data['cpu'], QColor(0, 0, 255))
        
        # Draw memory graph (yellow)
        self._draw_single_graph(painter, self._data['memory'], QColor(255, 255, 0))
    
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
            y = height - (value / 100 * height)  # Scale to percentage
            points.append((x, y))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
    
    def get_data(self) -> Dict[str, Any]:
        """Get the current process data."""
        data = super().get_data()
        data['selected_pid'] = self._selected_pid
        return data
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the process data."""
        super().set_data(data)
        if 'selected_pid' in data and data['selected_pid'] != self._selected_pid:
            self._selected_pid = data['selected_pid']
            if self._selected_pid in self._process_list:
                self._process_selector.setCurrentText(self._process_list[self._selected_pid])
    
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
    
    def set_version(self, version: str) -> None:
        """Set the version for the graph."""
        self._version = version
        self.update()  # Redraw with version-specific styling
    
    def shutdown(self) -> None:
        """Shutdown the graph."""
        self._update_timer.stop()
        self._data['cpu'].clear()
        self._data['memory'].clear() 
Process graph visualization component for the Lumina Frontend System.
Displays real-time resource usage for selected processes.
"""

from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QProgressBar
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import psutil
import numpy as np
from .base_visualization import BaseVisualization

class ProcessGraph(BaseVisualization):
    """Widget for displaying process resource usage."""
    
    # Signal emitted when process data is updated
    data_updated = Signal(dict)
    
    def __init__(self):
        super().__init__("Process Usage")
        self._data: Dict[str, List[float]] = {
            'cpu': [],
            'memory': []
        }
        self._max_points = 100
        self._selected_pid: Optional[int] = None
        self._process_list: Dict[int, str] = {}
        
        # Create process selector
        self._process_selector = QComboBox()
        self._process_selector.currentTextChanged.connect(self._on_process_changed)
        self.layout().addWidget(self._process_selector)
        
        # Create progress bars for current values
        self._cpu_bar = QProgressBar()
        self._cpu_bar.setRange(0, 100)
        self._cpu_bar.setTextVisible(True)
        self._cpu_bar.setFormat("CPU: %p%")
        self.layout().addWidget(self._cpu_bar)
        
        self._memory_bar = QProgressBar()
        self._memory_bar.setRange(0, 100)
        self._memory_bar.setTextVisible(True)
        self._memory_bar.setFormat("Memory: %p%")
        self.layout().addWidget(self._memory_bar)
        
        # Set minimum size
        self.setMinimumSize(200, 100)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(30, 30, 30))
        self.setPalette(palette)
        
        # Populate process list
        self._populate_processes()
    
    def _populate_processes(self) -> None:
        """Populate the process selector with running processes."""
        self._process_selector.clear()
        self._process_list.clear()
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = proc.info
                if info['cpu_percent'] is not None and info['memory_percent'] is not None:
                    display_name = f"{info['name']} (PID: {info['pid']})"
                    self._process_selector.addItem(display_name)
                    self._process_list[info['pid']] = display_name
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if self._process_selector.count() > 0:
            self._selected_pid = int(self._process_selector.currentText().split('(PID: ')[1].rstrip(')'))
    
    def _on_process_changed(self, process_text: str) -> None:
        """Handle process selection change."""
        if not process_text:
            return
            
        try:
            self._selected_pid = int(process_text.split('(PID: ')[1].rstrip(')'))
            self._data['cpu'].clear()
            self._data['memory'].clear()
        except (IndexError, ValueError):
            self._selected_pid = None
    
    def _update_data(self) -> None:
        """Update process data."""
        if self._paused or not self._selected_pid:
            return
            
        try:
            process = psutil.Process(self._selected_pid)
            cpu_percent = process.cpu_percent()
            memory_percent = process.memory_percent()
            
            # Update data arrays
            self._data['cpu'].append(cpu_percent)
            self._data['memory'].append(memory_percent)
            if len(self._data['cpu']) > self._max_points:
                self._data['cpu'].pop(0)
            if len(self._data['memory']) > self._max_points:
                self._data['memory'].pop(0)
            
            # Update progress bars
            self._cpu_bar.setValue(int(cpu_percent))
            self._memory_bar.setValue(int(memory_percent))
            
            # Emit data update signal
            self.data_updated.emit({
                'pid': self._selected_pid,
                'name': process.name(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used': process.memory_info().rss / (1024 * 1024)  # MB
            })
            
            self.update()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process no longer exists, refresh process list
            self._populate_processes()
    
    def _draw_visualization(self, painter: QPainter) -> None:
        """Draw the CPU and memory graphs."""
        # Draw CPU graph (blue)
        self._draw_single_graph(painter, self._data['cpu'], QColor(0, 0, 255))
        
        # Draw memory graph (yellow)
        self._draw_single_graph(painter, self._data['memory'], QColor(255, 255, 0))
    
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
            y = height - (value / 100 * height)  # Scale to percentage
            points.append((x, y))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
    
    def get_data(self) -> Dict[str, Any]:
        """Get the current process data."""
        data = super().get_data()
        data['selected_pid'] = self._selected_pid
        return data
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the process data."""
        super().set_data(data)
        if 'selected_pid' in data and data['selected_pid'] != self._selected_pid:
            self._selected_pid = data['selected_pid']
            if self._selected_pid in self._process_list:
                self._process_selector.setCurrentText(self._process_list[self._selected_pid])
    
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
    
    def set_version(self, version: str) -> None:
        """Set the version for the graph."""
        self._version = version
        self.update()  # Redraw with version-specific styling
    
    def shutdown(self) -> None:
        """Shutdown the graph."""
        self._update_timer.stop()
        self._data['cpu'].clear()
        self._data['memory'].clear() 
 