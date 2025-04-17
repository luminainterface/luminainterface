"""
Disk I/O graph visualization component for the Lumina Frontend System.
Displays real-time disk read and write operations.
"""

from typing import List, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import psutil
import numpy as np
from .base_visualization import BaseVisualization

class DiskIOGraph(BaseVisualization):
    """Widget for displaying disk I/O operations."""
    
    # Signal emitted when disk I/O data is updated
    data_updated = Signal(dict)
    
    def __init__(self):
        super().__init__("Disk I/O")
        self._data: Dict[str, List[float]] = {
            'read': [],
            'write': []
        }
        self._disk_io_counters: Dict[str, psutil._common.sdiskio] = {}
        self._selected_disk = ""
        
        # Create disk selector
        self._disk_selector = QComboBox()
        self._disk_selector.currentTextChanged.connect(self._on_disk_changed)
        self.layout().addWidget(self._disk_selector)
        
        # Create labels for current values
        self._read_label = QLabel("Read: 0 MB/s")
        self._write_label = QLabel("Write: 0 MB/s")
        self.layout().addWidget(self._read_label)
        self.layout().addWidget(self._write_label)
        
        # Populate disk list
        self._populate_disks()
    
    def _populate_disks(self) -> None:
        """Populate the disk selector with available disks."""
        self._disk_selector.clear()
        disks = psutil.disk_partitions(all=False)
        for disk in disks:
            self._disk_selector.addItem(disk.device)
        
        if self._disk_selector.count() > 0:
            self._selected_disk = self._disk_selector.currentText()
            self._initialize_disk_counters()
    
    def _initialize_disk_counters(self) -> None:
        """Initialize disk I/O counters for the selected disk."""
        self._disk_io_counters = {}
        for disk in psutil.disk_io_counters(perdisk=True):
            self._disk_io_counters[disk] = psutil.disk_io_counters(perdisk=True)[disk]
    
    def _on_disk_changed(self, disk: str) -> None:
        """Handle disk selection change."""
        self._selected_disk = disk
        self._data['read'].clear()
        self._data['write'].clear()
        self._initialize_disk_counters()
    
    def _update_data(self) -> None:
        """Update disk I/O data."""
        if self._paused or not self._selected_disk:
            return
            
        # Get current disk I/O counters
        current_counters = psutil.disk_io_counters(perdisk=True)
        if self._selected_disk not in current_counters:
            return
            
        current = current_counters[self._selected_disk]
        previous = self._disk_io_counters.get(self._selected_disk, current)
        
        # Calculate read and write rates (MB/s)
        read_rate = (current.read_bytes - previous.read_bytes) / (1024 * 1024)
        write_rate = (current.write_bytes - previous.write_bytes) / (1024 * 1024)
        
        # Update data arrays
        self._data['read'].append(read_rate)
        self._data['write'].append(write_rate)
        if len(self._data['read']) > self._max_points:
            self._data['read'].pop(0)
        if len(self._data['write']) > self._max_points:
            self._data['write'].pop(0)
        
        # Update labels
        self._read_label.setText(f"Read: {read_rate:.1f} MB/s")
        self._write_label.setText(f"Write: {write_rate:.1f} MB/s")
        
        # Update counters
        self._disk_io_counters[self._selected_disk] = current
        
        # Emit data update signal
        self.data_updated.emit({
            'disk': self._selected_disk,
            'read_rate': read_rate,
            'write_rate': write_rate,
            'read_bytes': current.read_bytes,
            'write_bytes': current.write_bytes
        })
        
        self.update()
    
    def _draw_visualization(self, painter: QPainter) -> None:
        """Draw the read and write graphs."""
        # Draw read graph (blue)
        self._draw_single_graph(painter, self._data['read'], QColor(0, 0, 255))
        
        # Draw write graph (yellow)
        self._draw_single_graph(painter, self._data['write'], QColor(255, 255, 0))
    
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
        
        # Find maximum value for scaling
        max_value = max(data) if data else 1
        
        for i, value in enumerate(data):
            x = (i / (self._max_points - 1)) * width
            y = height - (value / max_value * height)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
    
    def get_data(self) -> Dict[str, Any]:
        """Get the current disk I/O data."""
        data = super().get_data()
        data['selected_disk'] = self._selected_disk
        return data
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the disk I/O data."""
        super().set_data(data)
        if 'selected_disk' in data and data['selected_disk'] != self._selected_disk:
            self._selected_disk = data['selected_disk']
            self._disk_selector.setCurrentText(self._selected_disk)
            self._initialize_disk_counters() 
Disk I/O graph visualization component for the Lumina Frontend System.
Displays real-time disk read and write operations.
"""

from typing import List, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import psutil
import numpy as np
from .base_visualization import BaseVisualization

class DiskIOGraph(BaseVisualization):
    """Widget for displaying disk I/O operations."""
    
    # Signal emitted when disk I/O data is updated
    data_updated = Signal(dict)
    
    def __init__(self):
        super().__init__("Disk I/O")
        self._data: Dict[str, List[float]] = {
            'read': [],
            'write': []
        }
        self._disk_io_counters: Dict[str, psutil._common.sdiskio] = {}
        self._selected_disk = ""
        
        # Create disk selector
        self._disk_selector = QComboBox()
        self._disk_selector.currentTextChanged.connect(self._on_disk_changed)
        self.layout().addWidget(self._disk_selector)
        
        # Create labels for current values
        self._read_label = QLabel("Read: 0 MB/s")
        self._write_label = QLabel("Write: 0 MB/s")
        self.layout().addWidget(self._read_label)
        self.layout().addWidget(self._write_label)
        
        # Populate disk list
        self._populate_disks()
    
    def _populate_disks(self) -> None:
        """Populate the disk selector with available disks."""
        self._disk_selector.clear()
        disks = psutil.disk_partitions(all=False)
        for disk in disks:
            self._disk_selector.addItem(disk.device)
        
        if self._disk_selector.count() > 0:
            self._selected_disk = self._disk_selector.currentText()
            self._initialize_disk_counters()
    
    def _initialize_disk_counters(self) -> None:
        """Initialize disk I/O counters for the selected disk."""
        self._disk_io_counters = {}
        for disk in psutil.disk_io_counters(perdisk=True):
            self._disk_io_counters[disk] = psutil.disk_io_counters(perdisk=True)[disk]
    
    def _on_disk_changed(self, disk: str) -> None:
        """Handle disk selection change."""
        self._selected_disk = disk
        self._data['read'].clear()
        self._data['write'].clear()
        self._initialize_disk_counters()
    
    def _update_data(self) -> None:
        """Update disk I/O data."""
        if self._paused or not self._selected_disk:
            return
            
        # Get current disk I/O counters
        current_counters = psutil.disk_io_counters(perdisk=True)
        if self._selected_disk not in current_counters:
            return
            
        current = current_counters[self._selected_disk]
        previous = self._disk_io_counters.get(self._selected_disk, current)
        
        # Calculate read and write rates (MB/s)
        read_rate = (current.read_bytes - previous.read_bytes) / (1024 * 1024)
        write_rate = (current.write_bytes - previous.write_bytes) / (1024 * 1024)
        
        # Update data arrays
        self._data['read'].append(read_rate)
        self._data['write'].append(write_rate)
        if len(self._data['read']) > self._max_points:
            self._data['read'].pop(0)
        if len(self._data['write']) > self._max_points:
            self._data['write'].pop(0)
        
        # Update labels
        self._read_label.setText(f"Read: {read_rate:.1f} MB/s")
        self._write_label.setText(f"Write: {write_rate:.1f} MB/s")
        
        # Update counters
        self._disk_io_counters[self._selected_disk] = current
        
        # Emit data update signal
        self.data_updated.emit({
            'disk': self._selected_disk,
            'read_rate': read_rate,
            'write_rate': write_rate,
            'read_bytes': current.read_bytes,
            'write_bytes': current.write_bytes
        })
        
        self.update()
    
    def _draw_visualization(self, painter: QPainter) -> None:
        """Draw the read and write graphs."""
        # Draw read graph (blue)
        self._draw_single_graph(painter, self._data['read'], QColor(0, 0, 255))
        
        # Draw write graph (yellow)
        self._draw_single_graph(painter, self._data['write'], QColor(255, 255, 0))
    
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
        
        # Find maximum value for scaling
        max_value = max(data) if data else 1
        
        for i, value in enumerate(data):
            x = (i / (self._max_points - 1)) * width
            y = height - (value / max_value * height)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
    
    def get_data(self) -> Dict[str, Any]:
        """Get the current disk I/O data."""
        data = super().get_data()
        data['selected_disk'] = self._selected_disk
        return data
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the disk I/O data."""
        super().set_data(data)
        if 'selected_disk' in data and data['selected_disk'] != self._selected_disk:
            self._selected_disk = data['selected_disk']
            self._disk_selector.setCurrentText(self._selected_disk)
            self._initialize_disk_counters() 
 