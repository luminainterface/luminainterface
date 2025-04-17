"""
Network graph visualization component for the Lumina Frontend System.
Displays real-time network traffic for selected network interfaces.
"""

from typing import List, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import psutil
import numpy as np
from .base_visualization import BaseVisualization

class NetworkGraph(BaseVisualization):
    """Widget for displaying network traffic."""
    
    # Signal emitted when network data is updated
    data_updated = Signal(dict)
    
    def __init__(self):
        super().__init__("Network Traffic")
        self._data: Dict[str, List[float]] = {
            'download': [],
            'upload': []
        }
        self._network_counters: Dict[str, psutil._common.snetio] = {}
        self._selected_interface = ""
        
        # Create interface selector
        self._interface_selector = QComboBox()
        self._interface_selector.currentTextChanged.connect(self._on_interface_changed)
        self.layout().addWidget(self._interface_selector)
        
        # Create labels for current values
        self._download_label = QLabel("Download: 0 MB/s")
        self._upload_label = QLabel("Upload: 0 MB/s")
        self.layout().addWidget(self._download_label)
        self.layout().addWidget(self._upload_label)
        
        # Populate interface list
        self._populate_interfaces()
    
    def _populate_interfaces(self) -> None:
        """Populate the interface selector with available network interfaces."""
        self._interface_selector.clear()
        interfaces = psutil.net_if_stats()
        for interface in interfaces:
            if interfaces[interface].isup:  # Only show active interfaces
                self._interface_selector.addItem(interface)
        
        if self._interface_selector.count() > 0:
            self._selected_interface = self._interface_selector.currentText()
            self._initialize_network_counters()
    
    def _initialize_network_counters(self) -> None:
        """Initialize network counters for the selected interface."""
        self._network_counters = {}
        for interface in psutil.net_io_counters(pernic=True):
            self._network_counters[interface] = psutil.net_io_counters(pernic=True)[interface]
    
    def _on_interface_changed(self, interface: str) -> None:
        """Handle interface selection change."""
        self._selected_interface = interface
        self._data['download'].clear()
        self._data['upload'].clear()
        self._initialize_network_counters()
    
    def _update_data(self) -> None:
        """Update network data."""
        if self._paused or not self._selected_interface:
            return
            
        # Get current network counters
        current_counters = psutil.net_io_counters(pernic=True)
        if self._selected_interface not in current_counters:
            return
            
        current = current_counters[self._selected_interface]
        previous = self._network_counters.get(self._selected_interface, current)
        
        # Calculate download and upload rates (MB/s)
        download_rate = (current.bytes_recv - previous.bytes_recv) / (1024 * 1024)
        upload_rate = (current.bytes_sent - previous.bytes_sent) / (1024 * 1024)
        
        # Update data arrays
        self._data['download'].append(download_rate)
        self._data['upload'].append(upload_rate)
        if len(self._data['download']) > self._max_points:
            self._data['download'].pop(0)
        if len(self._data['upload']) > self._max_points:
            self._data['upload'].pop(0)
        
        # Update labels
        self._download_label.setText(f"Download: {download_rate:.1f} MB/s")
        self._upload_label.setText(f"Upload: {upload_rate:.1f} MB/s")
        
        # Update counters
        self._network_counters[self._selected_interface] = current
        
        # Emit data update signal
        self.data_updated.emit({
            'interface': self._selected_interface,
            'download_rate': download_rate,
            'upload_rate': upload_rate,
            'bytes_recv': current.bytes_recv,
            'bytes_sent': current.bytes_sent
        })
        
        self.update()
    
    def _draw_visualization(self, painter: QPainter) -> None:
        """Draw the download and upload graphs."""
        # Draw download graph (green)
        self._draw_single_graph(painter, self._data['download'], QColor(0, 255, 0))
        
        # Draw upload graph (red)
        self._draw_single_graph(painter, self._data['upload'], QColor(255, 0, 0))
    
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
        """Get the current network data."""
        data = super().get_data()
        data['selected_interface'] = self._selected_interface
        return data
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the network data."""
        super().set_data(data)
        if 'selected_interface' in data and data['selected_interface'] != self._selected_interface:
            self._selected_interface = data['selected_interface']
            self._interface_selector.setCurrentText(self._selected_interface)
            self._initialize_network_counters() 
Network graph visualization component for the Lumina Frontend System.
Displays real-time network traffic for selected network interfaces.
"""

from typing import List, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QColor, QPen
import psutil
import numpy as np
from .base_visualization import BaseVisualization

class NetworkGraph(BaseVisualization):
    """Widget for displaying network traffic."""
    
    # Signal emitted when network data is updated
    data_updated = Signal(dict)
    
    def __init__(self):
        super().__init__("Network Traffic")
        self._data: Dict[str, List[float]] = {
            'download': [],
            'upload': []
        }
        self._network_counters: Dict[str, psutil._common.snetio] = {}
        self._selected_interface = ""
        
        # Create interface selector
        self._interface_selector = QComboBox()
        self._interface_selector.currentTextChanged.connect(self._on_interface_changed)
        self.layout().addWidget(self._interface_selector)
        
        # Create labels for current values
        self._download_label = QLabel("Download: 0 MB/s")
        self._upload_label = QLabel("Upload: 0 MB/s")
        self.layout().addWidget(self._download_label)
        self.layout().addWidget(self._upload_label)
        
        # Populate interface list
        self._populate_interfaces()
    
    def _populate_interfaces(self) -> None:
        """Populate the interface selector with available network interfaces."""
        self._interface_selector.clear()
        interfaces = psutil.net_if_stats()
        for interface in interfaces:
            if interfaces[interface].isup:  # Only show active interfaces
                self._interface_selector.addItem(interface)
        
        if self._interface_selector.count() > 0:
            self._selected_interface = self._interface_selector.currentText()
            self._initialize_network_counters()
    
    def _initialize_network_counters(self) -> None:
        """Initialize network counters for the selected interface."""
        self._network_counters = {}
        for interface in psutil.net_io_counters(pernic=True):
            self._network_counters[interface] = psutil.net_io_counters(pernic=True)[interface]
    
    def _on_interface_changed(self, interface: str) -> None:
        """Handle interface selection change."""
        self._selected_interface = interface
        self._data['download'].clear()
        self._data['upload'].clear()
        self._initialize_network_counters()
    
    def _update_data(self) -> None:
        """Update network data."""
        if self._paused or not self._selected_interface:
            return
            
        # Get current network counters
        current_counters = psutil.net_io_counters(pernic=True)
        if self._selected_interface not in current_counters:
            return
            
        current = current_counters[self._selected_interface]
        previous = self._network_counters.get(self._selected_interface, current)
        
        # Calculate download and upload rates (MB/s)
        download_rate = (current.bytes_recv - previous.bytes_recv) / (1024 * 1024)
        upload_rate = (current.bytes_sent - previous.bytes_sent) / (1024 * 1024)
        
        # Update data arrays
        self._data['download'].append(download_rate)
        self._data['upload'].append(upload_rate)
        if len(self._data['download']) > self._max_points:
            self._data['download'].pop(0)
        if len(self._data['upload']) > self._max_points:
            self._data['upload'].pop(0)
        
        # Update labels
        self._download_label.setText(f"Download: {download_rate:.1f} MB/s")
        self._upload_label.setText(f"Upload: {upload_rate:.1f} MB/s")
        
        # Update counters
        self._network_counters[self._selected_interface] = current
        
        # Emit data update signal
        self.data_updated.emit({
            'interface': self._selected_interface,
            'download_rate': download_rate,
            'upload_rate': upload_rate,
            'bytes_recv': current.bytes_recv,
            'bytes_sent': current.bytes_sent
        })
        
        self.update()
    
    def _draw_visualization(self, painter: QPainter) -> None:
        """Draw the download and upload graphs."""
        # Draw download graph (green)
        self._draw_single_graph(painter, self._data['download'], QColor(0, 255, 0))
        
        # Draw upload graph (red)
        self._draw_single_graph(painter, self._data['upload'], QColor(255, 0, 0))
    
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
        """Get the current network data."""
        data = super().get_data()
        data['selected_interface'] = self._selected_interface
        return data
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Set the network data."""
        super().set_data(data)
        if 'selected_interface' in data and data['selected_interface'] != self._selected_interface:
            self._selected_interface = data['selected_interface']
            self._interface_selector.setCurrentText(self._selected_interface)
            self._initialize_network_counters() 
 