"""
Throughput Graph for Lumina Frontend
===================================

This module contains the throughput graph visualization component
that displays network throughput metrics.
"""

import psutil
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QComboBox
)
from PySide6.QtCore import QTimer
import pyqtgraph as pg

class ThroughputGraph(QWidget):
    """Network throughput visualization."""
    
    def __init__(self, config, mini=False):
        super().__init__()
        
        self.config = config
        self.mini = mini
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create title
        self.title = QLabel("Network Throughput")
        self.title.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.layout.addWidget(self.title)
        
        # Create network interface selector
        self.interface_combo = QComboBox()
        self._populate_interface_list()
        self.interface_combo.currentTextChanged.connect(self._on_interface_changed)
        self.layout.addWidget(self.interface_combo)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(None)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Throughput (MB/s)')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        
        # Create plot curves
        self.download_curve = self.plot_widget.plot(pen='g')
        self.upload_curve = self.plot_widget.plot(pen='r')
        
        self.layout.addWidget(self.plot_widget)
        
        # Initialize data arrays
        self.time_data = np.zeros(100)
        self.download_data = np.zeros(100)
        self.upload_data = np.zeros(100)
        self.current_time = 0
        
        # Store previous values for rate calculation
        self.prev_download = 0
        self.prev_upload = 0
        self.prev_time = 0
    
    def _populate_interface_list(self):
        """Populate the network interface dropdown."""
        interfaces = psutil.net_if_stats().keys()
        self.interface_combo.addItems(interfaces)
    
    def _on_interface_changed(self, interface):
        """Handle interface selection change."""
        self.prev_download = 0
        self.prev_upload = 0
        self.prev_time = 0
    
    def initialize(self):
        """Initialize the graph."""
        # Set plot ranges
        self.plot_widget.setXRange(0, 100)
        self.plot_widget.setYRange(0, 100)
        
        # Start update timer
        update_interval = self.config.get("visualization.update_interval", 100)
        self.timer.start(update_interval)
    
    def update(self):
        """Update the throughput graph."""
        # Get current time
        current_time = psutil.cpu_times().user
        
        # Get selected interface
        interface = self.interface_combo.currentText()
        
        # Get network I/O counters
        net_io = psutil.net_io_counters(pernic=True)[interface]
        
        # Calculate rates
        time_diff = current_time - self.prev_time
        if time_diff > 0:
            download_rate = (net_io.bytes_recv - self.prev_download) / time_diff / 1024 / 1024  # MB/s
            upload_rate = (net_io.bytes_sent - self.prev_upload) / time_diff / 1024 / 1024  # MB/s
            
            # Update data arrays
            self.time_data = np.roll(self.time_data, -1)
            self.download_data = np.roll(self.download_data, -1)
            self.upload_data = np.roll(self.upload_data, -1)
            
            self.time_data[-1] = self.current_time
            self.download_data[-1] = download_rate
            self.upload_data[-1] = upload_rate
            
            # Update plot
            self.download_curve.setData(self.time_data, self.download_data)
            self.upload_curve.setData(self.time_data, self.upload_data)
            
            # Update title
            self.title.setText(f"Network Throughput - Download: {download_rate:.2f} MB/s, Upload: {upload_rate:.2f} MB/s")
            
            # Store current values
            self.prev_download = net_io.bytes_recv
            self.prev_upload = net_io.bytes_sent
            self.prev_time = current_time
            self.current_time += 1
    
    def resizeEvent(self, event):
        """Handle resize events."""
        if self.mini:
            # Adjust plot size for mini version
            self.plot_widget.setMinimumHeight(100)
            self.plot_widget.setMaximumHeight(150)
        super().resizeEvent(event) 