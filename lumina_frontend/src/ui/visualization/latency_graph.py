"""
Latency Graph for Lumina Frontend
================================

This module contains the latency graph visualization component
that displays network latency metrics.
"""

import psutil
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QComboBox, QPushButton
)
from PySide6.QtCore import QTimer
import pyqtgraph as pg
import socket
import time

class LatencyGraph(QWidget):
    """Network latency visualization."""
    
    def __init__(self, config, mini=False):
        super().__init__()
        
        self.config = config
        self.mini = mini
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create title
        self.title = QLabel("Network Latency")
        self.title.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.layout.addWidget(self.title)
        
        # Create host input
        self.host_combo = QComboBox()
        self.host_combo.setEditable(True)
        self.host_combo.addItems(["8.8.8.8", "1.1.1.1", "google.com"])
        self.layout.addWidget(self.host_combo)
        
        # Create ping button
        self.ping_button = QPushButton("Ping")
        self.ping_button.clicked.connect(self._on_ping_clicked)
        self.layout.addWidget(self.ping_button)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(None)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Latency (ms)')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        
        # Create plot curve
        self.latency_curve = self.plot_widget.plot(pen='b')
        
        self.layout.addWidget(self.plot_widget)
        
        # Initialize data arrays
        self.time_data = np.zeros(100)
        self.latency_data = np.zeros(100)
        self.current_time = 0
        
        # Store ping results
        self.ping_results = []
    
    def _on_ping_clicked(self):
        """Handle ping button click."""
        host = self.host_combo.currentText()
        self._ping_host(host)
    
    def _ping_host(self, host):
        """Ping a host and measure latency."""
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            
            # Start timer
            start_time = time.time()
            
            # Connect to host
            sock.connect((host, 80))
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Store result
            self.ping_results.append(latency)
            if len(self.ping_results) > 100:
                self.ping_results.pop(0)
            
            # Update plot
            self._update_plot()
            
        except socket.error:
            # Connection failed
            self.ping_results.append(0)  # Use 0 to indicate failure
            if len(self.ping_results) > 100:
                self.ping_results.pop(0)
            
            # Update plot
            self._update_plot()
        
        finally:
            sock.close()
    
    def _update_plot(self):
        """Update the latency plot."""
        # Update data arrays
        self.time_data = np.roll(self.time_data, -1)
        self.latency_data = np.roll(self.latency_data, -1)
        
        self.time_data[-1] = self.current_time
        self.latency_data[-1] = self.ping_results[-1] if self.ping_results else 0
        
        # Update plot
        self.latency_curve.setData(self.time_data, self.latency_data)
        
        # Update title
        avg_latency = np.mean(self.ping_results) if self.ping_results else 0
        self.title.setText(f"Network Latency - Average: {avg_latency:.2f} ms")
        
        self.current_time += 1
    
    def initialize(self):
        """Initialize the graph."""
        # Set plot ranges
        self.plot_widget.setXRange(0, 100)
        self.plot_widget.setYRange(0, 1000)  # 1000ms max
        
        # Start update timer
        update_interval = self.config.get("visualization.update_interval", 1000)
        self.timer.start(update_interval)
    
    def update(self):
        """Update the latency graph."""
        host = self.host_combo.currentText()
        self._ping_host(host)
    
    def resizeEvent(self, event):
        """Handle resize events."""
        if self.mini:
            # Adjust plot size for mini version
            self.plot_widget.setMinimumHeight(100)
            self.plot_widget.setMaximumHeight(150)
        super().resizeEvent(event) 