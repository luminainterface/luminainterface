"""
Connections Graph for Lumina Frontend
====================================

This module contains the connections graph visualization component
that displays active network connections and their status.
"""

import psutil
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PySide6.QtCore import QTimer
import pyqtgraph as pg

class ConnectionsGraph(QWidget):
    """Network connections visualization."""
    
    def __init__(self, config, mini=False):
        super().__init__()
        
        self.config = config
        self.mini = mini
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create title
        self.title = QLabel("Active Connections")
        self.title.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.layout.addWidget(self.title)
        
        # Create connections table
        self.connections_table = QTableWidget()
        self.connections_table.setColumnCount(5)
        self.connections_table.setHorizontalHeaderLabels([
            "Protocol", "Local Address", "Remote Address", "Status", "PID"
        ])
        self.connections_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.connections_table)
        
        # Create plot widget for connection count
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(None)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Connections')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        
        # Create plot curve
        self.connections_curve = self.plot_widget.plot(pen='y')
        
        self.layout.addWidget(self.plot_widget)
        
        # Initialize data arrays
        self.time_data = np.zeros(100)
        self.connections_data = np.zeros(100)
        self.current_time = 0
    
    def _update_connections_table(self):
        """Update the connections table with current connections."""
        # Get current connections
        connections = psutil.net_connections()
        
        # Clear table
        self.connections_table.setRowCount(0)
        
        # Add connections to table
        for conn in connections:
            row = self.connections_table.rowCount()
            self.connections_table.insertRow(row)
            
            # Protocol
            protocol = "TCP" if conn.type == socket.SOCK_STREAM else "UDP"
            self.connections_table.setItem(row, 0, QTableWidgetItem(protocol))
            
            # Local address
            local_addr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "-"
            self.connections_table.setItem(row, 1, QTableWidgetItem(local_addr))
            
            # Remote address
            remote_addr = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "-"
            self.connections_table.setItem(row, 2, QTableWidgetItem(remote_addr))
            
            # Status
            status = conn.status if hasattr(conn, 'status') else "-"
            self.connections_table.setItem(row, 3, QTableWidgetItem(status))
            
            # PID
            pid = str(conn.pid) if conn.pid else "-"
            self.connections_table.setItem(row, 4, QTableWidgetItem(pid))
    
    def _update_plot(self):
        """Update the connections plot."""
        # Get current connection count
        connection_count = len(psutil.net_connections())
        
        # Update data arrays
        self.time_data = np.roll(self.time_data, -1)
        self.connections_data = np.roll(self.connections_data, -1)
        
        self.time_data[-1] = self.current_time
        self.connections_data[-1] = connection_count
        
        # Update plot
        self.connections_curve.setData(self.time_data, self.connections_data)
        
        # Update title
        self.title.setText(f"Active Connections: {connection_count}")
        
        self.current_time += 1
    
    def initialize(self):
        """Initialize the connections graph."""
        # Set plot ranges
        self.plot_widget.setXRange(0, 100)
        self.plot_widget.setYRange(0, 100)  # Adjust based on expected max connections
        
        # Start update timer
        update_interval = self.config.get("visualization.update_interval", 1000)
        self.timer.start(update_interval)
    
    def update(self):
        """Update the connections graph."""
        self._update_connections_table()
        self._update_plot()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        if self.mini:
            # Adjust plot size for mini version
            self.plot_widget.setMinimumHeight(100)
            self.plot_widget.setMaximumHeight(150)
            self.connections_table.hide()
        super().resizeEvent(event) 