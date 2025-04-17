"""
Process Graph for Lumina Frontend
================================

This module contains the process graph visualization component
that displays running processes and their resource usage.
"""

import psutil
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox
)
from PySide6.QtCore import QTimer, Qt
import pyqtgraph as pg

class ProcessGraph(QWidget):
    """Process visualization."""
    
    def __init__(self, config, mini=False):
        super().__init__()
        
        self.config = config
        self.mini = mini
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create title
        self.title = QLabel("Process Monitor")
        self.title.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.layout.addWidget(self.title)
        
        # Create sort selector
        self.sort_combo = QComboBox()
        self.sort_combo.addItems([
            "CPU Usage",
            "Memory Usage",
            "Process Name"
        ])
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        self.layout.addWidget(self.sort_combo)
        
        # Create process table
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(5)
        self.process_table.setHorizontalHeaderLabels([
            "PID", "Name", "CPU %", "Memory %", "Status"
        ])
        self.process_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.process_table)
        
        # Create plot widget for top processes
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(None)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Usage (%)')
        self.plot_widget.setLabel('bottom', 'Process')
        
        # Create plot bars
        self.process_bars = pg.BarGraphItem(
            x=np.arange(5),
            height=np.zeros(5),
            width=0.6,
            brush='g'
        )
        self.plot_widget.addItem(self.process_bars)
        
        self.layout.addWidget(self.plot_widget)
        
        # Initialize data
        self.top_processes = []
        self.sort_key = "cpu_percent"
    
    def _on_sort_changed(self, sort_key):
        """Handle sort selection change."""
        if sort_key == "CPU Usage":
            self.sort_key = "cpu_percent"
        elif sort_key == "Memory Usage":
            self.sort_key = "memory_percent"
        else:
            self.sort_key = "name"
        self.update()
    
    def _update_process_table(self):
        """Update the process table with current processes."""
        # Get all processes
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort processes
        processes.sort(key=lambda x: x[self.sort_key], reverse=True)
        
        # Clear table
        self.process_table.setRowCount(0)
        
        # Add processes to table
        for proc in processes[:50]:  # Show top 50 processes
            row = self.process_table.rowCount()
            self.process_table.insertRow(row)
            
            # PID
            self.process_table.setItem(row, 0, QTableWidgetItem(str(proc['pid'])))
            
            # Name
            self.process_table.setItem(row, 1, QTableWidgetItem(proc['name']))
            
            # CPU %
            cpu_item = QTableWidgetItem(f"{proc['cpu_percent']:.1f}")
            cpu_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.process_table.setItem(row, 2, cpu_item)
            
            # Memory %
            mem_item = QTableWidgetItem(f"{proc['memory_percent']:.1f}")
            mem_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.process_table.setItem(row, 3, mem_item)
            
            # Status
            self.process_table.setItem(row, 4, QTableWidgetItem(proc['status']))
        
        # Store top 5 processes for the graph
        self.top_processes = processes[:5]
    
    def _update_plot(self):
        """Update the process plot."""
        if not self.top_processes:
            return
        
        # Update bar graph
        x = np.arange(len(self.top_processes))
        heights = [p[self.sort_key] for p in self.top_processes]
        names = [p['name'][:15] for p in self.top_processes]  # Truncate long names
        
        self.process_bars.setOpts(
            x=x,
            height=heights,
            width=0.6
        )
        
        # Update x-axis labels
        self.plot_widget.getAxis('bottom').setTicks([[(i, name) for i, name in enumerate(names)]])
        
        # Update title
        self.title.setText(f"Top 5 Processes by {self.sort_combo.currentText()}")
    
    def initialize(self):
        """Initialize the process graph."""
        # Set plot ranges
        self.plot_widget.setYRange(0, 100)
        
        # Start update timer
        update_interval = self.config.get("visualization.update_interval", 1000)
        self.timer.start(update_interval)
    
    def update(self):
        """Update the process graph."""
        self._update_process_table()
        self._update_plot()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        if self.mini:
            # Adjust plot size for mini version
            self.plot_widget.setMinimumHeight(100)
            self.plot_widget.setMaximumHeight(150)
            self.process_table.hide()
            self.sort_combo.hide()
        super().resizeEvent(event) 