"""
Disk I/O Graph for Lumina Frontend
=================================

This module contains the disk I/O graph visualization component
that displays disk usage metrics and I/O operations.
"""

import psutil
import numpy as np
from PySide6.QtWidgets import QGridLayout, QProgressBar, QComboBox
import pyqtgraph as pg

from .base_visualization import BaseVisualization

class DiskGraph(BaseVisualization):
    """Disk I/O visualization."""
    
    def __init__(self, config, mini=False):
        super().__init__(config, mini)
        
        # Set title
        self.title.setText("Disk I/O")
        
        # Set plot labels
        self.plot_widget.setLabel('left', 'Rate (MB/s)')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        
        # Create disk selector
        self.disk_selector = QComboBox()
        self.disk_selector.currentIndexChanged.connect(self._on_disk_change)
        self.layout.addWidget(self.disk_selector)
        
        # Create plot curves for different metrics
        self.io_curves = {
            'read': self.plot_widget.plot(pen='g', name='Read'),
            'write': self.plot_widget.plot(pen='r', name='Write'),
            'total': self.plot_widget.plot(pen='b', name='Total')
        }
        
        # Create I/O bars
        self.io_bars_layout = QGridLayout()
        self.io_bars = {}
        
        io_types = ['read', 'write', 'total']
        for i, io_type in enumerate(io_types):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            bar.setFormat(f"{io_type.capitalize()}: %p%")
            self.io_bars[io_type] = bar
            self.io_bars_layout.addWidget(bar, i, 0)
        
        self.layout.addLayout(self.io_bars_layout)
        
        # Initialize data arrays
        self.io_data = {key: np.zeros(100) for key in self.io_curves.keys()}
        
        # Populate disk list
        self._populate_disk_list()
        
        # Set up quantum/cosmic visualization if enabled
        if self.config.get("visualization.style") in ["quantum", "cosmic"]:
            self._setup_quantum_visualization()
    
    def _populate_disk_list(self):
        """Populate the disk selector with available disks."""
        self.disk_selector.clear()
        for partition in psutil.disk_partitions():
            if 'fixed' in partition.opts:
                self.disk_selector.addItem(partition.device)
    
    def _on_disk_change(self, index):
        """Handle disk selection change."""
        if index >= 0:
            self.selected_disk = self.disk_selector.currentText()
            # Reset data arrays
            self.io_data = {key: np.zeros(100) for key in self.io_curves.keys()}
    
    def _setup_quantum_visualization(self):
        """Set up quantum/cosmic visualization features."""
        style = self.config.get("visualization.style")
        
        if style == "quantum":
            # Quantum visualization
            for curve in self.io_curves.values():
                curve.setPen(pg.mkPen(width=2, style=Qt.DashLine))
            
            # Add quantum field effect
            self.quantum_field = pg.PlotCurveItem(pen=pg.mkPen('w', width=1, style=Qt.DotLine))
            self.plot_widget.addItem(self.quantum_field)
            
        elif style == "cosmic":
            # Cosmic visualization
            for curve in self.io_curves.values():
                curve.setPen(pg.mkPen(width=3))
            
            # Add cosmic field effect
            self.cosmic_field = pg.PlotCurveItem(pen=pg.mkPen('y', width=2, style=Qt.DashDotLine))
            self.plot_widget.addItem(self.cosmic_field)
    
    def _update_plot(self):
        """Update the disk I/O plot."""
        if not hasattr(self, 'selected_disk'):
            return
        
        # Get disk I/O counters
        disk_io = psutil.disk_io_counters(perdisk=True)
        if self.selected_disk not in disk_io:
            return
        
        io = disk_io[self.selected_disk]
        
        # Calculate rates (MB/s)
        read_rate = io.read_bytes / (1024**2)  # Convert to MB
        write_rate = io.write_bytes / (1024**2)
        total_rate = read_rate + write_rate
        
        # Update time data
        self._update_time_data()
        
        # Update I/O data
        self.io_data['read'] = np.roll(self.io_data['read'], -1)
        self.io_data['read'][-1] = read_rate
        
        self.io_data['write'] = np.roll(self.io_data['write'], -1)
        self.io_data['write'][-1] = write_rate
        
        self.io_data['total'] = np.roll(self.io_data['total'], -1)
        self.io_data['total'][-1] = total_rate
        
        # Update plot curves
        for key, curve in self.io_curves.items():
            curve.setData(self.time_data, self.io_data[key])
        
        # Update progress bars
        max_rate = max(self.io_data['total'])
        if max_rate > 0:
            self.io_bars['read'].setValue(int((read_rate / max_rate) * 100))
            self.io_bars['write'].setValue(int((write_rate / max_rate) * 100))
            self.io_bars['total'].setValue(int((total_rate / max_rate) * 100))
        
        # Update quantum/cosmic field if enabled
        if hasattr(self, 'quantum_field'):
            # Create quantum field effect
            field_data = np.mean([self.io_data['read'], self.io_data['write']], axis=0)
            self.quantum_field.setData(self.time_data, field_data)
        
        elif hasattr(self, 'cosmic_field'):
            # Create cosmic field effect
            field_data = np.max([self.io_data['read'], self.io_data['write']], axis=0)
            self.cosmic_field.setData(self.time_data, field_data)
        
        # Update title with current rates
        self.title.setText(
            f"Disk I/O - {self.selected_disk}\n"
            f"Read: {read_rate:.1f} MB/s | Write: {write_rate:.1f} MB/s"
        )
    
    def update(self):
        """Update the disk I/O graph."""
        self._update_plot()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        if self.mini:
            # Adjust plot size for mini version
            self.plot_widget.setMinimumHeight(100)
            self.plot_widget.setMaximumHeight(150)
            self.io_bars_layout.hide()
            self.disk_selector.hide()
        super().resizeEvent(event) 