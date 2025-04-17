"""
CPU usage visualization component for the Lumina Frontend.
Displays real-time CPU usage metrics with optional quantum/cosmic visualization.
"""

from typing import Dict, Optional
import psutil
import numpy as np
from PySide6.QtWidgets import QWidget
import pyqtgraph as pg
from .base_visualization import BaseVisualization

class CPUGraph(BaseVisualization):
    """Visualization component for CPU usage metrics."""
    
    def __init__(self, parent: Optional[QWidget] = None, mini: bool = False):
        """Initialize the CPU graph visualization.
        
        Args:
            parent: Parent widget
            mini: Whether this is a mini version
        """
        super().__init__("CPU Usage", parent, mini)
        
        # Initialize CPU-specific properties
        self.num_cores = psutil.cpu_count()
        self.core_data = np.zeros((self.num_cores, self.time_range))
        
        # Setup plot
        self._setup_plot()
        
    def _setup_plot(self) -> None:
        """Setup the CPU usage plot."""
        # Set plot labels
        self.plot_widget.setLabel('left', 'Usage (%)')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        
        # Create curves for each core
        self.core_curves = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for i in range(self.num_cores):
            color = colors[i % len(colors)]
            curve = self.plot_widget.plot(
                pen=pg.mkPen(color=color, width=2),
                name=f'Core {i}'
            )
            self.core_curves.append(curve)
            
        # Add total CPU usage curve
        self.total_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#000000', width=3),
            name='Total'
        )
        
        # Setup quantum/cosmic visualization if enabled
        if hasattr(self, 'quantum_style'):
            self._setup_quantum_visualization()
            
    def _setup_quantum_visualization(self) -> None:
        """Setup quantum or cosmic visualization effects."""
        if self.quantum_style == "quantum":
            # Add quantum field effect
            self.quantum_field = pg.PlotCurveItem(
                pen=pg.mkPen(color='#00ffff', width=1, style=Qt.DashLine)
            )
            self.plot_widget.addItem(self.quantum_field)
            
        elif self.quantum_style == "cosmic":
            # Add cosmic field effect
            self.cosmic_field = pg.PlotCurveItem(
                pen=pg.mkPen(color='#ff00ff', width=1, style=Qt.DashLine)
            )
            self.plot_widget.addItem(self.cosmic_field)
            
    def _update_plot(self) -> None:
        """Update the plot with current CPU usage data."""
        # Get current CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        total_percent = sum(cpu_percent) / self.num_cores
        
        # Update core data
        self.core_data = np.roll(self.core_data, -1, axis=1)
        for i, percent in enumerate(cpu_percent):
            self.core_data[i, -1] = percent
            
        # Update time data
        self.time_data = np.roll(self.time_data, -1)
        self.time_data[-1] = self.time_data[-2] + 1 if len(self.time_data) > 1 else 0
        
        # Update curves
        for i, curve in enumerate(self.core_curves):
            curve.setData(self.time_data, self.core_data[i])
            
        self.total_curve.setData(self.time_data, np.mean(self.core_data, axis=0))
        
        # Update quantum/cosmic effects if enabled
        if hasattr(self, 'quantum_field'):
            # Create quantum field effect
            field_data = np.sin(self.time_data * 0.1) * 10 + 50
            self.quantum_field.setData(self.time_data, field_data)
            
        elif hasattr(self, 'cosmic_field'):
            # Create cosmic field effect
            field_data = np.cos(self.time_data * 0.1) * 10 + 50
            self.cosmic_field.setData(self.time_data, field_data)
            
        # Update data dictionary
        self.current_data = {
            'time': self.time_data,
            'cores': self.core_data,
            'total': np.mean(self.core_data, axis=0)
        }
        
        # Emit data updated signal
        self.data_updated.emit(self.current_data)
    
    def update(self):
        """Update the CPU graph."""
        self._update_plot()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        if self.mini:
            # Adjust plot size for mini version
            self.plot_widget.setMinimumHeight(100)
            self.plot_widget.setMaximumHeight(150)
        super().resizeEvent(event) 