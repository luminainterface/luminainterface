"""
Memory Graph for Lumina Frontend
===============================

This module contains the memory graph visualization component
that displays memory usage metrics and memory pressure.
"""

import psutil
import numpy as np
from PySide6.QtWidgets import QGridLayout, QProgressBar
import pyqtgraph as pg

from .base_visualization import BaseVisualization

class MemoryGraph(BaseVisualization):
    """Memory usage visualization."""
    
    def __init__(self, config, mini=False):
        super().__init__(config, mini)
        
        # Set title
        self.title.setText("Memory Usage")
        
        # Set plot labels
        self.plot_widget.setLabel('left', 'Usage (GB)')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        
        # Create memory curves in main region
        self.memory_curves = {
            'total': self.add_curve('main', 'Total', 'r'),
            'available': self.add_curve('main', 'Available', 'g'),
            'used': self.add_curve('main', 'Used', 'b'),
            'cached': self.add_curve('main', 'Cached', 'y')
        }
        
        # Create pressure curve in mini region if enabled
        if self.mini_plot:
            self.pressure_curve = self.add_curve('mini', 'Pressure', 'w', width=3)
        
        # Create memory bars
        self.memory_bars_layout = QGridLayout()
        self.memory_bars = {}
        
        memory_types = ['used', 'available', 'cached']
        for i, mem_type in enumerate(memory_types):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            bar.setFormat(f"{mem_type.capitalize()}: %p%")
            self.memory_bars[mem_type] = bar
            self.memory_bars_layout.addWidget(bar, i, 0)
        
        self.layout.addLayout(self.memory_bars_layout)
        
        # Set up quantum/cosmic visualization if enabled
        if self.config.get("visualization.style") in ["quantum", "cosmic"]:
            self._setup_quantum_visualization()
    
    def _setup_quantum_visualization(self):
        """Set up quantum/cosmic visualization features."""
        style = self.config.get("visualization.style")
        
        if style == "quantum":
            # Quantum visualization
            for curve in self.memory_curves.values():
                curve.setPen(pg.mkPen(width=2, style=Qt.DashLine))
            
            # Add quantum field effect
            self.quantum_field = self.add_curve('main', 'Quantum Field', 'w', width=1)
            self.quantum_field.setPen(pg.mkPen('w', width=1, style=Qt.DotLine))
            
        elif style == "cosmic":
            # Cosmic visualization
            for curve in self.memory_curves.values():
                curve.setPen(pg.mkPen(width=3))
            
            # Add cosmic field effect
            self.cosmic_field = self.add_curve('main', 'Cosmic Field', 'y', width=2)
            self.cosmic_field.setPen(pg.mkPen('y', width=2, style=Qt.DashDotLine))
    
    def _update_plot(self):
        """Update the memory plot."""
        # Get memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Update time data
        self._update_time_data()
        
        # Update memory data
        self.update_curve('main', 'Total', memory.total / (1024**3))  # Convert to GB
        self.update_curve('main', 'Available', memory.available / (1024**3))
        self.update_curve('main', 'Used', memory.used / (1024**3))
        self.update_curve('main', 'Cached', memory.cached / (1024**3))
        
        # Update pressure in mini region if enabled
        if self.mini_plot:
            self.update_curve('mini', 'Pressure', memory.percent)
        
        # Update progress bars
        self.memory_bars['used'].setValue(int(memory.percent))
        self.memory_bars['available'].setValue(int(100 - memory.percent))
        self.memory_bars['cached'].setValue(int((memory.cached / memory.total) * 100))
        
        # Update quantum/cosmic field if enabled
        if hasattr(self, 'quantum_field'):
            # Create quantum field effect
            field_data = np.mean([memory.used, memory.available]) / (1024**3)
            self.update_curve('main', 'Quantum Field', field_data)
        
        elif hasattr(self, 'cosmic_field'):
            # Create cosmic field effect
            field_data = np.max([memory.used, memory.available]) / (1024**3)
            self.update_curve('main', 'Cosmic Field', field_data)
        
        # Update title with memory pressure
        pressure = memory.percent
        self.title.setText(f"Memory Usage - Pressure: {pressure:.1f}%")
        
        # Emit data updated signal
        self.data_updated.emit({
            'total': memory.total / (1024**3),
            'available': memory.available / (1024**3),
            'used': memory.used / (1024**3),
            'cached': memory.cached / (1024**3),
            'pressure': pressure,
            'time': self.current_time
        })
    
    def update(self):
        """Update the memory graph."""
        self._update_plot()
    
    def resizeEvent(self, event):
        """Handle resize events."""
        if self.mini:
            # Adjust plot size for mini version
            self.plot_widget.setMinimumHeight(100)
            self.plot_widget.setMaximumHeight(150)
            self.memory_bars_layout.hide()
        super().resizeEvent(event) 