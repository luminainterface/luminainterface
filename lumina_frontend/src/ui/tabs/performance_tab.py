"""
Performance Tab for Lumina Frontend
==================================

This module contains the performance tab that displays system performance
metrics and visualizations.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QGridLayout
)
from PySide6.QtCore import QTimer

from ..visualization.cpu_graph import CPUGraph
from ..visualization.memory_graph import MemoryGraph
from ..visualization.disk_graph import DiskGraph

class PerformanceTab(QWidget):
    """Performance monitoring tab."""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create tab widget for sub-tabs
        self.tab_widget = QTabWidget()
        
        # Create sub-tabs
        self._create_overview_tab()
        self._create_cpu_tab()
        self._create_memory_tab()
        self._create_disk_tab()
        
        # Add tab widget to layout
        self.layout.addWidget(self.tab_widget)
    
    def _create_overview_tab(self):
        """Create the overview tab."""
        overview_tab = QWidget()
        layout = QGridLayout(overview_tab)
        
        # Create mini graphs
        self.cpu_mini = CPUGraph(self.config, mini=True)
        self.memory_mini = MemoryGraph(self.config, mini=True)
        self.disk_mini = DiskGraph(self.config, mini=True)
        
        # Add to layout
        layout.addWidget(self.cpu_mini, 0, 0)
        layout.addWidget(self.memory_mini, 0, 1)
        layout.addWidget(self.disk_mini, 1, 0, 1, 2)
        
        self.tab_widget.addTab(overview_tab, "Overview")
    
    def _create_cpu_tab(self):
        """Create the CPU tab."""
        cpu_tab = QWidget()
        layout = QVBoxLayout(cpu_tab)
        
        # Create CPU graph
        self.cpu_graph = CPUGraph(self.config)
        layout.addWidget(self.cpu_graph)
        
        self.tab_widget.addTab(cpu_tab, "CPU")
    
    def _create_memory_tab(self):
        """Create the memory tab."""
        memory_tab = QWidget()
        layout = QVBoxLayout(memory_tab)
        
        # Create memory graph
        self.memory_graph = MemoryGraph(self.config)
        layout.addWidget(self.memory_graph)
        
        self.tab_widget.addTab(memory_tab, "Memory")
    
    def _create_disk_tab(self):
        """Create the disk tab."""
        disk_tab = QWidget()
        layout = QVBoxLayout(disk_tab)
        
        # Create disk graph
        self.disk_graph = DiskGraph(self.config)
        layout.addWidget(self.disk_graph)
        
        self.tab_widget.addTab(disk_tab, "Disk")
    
    def initialize(self):
        """Initialize the performance tab."""
        # Initialize graphs
        self.cpu_mini.initialize()
        self.memory_mini.initialize()
        self.disk_mini.initialize()
        self.cpu_graph.initialize()
        self.memory_graph.initialize()
        self.disk_graph.initialize()
        
        # Start update timer
        update_interval = self.config.get("visualization.update_interval", 100)
        self.timer.start(update_interval)
    
    def update_metrics(self):
        """Update performance metrics."""
        # Update mini graphs
        self.cpu_mini.update()
        self.memory_mini.update()
        self.disk_mini.update()
        
        # Update full graphs if visible
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, "children") and self.cpu_graph in current_tab.children():
            self.cpu_graph.update()
        elif hasattr(current_tab, "children") and self.memory_graph in current_tab.children():
            self.memory_graph.update()
        elif hasattr(current_tab, "children") and self.disk_graph in current_tab.children():
            self.disk_graph.update() 