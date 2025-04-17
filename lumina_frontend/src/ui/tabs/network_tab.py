"""
Network Tab for Lumina Frontend
==============================

This module contains the network tab that displays network metrics
and visualizations.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QGridLayout
)
from PySide6.QtCore import QTimer

from ..visualization.throughput_graph import ThroughputGraph
from ..visualization.latency_graph import LatencyGraph
from ..visualization.connections_graph import ConnectionsGraph

class NetworkTab(QWidget):
    """Network monitoring tab."""
    
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
        self._create_throughput_tab()
        self._create_latency_tab()
        self._create_connections_tab()
        
        # Add tab widget to layout
        self.layout.addWidget(self.tab_widget)
    
    def _create_overview_tab(self):
        """Create the overview tab."""
        overview_tab = QWidget()
        layout = QGridLayout(overview_tab)
        
        # Create mini graphs
        self.throughput_mini = ThroughputGraph(self.config, mini=True)
        self.latency_mini = LatencyGraph(self.config, mini=True)
        self.connections_mini = ConnectionsGraph(self.config, mini=True)
        
        # Add to layout
        layout.addWidget(self.throughput_mini, 0, 0)
        layout.addWidget(self.latency_mini, 0, 1)
        layout.addWidget(self.connections_mini, 1, 0, 1, 2)
        
        self.tab_widget.addTab(overview_tab, "Overview")
    
    def _create_throughput_tab(self):
        """Create the throughput tab."""
        throughput_tab = QWidget()
        layout = QVBoxLayout(throughput_tab)
        
        # Create throughput graph
        self.throughput_graph = ThroughputGraph(self.config)
        layout.addWidget(self.throughput_graph)
        
        self.tab_widget.addTab(throughput_tab, "Throughput")
    
    def _create_latency_tab(self):
        """Create the latency tab."""
        latency_tab = QWidget()
        layout = QVBoxLayout(latency_tab)
        
        # Create latency graph
        self.latency_graph = LatencyGraph(self.config)
        layout.addWidget(self.latency_graph)
        
        self.tab_widget.addTab(latency_tab, "Latency")
    
    def _create_connections_tab(self):
        """Create the connections tab."""
        connections_tab = QWidget()
        layout = QVBoxLayout(connections_tab)
        
        # Create connections graph
        self.connections_graph = ConnectionsGraph(self.config)
        layout.addWidget(self.connections_graph)
        
        self.tab_widget.addTab(connections_tab, "Connections")
    
    def initialize(self):
        """Initialize the network tab."""
        # Initialize graphs
        self.throughput_mini.initialize()
        self.latency_mini.initialize()
        self.connections_mini.initialize()
        self.throughput_graph.initialize()
        self.latency_graph.initialize()
        self.connections_graph.initialize()
        
        # Start update timer
        update_interval = self.config.get("visualization.update_interval", 100)
        self.timer.start(update_interval)
    
    def update_metrics(self):
        """Update network metrics."""
        # Update mini graphs
        self.throughput_mini.update()
        self.latency_mini.update()
        self.connections_mini.update()
        
        # Update full graphs if visible
        current_tab = self.tab_widget.currentWidget()
        if hasattr(current_tab, "children") and self.throughput_graph in current_tab.children():
            self.throughput_graph.update()
        elif hasattr(current_tab, "children") and self.latency_graph in current_tab.children():
            self.latency_graph.update()
        elif hasattr(current_tab, "children") and self.connections_graph in current_tab.children():
            self.connections_graph.update() 