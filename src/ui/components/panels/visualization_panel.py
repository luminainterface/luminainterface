"""
Visualization panel for the Lumina Frontend System.
Displays system metrics and visualizations in real-time.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from ...core.main_controller import MainController
from ...visualization.manager import VisualizationManager
from ...visualization.cpu_graph import CPUGraph
from ...visualization.memory_graph import MemoryGraph
from ...visualization.disk_graph import DiskGraph
from ...visualization.network_graph import NetworkGraph

class VisualizationPanel(QWidget):
    """Panel for displaying system visualizations."""
    
    def __init__(self, controller: MainController):
        super().__init__()
        self._controller = controller
        self._visualization_manager: Optional[VisualizationManager] = None
        self._graphs = {}
        
        self._initialize_ui()
        self._setup_layout()
        self._connect_signals()
    
    def _initialize_ui(self) -> None:
        """Initialize the user interface."""
        # Create main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(10, 10, 10, 10)
        self._main_layout.setSpacing(10)
        
        # Create title
        self._title_label = QLabel("System Visualization")
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self._main_layout.addWidget(self._title_label)
        
        # Create graph container
        self._graph_container = QWidget()
        self._graph_layout = QHBoxLayout(self._graph_container)
        self._graph_layout.setContentsMargins(0, 0, 0, 0)
        self._graph_layout.setSpacing(10)
        
        # Add graph container to main layout
        self._main_layout.addWidget(self._graph_container)
    
    def _setup_layout(self) -> None:
        """Set up the panel layout."""
        # Create graphs
        self._graphs['cpu'] = CPUGraph()
        self._graphs['memory'] = MemoryGraph()
        self._graphs['disk'] = DiskGraph()
        self._graphs['network'] = NetworkGraph()
        
        # Add graphs to layout
        for graph in self._graphs.values():
            self._graph_layout.addWidget(graph)
    
    def _connect_signals(self) -> None:
        """Connect signals and slots."""
        # Get visualization manager
        self._visualization_manager = self._controller.get_component('visualization')
        
        if self._visualization_manager:
            # Connect manager signals
            self._visualization_manager.metrics_updated.connect(self._on_metrics_updated)
            self._visualization_manager.health_warning.connect(self._on_health_warning)
    
    def _on_metrics_updated(self, metrics) -> None:
        """Handle metrics update."""
        # Update graphs
        self._graphs['cpu'].update(metrics.cpu_usage)
        self._graphs['memory'].update(metrics.memory_usage)
        self._graphs['disk'].update(metrics.disk_io)
        self._graphs['network'].update(metrics.network_io)
    
    def _on_health_warning(self, message: str) -> None:
        """Handle health warning."""
        # Update title with warning
        self._title_label.setText(f"System Visualization - {message}")
        self._title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
    
    def on_system_initialized(self) -> None:
        """Handle system initialization."""
        # Update graphs
        for graph in self._graphs.values():
            graph.initialize()
    
    def on_version_changed(self, version: str) -> None:
        """Handle version change."""
        # Update graphs based on version
        for graph in self._graphs.values():
            graph.set_version(version)
    
    def get_graph(self, name: str) -> Optional[QWidget]:
        """Get a graph by name."""
        return self._graphs.get(name) 
Visualization panel for the Lumina Frontend System.
Displays system metrics and visualizations in real-time.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from ...core.main_controller import MainController
from ...visualization.manager import VisualizationManager
from ...visualization.cpu_graph import CPUGraph
from ...visualization.memory_graph import MemoryGraph
from ...visualization.disk_graph import DiskGraph
from ...visualization.network_graph import NetworkGraph

class VisualizationPanel(QWidget):
    """Panel for displaying system visualizations."""
    
    def __init__(self, controller: MainController):
        super().__init__()
        self._controller = controller
        self._visualization_manager: Optional[VisualizationManager] = None
        self._graphs = {}
        
        self._initialize_ui()
        self._setup_layout()
        self._connect_signals()
    
    def _initialize_ui(self) -> None:
        """Initialize the user interface."""
        # Create main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(10, 10, 10, 10)
        self._main_layout.setSpacing(10)
        
        # Create title
        self._title_label = QLabel("System Visualization")
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self._main_layout.addWidget(self._title_label)
        
        # Create graph container
        self._graph_container = QWidget()
        self._graph_layout = QHBoxLayout(self._graph_container)
        self._graph_layout.setContentsMargins(0, 0, 0, 0)
        self._graph_layout.setSpacing(10)
        
        # Add graph container to main layout
        self._main_layout.addWidget(self._graph_container)
    
    def _setup_layout(self) -> None:
        """Set up the panel layout."""
        # Create graphs
        self._graphs['cpu'] = CPUGraph()
        self._graphs['memory'] = MemoryGraph()
        self._graphs['disk'] = DiskGraph()
        self._graphs['network'] = NetworkGraph()
        
        # Add graphs to layout
        for graph in self._graphs.values():
            self._graph_layout.addWidget(graph)
    
    def _connect_signals(self) -> None:
        """Connect signals and slots."""
        # Get visualization manager
        self._visualization_manager = self._controller.get_component('visualization')
        
        if self._visualization_manager:
            # Connect manager signals
            self._visualization_manager.metrics_updated.connect(self._on_metrics_updated)
            self._visualization_manager.health_warning.connect(self._on_health_warning)
    
    def _on_metrics_updated(self, metrics) -> None:
        """Handle metrics update."""
        # Update graphs
        self._graphs['cpu'].update(metrics.cpu_usage)
        self._graphs['memory'].update(metrics.memory_usage)
        self._graphs['disk'].update(metrics.disk_io)
        self._graphs['network'].update(metrics.network_io)
    
    def _on_health_warning(self, message: str) -> None:
        """Handle health warning."""
        # Update title with warning
        self._title_label.setText(f"System Visualization - {message}")
        self._title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
    
    def on_system_initialized(self) -> None:
        """Handle system initialization."""
        # Update graphs
        for graph in self._graphs.values():
            graph.initialize()
    
    def on_version_changed(self, version: str) -> None:
        """Handle version change."""
        # Update graphs based on version
        for graph in self._graphs.values():
            graph.set_version(version)
    
    def get_graph(self, name: str) -> Optional[QWidget]:
        """Get a graph by name."""
        return self._graphs.get(name) 
 