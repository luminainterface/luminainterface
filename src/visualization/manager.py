"""
Visualization manager for the Lumina Frontend System.
Manages visualization components and real-time updates.
"""

from typing import Dict, Optional
from PySide6.QtCore import QObject, Signal, QTimer
from .cpu_graph import CPUGraph
from .memory_graph import MemoryGraph
from .disk_graph import DiskGraph
from .network_graph import NetworkGraph
from ...core.system_monitor import SystemMonitor, SystemMetrics

class VisualizationManager(QObject):
    """Manages visualization components and updates."""
    
    # Signals
    metrics_updated = Signal(SystemMetrics)
    health_warning = Signal(str)
    visualization_ready = Signal()
    
    def __init__(self):
        super().__init__()
        self._graphs: Dict[str, QObject] = {}
        self._system_monitor: Optional[SystemMonitor] = None
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_visualizations)
    
    def initialize(self) -> None:
        """Initialize visualization components."""
        # Create graphs
        self._graphs['cpu'] = CPUGraph()
        self._graphs['memory'] = MemoryGraph()
        self._graphs['disk'] = DiskGraph()
        self._graphs['network'] = NetworkGraph()
        
        # Initialize graphs
        for graph in self._graphs.values():
            graph.initialize()
        
        # Start update timer
        self._update_timer.start(100)  # Update every 100ms
        
        self.visualization_ready.emit()
    
    def _update_visualizations(self) -> None:
        """Update all visualizations."""
        if not self._system_monitor:
            return
            
        # Get current metrics
        metrics = self._system_monitor.get_metrics()
        
        # Update graphs
        self._graphs['cpu'].update(metrics.cpu_usage)
        self._graphs['memory'].update(metrics.memory_usage)
        self._graphs['disk'].update(metrics.disk_io)
        self._graphs['network'].update(metrics.network_io)
        
        # Emit metrics update
        self.metrics_updated.emit(metrics)
    
    def set_system_monitor(self, monitor: SystemMonitor) -> None:
        """Set the system monitor."""
        self._system_monitor = monitor
        
        # Connect monitor signals
        self._system_monitor.metrics_updated.connect(self._on_metrics_updated)
        self._system_monitor.health_warning.connect(self._on_health_warning)
    
    def _on_metrics_updated(self, metrics: SystemMetrics) -> None:
        """Handle metrics update from system monitor."""
        self.metrics_updated.emit(metrics)
    
    def _on_health_warning(self, message: str) -> None:
        """Handle health warning from system monitor."""
        self.health_warning.emit(message)
    
    def get_graph(self, name: str) -> Optional[QObject]:
        """Get a graph by name."""
        return self._graphs.get(name)
    
    def set_version(self, version: str) -> None:
        """Set version for all visualizations."""
        for graph in self._graphs.values():
            graph.set_version(version)
    
    def shutdown(self) -> None:
        """Shutdown visualization components."""
        # Stop update timer
        self._update_timer.stop()
        
        # Shutdown graphs
        for graph in self._graphs.values():
            graph.shutdown()
        
        # Clear references
        self._graphs.clear()
        self._system_monitor = None 
Visualization manager for the Lumina Frontend System.
Manages visualization components and real-time updates.
"""

from typing import Dict, Optional
from PySide6.QtCore import QObject, Signal, QTimer
from .cpu_graph import CPUGraph
from .memory_graph import MemoryGraph
from .disk_graph import DiskGraph
from .network_graph import NetworkGraph
from ...core.system_monitor import SystemMonitor, SystemMetrics

class VisualizationManager(QObject):
    """Manages visualization components and updates."""
    
    # Signals
    metrics_updated = Signal(SystemMetrics)
    health_warning = Signal(str)
    visualization_ready = Signal()
    
    def __init__(self):
        super().__init__()
        self._graphs: Dict[str, QObject] = {}
        self._system_monitor: Optional[SystemMonitor] = None
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_visualizations)
    
    def initialize(self) -> None:
        """Initialize visualization components."""
        # Create graphs
        self._graphs['cpu'] = CPUGraph()
        self._graphs['memory'] = MemoryGraph()
        self._graphs['disk'] = DiskGraph()
        self._graphs['network'] = NetworkGraph()
        
        # Initialize graphs
        for graph in self._graphs.values():
            graph.initialize()
        
        # Start update timer
        self._update_timer.start(100)  # Update every 100ms
        
        self.visualization_ready.emit()
    
    def _update_visualizations(self) -> None:
        """Update all visualizations."""
        if not self._system_monitor:
            return
            
        # Get current metrics
        metrics = self._system_monitor.get_metrics()
        
        # Update graphs
        self._graphs['cpu'].update(metrics.cpu_usage)
        self._graphs['memory'].update(metrics.memory_usage)
        self._graphs['disk'].update(metrics.disk_io)
        self._graphs['network'].update(metrics.network_io)
        
        # Emit metrics update
        self.metrics_updated.emit(metrics)
    
    def set_system_monitor(self, monitor: SystemMonitor) -> None:
        """Set the system monitor."""
        self._system_monitor = monitor
        
        # Connect monitor signals
        self._system_monitor.metrics_updated.connect(self._on_metrics_updated)
        self._system_monitor.health_warning.connect(self._on_health_warning)
    
    def _on_metrics_updated(self, metrics: SystemMetrics) -> None:
        """Handle metrics update from system monitor."""
        self.metrics_updated.emit(metrics)
    
    def _on_health_warning(self, message: str) -> None:
        """Handle health warning from system monitor."""
        self.health_warning.emit(message)
    
    def get_graph(self, name: str) -> Optional[QObject]:
        """Get a graph by name."""
        return self._graphs.get(name)
    
    def set_version(self, version: str) -> None:
        """Set version for all visualizations."""
        for graph in self._graphs.values():
            graph.set_version(version)
    
    def shutdown(self) -> None:
        """Shutdown visualization components."""
        # Stop update timer
        self._update_timer.stop()
        
        # Shutdown graphs
        for graph in self._graphs.values():
            graph.shutdown()
        
        # Clear references
        self._graphs.clear()
        self._system_monitor = None 
 