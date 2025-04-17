"""
System monitor for the Lumina Frontend System.
Handles system health monitoring, performance metrics, and resource usage.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal, QTimer
import psutil
import time

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    gpu_usage: Optional[float]
    timestamp: float

class SystemMonitor(QObject):
    """Monitors system health and performance."""
    
    # Signals
    metrics_updated = Signal(SystemMetrics)
    health_warning = Signal(str)
    resource_threshold = Signal(str, float)
    
    def __init__(self):
        super().__init__()
        self._metrics = SystemMetrics(0.0, 0.0, 0.0, 0.0, None, time.time())
        self._thresholds = {
            'cpu': 80.0,
            'memory': 85.0,
            'disk': 90.0,
            'network': 75.0,
            'gpu': 85.0
        }
        
        # Initialize monitoring timer
        self._monitor_timer = QTimer()
        self._monitor_timer.timeout.connect(self._update_metrics)
        self._monitor_timer.start(1000)  # Update every second
    
    def _update_metrics(self) -> None:
        """Update system metrics."""
        # Get CPU usage
        cpu_usage = psutil.cpu_percent()
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Get disk I/O
        disk_io = psutil.disk_io_counters()
        disk_usage = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)  # MB/s
        
        # Get network I/O
        net_io = psutil.net_io_counters()
        network_usage = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB/s
        
        # Get GPU usage if available
        gpu_usage = self._get_gpu_usage()
        
        # Update metrics
        self._metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_io=disk_usage,
            network_io=network_usage,
            gpu_usage=gpu_usage,
            timestamp=time.time()
        )
        
        # Emit updated metrics
        self.metrics_updated.emit(self._metrics)
        
        # Check thresholds
        self._check_thresholds()
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        return None
    
    def _check_thresholds(self) -> None:
        """Check if any metrics exceed thresholds."""
        if self._metrics.cpu_usage > self._thresholds['cpu']:
            self.resource_threshold.emit('cpu', self._metrics.cpu_usage)
            
        if self._metrics.memory_usage > self._thresholds['memory']:
            self.resource_threshold.emit('memory', self._metrics.memory_usage)
            
        if self._metrics.disk_io > self._thresholds['disk']:
            self.resource_threshold.emit('disk', self._metrics.disk_io)
            
        if self._metrics.network_io > self._thresholds['network']:
            self.resource_threshold.emit('network', self._metrics.network_io)
            
        if self._metrics.gpu_usage and self._metrics.gpu_usage > self._thresholds['gpu']:
            self.resource_threshold.emit('gpu', self._metrics.gpu_usage)
    
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self._metrics
    
    def set_threshold(self, resource: str, threshold: float) -> None:
        """Set threshold for a resource."""
        if resource in self._thresholds:
            self._thresholds[resource] = threshold
    
    def get_threshold(self, resource: str) -> Optional[float]:
        """Get threshold for a resource."""
        return self._thresholds.get(resource)
    
    def start_monitoring(self) -> None:
        """Start system monitoring."""
        self._monitor_timer.start()
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._monitor_timer.stop() 
System monitor for the Lumina Frontend System.
Handles system health monitoring, performance metrics, and resource usage.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal, QTimer
import psutil
import time

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    gpu_usage: Optional[float]
    timestamp: float

class SystemMonitor(QObject):
    """Monitors system health and performance."""
    
    # Signals
    metrics_updated = Signal(SystemMetrics)
    health_warning = Signal(str)
    resource_threshold = Signal(str, float)
    
    def __init__(self):
        super().__init__()
        self._metrics = SystemMetrics(0.0, 0.0, 0.0, 0.0, None, time.time())
        self._thresholds = {
            'cpu': 80.0,
            'memory': 85.0,
            'disk': 90.0,
            'network': 75.0,
            'gpu': 85.0
        }
        
        # Initialize monitoring timer
        self._monitor_timer = QTimer()
        self._monitor_timer.timeout.connect(self._update_metrics)
        self._monitor_timer.start(1000)  # Update every second
    
    def _update_metrics(self) -> None:
        """Update system metrics."""
        # Get CPU usage
        cpu_usage = psutil.cpu_percent()
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Get disk I/O
        disk_io = psutil.disk_io_counters()
        disk_usage = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)  # MB/s
        
        # Get network I/O
        net_io = psutil.net_io_counters()
        network_usage = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB/s
        
        # Get GPU usage if available
        gpu_usage = self._get_gpu_usage()
        
        # Update metrics
        self._metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_io=disk_usage,
            network_io=network_usage,
            gpu_usage=gpu_usage,
            timestamp=time.time()
        )
        
        # Emit updated metrics
        self.metrics_updated.emit(self._metrics)
        
        # Check thresholds
        self._check_thresholds()
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        return None
    
    def _check_thresholds(self) -> None:
        """Check if any metrics exceed thresholds."""
        if self._metrics.cpu_usage > self._thresholds['cpu']:
            self.resource_threshold.emit('cpu', self._metrics.cpu_usage)
            
        if self._metrics.memory_usage > self._thresholds['memory']:
            self.resource_threshold.emit('memory', self._metrics.memory_usage)
            
        if self._metrics.disk_io > self._thresholds['disk']:
            self.resource_threshold.emit('disk', self._metrics.disk_io)
            
        if self._metrics.network_io > self._thresholds['network']:
            self.resource_threshold.emit('network', self._metrics.network_io)
            
        if self._metrics.gpu_usage and self._metrics.gpu_usage > self._thresholds['gpu']:
            self.resource_threshold.emit('gpu', self._metrics.gpu_usage)
    
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self._metrics
    
    def set_threshold(self, resource: str, threshold: float) -> None:
        """Set threshold for a resource."""
        if resource in self._thresholds:
            self._thresholds[resource] = threshold
    
    def get_threshold(self, resource: str) -> Optional[float]:
        """Get threshold for a resource."""
        return self._thresholds.get(resource)
    
    def start_monitoring(self) -> None:
        """Start system monitoring."""
        self._monitor_timer.start()
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._monitor_timer.stop() 
 