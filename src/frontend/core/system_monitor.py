from PySide6.QtCore import QObject, Signal, QTimer
from typing import Dict, Any
import logging
import psutil
import time

class SystemMonitor(QObject):
    """Monitors system health and performance metrics."""
    
    # Signals
    metrics_updated = Signal(dict)
    system_warning = Signal(str)
    critical_error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, Any] = {}
        self.monitoring = False
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        
    def start_monitoring(self, interval_ms: int = 1000):
        """Start monitoring system metrics."""
        try:
            self.logger.info("Starting system monitoring...")
            self.monitoring = True
            self.update_timer.start(interval_ms)
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {str(e)}")
            self.critical_error.emit(str(e))
            
    def stop_monitoring(self):
        """Stop monitoring system metrics."""
        try:
            self.logger.info("Stopping system monitoring...")
            self.monitoring = False
            self.update_timer.stop()
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {str(e)}")
            self.critical_error.emit(str(e))
            
    def update_metrics(self):
        """Update system metrics."""
        try:
            metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "timestamp": time.time()
            }
            
            self.metrics = metrics
            self.metrics_updated.emit(metrics)
            
            # Check for warnings
            if metrics["cpu_percent"] > 90:
                self.system_warning.emit("High CPU usage detected")
            if metrics["memory_percent"] > 90:
                self.system_warning.emit("High memory usage detected")
            if metrics["disk_percent"] > 90:
                self.system_warning.emit("High disk usage detected")
                
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            self.critical_error.emit(str(e))
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return self.metrics
        
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self.monitoring 