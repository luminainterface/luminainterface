#!/usr/bin/env python3
"""
Metrics Collector

This module collects and processes metrics from the backend system,
providing real-time monitoring and analysis capabilities.
"""

import logging
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from .config import METRICS_CONFIG

logger = logging.getLogger(__name__)

class MetricsCollector(QObject):
    """Collects and processes metrics from the backend."""
    
    # Signals
    metrics_updated = Signal(dict)  # Emitted when new metrics are available
    threshold_exceeded = Signal(str, float)  # Emitted when a threshold is exceeded
    
    def __init__(self):
        """Initialize the metrics collector."""
        super().__init__()
        self.metrics_history = {
            'cpu_usage': deque(maxlen=METRICS_CONFIG['history_length']),
            'memory_usage': deque(maxlen=METRICS_CONFIG['history_length']),
            'network_traffic': deque(maxlen=METRICS_CONFIG['history_length']),
            'disk_io': deque(maxlen=METRICS_CONFIG['history_length'])
        }
        self.thresholds = METRICS_CONFIG['thresholds']
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update metrics with new data."""
        try:
            # Update history
            for key, value in metrics.items():
                if key in self.metrics_history:
                    self.metrics_history[key].append(value)
                    
            # Check thresholds
            self._check_thresholds(metrics)
            
            # Emit updated metrics
            self.metrics_updated.emit(metrics)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            
    def _check_thresholds(self, metrics: Dict[str, float]) -> None:
        """Check if any metrics exceed their thresholds."""
        for key, value in metrics.items():
            if key in self.thresholds and value > self.thresholds[key]:
                self.threshold_exceeded.emit(key, value)
                
    def get_metric_history(self, metric_name: str) -> Optional[List[float]]:
        """Get history for a specific metric."""
        return list(self.metrics_history.get(metric_name, []))
        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        return {
            key: values[-1] if values else 0.0
            for key, values in self.metrics_history.items()
        }
        
    def get_metric_average(self, metric_name: str) -> float:
        """Get average value for a metric."""
        values = self.metrics_history.get(metric_name, [])
        return sum(values) / len(values) if values else 0.0
        
    def get_metric_trend(self, metric_name: str) -> float:
        """Get trend for a metric (positive or negative)."""
        values = self.metrics_history.get(metric_name, [])
        if len(values) < 2:
            return 0.0
        return values[-1] - values[0] 