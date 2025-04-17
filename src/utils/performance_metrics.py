"""
Performance Metrics and Benchmarking System for Lumina Neural Network Project

This module provides tools for measuring and tracking performance metrics
across different components of the system.
"""

import time
import functools
import threading
from typing import Dict, Any, Callable, Optional
from datetime import datetime
from collections import defaultdict
import psutil
import json
from pathlib import Path

class PerformanceMetrics:
    """Performance metrics collection and analysis"""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = defaultdict(list)
        self.lock = threading.Lock()
        self.process = psutil.Process()
    
    def measure_execution_time(self, func: Callable) -> Callable:
        """Decorator to measure function execution time"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self.process.memory_info().rss
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            metric_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "function": func.__name__,
                "execution_time": execution_time,
                "memory_used": memory_used,
                "args": str(args),
                "kwargs": str(kwargs)
            }
            
            with self.lock:
                self.metrics[func.__name__].append(metric_data)
            
            return result
        return wrapper
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a custom metric"""
        metric_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "name": name,
            "value": value
        }
        
        if metadata:
            metric_data["metadata"] = metadata
        
        with self.lock:
            self.metrics[name].append(metric_data)
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get recorded metrics"""
        with self.lock:
            if name:
                return {name: self.metrics.get(name, [])}
            return dict(self.metrics)
    
    def save_metrics(self, filename: Optional[str] = None):
        """Save metrics to file"""
        if not filename:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        metrics_file = self.metrics_dir / filename
        
        with self.lock:
            metrics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": dict(self.metrics)
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "process_memory": self.process.memory_info().rss,
            "process_cpu": self.process.cpu_percent()
        }

# Create global instance
performance_metrics = PerformanceMetrics()

def measure_time(func: Callable) -> Callable:
    """Decorator to measure function execution time using global metrics"""
    return performance_metrics.measure_execution_time(func)

def record_metric(name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
    """Record a custom metric using global metrics"""
    performance_metrics.record_metric(name, value, metadata)

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics using global metrics"""
    return performance_metrics.get_system_metrics() 