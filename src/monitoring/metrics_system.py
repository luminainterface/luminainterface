"""
Monitoring Metrics System

This module provides a centralized metrics collection, storage and retrieval system 
for monitoring the Lumina Neural Network System.
"""

import logging
import time
import threading
import json
import statistics
import os
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)

# Ensure metrics directory exists
metrics_dir = Path("data/metrics")
metrics_dir.mkdir(parents=True, exist_ok=True)

class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"         # Incremental counter
    GAUGE = "gauge"             # Value that can go up and down
    HISTOGRAM = "histogram"     # Distribution of values
    TIMER = "timer"             # Time duration
    EVENT = "event"             # Discrete events


class MetricsManager:
    """Centralized metrics collection and management system"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton implementation"""
        if cls._instance is None:
            cls._instance = super(MetricsManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the metrics manager"""
        if self._initialized:
            return
            
        self._initialized = True
        self.config = config or {}
        
        # Metrics storage by category and name
        self._metrics = {
            "system": {},
            "neural": {},
            "memory": {},
            "api": {},
            "integration": {},
            "performance": {},
            "components": {},
            "user": {}
        }
        
        # Historical metrics (time series)
        self._history = {}
        
        # Historical storage configuration
        self._history_max_points = self.config.get("history_max_points", 1000)
        self._history_retention = self.config.get("history_retention_days", 7)
        
        # Real-time value storage
        self._current_values = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background collection thread
        self._collector_thread = None
        self._should_stop = threading.Event()
        self._collection_interval = self.config.get("collection_interval", 60)  # seconds
        
        # Start background collection if enabled
        if self.config.get("auto_collect", True):
            self.start_collection()
            
        logger.info("MetricsManager initialized")
    
    def register_metric(self, 
                      category: str, 
                      name: str, 
                      metric_type: MetricType,
                      description: str = None,
                      unit: str = None,
                      labels: List[str] = None,
                      aggregation: str = "last") -> None:
        """
        Register a new metric for collection.
        
        Args:
            category: Metric category (system, neural, memory, etc.)
            name: Metric name
            metric_type: Type of metric
            description: Human-readable description
            unit: Unit of measurement
            labels: Label dimensions for this metric
            aggregation: How to aggregate this metric (last, sum, avg, min, max)
        """
        with self._lock:
            if category not in self._metrics:
                self._metrics[category] = {}
                
            self._metrics[category][name] = {
                "type": metric_type,
                "description": description or f"{category}.{name} metric",
                "unit": unit,
                "labels": labels or [],
                "aggregation": aggregation,
                "created": datetime.now().isoformat()
            }
            
            # Initialize history for this metric
            history_key = f"{category}.{name}"
            if history_key not in self._history:
                self._history[history_key] = []
                
            logger.debug(f"Registered metric: {category}.{name} ({metric_type.value})")
    
    def record_metric(self, 
                    category: str, 
                    name: str, 
                    value: Any,
                    timestamp: datetime = None,
                    labels: Dict[str, str] = None) -> None:
        """
        Record a metric value.
        
        Args:
            category: Metric category
            name: Metric name
            value: Metric value
            timestamp: Timestamp (default: now)
            labels: Label values for this metric
        """
        if category not in self._metrics or name not in self._metrics[category]:
            # Auto-register unknown metrics as gauges
            self.register_metric(category, name, MetricType.GAUGE)
            
        with self._lock:
            timestamp = timestamp or datetime.now()
            metric_key = f"{category}.{name}"
            
            # Store current value
            if metric_key not in self._current_values:
                self._current_values[metric_key] = {}
                
            # Use labels as a compound key if provided
            label_key = "_".join(f"{k}:{v}" for k, v in (labels or {}).items()) or "default"
            
            self._current_values[metric_key][label_key] = {
                "value": value,
                "timestamp": timestamp.isoformat(),
                "labels": labels or {}
            }
            
            # Add to history
            if metric_key in self._history:
                self._history[metric_key].append({
                    "timestamp": timestamp.isoformat(),
                    "value": value,
                    "labels": labels or {}
                })
                
                # Trim history if needed
                if len(self._history[metric_key]) > self._history_max_points:
                    self._history[metric_key] = self._history[metric_key][-self._history_max_points:]
    
    def increment_counter(self, 
                         category: str, 
                         name: str, 
                         value: int = 1,
                         labels: Dict[str, str] = None) -> int:
        """
        Increment a counter metric.
        
        Args:
            category: Metric category
            name: Metric name
            value: Increment amount (default: 1)
            labels: Label values for this metric
            
        Returns:
            New counter value
        """
        # Auto-register if not exists
        if category not in self._metrics or name not in self._metrics[category]:
            self.register_metric(category, name, MetricType.COUNTER)
        
        metric_key = f"{category}.{name}"
        label_key = "_".join(f"{k}:{v}" for k, v in (labels or {}).items()) or "default"
        
        with self._lock:
            # Initialize if not exists
            if metric_key not in self._current_values:
                self._current_values[metric_key] = {}
                
            if label_key not in self._current_values[metric_key]:
                self._current_values[metric_key][label_key] = {
                    "value": 0,
                    "timestamp": datetime.now().isoformat(),
                    "labels": labels or {}
                }
                
            # Increment value
            current = self._current_values[metric_key][label_key]["value"]
            new_value = current + value
            self._current_values[metric_key][label_key]["value"] = new_value
            self._current_values[metric_key][label_key]["timestamp"] = datetime.now().isoformat()
            
            # Add to history
            self.record_metric(category, name, new_value, labels=labels)
            
            return new_value
    
    def record_timer(self, 
                    category: str, 
                    name: str, 
                    duration_ms: float,
                    labels: Dict[str, str] = None) -> None:
        """
        Record a timing metric in milliseconds.
        
        Args:
            category: Metric category
            name: Metric name
            duration_ms: Duration in milliseconds
            labels: Label values for this metric
        """
        # Auto-register if not exists
        if category not in self._metrics or name not in self._metrics[category]:
            self.register_metric(category, name, MetricType.TIMER, unit="ms")
            
        # Record the timing
        self.record_metric(category, name, duration_ms, labels=labels)
    
    def timed(self, category: str, name: str, labels: Dict[str, str] = None):
        """
        Decorator to time a function execution.
        
        Args:
            category: Metric category
            name: Metric name
            labels: Label values for this metric
            
        Returns:
            Decorated function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    self.record_timer(category, name, duration_ms, labels)
            return wrapper
        return decorator
    
    def get_current_value(self, 
                         category: str, 
                         name: str,
                         label_filter: Dict[str, str] = None) -> Any:
        """
        Get the current value of a metric.
        
        Args:
            category: Metric category
            name: Metric name
            label_filter: Filter by label values
            
        Returns:
            Current metric value or None if not found
        """
        metric_key = f"{category}.{name}"
        
        with self._lock:
            if metric_key not in self._current_values:
                return None
                
            # If no label filter, return the default or first value
            if not label_filter:
                if "default" in self._current_values[metric_key]:
                    return self._current_values[metric_key]["default"]["value"]
                elif self._current_values[metric_key]:
                    # Return the first value
                    return next(iter(self._current_values[metric_key].values()))["value"]
                return None
                
            # Filter by labels
            for label_key, data in self._current_values[metric_key].items():
                labels = data.get("labels", {})
                
                # Check if all filter labels match
                if all(labels.get(k) == v for k, v in label_filter.items()):
                    return data["value"]
                    
            return None
    
    def get_metric_history(self, 
                           category: str, 
                           name: str,
                           hours: int = 24,
                           aggregation: str = None,
                           label_filter: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Get historical values for a metric.
        
        Args:
            category: Metric category
            name: Metric name
            hours: Number of hours of history to retrieve
            aggregation: Override the default aggregation method
            label_filter: Filter by label values
            
        Returns:
            List of timestamp/value pairs
        """
        metric_key = f"{category}.{name}"
        
        with self._lock:
            if metric_key not in self._history:
                return []
                
            # Get metric definition
            metric_def = self._metrics.get(category, {}).get(name, {})
            agg_method = aggregation or metric_def.get("aggregation", "last")
            
            # Filter by time
            cutoff = datetime.now() - timedelta(hours=hours)
            history = [
                item for item in self._history[metric_key]
                if datetime.fromisoformat(item["timestamp"]) >= cutoff
            ]
            
            # Filter by labels if provided
            if label_filter:
                history = [
                    item for item in history
                    if all(item.get("labels", {}).get(k) == v for k, v in label_filter.items())
                ]
                
            return history
    
    def get_aggregated_history(self, 
                             category: str, 
                             name: str,
                             interval_minutes: int = 5,
                             hours: int = 24,
                             aggregation: str = None,
                             label_filter: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Get aggregated historical values for a metric.
        
        Args:
            category: Metric category
            name: Metric name
            interval_minutes: Aggregation interval in minutes
            hours: Number of hours of history to retrieve
            aggregation: Override the default aggregation method
            label_filter: Filter by label values
            
        Returns:
            List of timestamp/value pairs aggregated by interval
        """
        # Get raw history
        history = self.get_metric_history(category, name, hours, aggregation, label_filter)
        
        if not history:
            return []
            
        # Get metric definition
        metric_def = self._metrics.get(category, {}).get(name, {})
        agg_method = aggregation or metric_def.get("aggregation", "last")
        
        # Group by interval
        result = []
        interval_seconds = interval_minutes * 60
        
        # Determine start and end times
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Create time buckets
        current_time = start_time
        while current_time < end_time:
            bucket_end = current_time + timedelta(minutes=interval_minutes)
            
            # Filter points in this bucket
            bucket_points = [
                item for item in history
                if current_time <= datetime.fromisoformat(item["timestamp"]) < bucket_end
            ]
            
            if bucket_points:
                # Extract values
                values = [item["value"] for item in bucket_points]
                
                # Aggregate based on method
                if agg_method == "sum":
                    agg_value = sum(values)
                elif agg_method == "avg":
                    agg_value = sum(values) / len(values)
                elif agg_method == "min":
                    agg_value = min(values)
                elif agg_method == "max":
                    agg_value = max(values)
                else:  # "last" or default
                    agg_value = values[-1]
                    
                result.append({
                    "timestamp": current_time.isoformat(),
                    "value": agg_value,
                    "count": len(bucket_points)
                })
            else:
                # No data for this interval
                result.append({
                    "timestamp": current_time.isoformat(),
                    "value": None,
                    "count": 0
                })
                
            current_time = bucket_end
            
        return result
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all registered metrics and their current values.
        
        Returns:
            Dictionary of all metrics by category
        """
        result = {}
        
        with self._lock:
            for category, metrics in self._metrics.items():
                result[category] = {}
                
                for name, definition in metrics.items():
                    metric_key = f"{category}.{name}"
                    
                    # Get current values
                    values = {}
                    if metric_key in self._current_values:
                        for label_key, data in self._current_values[metric_key].items():
                            values[label_key] = {
                                "value": data["value"],
                                "timestamp": data["timestamp"],
                                "labels": data["labels"]
                            }
                    
                    # Add to result
                    result[category][name] = {
                        "definition": definition,
                        "values": values
                    }
                    
        return result
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system-level metrics (CPU, memory, disk, etc).
        
        Returns:
            Dictionary of system metrics
        """
        try:
            # Get system metrics using psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Record metrics
            self.record_metric("system", "cpu_usage", cpu_percent, unit="%")
            self.record_metric("system", "memory_usage", memory.percent, unit="%")
            self.record_metric("system", "memory_available", memory.available / (1024 * 1024), unit="MB")
            self.record_metric("system", "disk_usage", disk.percent, unit="%")
            self.record_metric("system", "disk_available", disk.free / (1024 * 1024 * 1024), unit="GB")
            
            # Return current values
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "memory_total_mb": memory.total / (1024 * 1024),
                "disk_usage": disk.percent,
                "disk_available_gb": disk.free / (1024 * 1024 * 1024),
                "disk_total_gb": disk.total / (1024 * 1024 * 1024),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def start_collection(self) -> None:
        """Start the background metrics collection thread"""
        if self._collector_thread is not None and self._collector_thread.is_alive():
            logger.warning("Metrics collector already running")
            return
            
        self._should_stop.clear()
        self._collector_thread = threading.Thread(
            target=self._collection_thread,
            daemon=True,
            name="MetricsCollector"
        )
        self._collector_thread.start()
        logger.info("Started metrics collection thread")
    
    def stop_collection(self) -> None:
        """Stop the background metrics collection thread"""
        if self._collector_thread is None or not self._collector_thread.is_alive():
            return
            
        self._should_stop.set()
        self._collector_thread.join(timeout=2.0)
        logger.info("Stopped metrics collection thread")
    
    def _collection_thread(self) -> None:
        """Background thread for collecting system metrics"""
        next_save_time = time.time() + 300  # Save every 5 minutes
        
        while not self._should_stop.is_set():
            try:
                # Collect system metrics
                self.get_system_metrics()
                
                # Save metrics to disk if it's time
                current_time = time.time()
                if current_time >= next_save_time:
                    self._save_metrics_to_disk()
                    next_save_time = current_time + 300
                
                # Sleep until next collection
                sleep_time = self._collection_interval
                if sleep_time <= 0:
                    sleep_time = 60  # Default to 60 seconds
                    
                # Use event with timeout to allow clean shutdown
                self._should_stop.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in metrics collection thread: {str(e)}")
                # Sleep briefly before retrying
                time.sleep(5)
    
    def _save_metrics_to_disk(self) -> None:
        """Save collected metrics to disk"""
        try:
            # Generate filename with date
            date_str = datetime.now().strftime("%Y%m%d")
            metrics_file = metrics_dir / f"metrics_{date_str}.json"
            
            # Get metrics to save
            metrics_data = self.get_all_metrics()
            
            # Save to file
            with open(metrics_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics_data
                }, f, indent=2)
                
            logger.debug(f"Saved metrics to {metrics_file}")
            
            # Clean up old metrics files
            self._cleanup_old_metrics_files()
            
        except Exception as e:
            logger.error(f"Error saving metrics to disk: {str(e)}")
    
    def _cleanup_old_metrics_files(self) -> None:
        """Remove metrics files older than retention period"""
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self._history_retention)
            
            # Find and remove old files
            for file in metrics_dir.glob("metrics_*.json"):
                try:
                    # Extract date from filename
                    date_str = file.stem.split("_")[1]
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    # Check if older than cutoff
                    if file_date < cutoff_date:
                        file.unlink()
                        logger.debug(f"Removed old metrics file: {file}")
                except Exception as e:
                    logger.warning(f"Error processing metrics file {file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error cleaning up old metrics files: {str(e)}")


# Singleton instance
metrics_manager = MetricsManager()

# Helper functions
def record_metric(category: str, name: str, value: Any, **kwargs) -> None:
    """Record a metric value"""
    metrics_manager.record_metric(category, name, value, **kwargs)

def increment_counter(category: str, name: str, value: int = 1, **kwargs) -> int:
    """Increment a counter metric"""
    return metrics_manager.increment_counter(category, name, value, **kwargs)

def record_timer(category: str, name: str, duration_ms: float, **kwargs) -> None:
    """Record a timing metric"""
    metrics_manager.record_timer(category, name, duration_ms, **kwargs)

def timed(category: str, name: str, **kwargs):
    """Decorator to time a function"""
    return metrics_manager.timed(category, name, **kwargs) 