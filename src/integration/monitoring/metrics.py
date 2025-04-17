"""
Monitoring Module

Handles system performance monitoring and health tracking.
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from dataclasses import dataclass
from datetime import datetime

from ..config import MONITORING_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System metrics data class."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    timestamp: datetime

class MetricsCollector:
    """Collects and exports system metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or MONITORING_CONFIG
        
        # Initialize Prometheus metrics
        self.gate_count = Gauge('logic_gate_count', 'Number of active logic gates')
        self.gate_state = Gauge('logic_gate_state', 'Gate state (0=closed, 1=open)', ['gate_id'])
        self.connection_count = Gauge('connection_count', 'Number of active connections')
        
        self.signal_counter = Counter('signal_count', 'Number of signals processed')
        self.error_counter = Counter('error_count', 'Number of errors encountered')
        
        self.processing_time = Histogram(
            'signal_processing_time',
            'Time taken to process signals',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
        )
        
        # System metrics
        self.cpu_usage = Gauge('cpu_usage', 'CPU usage percentage')
        self.memory_usage = Gauge('memory_usage', 'Memory usage percentage')
        self.disk_usage = Gauge('disk_usage', 'Disk usage percentage')
        
        # AutoWiki metrics
        self.article_count = Gauge('article_count', 'Number of articles in AutoWiki')
        self.suggestion_count = Gauge('suggestion_count', 'Number of pending suggestions')
        self.learning_progress = Gauge('learning_progress', 'Learning system progress')
        
        # Start metrics server
        self.start_server()
        
    def start_server(self) -> None:
        """Start the Prometheus metrics server."""
        try:
            start_http_server(
                self.config['metrics_port'],
                addr='localhost'
            )
            logger.info(f"Metrics server started on port {self.config['metrics_port']}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            metrics = SystemMetrics(
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                disk_usage=psutil.disk_usage('/').percent,
                network_io={
                    'bytes_sent': psutil.net_io_counters().bytes_sent,
                    'bytes_recv': psutil.net_io_counters().bytes_recv
                },
                timestamp=datetime.now()
            )
            
            # Update Prometheus metrics
            self.cpu_usage.set(metrics.cpu_usage)
            self.memory_usage.set(metrics.memory_usage)
            self.disk_usage.set(metrics.disk_usage)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
            
    def update_gate_metrics(
        self,
        gate_count: int,
        gate_states: Dict[str, bool],
        connection_count: int
    ) -> None:
        """Update logic gate metrics."""
        try:
            self.gate_count.set(gate_count)
            self.connection_count.set(connection_count)
            
            for gate_id, is_open in gate_states.items():
                self.gate_state.labels(gate_id=gate_id).set(1 if is_open else 0)
                
        except Exception as e:
            logger.error(f"Error updating gate metrics: {e}")
            
    def update_autowiki_metrics(
        self,
        article_count: int,
        suggestion_count: int,
        learning_progress: float
    ) -> None:
        """Update AutoWiki metrics."""
        try:
            self.article_count.set(article_count)
            self.suggestion_count.set(suggestion_count)
            self.learning_progress.set(learning_progress)
            
        except Exception as e:
            logger.error(f"Error updating AutoWiki metrics: {e}")
            
    def record_signal_processed(self, processing_time: float) -> None:
        """Record a processed signal."""
        try:
            self.signal_counter.inc()
            self.processing_time.observe(processing_time)
            
        except Exception as e:
            logger.error(f"Error recording signal: {e}")
            
    def record_error(self) -> None:
        """Record an error occurrence."""
        try:
            self.error_counter.inc()
        except Exception as e:
            logger.error(f"Error recording error: {e}")
            
class HealthMonitor:
    """Monitors system health and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks = {
            'cpu': self._check_cpu,
            'memory': self._check_memory,
            'disk': self._check_disk,
            'errors': self._check_errors
        }
        
    def check_health(self) -> Dict[str, Any]:
        """Perform a complete health check."""
        try:
            health_status = {}
            for check_name, check_func in self.health_checks.items():
                health_status[check_name] = check_func()
                
            # Calculate overall health
            healthy_checks = sum(1 for status in health_status.values() if status['healthy'])
            total_checks = len(health_status)
            overall_health = healthy_checks / total_checks
            
            health_status['overall'] = {
                'healthy': overall_health >= 0.7,
                'score': overall_health,
                'timestamp': datetime.now().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            return {'error': str(e)}
            
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU health."""
        try:
            cpu_usage = psutil.cpu_percent()
            return {
                'healthy': cpu_usage < 80,
                'value': cpu_usage,
                'threshold': 80
            }
        except Exception as e:
            logger.error(f"Error checking CPU: {e}")
            return {'healthy': False, 'error': str(e)}
            
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory health."""
        try:
            memory = psutil.virtual_memory()
            return {
                'healthy': memory.percent < 90,
                'value': memory.percent,
                'threshold': 90
            }
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return {'healthy': False, 'error': str(e)}
            
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk health."""
        try:
            disk = psutil.disk_usage('/')
            return {
                'healthy': disk.percent < 95,
                'value': disk.percent,
                'threshold': 95
            }
        except Exception as e:
            logger.error(f"Error checking disk: {e}")
            return {'healthy': False, 'error': str(e)}
            
    def _check_errors(self) -> Dict[str, Any]:
        """Check error rate."""
        try:
            error_count = self.metrics_collector.error_counter._value.get()
            signal_count = self.metrics_collector.signal_counter._value.get()
            
            if signal_count == 0:
                error_rate = 0
            else:
                error_rate = error_count / signal_count
                
            return {
                'healthy': error_rate < 0.01,
                'value': error_rate,
                'threshold': 0.01
            }
        except Exception as e:
            logger.error(f"Error checking error rate: {e}")
            return {'healthy': False, 'error': str(e)}
            
@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    signal_processing_time: float
    gate_update_time: float
    pattern_recognition_time: float
    state_prediction_time: float
    timestamp: datetime

class PerformanceMonitor:
    """Monitors and analyzes system performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.performance_history: List[PerformanceMetrics] = []
        
    def record_performance(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        try:
            self.performance_history.append(metrics)
            
            # Trim history if too long
            max_history = 1000
            if len(self.performance_history) > max_history:
                self.performance_history = self.performance_history[-max_history:]
                
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
            
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics."""
        try:
            if not self.performance_history:
                return {}
                
            # Calculate statistics
            signal_times = [m.signal_processing_time for m in self.performance_history]
            gate_times = [m.gate_update_time for m in self.performance_history]
            pattern_times = [m.pattern_recognition_time for m in self.performance_history]
            prediction_times = [m.state_prediction_time for m in self.performance_history]
            
            return {
                'signal_processing': {
                    'avg': sum(signal_times) / len(signal_times),
                    'max': max(signal_times),
                    'min': min(signal_times)
                },
                'gate_update': {
                    'avg': sum(gate_times) / len(gate_times),
                    'max': max(gate_times),
                    'min': min(gate_times)
                },
                'pattern_recognition': {
                    'avg': sum(pattern_times) / len(pattern_times),
                    'max': max(pattern_times),
                    'min': min(pattern_times)
                },
                'state_prediction': {
                    'avg': sum(prediction_times) / len(prediction_times),
                    'max': max(prediction_times),
                    'min': min(prediction_times)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {'error': str(e)} 