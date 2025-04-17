#!/usr/bin/env python3
"""
LUMINA v7.5 Bridge Monitor
Monitors version bridge performance and health
"""

import os
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class MessageMetrics:
    """Metrics for message processing"""
    total_messages: int = 0
    successful_messages: int = 0
    failed_messages: int = 0
    total_processing_time: float = 0.0
    last_message_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time"""
        if self.total_messages == 0:
            return 0.0
        return self.total_processing_time / self.total_messages
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_messages == 0:
            return 0.0
        return self.successful_messages / self.total_messages * 100

class BridgeMonitor:
    """Monitors version bridge performance and health"""
    
    def __init__(self):
        """Initialize the bridge monitor"""
        self.logger = logging.getLogger("BridgeMonitor")
        self.start_time = datetime.now()
        self.version_metrics: Dict[str, MessageMetrics] = {}
        self.global_metrics = MessageMetrics()
        self._bottleneck_threshold = 1.0  # seconds
        
        # Configure file handler
        fh = logging.FileHandler(os.path.join('logs', 'bridge_monitor.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
    def record_message_start(self, source: str, target: str) -> float:
        """Record the start of message processing"""
        start_time = time.time()
        
        # Initialize metrics if needed
        if source not in self.version_metrics:
            self.version_metrics[source] = MessageMetrics()
            
        # Update message counts
        self.version_metrics[source].total_messages += 1
        self.global_metrics.total_messages += 1
        
        return start_time
        
    def record_message_complete(self, source: str, target: str, start_time: float, success: bool):
        """Record message completion"""
        end_time = time.time()
        processing_time = end_time - start_time
        
        metrics = self.version_metrics[source]
        metrics.total_processing_time += processing_time
        metrics.last_message_time = end_time
        
        if success:
            metrics.successful_messages += 1
            self.global_metrics.successful_messages += 1
        else:
            metrics.failed_messages += 1
            self.global_metrics.failed_messages += 1
            
        self.global_metrics.total_processing_time += processing_time
        self.global_metrics.last_message_time = end_time
        
        # Log if processing time exceeds threshold
        if processing_time > self._bottleneck_threshold:
            self.logger.warning(
                f"Slow message processing detected: {processing_time:.2f}s "
                f"from {source} to {target}"
            )
            
    def record_error(self, version: str, error: str):
        """Record an error for a version"""
        if version not in self.version_metrics:
            self.version_metrics[version] = MessageMetrics()
            
        self.version_metrics[version].errors.append(error)
        self.global_metrics.errors.append(f"{version}: {error}")
        self.logger.error(f"Error in version {version}: {error}")
        
    def detect_bottlenecks(self) -> List[Dict[str, any]]:
        """Detect performance bottlenecks"""
        bottlenecks = []
        
        for version, metrics in self.version_metrics.items():
            # Check for high failure rate (>10%)
            if metrics.total_messages > 10 and metrics.success_rate < 90:
                bottlenecks.append({
                    "version": version,
                    "type": "high_failure_rate",
                    "success_rate": metrics.success_rate,
                    "total_messages": metrics.total_messages
                })
                
            # Check for slow processing (avg > threshold)
            if metrics.average_processing_time > self._bottleneck_threshold:
                bottlenecks.append({
                    "version": version,
                    "type": "slow_processing",
                    "avg_time": metrics.average_processing_time,
                    "threshold": self._bottleneck_threshold
                })
                
            # Check for recent errors
            recent_errors = [
                error for error in metrics.errors
                if error.startswith(datetime.now().strftime("%Y-%m-%d"))
            ]
            if len(recent_errors) > 5:  # More than 5 errors today
                bottlenecks.append({
                    "version": version,
                    "type": "high_error_rate",
                    "error_count": len(recent_errors),
                    "recent_errors": recent_errors[-5:]  # Last 5 errors
                })
                
        return bottlenecks
        
    def get_system_health(self) -> Dict[str, any]:
        """Get overall system health metrics"""
        return {
            "uptime": str(datetime.now() - self.start_time),
            "total_messages": self.global_metrics.total_messages,
            "success_rate": self.global_metrics.success_rate,
            "avg_processing_time": self.global_metrics.average_processing_time,
            "recent_errors": self.global_metrics.errors[-5:] if self.global_metrics.errors else [],
            "version_metrics": {
                version: {
                    "success_rate": metrics.success_rate,
                    "avg_processing_time": metrics.average_processing_time,
                    "total_messages": metrics.total_messages,
                    "recent_errors": metrics.errors[-3:] if metrics.errors else []
                }
                for version, metrics in self.version_metrics.items()
            }
        }
        
    def get_version_health(self, version: str) -> Optional[Dict[str, any]]:
        """Get health metrics for a specific version"""
        if version not in self.version_metrics:
            return None
            
        metrics = self.version_metrics[version]
        return {
            "success_rate": metrics.success_rate,
            "avg_processing_time": metrics.average_processing_time,
            "total_messages": metrics.total_messages,
            "failed_messages": metrics.failed_messages,
            "recent_errors": metrics.errors[-5:] if metrics.errors else []
        } 