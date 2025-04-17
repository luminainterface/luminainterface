#!/usr/bin/env python3
"""
Health Monitor

This module monitors the health of the backend system components,
providing real-time status updates and alerts.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from PySide6.QtCore import QObject, Signal

from .config import HEALTH_CONFIG

logger = logging.getLogger(__name__)

class HealthMonitor(QObject):
    """Monitors the health of backend system components."""
    
    # Signals
    health_updated = Signal(dict)  # Emitted when health status changes
    component_status_changed = Signal(str, bool)  # Emitted when a component's status changes
    error_detected = Signal(str, str)  # Emitted when an error is detected
    
    def __init__(self):
        """Initialize the health monitor."""
        super().__init__()
        self.health_status = {
            'overall': {'healthy': True},
            'components': {
                component: True
                for component in HEALTH_CONFIG['components']
            }
        }
        self.last_check = datetime.now()
        self.error_counts = {component: 0 for component in HEALTH_CONFIG['components']}
        
    def update_health(self, health_data: Dict[str, Any]) -> None:
        """Update health status with new data."""
        try:
            # Update component statuses
            for component, status in health_data.get('components', {}).items():
                if component in self.health_status['components']:
                    old_status = self.health_status['components'][component]
                    if status != old_status:
                        self.component_status_changed.emit(component, status)
                    self.health_status['components'][component] = status
                    
            # Update overall health
            overall_healthy = all(self.health_status['components'].values())
            if overall_healthy != self.health_status['overall']['healthy']:
                self.health_status['overall']['healthy'] = overall_healthy
                
            # Update last check time
            self.last_check = datetime.now()
            
            # Emit updated health status
            self.health_updated.emit(self.health_status)
            
        except Exception as e:
            logger.error(f"Error updating health status: {e}")
            
    def check_health(self) -> Dict[str, Any]:
        """Check current health status."""
        # Check if any components have been unhealthy for too long
        for component, status in self.health_status['components'].items():
            if not status:
                self.error_counts[component] += 1
                if self.error_counts[component] > HEALTH_CONFIG['thresholds']['error_rate'] * 100:
                    self.error_detected.emit(component, "Component has been unhealthy for too long")
            else:
                self.error_counts[component] = 0
                
        return self.health_status
        
    def get_component_status(self, component: str) -> Optional[bool]:
        """Get status of a specific component."""
        return self.health_status['components'].get(component)
        
    def is_healthy(self) -> bool:
        """Check if the system is overall healthy."""
        return self.health_status['overall']['healthy']
        
    def get_unhealthy_components(self) -> List[str]:
        """Get list of unhealthy components."""
        return [
            component
            for component, status in self.health_status['components'].items()
            if not status
        ] 