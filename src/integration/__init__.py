"""
Integration Package

This package handles the integration between the frontend and backend systems,
providing communication, monitoring, and data processing capabilities.
"""

from .backend_connector import BackendConnector
from .metrics_collector import MetricsCollector
from .health_monitor import HealthMonitor
from .signal_processor import SignalProcessor
from .system_integrator import SystemIntegrator
from .config import (
    BACKEND_CONFIG,
    METRICS_CONFIG,
    HEALTH_CONFIG,
    SIGNAL_CONFIG,
    LOGGING_CONFIG
)

__all__ = [
    'BackendConnector',
    'MetricsCollector',
    'HealthMonitor',
    'SignalProcessor',
    'SystemIntegrator',
    'BACKEND_CONFIG',
    'METRICS_CONFIG',
    'HEALTH_CONFIG',
    'SIGNAL_CONFIG',
    'LOGGING_CONFIG'
]

"""Integration package for neural network project.""" 