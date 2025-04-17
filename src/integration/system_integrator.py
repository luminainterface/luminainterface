#!/usr/bin/env python3
"""
System Integrator

This module provides central integration between all system components:
- Frontend UI
- Backend System
- Neural Seed
- Signal System (V7.5)
- Spiderweb Bridge
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from PySide6.QtCore import QObject, Signal, Slot

from .backend_connector import BackendConnector
from .metrics_collector import MetricsCollector
from .health_monitor import HealthMonitor
from .signal_processor import SignalProcessor
from .config import (
    BACKEND_CONFIG,
    METRICS_CONFIG,
    HEALTH_CONFIG,
    SIGNAL_CONFIG
)

logger = logging.getLogger(__name__)

class SystemIntegrator(QObject):
    """Central integration manager for all system components."""
    
    # Integration signals
    system_ready = Signal()  # Emitted when all systems are initialized
    state_changed = Signal(dict)  # Emitted when system state changes
    error_occurred = Signal(str)  # Emitted when an error occurs
    
    def __init__(self):
        """Initialize the system integrator."""
        super().__init__()
        
        # Core components
        self.backend = BackendConnector()
        self.metrics = MetricsCollector()
        self.health = HealthMonitor()
        self.signal_processor = SignalProcessor()
        
        # System state
        self.state = {
            'neural_seed': {
                'connected': False,
                'consciousness_level': 0.0,
                'stability': 0.0,
                'growth_stage': 'seed'
            },
            'signal_system': {
                'connected': False,
                'message_count': 0,
                'error_rate': 0.0
            },
            'spiderweb': {
                'connected': False,
                'active_bridges': 0,
                'version_nodes': []
            }
        }
        
        # Connect signals
        self._connect_signals()
        
    def _connect_signals(self) -> None:
        """Connect internal signals between components."""
        # Backend signals
        self.backend.signal_received.connect(self._handle_backend_signal)
        self.backend.metrics_updated.connect(self.metrics.update_metrics)
        self.backend.health_updated.connect(self.health.update_health)
        self.backend.error_occurred.connect(self._handle_error)
        
        # Metrics signals
        self.metrics.threshold_exceeded.connect(self._handle_threshold_exceeded)
        
        # Health signals
        self.health.component_status_changed.connect(self._handle_component_status)
        self.health.error_detected.connect(self._handle_error)
        
        # Signal processor signals
        self.signal_processor.signal_processed.connect(self._handle_processed_signal)
        self.signal_processor.error_occurred.connect(self._handle_error)
        
    async def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            # Connect to backend
            if not await self.backend.connect():
                raise RuntimeError("Failed to connect to backend")
                
            # Register signal handlers
            self._register_signal_handlers()
            
            # Initialize Neural Seed integration
            self._initialize_neural_seed()
            
            # Initialize Signal System
            self._initialize_signal_system()
            
            # Initialize Spiderweb Bridge
            self._initialize_spiderweb()
            
            # Emit ready signal
            self.system_ready.emit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            self.error_occurred.emit(str(e))
            return False
            
    def _register_signal_handlers(self) -> None:
        """Register handlers for different signal types."""
        # Neural Seed signals
        self.signal_processor.register_handler(
            'neural_seed_update',
            self._handle_neural_seed_update
        )
        
        # Signal System signals
        self.signal_processor.register_handler(
            'signal_system_update',
            self._handle_signal_system_update
        )
        
        # Spiderweb signals
        self.signal_processor.register_handler(
            'spiderweb_update',
            self._handle_spiderweb_update
        )
        
    def _initialize_neural_seed(self) -> None:
        """Initialize Neural Seed integration."""
        try:
            # TODO: Implement actual Neural Seed initialization
            self.state['neural_seed']['connected'] = True
            self._update_state()
            
        except Exception as e:
            logger.error(f"Failed to initialize Neural Seed: {e}")
            self.error_occurred.emit(str(e))
            
    def _initialize_signal_system(self) -> None:
        """Initialize Signal System (V7.5) integration."""
        try:
            # TODO: Implement actual Signal System initialization
            self.state['signal_system']['connected'] = True
            self._update_state()
            
        except Exception as e:
            logger.error(f"Failed to initialize Signal System: {e}")
            self.error_occurred.emit(str(e))
            
    def _initialize_spiderweb(self) -> None:
        """Initialize Spiderweb Bridge integration."""
        try:
            # TODO: Implement actual Spiderweb initialization
            self.state['spiderweb']['connected'] = True
            self._update_state()
            
        except Exception as e:
            logger.error(f"Failed to initialize Spiderweb: {e}")
            self.error_occurred.emit(str(e))
            
    @Slot(dict)
    def _handle_backend_signal(self, signal: Dict[str, Any]) -> None:
        """Handle signals from backend."""
        self.signal_processor.process_signal(signal)
        
    @Slot(str, float)
    def _handle_threshold_exceeded(self, metric: str, value: float) -> None:
        """Handle metric threshold exceeded."""
        logger.warning(f"Metric threshold exceeded: {metric} = {value}")
        
    @Slot(str, bool)
    def _handle_component_status(self, component: str, status: bool) -> None:
        """Handle component status changes."""
        logger.info(f"Component status changed: {component} = {status}")
        
    @Slot(dict)
    def _handle_processed_signal(self, signal: Dict[str, Any]) -> None:
        """Handle processed signals."""
        signal_type = signal.get('type')
        if signal_type == 'neural_seed_update':
            self._handle_neural_seed_update(signal)
        elif signal_type == 'signal_system_update':
            self._handle_signal_system_update(signal)
        elif signal_type == 'spiderweb_update':
            self._handle_spiderweb_update(signal)
            
    def _handle_neural_seed_update(self, data: Dict[str, Any]) -> None:
        """Handle Neural Seed updates."""
        self.state['neural_seed'].update({
            'consciousness_level': data.get('consciousness_level', 0.0),
            'stability': data.get('stability', 0.0),
            'growth_stage': data.get('growth_stage', 'seed')
        })
        self._update_state()
        
    def _handle_signal_system_update(self, data: Dict[str, Any]) -> None:
        """Handle Signal System updates."""
        self.state['signal_system'].update({
            'message_count': data.get('message_count', 0),
            'error_rate': data.get('error_rate', 0.0)
        })
        self._update_state()
        
    def _handle_spiderweb_update(self, data: Dict[str, Any]) -> None:
        """Handle Spiderweb Bridge updates."""
        self.state['spiderweb'].update({
            'active_bridges': data.get('active_bridges', 0),
            'version_nodes': data.get('version_nodes', [])
        })
        self._update_state()
        
    def _handle_error(self, error: str) -> None:
        """Handle system errors."""
        logger.error(f"System error: {error}")
        self.error_occurred.emit(error)
        
    def _update_state(self) -> None:
        """Update and emit system state."""
        self.state_changed.emit(self.state)
        
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state."""
        return self.state.copy()
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        return self.metrics.get_current_metrics()
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.health.check_health()
        
    def get_pending_signals(self) -> List[Dict[str, Any]]:
        """Get pending signals."""
        return self.signal_processor.get_pending_signals() 