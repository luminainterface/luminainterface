import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import sys
import os
import json
from enum import Enum

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

class BackendConnector:
    """Connector for interfacing with the backend system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connection_state = ConnectionState.DISCONNECTED
        self.backend = None
        self.metrics = {}
        self.gate_states = {}
        self.health_status = {}
        self.last_update = datetime.now()
        self.monitoring_task = None
        self.connection_retry_count = 0
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds
        self.state_change_callbacks: List[Callable] = []
        self.metrics_change_callbacks: List[Callable] = []
        self.gate_change_callbacks: List[Callable] = []
        self.health_change_callbacks: List[Callable] = []
        self.connection_state_callbacks: List[Callable] = []
        self._monitoring_active = False
        self._last_metrics = {}
        self._last_gate_states = {}
        self._last_health_status = {}
        
    async def connect(self, backend_path: str) -> bool:
        """Connect to the backend system."""
        try:
            if self.connection_state == ConnectionState.CONNECTED:
                self.logger.warning("Already connected to backend")
                return True
                
            self.connection_state = ConnectionState.CONNECTING
            self._notify_connection_state_change()
            
            # Add backend directory to Python path
            backend_dir = os.path.dirname(backend_path)
            if backend_dir not in sys.path:
                sys.path.append(backend_dir)
                
            # Import backend module
            try:
                from src.integration.backend import BackendSystem
            except ImportError as e:
                self.logger.error(f"Failed to import backend module: {e}")
                self.connection_state = ConnectionState.ERROR
                self._notify_connection_state_change()
                return False
                
            # Initialize backend with retry logic
            while self.connection_retry_count < self.max_retries:
                try:
                    self.backend = BackendSystem()
                    await self.backend.start()
                    self.connection_state = ConnectionState.CONNECTED
                    self._notify_connection_state_change()
                    self.logger.info("Connected to backend system")
                    
                    # Start monitoring loop
                    self._monitoring_active = True
                    self.monitoring_task = asyncio.create_task(self._monitoring_loop())
                    return True
                except Exception as e:
                    self.connection_retry_count += 1
                    self.logger.warning(f"Connection attempt {self.connection_retry_count} failed: {e}")
                    if self.connection_retry_count < self.max_retries:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        self.logger.error("Max connection retries reached")
                        self.connection_state = ConnectionState.ERROR
                        self._notify_connection_state_change()
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to connect to backend: {e}")
            self.connection_state = ConnectionState.ERROR
            self._notify_connection_state_change()
            return False
            
    def register_connection_state_callback(self, callback: Callable):
        """Register a callback for connection state changes."""
        self.connection_state_callbacks.append(callback)
        
    def _notify_connection_state_change(self):
        """Notify all registered callbacks of connection state change."""
        for callback in self.connection_state_callbacks:
            callback(self.connection_state)
            
    async def disconnect(self):
        """Disconnect from the backend system."""
        try:
            if self.connection_state != ConnectionState.CONNECTED:
                return
                
            self._monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
                    
            if self.backend:
                await self.backend.stop()
                
            self.connection_state = ConnectionState.DISCONNECTED
            self._notify_connection_state_change()
            self.logger.info("Disconnected from backend system")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from backend: {e}")
            self.connection_state = ConnectionState.ERROR
            self._notify_connection_state_change()
            
    async def _monitoring_loop(self) -> None:
        """Monitor backend system state."""
        while self._monitoring_active:
            try:
                # Get system metrics
                if hasattr(self.backend, 'metrics_collector'):
                    new_metrics = self.backend.metrics_collector.collect_system_metrics()
                    if new_metrics != self._last_metrics:
                        self.metrics = new_metrics
                        self._last_metrics = new_metrics.copy()
                        for callback in self.metrics_change_callbacks:
                            callback(new_metrics)
                    
                # Get gate states
                if hasattr(self.backend, 'ping_system'):
                    new_gate_states = {
                        str(gate_id): {
                            'output': gate.output,
                            'state': 'OPEN' if gate.output > 0.8 else 'CLOSED',
                            'connections': len(gate.connections),
                            'visual_effects': getattr(gate, 'get_state', lambda: {})().get('visual_effects', [])
                        }
                        for gate_id, gate in self.backend.ping_system.logic_gates.items()
                    }
                    if new_gate_states != self._last_gate_states:
                        self.gate_states = new_gate_states
                        self._last_gate_states = new_gate_states.copy()
                        for callback in self.gate_change_callbacks:
                            callback(new_gate_states)
                    
                # Get health status
                if hasattr(self.backend, 'health_monitor'):
                    new_health_status = self.backend.health_monitor.check_health()
                    if new_health_status != self._last_health_status:
                        self.health_status = new_health_status
                        self._last_health_status = new_health_status.copy()
                        for callback in self.health_change_callbacks:
                            callback(new_health_status)
                    
                self.last_update = datetime.now()
                
                # Notify state change
                current_state = {
                    'metrics': self.metrics,
                    'gate_states': self.gate_states,
                    'health_status': self.health_status
                }
                for callback in self.state_change_callbacks:
                    callback(current_state)
                    
                await asyncio.sleep(0.1)  # Update every 100ms
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)
                
    def get_connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self.connection_state
        
    def is_connected(self) -> bool:
        """Check if connected to backend."""
        return self.connection_state == ConnectionState.CONNECTED
        
    def get_backend_info(self) -> Dict[str, Any]:
        """Get comprehensive backend information."""
        return {
            'connection_state': self.connection_state.value,
            'last_update': self.last_update.isoformat(),
            'metrics': self.metrics,
            'gate_states': self.gate_states,
            'health_status': self.health_status
        } 