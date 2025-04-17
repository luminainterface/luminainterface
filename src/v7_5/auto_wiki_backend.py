#!/usr/bin/env python3
"""
AutoWiki Backend Service for LUMINA v7.5
Handles integration between AutoWikiProcessor and monitoring system
"""

import logging
import threading
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from queue import Queue
from PySide6.QtCore import QObject, Signal, Slot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("logs") / f"auto_wiki_backend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("AutoWikiBackend")

class AutoWikiBackend(QObject):
    """Backend service for AutoWiki system integration"""
    
    # Signals for system integration
    stateChanged = Signal(dict)
    metricsUpdated = Signal(dict)
    systemAlert = Signal(str, str)  # severity, message
    
    def __init__(self):
        super().__init__()
        
        # State management
        self._system_state = {
            'status': 'initializing',
            'health': 1.0,
            'last_update': None,
            'components': {
                'processor': False,
                'monitor': False,
                'integration': False
            }
        }
        
        # Metrics collection
        self._metrics_queue = Queue()
        self._metrics_history = {
            'response_times': [],
            'success_rates': [],
            'error_rates': [],
            'queue_sizes': []
        }
        
        # Threading
        self._metrics_thread = None
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Component references
        self.processor = None
        self.monitor = None
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the backend system"""
        try:
            # Create required directories
            Path("logs").mkdir(exist_ok=True)
            Path("data/metrics").mkdir(parents=True, exist_ok=True)
            
            # Start monitoring threads
            self._start_monitoring()
            
            # Update system state
            self._update_system_state('initialized', 1.0)
            logger.info("AutoWiki backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            self._update_system_state('error', 0.0)
            self.systemAlert.emit('error', f"Backend initialization failed: {str(e)}")
    
    def _start_monitoring(self):
        """Start monitoring threads"""
        # Metrics collection thread
        self._metrics_thread = threading.Thread(
            target=self._metrics_collection_loop,
            daemon=True
        )
        self._metrics_thread.start()
        
        # System monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._system_monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Monitoring threads started")
    
    def _metrics_collection_loop(self):
        """Collect and process metrics"""
        while not self._stop_event.is_set():
            try:
                # Process metrics from queue
                while not self._metrics_queue.empty():
                    metrics = self._metrics_queue.get_nowait()
                    self._process_metrics(metrics)
                
                # Emit updated metrics
                self._emit_metrics()
                
                time.sleep(1)  # 1 second interval
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                self.systemAlert.emit('warning', f"Metrics collection error: {str(e)}")
    
    def _system_monitoring_loop(self):
        """Monitor overall system health"""
        while not self._stop_event.is_set():
            try:
                self._check_system_health()
                time.sleep(5)  # 5 second interval
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                self.systemAlert.emit('warning', f"System monitoring error: {str(e)}")
    
    def _process_metrics(self, metrics: Dict[str, Any]):
        """Process and store metrics"""
        try:
            # Update metrics history
            if 'response_time' in metrics:
                self._metrics_history['response_times'].append(metrics['response_time'])
                if len(self._metrics_history['response_times']) > 1000:
                    self._metrics_history['response_times'] = self._metrics_history['response_times'][-1000:]
            
            if 'success_rate' in metrics:
                self._metrics_history['success_rates'].append(metrics['success_rate'])
                if len(self._metrics_history['success_rates']) > 1000:
                    self._metrics_history['success_rates'] = self._metrics_history['success_rates'][-1000:]
            
            if 'error_rate' in metrics:
                self._metrics_history['error_rates'].append(metrics['error_rate'])
                if len(self._metrics_history['error_rates']) > 1000:
                    self._metrics_history['error_rates'] = self._metrics_history['error_rates'][-1000:]
            
            if 'queue_size' in metrics:
                self._metrics_history['queue_sizes'].append(metrics['queue_size'])
                if len(self._metrics_history['queue_sizes']) > 1000:
                    self._metrics_history['queue_sizes'] = self._metrics_history['queue_sizes'][-1000:]
            
        except Exception as e:
            logger.error(f"Error processing metrics: {e}")
            self.systemAlert.emit('warning', f"Metrics processing error: {str(e)}")
    
    def _emit_metrics(self):
        """Emit current metrics"""
        try:
            metrics = {
                'response_times': self._metrics_history['response_times'][-100:],
                'success_rates': self._metrics_history['success_rates'][-100:],
                'error_rates': self._metrics_history['error_rates'][-100:],
                'queue_sizes': self._metrics_history['queue_sizes'][-100:],
                'timestamp': datetime.now().isoformat()
            }
            
            self.metricsUpdated.emit(metrics)
            
        except Exception as e:
            logger.error(f"Error emitting metrics: {e}")
    
    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check component status
            components_status = {
                'processor': self.processor is not None and hasattr(self.processor, '_processing'),
                'monitor': self.monitor is not None,
                'integration': all(self._system_state['components'].values())
            }
            
            # Update component status
            self._system_state['components'].update(components_status)
            
            # Calculate health score
            if all(components_status.values()):
                health = 1.0
                status = 'active'
            elif any(components_status.values()):
                health = 0.5
                status = 'degraded'
            else:
                health = 0.0
                status = 'error'
            
            self._update_system_state(status, health)
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            self._update_system_state('error', 0.0)
            self.systemAlert.emit('error', f"Health check failed: {str(e)}")
    
    def _update_system_state(self, status: str, health: float):
        """Update system state and emit changes"""
        try:
            old_status = self._system_state['status']
            old_health = self._system_state['health']
            
            self._system_state.update({
                'status': status,
                'health': health,
                'last_update': datetime.now().isoformat()
            })
            
            # Emit state change if significant
            if old_status != status or abs(old_health - health) >= 0.1:
                self.stateChanged.emit(self._system_state)
                
                # Emit alerts for significant changes
                if old_status != status:
                    severity = 'info' if status == 'active' else 'warning' if status == 'degraded' else 'error'
                    self.systemAlert.emit(severity, f"System status changed to {status}")
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")
    
    @Slot(dict)
    def process_metrics(self, metrics: Dict[str, Any]):
        """Process metrics from processor"""
        self._metrics_queue.put(metrics)
    
    @Slot(str, str)
    def handle_alert(self, severity: str, message: str):
        """Handle system alerts"""
        logger.log(
            logging.ERROR if severity == 'error' else
            logging.WARNING if severity == 'warning' else
            logging.INFO,
            f"System alert ({severity}): {message}"
        )
        self.systemAlert.emit(severity, message)
    
    def register_processor(self, processor):
        """Register AutoWikiProcessor instance"""
        self.processor = processor
        self._system_state['components']['processor'] = True
        processor.metricsUpdated.connect(self.process_metrics)
        processor.errorOccurred.connect(lambda msg: self.handle_alert('error', msg))
        logger.info("Registered AutoWikiProcessor")
    
    def register_monitor(self, monitor):
        """Register AutoWikiMonitor instance"""
        self.monitor = monitor
        self._system_state['components']['monitor'] = True
        self.stateChanged.connect(monitor.update_status)
        self.metricsUpdated.connect(monitor.update_metrics)
        self.systemAlert.connect(lambda sev, msg: monitor.show_error(msg))
        logger.info("Registered AutoWikiMonitor")
    
    def shutdown(self):
        """Shutdown the backend system"""
        try:
            logger.info("Shutting down AutoWiki backend...")
            self._stop_event.set()
            
            if self._metrics_thread and self._metrics_thread.is_alive():
                self._metrics_thread.join(timeout=5)
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5)
            
            self._update_system_state('shutdown', 0.0)
            logger.info("AutoWiki backend shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.systemAlert.emit('error', f"Shutdown error: {str(e)}")
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        return self._system_state.copy()
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            'history': self._metrics_history.copy(),
            'timestamp': datetime.now().isoformat()
        } 
 
 
"""
AutoWiki Backend Service for LUMINA v7.5
Handles integration between AutoWikiProcessor and monitoring system
"""

import logging
import threading
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from queue import Queue
from PySide6.QtCore import QObject, Signal, Slot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path("logs") / f"auto_wiki_backend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("AutoWikiBackend")

class AutoWikiBackend(QObject):
    """Backend service for AutoWiki system integration"""
    
    # Signals for system integration
    stateChanged = Signal(dict)
    metricsUpdated = Signal(dict)
    systemAlert = Signal(str, str)  # severity, message
    
    def __init__(self):
        super().__init__()
        
        # State management
        self._system_state = {
            'status': 'initializing',
            'health': 1.0,
            'last_update': None,
            'components': {
                'processor': False,
                'monitor': False,
                'integration': False
            }
        }
        
        # Metrics collection
        self._metrics_queue = Queue()
        self._metrics_history = {
            'response_times': [],
            'success_rates': [],
            'error_rates': [],
            'queue_sizes': []
        }
        
        # Threading
        self._metrics_thread = None
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Component references
        self.processor = None
        self.monitor = None
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the backend system"""
        try:
            # Create required directories
            Path("logs").mkdir(exist_ok=True)
            Path("data/metrics").mkdir(parents=True, exist_ok=True)
            
            # Start monitoring threads
            self._start_monitoring()
            
            # Update system state
            self._update_system_state('initialized', 1.0)
            logger.info("AutoWiki backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            self._update_system_state('error', 0.0)
            self.systemAlert.emit('error', f"Backend initialization failed: {str(e)}")
    
    def _start_monitoring(self):
        """Start monitoring threads"""
        # Metrics collection thread
        self._metrics_thread = threading.Thread(
            target=self._metrics_collection_loop,
            daemon=True
        )
        self._metrics_thread.start()
        
        # System monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._system_monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Monitoring threads started")
    
    def _metrics_collection_loop(self):
        """Collect and process metrics"""
        while not self._stop_event.is_set():
            try:
                # Process metrics from queue
                while not self._metrics_queue.empty():
                    metrics = self._metrics_queue.get_nowait()
                    self._process_metrics(metrics)
                
                # Emit updated metrics
                self._emit_metrics()
                
                time.sleep(1)  # 1 second interval
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                self.systemAlert.emit('warning', f"Metrics collection error: {str(e)}")
    
    def _system_monitoring_loop(self):
        """Monitor overall system health"""
        while not self._stop_event.is_set():
            try:
                self._check_system_health()
                time.sleep(5)  # 5 second interval
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                self.systemAlert.emit('warning', f"System monitoring error: {str(e)}")
    
    def _process_metrics(self, metrics: Dict[str, Any]):
        """Process and store metrics"""
        try:
            # Update metrics history
            if 'response_time' in metrics:
                self._metrics_history['response_times'].append(metrics['response_time'])
                if len(self._metrics_history['response_times']) > 1000:
                    self._metrics_history['response_times'] = self._metrics_history['response_times'][-1000:]
            
            if 'success_rate' in metrics:
                self._metrics_history['success_rates'].append(metrics['success_rate'])
                if len(self._metrics_history['success_rates']) > 1000:
                    self._metrics_history['success_rates'] = self._metrics_history['success_rates'][-1000:]
            
            if 'error_rate' in metrics:
                self._metrics_history['error_rates'].append(metrics['error_rate'])
                if len(self._metrics_history['error_rates']) > 1000:
                    self._metrics_history['error_rates'] = self._metrics_history['error_rates'][-1000:]
            
            if 'queue_size' in metrics:
                self._metrics_history['queue_sizes'].append(metrics['queue_size'])
                if len(self._metrics_history['queue_sizes']) > 1000:
                    self._metrics_history['queue_sizes'] = self._metrics_history['queue_sizes'][-1000:]
            
        except Exception as e:
            logger.error(f"Error processing metrics: {e}")
            self.systemAlert.emit('warning', f"Metrics processing error: {str(e)}")
    
    def _emit_metrics(self):
        """Emit current metrics"""
        try:
            metrics = {
                'response_times': self._metrics_history['response_times'][-100:],
                'success_rates': self._metrics_history['success_rates'][-100:],
                'error_rates': self._metrics_history['error_rates'][-100:],
                'queue_sizes': self._metrics_history['queue_sizes'][-100:],
                'timestamp': datetime.now().isoformat()
            }
            
            self.metricsUpdated.emit(metrics)
            
        except Exception as e:
            logger.error(f"Error emitting metrics: {e}")
    
    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check component status
            components_status = {
                'processor': self.processor is not None and hasattr(self.processor, '_processing'),
                'monitor': self.monitor is not None,
                'integration': all(self._system_state['components'].values())
            }
            
            # Update component status
            self._system_state['components'].update(components_status)
            
            # Calculate health score
            if all(components_status.values()):
                health = 1.0
                status = 'active'
            elif any(components_status.values()):
                health = 0.5
                status = 'degraded'
            else:
                health = 0.0
                status = 'error'
            
            self._update_system_state(status, health)
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            self._update_system_state('error', 0.0)
            self.systemAlert.emit('error', f"Health check failed: {str(e)}")
    
    def _update_system_state(self, status: str, health: float):
        """Update system state and emit changes"""
        try:
            old_status = self._system_state['status']
            old_health = self._system_state['health']
            
            self._system_state.update({
                'status': status,
                'health': health,
                'last_update': datetime.now().isoformat()
            })
            
            # Emit state change if significant
            if old_status != status or abs(old_health - health) >= 0.1:
                self.stateChanged.emit(self._system_state)
                
                # Emit alerts for significant changes
                if old_status != status:
                    severity = 'info' if status == 'active' else 'warning' if status == 'degraded' else 'error'
                    self.systemAlert.emit(severity, f"System status changed to {status}")
            
        except Exception as e:
            logger.error(f"Error updating system state: {e}")
    
    @Slot(dict)
    def process_metrics(self, metrics: Dict[str, Any]):
        """Process metrics from processor"""
        self._metrics_queue.put(metrics)
    
    @Slot(str, str)
    def handle_alert(self, severity: str, message: str):
        """Handle system alerts"""
        logger.log(
            logging.ERROR if severity == 'error' else
            logging.WARNING if severity == 'warning' else
            logging.INFO,
            f"System alert ({severity}): {message}"
        )
        self.systemAlert.emit(severity, message)
    
    def register_processor(self, processor):
        """Register AutoWikiProcessor instance"""
        self.processor = processor
        self._system_state['components']['processor'] = True
        processor.metricsUpdated.connect(self.process_metrics)
        processor.errorOccurred.connect(lambda msg: self.handle_alert('error', msg))
        logger.info("Registered AutoWikiProcessor")
    
    def register_monitor(self, monitor):
        """Register AutoWikiMonitor instance"""
        self.monitor = monitor
        self._system_state['components']['monitor'] = True
        self.stateChanged.connect(monitor.update_status)
        self.metricsUpdated.connect(monitor.update_metrics)
        self.systemAlert.connect(lambda sev, msg: monitor.show_error(msg))
        logger.info("Registered AutoWikiMonitor")
    
    def shutdown(self):
        """Shutdown the backend system"""
        try:
            logger.info("Shutting down AutoWiki backend...")
            self._stop_event.set()
            
            if self._metrics_thread and self._metrics_thread.is_alive():
                self._metrics_thread.join(timeout=5)
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5)
            
            self._update_system_state('shutdown', 0.0)
            logger.info("AutoWiki backend shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.systemAlert.emit('error', f"Shutdown error: {str(e)}")
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        return self._system_state.copy()
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            'history': self._metrics_history.copy(),
            'timestamp': datetime.now().isoformat()
        } 
 