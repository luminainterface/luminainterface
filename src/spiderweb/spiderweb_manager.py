import logging
import threading
import time
import queue
from enum import Enum
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
import sqlite3
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class VersionState(Enum):
    """Possible states for a version node in the spiderweb"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTDOWN = "shutdown"
    ERROR = "error"

@dataclass
class VersionNode:
    """Represents a version node in the spiderweb"""
    version: str
    system: Any  # The system object (e.g., CentralNode instance)
    state: VersionState
    message_handlers: Dict[str, Callable]
    event_queue: queue.Queue
    processing_thread: Optional[threading.Thread]
    lock: threading.Lock

class SpiderwebManager:
    """Manager for the spiderweb architecture connecting different versions"""
    
    def __init__(self):
        """Initialize the spiderweb manager"""
        self.versions = {}  # Dict of version ID to VersionNode
        self.compatibility_matrix = {}  # Dict of version ID to list of compatible versions
        self.lock = threading.Lock()
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        self.db_path = Path("node_zero.db")
        self.quantum_manager = None
        self.cosmic_manager = None
        self._running = False
        self._sync_thread = None
        self._metrics_thread = None
        self._version_bridges = {}
        self._nodes = {}
        self._connections = {}
        
    def initialize(self) -> bool:
        """Initialize the Spiderweb system"""
        try:
            # Initialize database
            self._init_database()
            
            # Initialize version bridges
            self._init_version_bridges()
            
            # Start background threads
            self._start_background_threads()
            
            self.logger.info("Spiderweb system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Spiderweb system: {e}")
            return False
            
    def _init_database(self):
        """Initialize the database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables from schema
            from v1.database.schema import SCHEMA
            for table_name, create_sql in SCHEMA.items():
                cursor.execute(create_sql)
                
            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
            
    def _init_version_bridges(self):
        """Initialize version bridges"""
        try:
            # Initialize V11 (Quantum) bridge
            self._version_bridges['v11'] = {
                'type': 'quantum',
                'status': 'disconnected',
                'metrics': {}
            }
            
            # Initialize V12 (Cosmic) bridge
            self._version_bridges['v12'] = {
                'type': 'cosmic',
                'status': 'disconnected',
                'metrics': {}
            }
            
            self.logger.info("Version bridges initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize version bridges: {e}")
            raise
            
    def _start_background_threads(self):
        """Start background synchronization and metrics threads"""
        try:
            self._running = True
            
            # Start synchronization thread
            self._sync_thread = threading.Thread(target=self._sync_loop)
            self._sync_thread.daemon = True
            self._sync_thread.start()
            
            # Start metrics thread
            self._metrics_thread = threading.Thread(target=self._metrics_loop)
            self._metrics_thread.daemon = True
            self._metrics_thread.start()
            
            self.logger.info("Background threads started")
            
        except Exception as e:
            self.logger.error(f"Failed to start background threads: {e}")
            raise
            
    def _sync_loop(self):
        """Background synchronization loop"""
        while self._running:
            try:
                # Sync quantum components
                if self.quantum_manager:
                    self._sync_quantum()
                    
                # Sync cosmic components
                if self.cosmic_manager:
                    self._sync_cosmic()
                    
                time.sleep(1)  # Adjust sync frequency as needed
                
            except Exception as e:
                self.logger.error(f"Error in sync loop: {e}")
                time.sleep(5)  # Longer delay on error
                
    def _metrics_loop(self):
        """Background metrics collection loop"""
        while self._running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect quantum metrics
                if self.quantum_manager:
                    self._collect_quantum_metrics()
                    
                # Collect cosmic metrics
                if self.cosmic_manager:
                    self._collect_cosmic_metrics()
                    
                time.sleep(0.5)  # Adjust metrics collection frequency
                
            except Exception as e:
                self.logger.error(f"Error in metrics loop: {e}")
                time.sleep(5)  # Longer delay on error
                
    def _sync_quantum(self):
        """Synchronize quantum components"""
        try:
            # Get quantum state
            quantum_state = self.quantum_manager.get_state()
            
            # Update version bridge metrics
            self._version_bridges['v11']['metrics'].update({
                'field_strength': quantum_state.get('field_strength', 0),
                'entangled_nodes': quantum_state.get('entangled_nodes', []),
                'phase': quantum_state.get('phase', 0),
                'frequency': quantum_state.get('frequency', 0)
            })
            
            # Log sync event
            self._log_sync_event('quantum', 'success')
            
        except Exception as e:
            self.logger.error(f"Error syncing quantum components: {e}")
            self._log_sync_event('quantum', 'error', str(e))
            
    def _sync_cosmic(self):
        """Synchronize cosmic components"""
        try:
            # Get cosmic state
            cosmic_state = self.cosmic_manager.get_state()
            
            # Update version bridge metrics
            self._version_bridges['v12']['metrics'].update({
                'field_strength': cosmic_state.get('field_strength', 0),
                'dimensional_resonance': cosmic_state.get('resonance', 0),
                'universal_phase': cosmic_state.get('phase', 0),
                'cosmic_frequency': cosmic_state.get('frequency', 0)
            })
            
            # Log sync event
            self._log_sync_event('cosmic', 'success')
            
        except Exception as e:
            self.logger.error(f"Error syncing cosmic components: {e}")
            self._log_sync_event('cosmic', 'error', str(e))
            
    def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get node count
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            
            # Get active connections
            cursor.execute("SELECT COUNT(*) FROM connections WHERE status = 'active'")
            active_connections = cursor.fetchone()[0]
            
            # Get version bridge status
            version_count = len(self._version_bridges)
            active_versions = sum(1 for v in self._version_bridges.values() if v['status'] == 'connected')
            
            # Store metrics
            metrics = {
                'node_count': node_count,
                'active_connections': active_connections,
                'version_count': version_count,
                'active_versions': active_versions,
                'timestamp': datetime.now().isoformat()
            }
            
            cursor.execute("""
                INSERT INTO metrics (metric_type, value, metadata)
                VALUES (?, ?, ?)
            """, ('system', 1.0, json.dumps(metrics)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            
    def _collect_quantum_metrics(self):
        """Collect quantum-specific metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            metrics = self._version_bridges['v11']['metrics']
            
            cursor.execute("""
                INSERT INTO metrics (metric_type, value, metadata)
                VALUES (?, ?, ?)
            """, ('quantum', 1.0, json.dumps(metrics)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error collecting quantum metrics: {e}")
            
    def _collect_cosmic_metrics(self):
        """Collect cosmic-specific metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            metrics = self._version_bridges['v12']['metrics']
            
            cursor.execute("""
                INSERT INTO metrics (metric_type, value, metadata)
                VALUES (?, ?, ?)
            """, ('cosmic', 1.0, json.dumps(metrics)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error collecting cosmic metrics: {e}")
            
    def _log_sync_event(self, event_type: str, status: str, error_message: str = None):
        """Log synchronization events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sync_events (event_type, status, source_version, target_version, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (event_type, status, 'v11' if event_type == 'quantum' else 'v12', 
                 'v12' if event_type == 'cosmic' else 'v11', error_message))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging sync event: {e}")
            
    def connect_version(self, version: str, config: Dict[str, Any]) -> bool:
        """Connect a version bridge"""
        try:
            if version not in self._version_bridges:
                raise ValueError(f"Unsupported version: {version}")
                
            # Update version bridge status
            self._version_bridges[version]['status'] = 'connected'
            self._version_bridges[version]['config'] = config
            
            # Initialize appropriate manager
            if version == 'v11':
                self.quantum_manager = QuantumManager(config)
            elif version == 'v12':
                self.cosmic_manager = CosmicManager(config)
                
            self.logger.info(f"Version bridge {version} connected successfully")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect version bridge {version}: {e}")
            return False
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            metrics = {
                'system': {
                    'node_count': len(self._nodes),
                    'active_connections': len([c for c in self._connections.values() if c['status'] == 'active']),
                    'version_count': len(self._version_bridges),
                    'active_versions': sum(1 for v in self._version_bridges.values() if v['status'] == 'connected')
                }
            }
            
            # Add quantum metrics if available
            if 'v11' in self._version_bridges:
                metrics['quantum'] = self._version_bridges['v11']['metrics']
                
            # Add cosmic metrics if available
            if 'v12' in self._version_bridges:
                metrics['cosmic'] = self._version_bridges['v12']['metrics']
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {}
            
    def cleanup(self):
        """Clean up resources"""
        try:
            self._running = False
            
            if self._sync_thread:
                self._sync_thread.join()
                
            if self._metrics_thread:
                self._metrics_thread.join()
                
            self.logger.info("Spiderweb system cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def _build_compatibility_matrix(self):
        """Build the compatibility matrix for connected versions"""
        # This is a simplified version - in a real implementation, this would
        # consider semantic versioning rules, version distances, etc.
        version_list = sorted(self.versions.keys())
        self.compatibility_matrix = {}
        
        for i, v1 in enumerate(version_list):
            compatible = []
            
            # Consider versions nearby as compatible
            for j, v2 in enumerate(version_list):
                if i == j:
                    # Skip self
                    continue
                    
                # Simple rule: versions within 2 steps are compatible
                # This is just an example - real implementation would be more sophisticated
                if abs(i - j) <= 2:
                    compatible.append(v2)
                    
            self.compatibility_matrix[v1] = compatible
            self.logger.debug(f"Version {v1} is compatible with: {compatible}")
            
    def register_message_handler(self, version: str, message_type: str, handler: Callable):
        """Register a message handler for a specific version"""
        try:
            with self.lock:
                if version not in self.versions:
                    self.logger.error(f"Cannot register handler for unknown version {version}")
                    return False
                    
                self.versions[version].message_handlers[message_type] = handler
                self.logger.debug(f"Registered handler for {message_type} messages on version {version}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to register message handler: {str(e)}")
            return False
            
    def send_data(self, source_version: str, target_version: str, data: dict) -> bool:
        """Send data from one version to another, checking compatibility"""
        try:
            with self.lock:
                # Check if versions exist
                if source_version not in self.versions:
                    self.logger.error(f"Source version {source_version} not found")
                    return False
                    
                if target_version not in self.versions:
                    self.logger.error(f"Target version {target_version} not found")
                    return False
                    
                # Check compatibility
                if target_version not in self.compatibility_matrix.get(source_version, []):
                    self.logger.warning(f"Version {source_version} is not compatible with {target_version}")
                    return False
                    
                # Add to target's queue
                target = self.versions[target_version]
                with target.lock:
                    target.event_queue.put({
                        'source': source_version,
                        'type': data.get('type', 'unknown'),
                        'data': data,
                        'timestamp': time.time()
                    })
                    
                self.logger.debug(f"Sent data from {source_version} to {target_version}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to send data: {str(e)}")
            return False
            
    def broadcast(self, source_version: str, data: dict) -> bool:
        """Broadcast data to all compatible versions"""
        try:
            with self.lock:
                if source_version not in self.versions:
                    self.logger.error(f"Source version {source_version} not found")
                    return False
                    
                compatible_versions = self.compatibility_matrix.get(source_version, [])
                success = True
                
                for target_version in compatible_versions:
                    success = success and self.send_data(source_version, target_version, data)
                    
                return success
                
        except Exception as e:
            self.logger.error(f"Failed to broadcast data: {str(e)}")
            return False
            
    def _process_events(self, version: str):
        """Process events for a specific version"""
        try:
            self.logger.info(f"Starting event processing for version {version}")
            node = self.versions[version]
            
            while self.is_running and node.state == VersionState.RUNNING:
                try:
                    # Get event with timeout
                    try:
                        event = node.event_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                        
                    # Process event
                    event_type = event.get('type', 'unknown')
                    handler = node.message_handlers.get(event_type)
                    
                    if handler:
                        try:
                            with node.lock:
                                handler(event['data'])
                        except Exception as e:
                            self.logger.error(f"Error in handler for {event_type} on version {version}: {str(e)}")
                    else:
                        self.logger.warning(f"No handler for {event_type} on version {version}")
                        
                    # Mark as done
                    node.event_queue.task_done()
                    
                except Exception as e:
                    self.logger.error(f"Error processing events for version {version}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Fatal error in processing thread for version {version}: {str(e)}")
            
            # Update version state
            with self.lock:
                if version in self.versions:
                    self.versions[version].state = VersionState.ERROR
                    
    def shutdown(self):
        """Shutdown the spiderweb system"""
        try:
            with self.lock:
                self.is_running = False
                
                # Stop all processing threads
                for version, node in self.versions.items():
                    node.state = VersionState.SHUTDOWN
                    
                # Wait for threads to finish
                for version, node in self.versions.items():
                    if node.processing_thread and node.processing_thread.is_alive():
                        node.processing_thread.join(timeout=2.0)
                        
                self.logger.info("Spiderweb system shut down")
                return True
                
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            return False
            
    def get_version_status(self, version: str) -> dict:
        """Get the status of a specific version"""
        try:
            with self.lock:
                if version not in self.versions:
                    return {'error': 'Version not found'}
                    
                node = self.versions[version]
                return {
                    'state': node.state.value,
                    'queue_size': node.event_queue.qsize(),
                    'handlers': list(node.message_handlers.keys())
                }
                
        except Exception as e:
            self.logger.error(f"Error getting version status: {str(e)}")
            return {'error': str(e)}
            
    def get_compatibility_matrix(self) -> dict:
        """Get a copy of the compatibility matrix"""
        with self.lock:
            return self.compatibility_matrix.copy() 