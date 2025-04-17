"""
LUMINA Version Bridge System

This module provides a spiderweb-like connection system between all versions of the Lumina system,
enabling seamless data exchange and version compatibility across the entire version spectrum.
"""

import os
import sys
import time
import logging
import threading
import json
from queue import Queue
from typing import Dict, Any, List, Callable, Optional, Union, Tuple, Set

def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with proper configuration"""
    logger = logging.getLogger(name)
    logger.propagate = False
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

class VersionBridgeSystem:
    """
    Spiderweb-like connection system for all Lumina versions (V1-V7).
    This system manages connections and data flow between any versions of the Lumina system.
    """
    
    def __init__(self, mock_mode: bool = False):
        """
        Initialize the Version Bridge System
        
        Args:
            mock_mode: Enable mock mode for testing
        """
        self.logger = setup_logger("lumina_v7.version_bridge_system")
        self.mock_mode = mock_mode
        self.running = False
        
        # Version connections
        self.connections: Dict[str, Any] = {}  # version -> system instance
        self.versions: Dict[str, str] = {}     # version -> version string
        self.compatibility_matrix: Dict[str, List[str]] = {}  # version -> compatible versions
        
        # Message handling
        self.message_handlers: Dict[str, Dict[str, Callable]] = {}  # version -> {type -> handler}
        self.event_queues: Dict[str, Queue] = {}  # version -> queue
        self.processing_threads: Dict[str, Optional[threading.Thread]] = {}  # version -> thread
        
        self.logger.info(f"Version Bridge System initialized (mock_mode={mock_mode})")
    
    def connect_version(self, version_id: str, system: Any) -> bool:
        """
        Connect a version to the bridge system
        
        Args:
            version_id: Version identifier (e.g., "v1", "v2", etc.)
            system: System instance
            
        Returns:
            bool: True if connection successful
        """
        try:
            if not hasattr(system, 'get_version'):
                self.logger.error(f"System {version_id} does not have get_version method")
                return False
            
            version = system.get_version()
            if not version:
                self.logger.error(f"Invalid version for system {version_id}")
                return False
            
            # Validate version string format (should be X.Y.Z)
            try:
                major = int(version.split('.')[0])
                if major <= 0:
                    self.logger.error(f"Invalid major version number for {version_id}: {major}")
                    return False
            except (ValueError, IndexError):
                self.logger.error(f"Invalid version string format for {version_id}: {version}")
                return False
            
            # Add mock suffix if in mock mode
            if self.mock_mode:
                version = f"{version}-mock"
            
            self.connections[version_id] = system
            self.versions[version_id] = version
            self.event_queues[version_id] = Queue()
            self.message_handlers[version_id] = {}
            
            # Update compatibility matrix
            self._update_compatibility_matrix()
            
            self.logger.info(f"Connected version {version_id} (version: {version})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect version {version_id}: {str(e)}")
            return False
    
    def _update_compatibility_matrix(self):
        """Update the compatibility matrix based on connected versions"""
        self.compatibility_matrix = {}
        for v1 in self.versions:
            compatible_versions = []
            v1_major = int(self.versions[v1].split('.')[0].split('-')[0])  # Handle mock suffix
            
            for v2 in self.versions:
                if v1 == v2:
                    continue
                
                v2_major = int(self.versions[v2].split('.')[0].split('-')[0])  # Handle mock suffix
                if abs(v1_major - v2_major) <= 1:  # Versions within 1 major version are compatible
                    compatible_versions.append(v2)
            
            self.compatibility_matrix[v1] = compatible_versions
    
    def check_compatibility(self, source_version: str, target_version: str) -> bool:
        """Check if two versions are compatible"""
        return target_version in self.compatibility_matrix.get(source_version, [])
    
    def send_data(self, source_version: str, target_version: str, data: Dict[str, Any]) -> bool:
        """
        Send data from one version to another
        
        Args:
            source_version: Source version identifier
            target_version: Target version identifier
            data: Data to send
            
        Returns:
            bool: True if send successful
        """
        try:
            if source_version not in self.connections:
                self.logger.error(f"Source version {source_version} not connected")
                return False
            
            if target_version not in self.connections:
                self.logger.error(f"Target version {target_version} not connected")
                return False
            
            if not self.check_compatibility(source_version, target_version):
                self.logger.error(f"Versions {source_version} and {target_version} are not compatible")
                return False
            
            # Add metadata to data
            data_with_metadata = {
                **data,
                "source_version": self.versions[source_version],
                "target_version": self.versions[target_version],
                "timestamp": time.time()
            }
            
            # Check for message handlers
            if data.get("type") in self.message_handlers.get(target_version, {}):
                handler = self.message_handlers[target_version][data["type"]]
                try:
                    handler(data_with_metadata)
                except Exception as e:
                    self.logger.error(f"Error executing message handler: {str(e)}")
            
            # Send data to target version
            target_system = self.connections[target_version]
            if hasattr(target_system, 'receive_data'):
                target_system.receive_data(data_with_metadata)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending data: {str(e)}")
            return False
    
    def register_message_handler(self, version_id: str, message_type: str, handler: Callable):
        """Register a message handler for a specific version and message type"""
        if version_id not in self.message_handlers:
            self.message_handlers[version_id] = {}
        self.message_handlers[version_id][message_type] = handler
    
    def _process_events(self, version_id: str):
        """Process events for a specific version"""
        while self.running:
            try:
                event = self.event_queues[version_id].get(timeout=0.1)
                if event:
                    self.send_data(
                        event.get("source_version"),
                        version_id,
                        event.get("data", {})
                    )
            except Exception as e:
                self.logger.error(f"Error processing events for {version_id}: {str(e)}")
                time.sleep(0.1)
    
    def start(self) -> bool:
        """Start the version bridge system"""
        if self.running:
            return True
        
        try:
            self.running = True
            for version_id in self.connections:
                thread = threading.Thread(
                    target=self._process_events,
                    args=(version_id,),
                    daemon=True
                )
                self.processing_threads[version_id] = thread
                thread.start()
            
            self.logger.info("Version Bridge System started")
            return True
        except Exception as e:
            self.logger.error(f"Error starting version bridge system: {str(e)}")
            return False
    
    def stop(self):
        """Stop the version bridge system"""
        if not self.running:
            return
        
        self.running = False
        for version_id in self.processing_threads:
            if self.processing_threads[version_id]:
                self.processing_threads[version_id].join(timeout=1.0)
                self.processing_threads[version_id] = None
        
        self.logger.info("Version Bridge System stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "running": self.running,
            "connected_versions": list(self.versions.keys()),
            "versions": self.versions,
            "compatibility_matrix": self.compatibility_matrix,
            "message_handlers": {k: list(v.keys()) for k, v in self.message_handlers.items()},
            "queue_sizes": {k: v.qsize() for k, v in self.event_queues.items()},
            "processing_threads": {
                version: thread.is_alive() if thread else False
                for version, thread in self.processing_threads.items()
            }
        }
    
    def broadcast(self, source_version: str, data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Broadcast data to all compatible versions
        
        Args:
            source_version: Source version identifier
            data: Data to broadcast
            
        Returns:
            Dict[str, bool]: Results for each target version
        """
        results = {}
        for target_version in self.compatibility_matrix.get(source_version, []):
            results[target_version] = self.send_data(source_version, target_version, data)
        return results 