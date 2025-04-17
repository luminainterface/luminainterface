"""
Spiderweb Bridge System for Version 2
This module implements the spiderweb architecture for connecting different components
and versions of the system, enabling seamless communication and data flow.
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from threading import Thread, Lock
import time
from pathlib import Path
import importlib.util
import json

logger = logging.getLogger(__name__)

class VersionType(Enum):
    V1 = 'v1'
    V2 = 'v2'
    V3 = 'v3'
    V4 = 'v4'
    V5 = 'v5'
    V6 = 'v6'
    V7 = 'v7'

@dataclass
class VersionInfo:
    """Information about a version node."""
    version: VersionType
    system: Any
    message_handlers: Dict[str, Callable]
    event_queue: Queue
    processing_thread: Optional[Thread]
    is_running: bool = False

class SpiderwebBridge:
    def __init__(self):
        """Initialize the spiderweb bridge system."""
        self.versions: Dict[VersionType, VersionInfo] = {}
        self.compatibility_matrix: Dict[VersionType, Set[VersionType]] = {}
        self._init_compatibility_matrix()
        self.lock = Lock()
        
        logger.info("Initialized SpiderwebBridge")
    
    def _init_compatibility_matrix(self) -> None:
        """Initialize the compatibility matrix."""
        for version in VersionType:
            compatible_versions = set()
            version_num = int(version.value[1:])  # Extract number from 'vX'
            
            # Add versions within 2 major versions
            for v in VersionType:
                v_num = int(v.value[1:])
                if abs(v_num - version_num) <= 2:
                    compatible_versions.add(v)
            
            self.compatibility_matrix[version] = compatible_versions
    
    def connect_version(self, version: VersionType, system: Any) -> bool:
        """
        Connect a version to the bridge.
        
        Args:
            version: Version to connect
            system: System instance
            
        Returns:
            True if connection successful, False otherwise
        """
        with self.lock:
            if version in self.versions:
                logger.warning(f"Version {version} already connected")
                return False
            
            self.versions[version] = VersionInfo(
                version=version,
                system=system,
                message_handlers={},
                event_queue=Queue(),
                processing_thread=None
            )
            
            logger.info(f"Connected version {version}")
            return True
    
    def disconnect_version(self, version: VersionType) -> bool:
        """
        Disconnect a version from the bridge.
        
        Args:
            version: Version to disconnect
            
        Returns:
            True if disconnection successful, False otherwise
        """
        with self.lock:
            if version not in self.versions:
                logger.warning(f"Version {version} not connected")
                return False
            
            version_info = self.versions[version]
            if version_info.is_running:
                self.stop_version(version)
            
            del self.versions[version]
            logger.info(f"Disconnected version {version}")
            return True
    
    def register_message_handler(self, version: VersionType, message_type: str,
                               handler: Callable) -> bool:
        """
        Register a message handler for a version.
        
        Args:
            version: Version to register handler for
            message_type: Type of message to handle
            handler: Handler function
            
        Returns:
            True if registration successful, False otherwise
        """
        with self.lock:
            if version not in self.versions:
                logger.warning(f"Version {version} not connected")
                return False
            
            self.versions[version].message_handlers[message_type] = handler
            logger.info(f"Registered handler for {message_type} in {version}")
            return True
    
    def start_version(self, version: VersionType) -> bool:
        """
        Start processing for a version.
        
        Args:
            version: Version to start
            
        Returns:
            True if start successful, False otherwise
        """
        with self.lock:
            if version not in self.versions:
                logger.warning(f"Version {version} not connected")
                return False
            
            version_info = self.versions[version]
            if version_info.is_running:
                logger.warning(f"Version {version} already running")
                return False
            
            version_info.is_running = True
            version_info.processing_thread = Thread(
                target=self._process_events,
                args=(version,),
                daemon=True
            )
            version_info.processing_thread.start()
            
            logger.info(f"Started version {version}")
            return True
    
    def stop_version(self, version: VersionType) -> bool:
        """
        Stop processing for a version.
        
        Args:
            version: Version to stop
            
        Returns:
            True if stop successful, False otherwise
        """
        with self.lock:
            if version not in self.versions:
                logger.warning(f"Version {version} not connected")
                return False
            
            version_info = self.versions[version]
            if not version_info.is_running:
                logger.warning(f"Version {version} not running")
                return False
            
            version_info.is_running = False
            version_info.event_queue.put(None)  # Signal thread to stop
            if version_info.processing_thread:
                version_info.processing_thread.join()
            
            logger.info(f"Stopped version {version}")
            return True
    
    def _process_events(self, version: VersionType) -> None:
        """Process events for a version."""
        version_info = self.versions[version]
        
        while version_info.is_running:
            try:
                event = version_info.event_queue.get(timeout=1.0)
                if event is None:  # Stop signal
                    break
                
                message_type = event.get('type')
                if message_type in version_info.message_handlers:
                    handler = version_info.message_handlers[message_type]
                    handler(event)
                
                version_info.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing event in {version}: {str(e)}")
    
    def send_data(self, from_version: VersionType, to_version: VersionType,
                 data: Dict[str, Any]) -> bool:
        """
        Send data between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            data: Data to send
            
        Returns:
            True if send successful, False otherwise
        """
        with self.lock:
            if from_version not in self.versions or to_version not in self.versions:
                logger.warning("One or both versions not connected")
                return False
            
            if to_version not in self.compatibility_matrix[from_version]:
                logger.warning(f"Versions {from_version} and {to_version} are not compatible")
                return False
            
            self.versions[to_version].event_queue.put(data)
            logger.debug(f"Sent data from {from_version} to {to_version}")
            return True
    
    def broadcast(self, from_version: VersionType, data: Dict[str, Any],
                 exclude_versions: Optional[List[VersionType]] = None) -> None:
        """
        Broadcast data to compatible versions.
        
        Args:
            from_version: Source version
            data: Data to broadcast
            exclude_versions: Versions to exclude
        """
        exclude_versions = exclude_versions or []
        
        with self.lock:
            if from_version not in self.versions:
                logger.warning(f"Version {from_version} not connected")
                return
            
            compatible_versions = self.compatibility_matrix[from_version]
            for version in compatible_versions:
                if version not in exclude_versions and version in self.versions:
                    self.versions[version].event_queue.put(data)
            
            logger.debug(f"Broadcasted data from {from_version}")
    
    def get_version_status(self, version: VersionType) -> Optional[Dict[str, Any]]:
        """
        Get status of a version.
        
        Args:
            version: Version to get status for
            
        Returns:
            Status dictionary if version exists, None otherwise
        """
        with self.lock:
            if version not in self.versions:
                return None
            
            version_info = self.versions[version]
            return {
                'version': version.value,
                'is_running': version_info.is_running,
                'message_handlers': list(version_info.message_handlers.keys()),
                'queue_size': version_info.event_queue.qsize()
            }
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """
        Get status of the entire bridge.
        
        Returns:
            Bridge status dictionary
        """
        with self.lock:
            return {
                'connected_versions': [v.value for v in self.versions.keys()],
                'compatibility_matrix': {
                    v.value: [c.value for c in compatible]
                    for v, compatible in self.compatibility_matrix.items()
                }
            }

# Export functionality for node integration
functionality = {
    'classes': {
        'SpiderwebBridge': SpiderwebBridge,
        'VersionType': VersionType,
        'VersionInfo': VersionInfo
    },
    'description': 'Spiderweb bridge system for version interoperability'
} 