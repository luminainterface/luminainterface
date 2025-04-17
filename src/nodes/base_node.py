"""
Base Node class for Neural Network Node Manager
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

class BaseNode:
    """Base class for all neural network nodes"""
    
    def __init__(self, node_id: Optional[str] = None):
        self._initialized = False
        self._active = False
        self.node_id = node_id or self.__class__.__name__
        self.dependencies: Dict[str, Any] = {}
        self.connections: List[str] = []
        self.last_process_time: Optional[str] = None
        self.activation_level: float = 0.0
        self.logger = logging.getLogger(f"node.{self.node_id}")
        
    def initialize(self) -> bool:
        """Initialize the node"""
        try:
            self._initialized = True
            self.logger.info(f"Node {self.node_id} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize node {self.node_id}: {str(e)}")
            self._initialized = False
            return False
            
    def activate(self) -> bool:
        """Activate the node"""
        if not self._initialized:
            self.logger.error(f"Cannot activate uninitialized node {self.node_id}")
            return False
            
        try:
            self._active = True
            self.activation_level = 1.0
            self.logger.info(f"Node {self.node_id} activated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to activate node {self.node_id}: {str(e)}")
            self._active = False
            self.activation_level = 0.0
            return False
            
    def deactivate(self) -> bool:
        """Deactivate the node"""
        self._active = False
        self.activation_level = 0.0
        self.logger.info(f"Node {self.node_id} deactivated")
        return True
        
    def process(self, data: Any) -> Optional[Dict[str, Any]]:
        """Process input data"""
        if not self._active:
            self.logger.error(f"Cannot process data - node {self.node_id} is not active")
            return None
            
        try:
            self.last_process_time = datetime.now().isoformat()
            result = {
                'status': 'success',
                'node_id': self.node_id,
                'data': data,
                'timestamp': self.last_process_time
            }
            return result
        except Exception as e:
            self.logger.error(f"Error processing data in node {self.node_id}: {str(e)}")
            return None
            
    def connect(self, node_id: str) -> bool:
        """Connect to another node"""
        if node_id not in self.connections:
            self.connections.append(node_id)
            self.logger.info(f"Node {self.node_id} connected to {node_id}")
            return True
        return False
        
    def disconnect(self, node_id: str) -> bool:
        """Disconnect from another node"""
        if node_id in self.connections:
            self.connections.remove(node_id)
            self.logger.info(f"Node {self.node_id} disconnected from {node_id}")
            return True
        return False
        
    def add_dependency(self, name: str, component: Any) -> bool:
        """Add a dependency to the node"""
        try:
            self.dependencies[name] = component
            self.logger.info(f"Added dependency {name} to node {self.node_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add dependency {name}: {str(e)}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get node status"""
        return {
            'node_id': self.node_id,
            'status': 'active' if self._active else 'inactive',
            'initialized': self._initialized,
            'activation_level': self.activation_level,
            'connections': self.connections,
            'last_process_time': self.last_process_time,
            'dependencies': list(self.dependencies.keys())
        }
        
    def is_initialized(self) -> bool:
        """Check if node is initialized"""
        return self._initialized
        
    def is_active(self) -> bool:
        """Check if node is active"""
        return self._active 