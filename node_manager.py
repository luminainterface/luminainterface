"""
Node Manager for Neural Network Node Manager
"""

import logging
import importlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

class NodeManager:
    """Manager class for handling neural network nodes"""
    
    def __init__(self):
        self.logger = logging.getLogger("NodeManager")
        self.nodes: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.active_nodes: Dict[str, Any] = {}
        self.node_states: Dict[str, Dict[str, Any]] = {}
        self.last_update = datetime.now()
        
    def initialize(self) -> bool:
        """Initialize the node manager"""
        try:
            # Create required directories
            self._create_directories()
            
            # Load available nodes
            self._load_available_nodes()
            
            # Initialize logging
            self._setup_logging()
            
            self.logger.info("NodeManager initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize NodeManager: {str(e)}")
            return False
            
    def _create_directories(self):
        """Create required directories"""
        directories = ['logs', 'data', 'models']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = Path('logs/node_manager.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def _load_available_nodes(self):
        """Load available node types"""
        try:
            # Load core nodes
            node_types = {
                'RSEN': 'nodes.RSEN_node',
                'HybridNode': 'nodes.hybrid_node',
                'FractalNode': 'nodes.fractal_node',
                'InfiniteMindsNode': 'nodes.infinite_minds_node',
                'IsomorphNode': 'nodes.isomorph_node',
                'VortexNode': 'nodes.vortex_node'
            }
            
            for node_name, module_path in node_types.items():
                try:
                    module = importlib.import_module(module_path)
                    node_class = getattr(module, node_name)
                    self.nodes[node_name] = node_class
                    self.logger.info(f"Loaded node type: {node_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load node type {node_name}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error loading available nodes: {str(e)}")
            raise
            
    def create_node(self, node_type: str, node_id: Optional[str] = None, **kwargs) -> Optional[Any]:
        """Create a new node instance"""
        try:
            if node_type not in self.nodes:
                raise ValueError(f"Unknown node type: {node_type}")
                
            # Create node instance
            node_class = self.nodes[node_type]
            node = node_class(node_id=node_id, **kwargs)
            
            # Initialize node
            if node.initialize():
                node_id = node_id or f"{node_type}_{len(self.active_nodes)}"
                self.active_nodes[node_id] = node
                self.node_states[node_id] = node.get_status()
                self.logger.info(f"Created and initialized node: {node_id}")
                return node
            else:
                self.logger.error(f"Failed to initialize node: {node_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating node: {str(e)}")
            return None
            
    def get_node(self, node_id: str) -> Optional[Any]:
        """Get a node by ID"""
        return self.active_nodes.get(node_id)
        
    def activate_node(self, node_id: str) -> bool:
        """Activate a node"""
        node = self.get_node(node_id)
        if node:
            try:
                if node.activate():
                    self.node_states[node_id] = node.get_status()
                    self.logger.info(f"Activated node: {node_id}")
                    return True
                else:
                    self.logger.error(f"Failed to activate node: {node_id}")
            except Exception as e:
                self.logger.error(f"Error activating node {node_id}: {str(e)}")
        return False
        
    def deactivate_node(self, node_id: str) -> bool:
        """Deactivate a node"""
        node = self.get_node(node_id)
        if node:
            try:
                if node.deactivate():
                    self.node_states[node_id] = node.get_status()
                    self.logger.info(f"Deactivated node: {node_id}")
                    return True
                else:
                    self.logger.error(f"Failed to deactivate node: {node_id}")
            except Exception as e:
                self.logger.error(f"Error deactivating node {node_id}: {str(e)}")
        return False
        
    def remove_node(self, node_id: str) -> bool:
        """Remove a node"""
        if node_id in self.active_nodes:
            try:
                node = self.active_nodes[node_id]
                node.deactivate()
                del self.active_nodes[node_id]
                del self.node_states[node_id]
                self.logger.info(f"Removed node: {node_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error removing node {node_id}: {str(e)}")
        return False
        
    def connect_nodes(self, source_id: str, target_id: str) -> bool:
        """Connect two nodes"""
        source = self.get_node(source_id)
        target = self.get_node(target_id)
        
        if source and target:
            try:
                if source.connect(target_id) and target.connect(source_id):
                    self.node_states[source_id] = source.get_status()
                    self.node_states[target_id] = target.get_status()
                    self.logger.info(f"Connected nodes: {source_id} <-> {target_id}")
                    return True
                else:
                    self.logger.error(f"Failed to connect nodes: {source_id} <-> {target_id}")
            except Exception as e:
                self.logger.error(f"Error connecting nodes: {str(e)}")
        return False
        
    def disconnect_nodes(self, source_id: str, target_id: str) -> bool:
        """Disconnect two nodes"""
        source = self.get_node(source_id)
        target = self.get_node(target_id)
        
        if source and target:
            try:
                if source.disconnect(target_id) and target.disconnect(source_id):
                    self.node_states[source_id] = source.get_status()
                    self.node_states[target_id] = target.get_status()
                    self.logger.info(f"Disconnected nodes: {source_id} <-> {target_id}")
                    return True
                else:
                    self.logger.error(f"Failed to disconnect nodes: {source_id} <-> {target_id}")
            except Exception as e:
                self.logger.error(f"Error disconnecting nodes: {str(e)}")
        return False
        
    def process_node(self, node_id: str, data: Any) -> Optional[Dict[str, Any]]:
        """Process data through a node"""
        node = self.get_node(node_id)
        if node:
            try:
                result = node.process(data)
                if result:
                    self.node_states[node_id] = node.get_status()
                    return result
                else:
                    self.logger.error(f"Node {node_id} processing returned no result")
            except Exception as e:
                self.logger.error(f"Error processing through node {node_id}: {str(e)}")
        return None
        
    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a node"""
        node = self.get_node(node_id)
        if node:
            try:
                status = node.get_status()
                self.node_states[node_id] = status
                return status
            except Exception as e:
                self.logger.error(f"Error getting status for node {node_id}: {str(e)}")
        return None
        
    def get_all_node_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all nodes"""
        # Update all node states
        for node_id, node in self.active_nodes.items():
            try:
                self.node_states[node_id] = node.get_status()
            except Exception as e:
                self.logger.error(f"Error updating state for node {node_id}: {str(e)}")
                
        return self.node_states
        
    def get_active_nodes(self) -> List[str]:
        """Get list of active node IDs"""
        return [
            node_id for node_id, node in self.active_nodes.items()
            if node.is_active()
        ]
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'total_nodes': len(self.active_nodes),
            'active_nodes': len(self.get_active_nodes()),
            'available_node_types': list(self.nodes.keys()),
            'last_update': self.last_update.isoformat()
        } 