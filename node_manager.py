import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import os
import importlib.util
import glob
import inspect

logger = logging.getLogger(__name__)

class NodeManager:
    def __init__(self):
        self.nodes = {}
        self.connections = {}
        self.node_states = {}
        self.last_sync = datetime.now()
        self.connection_types = {
            'default': {'bidirectional': True, 'weight_range': (0.0, 1.0)},
            'resonance': {'bidirectional': True, 'weight_range': (0.0, 1.0)},
            'knowledge': {'bidirectional': False, 'weight_range': (0.0, 1.0)},
            'feedback': {'bidirectional': True, 'weight_range': (-1.0, 1.0)},
            'control': {'bidirectional': False, 'weight_range': (0.0, 1.0)}
        }
        self.socket_status = {}
        logger.info("Initialized NodeManager with enhanced connection and socket support")
        
    def discover_node_files(self, directory: str = None) -> List[str]:
        """Discover all Python files containing 'node' in their name"""
        if directory is None:
            directory = os.path.dirname(os.path.abspath(__file__))
        
        node_files = []
        for root, _, _ in os.walk(directory):
            pattern = os.path.join(root, '*node*.py')
            node_files.extend(glob.glob(pattern))
        
        # Exclude this manager file
        node_files = [f for f in node_files if os.path.basename(f) != 'node_manager.py']
        logger.info(f"Discovered {len(node_files)} node files: {node_files}")
        return node_files

    def load_node_module(self, file_path: str) -> Optional[Any]:
        """Dynamically load a Python module from file path"""
        try:
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load spec for {file_path}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
            
        except Exception as e:
            logger.error(f"Error loading module {file_path}: {str(e)}")
            return None

    def discover_and_register_nodes(self, directory: str = None) -> int:
        """Discover and register all nodes in the given directory"""
        node_files = self.discover_node_files(directory)
        registered_count = 0
        
        for file_path in node_files:
            module = self.load_node_module(file_path)
            if module is None:
                continue
                
            # Look for classes in the module
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and 'node' in name.lower():
                    try:
                        # Create instance and register it
                        instance = obj()
                        node_id = f"{os.path.basename(file_path)}:{name}"
                        if self.register_node(node_id, instance):
                            registered_count += 1
                            logger.info(f"Successfully registered node {node_id}")
                    except Exception as e:
                        logger.error(f"Error instantiating node class {name}: {str(e)}")
        
        return registered_count

    def register_node(self, node_id: str, node_instance: Any) -> bool:
        """Register a new node with the manager"""
        try:
            if node_id in self.nodes:
                logger.warning(f"Node {node_id} already registered")
                return False
                
            self.nodes[node_id] = node_instance
            self.node_states[node_id] = {
                'status': 'initialized',
                'last_active': datetime.now().isoformat(),
                'processing_count': 0,
                'error_count': 0,
                'connections': {
                    'incoming': [],
                    'outgoing': []
                }
            }
            
            # Initialize socket for new node
            if hasattr(node_instance, 'initialize_socket'):
                success = node_instance.initialize_socket()
                self.socket_status[node_id] = 'active' if success else 'failed'
            else:
                self.socket_status[node_id] = 'default'
                
            logger.info(f"Successfully registered node: {node_id} with socket status: {self.socket_status[node_id]}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering node {node_id}: {str(e)}")
            return False
            
    def register_connection_type(self, type_name: str, bidirectional: bool = True, weight_range: Tuple[float, float] = (0.0, 1.0)) -> bool:
        """Register a new connection type"""
        try:
            if type_name in self.connection_types:
                logger.warning(f"Connection type {type_name} already exists")
                return False
                
            self.connection_types[type_name] = {
                'bidirectional': bidirectional,
                'weight_range': weight_range
            }
            logger.info(f"Registered new connection type: {type_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering connection type {type_name}: {str(e)}")
            return False
            
    def connect_nodes(self, source_id: str, target_id: str, connection_type: str = 'default', 
                     weight: float = 1.0, metadata: Optional[Dict] = None) -> bool:
        """Create a connection between two nodes with enhanced properties"""
        try:
            if source_id not in self.nodes or target_id not in self.nodes:
                logger.error(f"Cannot connect: one or both nodes not found - {source_id}, {target_id}")
                return False
                
            if connection_type not in self.connection_types:
                logger.error(f"Invalid connection type: {connection_type}")
                return False
                
            # Validate weight range
            weight_range = self.connection_types[connection_type]['weight_range']
            weight = max(min(weight, weight_range[1]), weight_range[0])
            
            # Create forward connection
            connection_key = f"{source_id}_{target_id}_{connection_type}"
            if connection_key in self.connections:
                logger.warning(f"Connection {connection_key} already exists")
                return False
                
            self.connections[connection_key] = {
                'type': connection_type,
                'weight': weight,
                'created_at': datetime.now().isoformat(),
                'status': 'active',
                'metadata': metadata or {}
            }
            
            # Update node states
            self.node_states[source_id]['connections']['outgoing'].append({
                'target': target_id,
                'type': connection_type,
                'weight': weight
            })
            self.node_states[target_id]['connections']['incoming'].append({
                'source': source_id,
                'type': connection_type,
                'weight': weight
            })
            
            # Create reverse connection if bidirectional
            if self.connection_types[connection_type]['bidirectional']:
                reverse_key = f"{target_id}_{source_id}_{connection_type}"
                self.connections[reverse_key] = {
                    'type': connection_type,
                    'weight': weight,
                    'created_at': datetime.now().isoformat(),
                    'status': 'active',
                    'metadata': metadata or {}
                }
                
                # Update node states for reverse connection
                self.node_states[target_id]['connections']['outgoing'].append({
                    'target': source_id,
                    'type': connection_type,
                    'weight': weight
                })
                self.node_states[source_id]['connections']['incoming'].append({
                    'source': target_id,
                    'type': connection_type,
                    'weight': weight
                })
            
            logger.info(f"Created {'bidirectional' if self.connection_types[connection_type]['bidirectional'] else 'one-way'} "
                       f"connection: {connection_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting nodes {source_id} to {target_id}: {str(e)}")
            return False
            
    def update_connection(self, source_id: str, target_id: str, connection_type: str,
                         weight: Optional[float] = None, status: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> bool:
        """Update properties of an existing connection"""
        try:
            connection_key = f"{source_id}_{target_id}_{connection_type}"
            if connection_key not in self.connections:
                logger.error(f"Connection {connection_key} not found")
                return False
                
            connection = self.connections[connection_key]
            
            if weight is not None:
                weight_range = self.connection_types[connection_type]['weight_range']
                connection['weight'] = max(min(weight, weight_range[1]), weight_range[0])
                
            if status is not None:
                connection['status'] = status
                
            if metadata is not None:
                connection['metadata'].update(metadata)
                
            # Update reverse connection if bidirectional
            if self.connection_types[connection_type]['bidirectional']:
                reverse_key = f"{target_id}_{source_id}_{connection_type}"
                if reverse_key in self.connections:
                    reverse_conn = self.connections[reverse_key]
                    if weight is not None:
                        reverse_conn['weight'] = connection['weight']
                    if status is not None:
                        reverse_conn['status'] = status
                    if metadata is not None:
                        reverse_conn['metadata'].update(metadata)
            
            logger.info(f"Updated connection: {connection_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating connection {source_id} to {target_id}: {str(e)}")
            return False
            
    def get_node_connections(self, node_id: str, connection_type: Optional[str] = None,
                           include_inactive: bool = False) -> Dict:
        """Get detailed connection information for a node"""
        try:
            if node_id not in self.nodes:
                logger.error(f"Node {node_id} not found")
                return {}
                
            connections = {
                'incoming': [],
                'outgoing': []
            }
            
            # Get all connections involving this node
            for conn_key, conn_data in self.connections.items():
                if not include_inactive and conn_data['status'] != 'active':
                    continue
                    
                if connection_type and conn_data['type'] != connection_type:
                    continue
                    
                source, target, conn_type = conn_key.split('_')
                
                if source == node_id:
                    connections['outgoing'].append({
                        'target': target,
                        'type': conn_type,
                        'weight': conn_data['weight'],
                        'status': conn_data['status'],
                        'metadata': conn_data['metadata']
                    })
                elif target == node_id:
                    connections['incoming'].append({
                        'source': source,
                        'type': conn_type,
                        'weight': conn_data['weight'],
                        'status': conn_data['status'],
                        'metadata': conn_data['metadata']
                    })
            
            return connections
            
        except Exception as e:
            logger.error(f"Error getting connections for node {node_id}: {str(e)}")
            return {}
            
    def get_connection_types(self) -> Dict:
        """Get all registered connection types and their properties"""
        return self.connection_types
            
    def update_node_state(self, node_id: str, state_update: Dict) -> bool:
        """Update the state of a node"""
        try:
            if node_id not in self.nodes:
                logger.error(f"Node {node_id} not found")
                return False
                
            current_state = self.node_states.get(node_id, {})
            current_state.update(state_update)
            current_state['last_active'] = datetime.now().isoformat()
            
            self.node_states[node_id] = current_state
            logger.info(f"Updated state for node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating state for node {node_id}: {str(e)}")
            return False
            
    def get_node_state(self, node_id: str) -> Dict:
        """Get the current state of a node"""
        try:
            if node_id not in self.nodes:
                logger.error(f"Node {node_id} not found")
                return {}
                
            return self.node_states.get(node_id, {})
            
        except Exception as e:
            logger.error(f"Error getting state for node {node_id}: {str(e)}")
            return {}
            
    def process_node(self, node_id: str, input_data: Any) -> Dict:
        """Process data through a specific node"""
        try:
            if node_id not in self.nodes:
                logger.error(f"Node {node_id} not found")
                return {'error': 'Node not found'}
                
            node = self.nodes[node_id]
            result = node.process(input_data)
            
            # Update node state
            self.node_states[node_id]['processing_count'] += 1
            self.node_states[node_id]['last_active'] = datetime.now().isoformat()
            
            return {'success': True, 'result': result}
            
        except Exception as e:
            logger.error(f"Error processing node {node_id}: {str(e)}")
            self.node_states[node_id]['error_count'] += 1
            return {'error': str(e)}
            
    def sync_nodes(self) -> bool:
        """Synchronize all registered nodes"""
        try:
            current_time = datetime.now()
            if (current_time - self.last_sync).total_seconds() < 60:
                logger.info("Skipping sync - last sync was less than 60 seconds ago")
                return True
                
            for node_id, node in self.nodes.items():
                try:
                    if hasattr(node, 'sync'):
                        node.sync()
                    self.node_states[node_id]['last_sync'] = current_time.isoformat()
                except Exception as e:
                    logger.error(f"Error syncing node {node_id}: {str(e)}")
                    
            self.last_sync = current_time
            logger.info("Successfully synchronized all nodes")
            return True
            
        except Exception as e:
            logger.error(f"Error during node synchronization: {str(e)}")
            return False
            
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        try:
            total_nodes = len(self.nodes)
            active_nodes = sum(1 for state in self.node_states.values() 
                             if state.get('status') == 'active')
            total_connections = len(self.connections)
            active_connections = sum(1 for conn in self.connections.values() 
                                   if conn.get('status') == 'active')
            total_processing = sum(state.get('processing_count', 0) 
                                 for state in self.node_states.values())
            total_errors = sum(state.get('error_count', 0) 
                             for state in self.node_states.values())
            
            # Group connections by type
            connections_by_type = {}
            for conn_key, conn_data in self.connections.items():
                conn_type = conn_data['type']
                if conn_type not in connections_by_type:
                    connections_by_type[conn_type] = 0
                connections_by_type[conn_type] += 1
            
            return {
                'total_nodes': total_nodes,
                'active_nodes': active_nodes,
                'total_connections': total_connections,
                'active_connections': active_connections,
                'connections_by_type': connections_by_type,
                'total_processing': total_processing,
                'total_errors': total_errors,
                'last_sync': self.last_sync.isoformat(),
                'nodes': self.node_states,
                'connection_types': self.connection_types
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {}
            
    def save_state(self, filepath: str) -> bool:
        """Save current system state to file"""
        try:
            state = {
                'nodes': self.node_states,
                'connections': self.connections,
                'connection_types': self.connection_types,
                'last_sync': self.last_sync.isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Successfully saved system state to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
            return False
            
    def load_state(self, filepath: str) -> bool:
        """Load system state from file"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"State file not found: {filepath}")
                return False
                
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.node_states = state.get('nodes', {})
            self.connections = state.get('connections', {})
            self.connection_types = state.get('connection_types', self.connection_types)
            self.last_sync = datetime.fromisoformat(state.get('last_sync', datetime.now().isoformat()))
            
            logger.info(f"Successfully loaded system state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading system state: {str(e)}")
            return False

    def initialize_sockets(self):
        """Initialize sockets for all registered nodes"""
        try:
            for node_id, node in self.nodes.items():
                if hasattr(node, 'initialize_socket'):
                    success = node.initialize_socket()
                    self.socket_status[node_id] = 'active' if success else 'failed'
                else:
                    # Create default socket if not implemented
                    self.socket_status[node_id] = 'default'
                logger.info(f"Initialized socket for node {node_id}: {self.socket_status[node_id]}")
            return True
        except Exception as e:
            logger.error(f"Error initializing sockets: {str(e)}")
            return False
            
    def validate_connections(self) -> Dict[str, List[str]]:
        """Validate all node connections and their socket status"""
        validation_results = {
            'valid': [],
            'invalid': [],
            'missing_socket': []
        }
        
        for connection_key, connection in self.connections.items():
            source_id, target_id, _ = connection_key.split('_')
            
            # Check if both nodes exist
            if source_id not in self.nodes or target_id not in self.nodes:
                validation_results['invalid'].append(connection_key)
                continue
                
            # Check socket status
            if source_id not in self.socket_status or target_id not in self.socket_status:
                validation_results['missing_socket'].append(connection_key)
                continue
                
            # Validate connection
            if self.socket_status[source_id] == 'active' and self.socket_status[target_id] == 'active':
                validation_results['valid'].append(connection_key)
            else:
                validation_results['invalid'].append(connection_key)
        
        return validation_results

def main():
    """Main function for testing"""
    logging.basicConfig(level=logging.INFO)
    manager = NodeManager()
    
    # Discover and register all nodes
    registered_count = manager.discover_and_register_nodes()
    logger.info(f"Registered {registered_count} nodes from discovered files")
    
    if registered_count == 0:
        # If no nodes found, run the test code
        logger.info("No nodes found, running test code...")
        # Register custom connection type
        manager.register_connection_type('quantum', bidirectional=True, weight_range=(-1.0, 1.0))
        
        # Test node registration
        manager.register_node('test_node_1', object())
        manager.register_node('test_node_2', object())
        manager.register_node('test_node_3', object())
        
        # Test connections with different types
        manager.connect_nodes('test_node_1', 'test_node_2', 'resonance', weight=0.8)
        manager.connect_nodes('test_node_2', 'test_node_3', 'knowledge', weight=0.6)
        manager.connect_nodes('test_node_1', 'test_node_3', 'quantum', weight=0.7)
        
        # Update connection
        manager.update_connection('test_node_1', 'test_node_2', 'resonance', weight=0.9)
        
        # Test state updates
        manager.update_node_state('test_node_1', {'status': 'active'})
    
    # Print system status
    print(json.dumps(manager.get_system_status(), indent=2))
    
if __name__ == '__main__':
    main()