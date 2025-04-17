from typing import Dict, List, Type, Optional
from uuid import UUID
import asyncio
from PySide6.QtCore import QObject, Signal
from .base_node import Node, NodePort

class NodeManager(QObject):
    """Manages the node system, including creation, connection, and execution of nodes"""
    
    # Signals
    node_added = Signal(UUID)
    node_removed = Signal(UUID)
    node_updated = Signal(UUID)
    connection_made = Signal(UUID, str, UUID, str)  # source_id, output_port, target_id, input_port
    connection_removed = Signal(UUID, str, UUID, str)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.nodes: Dict[UUID, Node] = {}
        self._registered_node_types: Dict[str, Type[Node]] = {}
        self._execution_order: List[UUID] = []
        self._is_running = False
        
    def register_node_type(self, name: str, node_class: Type[Node]) -> None:
        """Register a node type for creation"""
        self._registered_node_types[name] = node_class
        
    def create_node(self, node_type: str) -> Optional[Node]:
        """Create a new node of the specified type"""
        if node_type not in self._registered_node_types:
            self.error_occurred.emit(f"Unknown node type: {node_type}")
            return None
            
        try:
            node = self._registered_node_types[node_type]()
            self.nodes[node.id] = node
            self._update_execution_order()
            
            # Connect node signals
            node.node_updated.connect(lambda: self.node_updated.emit(node.id))
            node.error_occurred.connect(self.error_occurred.emit)
            
            self.node_added.emit(node.id)
            return node
            
        except Exception as e:
            self.error_occurred.emit(f"Error creating node: {str(e)}")
            return None
            
    def remove_node(self, node_id: UUID) -> bool:
        """Remove a node and its connections"""
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        
        # Remove connections
        for port in node.input_ports.values():
            for source_id in list(port.connections):
                self.disconnect_nodes(source_id, node_id)
                
        for port in node.output_ports.values():
            for target_id in list(port.connections):
                self.disconnect_nodes(node_id, target_id)
                
        # Clean up node
        if hasattr(node, 'cleanup'):
            node.cleanup()
            
        # Remove node
        del self.nodes[node_id]
        self._update_execution_order()
        self.node_removed.emit(node_id)
        return True
        
    def connect_nodes(self, source_id: UUID, output_port: str,
                     target_id: UUID, input_port: str) -> bool:
        """Connect two nodes together"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
            
        source_node = self.nodes[source_id]
        target_node = self.nodes[target_id]
        
        if output_port not in source_node.output_ports:
            self.error_occurred.emit(f"Output port {output_port} not found")
            return False
            
        if input_port not in target_node.input_ports:
            self.error_occurred.emit(f"Input port {input_port} not found")
            return False
            
        # Check port type compatibility
        source_type = source_node.output_ports[output_port].port_type
        target_type = target_node.input_ports[input_port].port_type
        if not issubclass(source_type, target_type):
            self.error_occurred.emit(f"Incompatible port types: {source_type} -> {target_type}")
            return False
            
        # Make connection
        source_node.connect_output(output_port, target_id)
        target_node.connect_input(input_port, source_id)
        
        self._update_execution_order()
        self.connection_made.emit(source_id, output_port, target_id, input_port)
        return True
        
    def disconnect_nodes(self, source_id: UUID, target_id: UUID) -> bool:
        """Disconnect two nodes"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
            
        source_node = self.nodes[source_id]
        target_node = self.nodes[target_id]
        
        # Find connected ports
        for output_port, port in source_node.output_ports.items():
            if target_id in port.connections:
                for input_port, target_port in target_node.input_ports.items():
                    if source_id in target_port.connections:
                        # Remove connections
                        source_node.disconnect_output(output_port, target_id)
                        target_node.disconnect_input(input_port, source_id)
                        self.connection_removed.emit(source_id, output_port, target_id, input_port)
                        
        self._update_execution_order()
        return True
        
    def _update_execution_order(self) -> None:
        """Update the execution order of nodes based on dependencies"""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node_id: UUID):
            if node_id in temp_visited:
                raise ValueError("Cyclic dependency detected")
            if node_id in visited:
                return
                
            temp_visited.add(node_id)
            node = self.nodes[node_id]
            
            # Visit all dependencies
            for port in node.output_ports.values():
                for target_id in port.connections:
                    visit(target_id)
                    
            temp_visited.remove(node_id)
            visited.add(node_id)
            order.append(node_id)
            
        # Visit all nodes
        try:
            for node_id in self.nodes:
                if node_id not in visited:
                    visit(node_id)
            self._execution_order = list(reversed(order))
        except ValueError as e:
            self.error_occurred.emit(str(e))
            self._execution_order = []
            
    async def execute(self) -> None:
        """Execute all nodes in the correct order"""
        self._is_running = True
        try:
            for node_id in self._execution_order:
                if not self._is_running:
                    break
                node = self.nodes[node_id]
                await node.process()
        except Exception as e:
            self.error_occurred.emit(f"Execution error: {str(e)}")
        finally:
            self._is_running = False
            
    def stop(self) -> None:
        """Stop node execution"""
        self._is_running = False
        
    def get_node(self, node_id: UUID) -> Optional[Node]:
        """Get a node by its ID"""
        return self.nodes.get(node_id)
        
    def get_all_nodes(self) -> List[Node]:
        """Get all nodes in the system"""
        return list(self.nodes.values())
        
    def clear(self) -> None:
        """Remove all nodes and connections"""
        node_ids = list(self.nodes.keys())
        for node_id in node_ids:
            self.remove_node(node_id)
        self._execution_order = [] 