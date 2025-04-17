from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from uuid import UUID, uuid4
from enum import Enum
from PySide6.QtCore import QObject, Signal

class NodePort:
    """Represents an input or output port on a node"""
    def __init__(self, name: str, port_type: type, description: str = ""):
        self.name = name
        self.port_type = port_type
        self.description = description
        self.connections: Set[UUID] = set()
        self.value: Any = None

class NodeType(Enum):
    PROCESSOR = "processor"
    INPUT = "input"
    OUTPUT = "output"
    UTILITY = "utility"

@dataclass
class NodeMetadata:
    """Metadata for node visualization and organization"""
    name: str
    description: str
    category: str
    type: NodeType
    color: str = "#808080"
    icon: str = "⚙️"

class Node(QObject):
    """Base class for all nodes in the system"""
    
    # Signals
    node_updated = Signal(UUID)  # Emitted when node state changes
    error_occurred = Signal(str)  # Emitted when an error occurs
    
    def __init__(self, metadata: NodeMetadata):
        super().__init__()
        self.id = uuid4()
        self.metadata = metadata
        self.input_ports: Dict[str, NodePort] = {}
        self.output_ports: Dict[str, NodePort] = {}
        self.is_processing = False
        self.error = None
        
    def add_input_port(self, name: str, port_type: type, description: str = "") -> None:
        """Add an input port to the node"""
        self.input_ports[name] = NodePort(name, port_type, description)
        
    def add_output_port(self, name: str, port_type: type, description: str = "") -> None:
        """Add an output port to the node"""
        self.output_ports[name] = NodePort(name, port_type, description)
        
    def connect_input(self, port_name: str, source_node_id: UUID) -> bool:
        """Connect an input port to a source node"""
        if port_name not in self.input_ports:
            self.error_occurred.emit(f"Input port {port_name} does not exist")
            return False
        self.input_ports[port_name].connections.add(source_node_id)
        return True
        
    def connect_output(self, port_name: str, target_node_id: UUID) -> bool:
        """Connect an output port to a target node"""
        if port_name not in self.output_ports:
            self.error_occurred.emit(f"Output port {port_name} does not exist")
            return False
        self.output_ports[port_name].connections.add(target_node_id)
        return True
        
    def disconnect_input(self, port_name: str, source_node_id: UUID) -> bool:
        """Disconnect an input port from a source node"""
        if port_name not in self.input_ports:
            return False
        self.input_ports[port_name].connections.discard(source_node_id)
        return True
        
    def disconnect_output(self, port_name: str, target_node_id: UUID) -> bool:
        """Disconnect an output port from a target node"""
        if port_name not in self.output_ports:
            return False
        self.output_ports[port_name].connections.discard(target_node_id)
        return True
        
    def get_input_value(self, port_name: str) -> Any:
        """Get the current value of an input port"""
        if port_name not in self.input_ports:
            return None
        return self.input_ports[port_name].value
        
    def set_output_value(self, port_name: str, value: Any) -> bool:
        """Set the value of an output port"""
        if port_name not in self.output_ports:
            return False
        self.output_ports[port_name].value = value
        self.node_updated.emit(self.id)
        return True
        
    async def process(self) -> None:
        """Process the node's inputs and generate outputs"""
        raise NotImplementedError("Nodes must implement process method")
        
    def validate_inputs(self) -> bool:
        """Check if all required inputs are connected and valid"""
        return all(
            len(port.connections) > 0 and port.value is not None
            for port in self.input_ports.values()
        )
        
    def to_dict(self) -> Dict:
        """Convert node to dictionary for serialization"""
        return {
            "id": str(self.id),
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "category": self.metadata.category,
                "type": self.metadata.type.value,
                "color": self.metadata.color,
                "icon": self.metadata.icon
            },
            "input_ports": {
                name: {
                    "type": str(port.port_type),
                    "description": port.description,
                    "connections": [str(conn) for conn in port.connections]
                }
                for name, port in self.input_ports.items()
            },
            "output_ports": {
                name: {
                    "type": str(port.port_type),
                    "description": port.description,
                    "connections": [str(conn) for conn in port.connections]
                }
                for name, port in self.output_ports.items()
            }
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        """Create node from dictionary representation"""
        metadata = NodeMetadata(
            name=data["metadata"]["name"],
            description=data["metadata"]["description"],
            category=data["metadata"]["category"],
            type=NodeType(data["metadata"]["type"]),
            color=data["metadata"]["color"],
            icon=data["metadata"]["icon"]
        )
        node = cls(metadata)
        node.id = UUID(data["id"])
        
        # Restore ports and connections
        for name, port_data in data["input_ports"].items():
            node.add_input_port(name, eval(port_data["type"]), port_data["description"])
            for conn in port_data["connections"]:
                node.connect_input(name, UUID(conn))
                
        for name, port_data in data["output_ports"].items():
            node.add_output_port(name, eval(port_data["type"]), port_data["description"])
            for conn in port_data["connections"]:
                node.connect_output(name, UUID(conn))
                
        return node 