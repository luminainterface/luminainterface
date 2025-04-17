import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from threading import RLock

# Set up logging
logger = logging.getLogger(__name__)

class NodeState(Enum):
    """Possible states of a node"""
    INITIALIZING = auto()
    READY = auto()
    PROCESSING = auto()
    PAUSED = auto()
    ERROR = auto()
    SHUTDOWN = auto()

@dataclass
class NodeStats:
    """Statistics tracked for each node"""
    created_time: float = field(default_factory=time.time)
    last_active_time: float = field(default_factory=time.time)
    processed_messages: int = 0
    sent_messages: int = 0
    errors: int = 0
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0

class NodeType(Enum):
    """Types of nodes in the system"""
    INPUT = auto()       # Receives external input
    PROCESSOR = auto()   # Processes data
    MEMORY = auto()      # Stores information
    OUTPUT = auto()      # Produces external output
    CONTROLLER = auto()  # Manages other nodes
    UTILITY = auto()     # Provides utility functions

class ConnectionType(Enum):
    """Types of connections between nodes"""
    DATA = auto()       # Regular data flow
    CONTROL = auto()    # Control signals
    FEEDBACK = auto()   # Feedback information
    MEMORY = auto()     # Memory access

@dataclass
class Connection:
    """Represents a connection between two nodes"""
    source_id: str
    target_id: str
    connection_type: ConnectionType = ConnectionType.DATA
    weight: float = 1.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseNode:
    """
    Base class for all nodes in the Lumina v1 system
    
    Provides standardized interfaces for:
    - Node identity and type management
    - Connection handling
    - Data processing
    - State management
    - Statistics and monitoring
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        node_type: NodeType = NodeType.PROCESSOR,
        name: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        # Core identity
        self.node_id = node_id or str(uuid.uuid4())
        self.node_type = node_type
        self.name = name or f"{self.__class__.__name__}_{self.node_id[:8]}"
        self.description = description
        self.metadata = metadata or {}
        
        # State management
        self._state = NodeState.INITIALIZING
        self._lock = RLock()
        
        # Connections
        self._incoming_connections: Dict[str, Connection] = {}  # source_id -> Connection
        self._outgoing_connections: Dict[str, Connection] = {}  # target_id -> Connection
        
        # Data storage
        self._input_buffer: Dict[str, List[Any]] = {}  # source_id -> buffer
        self._context: Dict[str, Any] = {}
        
        # Statistics and monitoring
        self.stats = NodeStats()
        
        # Callbacks
        self._pre_process_callbacks: List[Callable] = []
        self._post_process_callbacks: List[Callable] = []
        self._state_change_callbacks: List[Callable] = []
        
        logger.info(f"Initialized node: {self.name} (ID: {self.node_id}, Type: {self.node_type.name})")
    
    # State Management
    
    @property
    def state(self) -> NodeState:
        """Get the current state of the node"""
        with self._lock:
            return self._state
    
    @state.setter
    def state(self, new_state: NodeState) -> None:
        """Set the node state and trigger callbacks"""
        old_state = None
        with self._lock:
            if new_state != self._state:
                old_state = self._state
                self._state = new_state
                logger.debug(f"Node {self.name} state changed: {old_state.name} -> {new_state.name}")
        
        # Call callbacks outside the lock to prevent deadlocks
        if old_state is not None:
            for callback in self._state_change_callbacks:
                try:
                    callback(self, old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback for {self.name}: {str(e)}")
    
    def is_ready(self) -> bool:
        """Check if the node is in READY state"""
        return self.state == NodeState.READY
    
    # Connection Management
    
    def connect_to(
        self,
        target_node: 'BaseNode',
        connection_type: ConnectionType = ConnectionType.DATA,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Connection:
        """
        Connect this node to a target node
        
        Args:
            target_node: Node to connect to
            connection_type: Type of connection
            weight: Connection weight
            metadata: Additional connection data
            
        Returns:
            Connection: The created connection
        """
        with self._lock:
            # Create connection object
            connection = Connection(
                source_id=self.node_id,
                target_id=target_node.node_id,
                connection_type=connection_type,
                weight=weight,
                enabled=True,
                metadata=metadata or {}
            )
            
            # Add to our outgoing connections
            self._outgoing_connections[target_node.node_id] = connection
            
            # Add to target's incoming connections
            target_node.add_incoming_connection(connection)
            
            logger.debug(f"Connected {self.name} -> {target_node.name}")
            return connection
    
    def add_incoming_connection(self, connection: Connection) -> None:
        """
        Add an incoming connection from another node
        
        Args:
            connection: Connection object
        """
        if connection.target_id != self.node_id:
            raise ValueError(f"Connection target ID {connection.target_id} does not match this node's ID {self.node_id}")
        
        with self._lock:
            self._incoming_connections[connection.source_id] = connection
            self._input_buffer[connection.source_id] = []
    
    def disconnect_from(self, target_node_id: str) -> bool:
        """
        Disconnect this node from a target node
        
        Args:
            target_node_id: ID of the target node
            
        Returns:
            bool: True if disconnected, False if not connected
        """
        with self._lock:
            if target_node_id not in self._outgoing_connections:
                return False
            
            # Remove from our outgoing connections
            connection = self._outgoing_connections.pop(target_node_id)
            
            logger.debug(f"Disconnected {self.name} -> {target_node_id}")
            return True
    
    def disconnect_from_source(self, source_node_id: str) -> bool:
        """
        Disconnect a source node from this node
        
        Args:
            source_node_id: ID of the source node
            
        Returns:
            bool: True if disconnected, False if not connected
        """
        with self._lock:
            if source_node_id not in self._incoming_connections:
                return False
            
            # Remove from our incoming connections
            self._incoming_connections.pop(source_node_id)
            
            # Clean up input buffer
            if source_node_id in self._input_buffer:
                self._input_buffer.pop(source_node_id)
            
            logger.debug(f"Disconnected {source_node_id} -> {self.name}")
            return True
    
    def get_incoming_connections(self) -> List[Connection]:
        """Get all incoming connections"""
        with self._lock:
            return list(self._incoming_connections.values())
    
    def get_outgoing_connections(self) -> List[Connection]:
        """Get all outgoing connections"""
        with self._lock:
            return list(self._outgoing_connections.values())
    
    def has_connection_to(self, target_node_id: str) -> bool:
        """Check if this node has a connection to the target node"""
        with self._lock:
            return target_node_id in self._outgoing_connections
    
    def has_connection_from(self, source_node_id: str) -> bool:
        """Check if this node has a connection from the source node"""
        with self._lock:
            return source_node_id in self._incoming_connections
    
    # Data Processing
    
    def receive_data(self, source_id: str, data: Any) -> None:
        """
        Receive data from a source node
        
        Args:
            source_id: ID of the source node
            data: Data to be processed
        """
        with self._lock:
            # Verify connection exists
            if source_id not in self._incoming_connections:
                logger.warning(f"Node {self.name} received data from unconnected source {source_id}")
                return
            
            # Add to input buffer
            self._input_buffer.setdefault(source_id, []).append(data)
            
            # Update stats
            self.stats.last_active_time = time.time()
    
    def send_data(self, data: Any, target_id: Optional[str] = None) -> bool:
        """
        Send data to connected nodes
        
        Args:
            data: Data to send
            target_id: Optional target node ID, if None, send to all connected nodes
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if target_id is not None:
                # Send to specific target
                if target_id not in self._outgoing_connections:
                    logger.warning(f"Node {self.name} tried to send data to unconnected target {target_id}")
                    return False
                
                target_connection = self._outgoing_connections[target_id]
                if not target_connection.enabled:
                    return False
                
                try:
                    self._send_to_target(target_id, data)
                    self.stats.sent_messages += 1
                    self.stats.last_active_time = time.time()
                    return True
                except Exception as e:
                    logger.error(f"Error sending data from {self.name} to {target_id}: {str(e)}")
                    self.stats.errors += 1
                    return False
            else:
                # Send to all connected targets
                success = False
                for target_id, connection in self._outgoing_connections.items():
                    if connection.enabled:
                        try:
                            self._send_to_target(target_id, data)
                            self.stats.sent_messages += 1
                            success = True
                        except Exception as e:
                            logger.error(f"Error sending data from {self.name} to {target_id}: {str(e)}")
                            self.stats.errors += 1
                
                if success:
                    self.stats.last_active_time = time.time()
                
                return success
    
    def _send_to_target(self, target_id: str, data: Any) -> None:
        """
        Send data to a specific target node
        To be implemented by derived classes or connection managers
        
        Args:
            target_id: ID of the target node
            data: Data to send
        """
        raise NotImplementedError("Subclasses must implement _send_to_target")
    
    def process(self) -> None:
        """
        Process data in the input buffer
        This is the main method to override in derived classes
        """
        if self.state != NodeState.READY and self.state != NodeState.PROCESSING:
            return
        
        self.state = NodeState.PROCESSING
        start_time = time.time()
        
        try:
            # Call pre-process callbacks
            for callback in self._pre_process_callbacks:
                callback(self)
            
            # Process data (to be implemented by derived classes)
            data_processed = self._process_data()
            
            # Update statistics
            if data_processed:
                self.stats.processed_messages += 1
            
            # Call post-process callbacks
            for callback in self._post_process_callbacks:
                callback(self)
            
            self.state = NodeState.READY
            
        except Exception as e:
            logger.error(f"Error processing data in node {self.name}: {str(e)}")
            self.stats.errors += 1
            self.state = NodeState.ERROR
            raise
        finally:
            # Update processing time statistics
            processing_time = time.time() - start_time
            total_time = self.stats.total_processing_time + processing_time
            total_msgs = self.stats.processed_messages
            
            if total_msgs > 0:
                self.stats.avg_processing_time = total_time / total_msgs
            
            self.stats.total_processing_time = total_time
    
    def _process_data(self) -> bool:
        """
        Process data from input buffers
        To be implemented by derived classes
        
        Returns:
            bool: True if data was processed, False otherwise
        """
        raise NotImplementedError("Subclasses must implement _process_data")
    
    def clear_buffers(self) -> None:
        """Clear all input buffers"""
        with self._lock:
            for source_id in self._input_buffer:
                self._input_buffer[source_id] = []
    
    # Context Management
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a context value"""
        with self._lock:
            self._context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value"""
        with self._lock:
            return self._context.get(key, default)
    
    def clear_context(self) -> None:
        """Clear all context values"""
        with self._lock:
            self._context.clear()
    
    # Lifecycle Management
    
    def initialize(self) -> bool:
        """
        Initialize the node
        Override in derived classes, but call super().initialize()
        
        Returns:
            bool: Success status
        """
        with self._lock:
            self.state = NodeState.READY
            return True
    
    def shutdown(self) -> None:
        """
        Shutdown the node
        Override in derived classes, but call super().shutdown()
        """
        with self._lock:
            self.state = NodeState.SHUTDOWN
            self.clear_buffers()
    
    def pause(self) -> None:
        """Pause node processing"""
        if self.state == NodeState.READY or self.state == NodeState.PROCESSING:
            self.state = NodeState.PAUSED
    
    def resume(self) -> None:
        """Resume node processing"""
        if self.state == NodeState.PAUSED:
            self.state = NodeState.READY
    
    def reset(self) -> None:
        """
        Reset node state
        Override in derived classes, but call super().reset()
        """
        with self._lock:
            self.clear_buffers()
            self.clear_context()
            self.stats = NodeStats()
            self.state = NodeState.READY
    
    # Callback Registration
    
    def register_pre_process_callback(self, callback: Callable) -> None:
        """Register a callback to be called before processing data"""
        self._pre_process_callbacks.append(callback)
    
    def register_post_process_callback(self, callback: Callable) -> None:
        """Register a callback to be called after processing data"""
        self._post_process_callbacks.append(callback)
    
    def register_state_change_callback(self, callback: Callable) -> None:
        """Register a callback to be called when the node state changes"""
        self._state_change_callbacks.append(callback)
    
    # Serialization
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary representation
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        with self._lock:
            node_dict = {
                "node_id": self.node_id,
                "node_type": self.node_type.name,
                "name": self.name,
                "description": self.description,
                "class": self.__class__.__name__,
                "state": self._state.name,
                "metadata": self.metadata,
                "stats": {
                    "created_time": self.stats.created_time,
                    "last_active_time": self.stats.last_active_time,
                    "processed_messages": self.stats.processed_messages,
                    "sent_messages": self.stats.sent_messages,
                    "errors": self.stats.errors,
                    "avg_processing_time": self.stats.avg_processing_time,
                    "total_processing_time": self.stats.total_processing_time
                },
                "connections": {
                    "incoming": [conn.source_id for conn in self._incoming_connections.values()],
                    "outgoing": [conn.target_id for conn in self._outgoing_connections.values()]
                }
            }
            
            return node_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseNode':
        """
        Create node from dictionary representation
        Note: This only creates the node, connections must be re-established separately
        
        Args:
            data: Dictionary representation of the node
            
        Returns:
            BaseNode: Created node
        """
        # Create node with basic properties
        node = cls(
            node_id=data.get("node_id"),
            node_type=NodeType[data.get("node_type", "PROCESSOR")],
            name=data.get("name"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )
        
        # Set state if provided
        if "state" in data:
            node.state = NodeState[data["state"]]
        
        return node


class SimpleProcessingNode(BaseNode):
    """
    A simple example processing node that demonstrates the BaseNode implementation
    
    This node applies a processing function to incoming data and forwards the result
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        process_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PROCESSOR,
            name=name,
            **kwargs
        )
        self.process_fn = process_fn or (lambda x: x)  # Default to identity function
    
    def _send_to_target(self, target_id: str, data: Any) -> None:
        """Send data to target node (expecting actual node objects)"""
        from connection_discovery import ConnectionDiscovery
        # Use the discovery service to locate the target node
        # Note: This is a placeholder implementation
        discovery = ConnectionDiscovery.get_instance()
        target_info = discovery.get_node_by_id(target_id)
        
        if target_info and hasattr(target_info, 'node'):
            target_node = target_info.node
            if hasattr(target_node, 'receive_data'):
                target_node.receive_data(self.node_id, data)
            else:
                logger.warning(f"Target node {target_id} does not have receive_data method")
        else:
            logger.warning(f"Target node {target_id} not found in discovery service")
    
    def _process_data(self) -> bool:
        """Process data from all input buffers"""
        with self._lock:
            # Check if there's any data to process
            has_data = False
            for buffer in self._input_buffer.values():
                if buffer:
                    has_data = True
                    break
            
            if not has_data:
                return False
            
            # Process data from each source
            for source_id, buffer in self._input_buffer.items():
                if not buffer:
                    continue
                
                # Get connection properties
                connection = self._incoming_connections.get(source_id)
                if not connection or not connection.enabled:
                    continue
                
                # Process each item in the buffer
                for data_item in buffer:
                    try:
                        # Apply the processing function
                        result = self.process_fn(data_item)
                        
                        # Forward to all outgoing connections
                        self.send_data(result)
                    except Exception as e:
                        logger.error(f"Error processing data in {self.name}: {str(e)}")
                        self.stats.errors += 1
                
                # Clear buffer after processing
                buffer.clear()
            
            return True


class InputNode(BaseNode):
    """
    An example input node that receives external data and forwards it into the network
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        transform_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.INPUT,
            name=name,
            **kwargs
        )
        self.transform_fn = transform_fn or (lambda x: x)  # Default to identity function
    
    def input_data(self, data: Any) -> None:
        """
        Receive external input data
        
        Args:
            data: Input data to process
        """
        if self.state != NodeState.READY and self.state != NodeState.PROCESSING:
            logger.warning(f"Input node {self.name} not ready to receive data (state: {self.state.name})")
            return
        
        try:
            # Transform the input data
            transformed_data = self.transform_fn(data)
            
            # Forward to all connected nodes
            self.send_data(transformed_data)
            
            # Update stats
            self.stats.processed_messages += 1
            self.stats.last_active_time = time.time()
            
        except Exception as e:
            logger.error(f"Error processing input data in {self.name}: {str(e)}")
            self.stats.errors += 1
    
    def _send_to_target(self, target_id: str, data: Any) -> None:
        """Send data to target node (expecting actual node objects)"""
        from connection_discovery import ConnectionDiscovery
        # Use the discovery service to locate the target node
        discovery = ConnectionDiscovery.get_instance()
        target_info = discovery.get_node_by_id(target_id)
        
        if target_info and hasattr(target_info, 'node'):
            target_node = target_info.node
            if hasattr(target_node, 'receive_data'):
                target_node.receive_data(self.node_id, data)
            else:
                logger.warning(f"Target node {target_id} does not have receive_data method")
        else:
            logger.warning(f"Target node {target_id} not found in discovery service")
    
    def _process_data(self) -> bool:
        """
        Input nodes don't process data from input buffers
        since they receive external data via input_data
        """
        return False


class OutputNode(BaseNode):
    """
    An example output node that receives network data and produces external output
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        output_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.OUTPUT,
            name=name,
            **kwargs
        )
        self.output_fn = output_fn or (lambda x: print(f"Output: {x}"))
        self._output_history: List[Any] = []
    
    def _send_to_target(self, target_id: str, data: Any) -> None:
        """Output nodes typically don't send data to other nodes"""
        logger.warning(f"Output node {self.name} attempting to send data to {target_id} (not supported)")
    
    def _process_data(self) -> bool:
        """Process incoming data and generate output"""
        with self._lock:
            # Check if there's any data to process
            has_data = False
            for buffer in self._input_buffer.values():
                if buffer:
                    has_data = True
                    break
            
            if not has_data:
                return False
            
            # Process data from each source
            for source_id, buffer in self._input_buffer.items():
                if not buffer:
                    continue
                
                # Get connection properties
                connection = self._incoming_connections.get(source_id)
                if not connection or not connection.enabled:
                    continue
                
                # Process each item in the buffer
                for data_item in buffer:
                    try:
                        # Call the output function
                        self.output_fn(data_item)
                        
                        # Store in history
                        self._output_history.append(data_item)
                        if len(self._output_history) > 100:  # Limit history size
                            self._output_history.pop(0)
                            
                    except Exception as e:
                        logger.error(f"Error generating output in {self.name}: {str(e)}")
                        self.stats.errors += 1
                
                # Clear buffer after processing
                buffer.clear()
            
            return True
    
    def get_history(self, limit: int = 10) -> List[Any]:
        """
        Get recent output history
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List[Any]: Recent output items
        """
        with self._lock:
            return self._output_history[-limit:] if self._output_history else [] 