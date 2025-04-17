"""
FlexNode - Adaptive Neural Network Node
=======================================

A flexible, adaptive neural network node that can dynamically connect with other nodes
and adjust its behavior based on the system's needs.

The FlexNode is designed to:
1. Automatically discover and connect to other nodes
2. Adapt its internal architecture based on input/output requirements
3. Optimize connections between nodes for better data flow
4. Provide insights about node efficiency and interactions
5. Suggest new connections or node modifications
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import threading
from queue import Queue
from pathlib import Path
import importlib
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("flex_node.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FlexNode")

# Load environment variables
load_dotenv()

# Feature flags from environment variables
ENABLE_HYBRID_NODE = os.getenv("ENABLE_HYBRID_NODE", "true").lower() in ("true", "1", "yes")
ENABLE_LLM_INTEGRATION = os.getenv("ENABLE_LLM_INTEGRATION", "true").lower() in ("true", "1", "yes")
ENABLE_WEIGHT_ADJUSTMENT_UI = os.getenv("ENABLE_WEIGHT_ADJUSTMENT_UI", "true").lower() in ("true", "1", "yes")

@dataclass
class NodeConnection:
    """Information about a connection between nodes"""
    source_id: str
    target_id: str
    connection_type: str = "default"
    weight: float = 0.5
    bidirectional: bool = False
    enabled: bool = True
    last_used: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "connection_type": self.connection_type,
            "weight": self.weight,
            "bidirectional": self.bidirectional,
            "enabled": self.enabled,
            "last_used": self.last_used,
            "metrics": self.metrics
        }

@dataclass
class ConnectionMetrics:
    """Metrics for node connections"""
    throughput: float = 0.0  # Messages per second
    latency: float = 0.0     # Average processing time
    error_rate: float = 0.0  # Percentage of failed messages
    utilization: float = 0.0 # How often the connection is used
    efficiency: float = 0.0  # Computational efficiency score
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "throughput": self.throughput,
            "latency": self.latency,
            "error_rate": self.error_rate,
            "utilization": self.utilization,
            "efficiency": self.efficiency
        }

class AdaptiveLayer(nn.Module):
    """An adaptive neural network layer that can resize itself based on input/output requirements"""
    
    def __init__(self, input_dim: int = 256, output_dim: int = 256, adaptation_rate: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adaptation_rate = adaptation_rate
        
        # Main layer components
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()
        
        # Attention mechanism for adaptive focus
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=4, batch_first=True)
        
        # Adaptation parameters
        self.input_importance = nn.Parameter(torch.ones(input_dim))
        self.output_importance = nn.Parameter(torch.ones(output_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive attention"""
        # Apply input importance weights
        weighted_input = x * self.input_importance.softmax(dim=0)
        
        # Linear transformation
        output = self.linear(weighted_input)
        
        # Apply normalization
        output = self.norm(output)
        
        # Apply activation
        output = self.activation(output)
        
        # Apply attention if input has sequence dimension
        if output.dim() > 2:
            attn_output, _ = self.attention(output, output, output)
            output = output + attn_output
            
        # Apply output importance weights
        output = output * self.output_importance.softmax(dim=0)
        
        return output
    
    def adapt(self, input_stats: torch.Tensor, output_stats: torch.Tensor) -> None:
        """Adapt the layer based on input and output statistics"""
        # Update input importance based on input statistics
        with torch.no_grad():
            self.input_importance.data += self.adaptation_rate * input_stats
            self.output_importance.data += self.adaptation_rate * output_stats

class FlexNode(nn.Module):
    """
    A flexible, adaptive neural network node that can dynamically connect 
    with other nodes and adjust its behavior based on the system's needs.
    """
    
    def __init__(self, 
                 embedding_dim: int = 256, 
                 hidden_dims: List[int] = [512, 256],
                 adaptation_rate: float = 0.01,
                 max_connections: int = 10):
        """
        Initialize the FlexNode.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
            adaptation_rate: Rate at which the node adapts to new inputs
            max_connections: Maximum number of connections to maintain
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.adaptation_rate = adaptation_rate
        self.max_connections = max_connections
        
        # Build adaptive layers
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(AdaptiveLayer(input_dim, hidden_dim, adaptation_rate))
            input_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dims[-1], embedding_dim)
        
        # Node state and metrics
        self.node_id = f"FlexNode_{int(time.time())}"
        self.connections: Dict[str, NodeConnection] = {}
        self.connected_nodes: Dict[str, Any] = {}
        self.message_queue = Queue()
        self.processed_messages = 0
        self.total_processing_time = 0.0
        self.is_active = False
        self.last_activity = time.time()
        self.metrics = {
            "throughput": 0.0,
            "avg_latency": 0.0,
            "error_rate": 0.0,
            "utilization": 0.0
        }
        
        # Start background processing thread
        self.stop_signal = threading.Event()
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        
        # Initialize central node reference
        self.central_node = None
        
    def set_central_node(self, central_node) -> None:
        """Set reference to the central node"""
        self.central_node = central_node
        logger.info(f"Connected to central node: {central_node}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FlexNode.
        
        Args:
            x: Input tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Processed tensor of shape (batch_size, embedding_dim)
        """
        # Track input statistics for adaptation
        input_stats = x.mean(dim=0) if x.dim() > 1 else x
        
        # Process through adaptive layers
        for layer in self.layers:
            x = layer(x)
            
        # Final projection back to embedding dimension
        output = self.output_projection(x)
        
        # Track output statistics for adaptation
        output_stats = output.mean(dim=0) if output.dim() > 1 else output
        
        # Adapt layers based on statistics
        if self.training:
            for layer in self.layers:
                layer.adapt(input_stats, output_stats)
        
        return output
    
    def process(self, data: Any) -> Any:
        """
        Process input data through the node.
        
        Args:
            data: Input data which can be various types
            
        Returns:
            Processed data
        """
        start_time = time.time()
        
        try:
            # Convert input to tensor if needed
            if isinstance(data, torch.Tensor):
                x = data
            elif isinstance(data, np.ndarray):
                x = torch.from_numpy(data).float()
            elif isinstance(data, (list, tuple)) and all(isinstance(item, (int, float)) for item in data):
                x = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, dict):
                # Try to extract embeddings or text for processing
                if "embedding" in data:
                    x = torch.tensor(data["embedding"], dtype=torch.float32)
                elif "text" in data:
                    # Simple text encoding as placeholder - in real system, use proper text embeddings
                    x = torch.randn(self.embedding_dim)  # Placeholder for demo
                else:
                    # Process numeric values if available
                    numeric_data = {k: v for k, v in data.items() if isinstance(v, (int, float))}
                    if numeric_data:
                        values = list(numeric_data.values())
                        x = torch.tensor(values, dtype=torch.float32)
                        # Pad or trim to match embedding dim
                        if len(values) < self.embedding_dim:
                            padding = torch.zeros(self.embedding_dim - len(values))
                            x = torch.cat([x, padding])
                        else:
                            x = x[:self.embedding_dim]
                    else:
                        # Default random embedding if no processable data
                        x = torch.randn(self.embedding_dim)
            else:
                # For unrecognized formats, return input unmodified
                logger.warning(f"Unprocessable input type: {type(data)}")
                return data
            
            # Forward pass through the model
            with torch.no_grad():
                output = self.forward(x)
            
            # Convert output to appropriate format
            if isinstance(data, dict):
                # Return enhanced version of input with additional features
                result = data.copy()
                result["flex_node_embedding"] = output.numpy().tolist()
                result["flex_node_processed"] = True
                result["timestamp"] = time.time()
            else:
                result = output.numpy() if isinstance(output, torch.Tensor) else output
                
            # Update metrics
            self.processed_messages += 1
            self.last_activity = time.time()
            process_time = time.time() - start_time
            self.total_processing_time += process_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            # Return original data on error
            return data
    
    def connect_to_node(self, node_id: str, node_instance: Any, 
                        connection_type: str = "default",
                        weight: float = 0.5,
                        bidirectional: bool = False) -> bool:
        """
        Connect this FlexNode to another node.
        
        Args:
            node_id: Unique identifier for the target node
            node_instance: The actual node object
            connection_type: Type of connection (default, data, control, etc.)
            weight: Initial connection weight
            bidirectional: Whether the connection is bidirectional
            
        Returns:
            True if connection was successful, False otherwise
        """
        if node_id in self.connections:
            logger.warning(f"Already connected to node {node_id}")
            return False
        
        # Create connection
        connection = NodeConnection(
            source_id=self.node_id,
            target_id=node_id,
            connection_type=connection_type,
            weight=weight,
            bidirectional=bidirectional
        )
        
        # Store connection and node
        self.connections[node_id] = connection
        self.connected_nodes[node_id] = node_instance
        
        logger.info(f"Connected to node {node_id} with {connection_type} connection")
        
        # If bidirectional, check if target node has a connect_to_node method
        if bidirectional and hasattr(node_instance, 'connect_to_node'):
            try:
                node_instance.connect_to_node(
                    self.node_id, 
                    self, 
                    connection_type, 
                    weight, 
                    False  # Prevent infinite recursion
                )
            except Exception as e:
                logger.error(f"Error creating bidirectional connection: {str(e)}")
        
        return True
    
    def send_to_node(self, node_id: str, data: Any) -> bool:
        """Send data to a connected node"""
        if node_id not in self.connections or not self.connections[node_id].enabled:
            logger.warning(f"Node {node_id} is not connected or enabled")
            return False
        
        node = self.connected_nodes.get(node_id)
        if not node:
            logger.warning(f"Node instance for {node_id} not found")
            return False
        
        try:
            # Update connection metrics
            connection = self.connections[node_id]
            connection.last_used = time.time()
            
            # Choose appropriate method to send data
            if hasattr(node, 'process'):
                node.process(data)
            elif hasattr(node, 'forward') and isinstance(data, torch.Tensor):
                node.forward(data)
            elif callable(node):
                node(data)
            else:
                logger.warning(f"No suitable method found to send data to {node_id}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error sending data to node {node_id}: {str(e)}")
            # Update error rate in connection metrics
            if 'error_rate' not in self.connections[node_id].metrics:
                self.connections[node_id].metrics['error_rate'] = 0.0
            self.connections[node_id].metrics['error_rate'] += 1.0
            return False
    
    def receive_message(self, source_id: str, data: Any) -> None:
        """Receive message from another node and queue for processing"""
        self.message_queue.put((source_id, data, time.time()))
        
    def _process_queue(self) -> None:
        """Background thread for processing queued messages"""
        while not self.stop_signal.is_set():
            try:
                if not self.message_queue.empty():
                    source_id, data, timestamp = self.message_queue.get(timeout=0.1)
                    
                    # Process the data
                    result = self.process(data)
                    
                    # Update metrics
                    latency = time.time() - timestamp
                    if source_id in self.connections:
                        conn = self.connections[source_id]
                        if 'latency' not in conn.metrics:
                            conn.metrics['latency'] = latency
                        else:
                            # Moving average
                            conn.metrics['latency'] = 0.9 * conn.metrics['latency'] + 0.1 * latency
                    
                    # Mark message as processed
                    self.message_queue.task_done()
                else:
                    # Sleep to prevent CPU spinning
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in message processing thread: {str(e)}")
    
    def start(self) -> None:
        """Start the node's processing thread"""
        if not self.is_active:
            self.stop_signal.clear()
            self.processing_thread.start()
            self.is_active = True
            logger.info(f"FlexNode {self.node_id} started")
    
    def stop(self) -> None:
        """Stop the node's processing thread"""
        if self.is_active:
            self.stop_signal.set()
            self.processing_thread.join(timeout=2.0)
            self.is_active = False
            logger.info(f"FlexNode {self.node_id} stopped")
    
    def optimize_connections(self) -> Dict[str, Any]:
        """
        Analyze and optimize the node's connections based on usage and performance metrics.
        Returns a report of changes made.
        """
        if not self.connections:
            return {"status": "no_connections", "changes": []}
        
        changes = []
        
        # Sort connections by activity (most recently used first)
        sorted_connections = sorted(
            self.connections.items(), 
            key=lambda x: x[1].last_used, 
            reverse=True
        )
        
        # Prune least used connections if we're over the limit
        if len(sorted_connections) > self.max_connections:
            for node_id, _ in sorted_connections[self.max_connections:]:
                changes.append({
                    "action": "removed",
                    "node_id": node_id,
                    "reason": "exceeded_max_connections"
                })
                del self.connections[node_id]
                if node_id in self.connected_nodes:
                    del self.connected_nodes[node_id]
        
        # Adjust weights based on usage patterns
        for node_id, connection in self.connections.items():
            # Calculate time since last use
            time_since_use = time.time() - connection.last_used
            
            # Adjust weight based on recency of use
            if time_since_use < 60:  # Used in the last minute
                new_weight = min(connection.weight * 1.05, 1.0)
            elif time_since_use > 3600:  # Not used in the last hour
                new_weight = max(connection.weight * 0.95, 0.1)
            else:
                new_weight = connection.weight
                
            # Apply the weight change if significant
            if abs(new_weight - connection.weight) > 0.01:
                changes.append({
                    "action": "weight_adjusted",
                    "node_id": node_id,
                    "old_weight": connection.weight,
                    "new_weight": new_weight,
                    "reason": "usage_pattern"
                })
                connection.weight = new_weight
        
        # Return optimization report
        return {
            "status": "success",
            "changes": changes,
            "timestamp": time.time(),
            "connections_count": len(self.connections)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get node performance metrics"""
        # Update overall metrics
        if self.processed_messages > 0:
            self.metrics["avg_latency"] = self.total_processing_time / self.processed_messages
        
        current_time = time.time()
        uptime = current_time - self.last_activity
        self.metrics["utilization"] = min(1.0, self.processed_messages / max(1, uptime))
        
        # Prepare full metrics report
        metrics_report = {
            "node_id": self.node_id,
            "processed_messages": self.processed_messages,
            "avg_processing_time": self.metrics["avg_latency"],
            "utilization": self.metrics["utilization"],
            "active_connections": len(self.connections),
            "last_activity": self.last_activity,
            "connections": {}
        }
        
        # Add connection-specific metrics
        for node_id, connection in self.connections.items():
            metrics_report["connections"][node_id] = {
                "weight": connection.weight,
                "last_used": connection.last_used,
                "metrics": connection.metrics
            }
            
        return metrics_report
    
    def save_state(self, filepath: str = None) -> str:
        """Save the node's state to a file"""
        if filepath is None:
            filepath = f"flex_node_{self.node_id}_state.json"
            
        try:
            # Prepare state dictionary
            state = {
                "node_id": self.node_id,
                "embedding_dim": self.embedding_dim,
                "hidden_dims": self.hidden_dims,
                "adaptation_rate": self.adaptation_rate,
                "max_connections": self.max_connections,
                "metrics": self.metrics,
                "processed_messages": self.processed_messages,
                "last_activity": self.last_activity,
                "connections": {
                    node_id: connection.to_dict() 
                    for node_id, connection in self.connections.items()
                },
                "timestamp": time.time()
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            # Also save model weights
            weights_path = filepath.replace('.json', '_weights.pt')
            torch.save(self.state_dict(), weights_path)
            
            logger.info(f"Saved FlexNode state to {filepath} and weights to {weights_path}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            return None
    
    def load_state(self, filepath: str) -> bool:
        """Load the node's state from a file"""
        try:
            # Load state dictionary
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            # Update node properties
            self.node_id = state.get("node_id", self.node_id)
            self.embedding_dim = state.get("embedding_dim", self.embedding_dim)
            self.hidden_dims = state.get("hidden_dims", self.hidden_dims)
            self.adaptation_rate = state.get("adaptation_rate", self.adaptation_rate)
            self.max_connections = state.get("max_connections", self.max_connections)
            self.metrics = state.get("metrics", self.metrics)
            self.processed_messages = state.get("processed_messages", self.processed_messages)
            self.last_activity = state.get("last_activity", self.last_activity)
            
            # Load connections (note: connected_nodes must be reconnected separately)
            self.connections = {}
            for node_id, conn_dict in state.get("connections", {}).items():
                self.connections[node_id] = NodeConnection(
                    source_id=conn_dict["source_id"],
                    target_id=conn_dict["target_id"],
                    connection_type=conn_dict["connection_type"],
                    weight=conn_dict["weight"],
                    bidirectional=conn_dict["bidirectional"],
                    enabled=conn_dict.get("enabled", True),
                    last_used=conn_dict.get("last_used", time.time()),
                    metrics=conn_dict.get("metrics", {})
                )
                
            # Load model weights if available
            weights_path = filepath.replace('.json', '_weights.pt')
            if os.path.exists(weights_path):
                self.load_state_dict(torch.load(weights_path))
                logger.info(f"Loaded FlexNode weights from {weights_path}")
                
            logger.info(f"Loaded FlexNode state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False
    
    def scan_for_nodes(self) -> List[str]:
        """
        Scan for available FlexNodes in the system.
        
        Returns:
            List[str]: List of node IDs that were discovered
        """
        discovered_nodes = []
        
        # Try multiple discovery methods to maximize node discovery
        try:
            # Method 1: Scan through registered packages (new enhanced method)
            node_packages = ["src", "src.v7", "src.v7.lumina_v7", "src.v7.lumina_v7.core", "src.v7.lumina_v7.nodes"]
            
            for package_name in node_packages:
                try:
                    package = importlib.import_module(package_name)
                    for module_name in dir(package):
                        # Skip private modules
                        if module_name.startswith('_'):
                            continue
                            
                        try:
                            # Try to import the module
                            module_path = f"{package_name}.{module_name}"
                            module = importlib.import_module(module_path)
                            
                            # Look for FlexNode or compatible classes
                            for attr_name in dir(module):
                                try:
                                    attr = getattr(module, attr_name)
                                    
                                    # Check if it's a class with node-like properties
                                    if (isinstance(attr, type) and 
                                        (hasattr(attr, 'node_id') or 
                                         hasattr(attr, 'receive_message') or
                                         hasattr(attr, 'connect_to_node'))):
                                        
                                        # Try to get the node_id
                                        if hasattr(attr, 'get_node_id'):
                                            node_id = attr.get_node_id()
                                            discovered_nodes.append(node_id)
                                            logger.info(f"Discovered node via package scanning: {node_id}")
                                except Exception as e:
                                    logger.debug(f"Error examining attribute {attr_name} in module {module_path}: {str(e)}")
                        except Exception as e:
                            logger.debug(f"Error importing module {module_name} from package {package_name}: {str(e)}")
                except ImportError:
                    logger.debug(f"Package {package_name} not found, skipping")
                
            # Method 2: Use the node consciousness manager if available
            try:
                from src.v7.lumina_v7.core.node_consciousness_manager import NodeConsciousnessManager
                
                # Try to get a global instance or create a temporary one
                try:
                    # Check if there's a global instance
                    manager = None
                    
                    # Look in various potential module locations
                    for module_name in ["src.v7.node_consciousness", "src.v7.lumina_v7.core.initialization"]:
                        try:
                            module = importlib.import_module(module_name)
                            if hasattr(module, "get_node_manager"):
                                manager = module.get_node_manager()
                                break
                        except ImportError:
                            pass
                    
                    # If no global instance, create a temporary one
                    if not manager:
                        manager = NodeConsciousnessManager()
                    
                    # Get all nodes from the manager
                    node_statuses = manager.get_system_status().get("node_statuses", {})
                    for node_id, status in node_statuses.items():
                        discovered_nodes.append(node_id)
                        logger.info(f"Discovered node via node consciousness manager: {node_id}")
                except Exception as e:
                    logger.debug(f"Error getting nodes from manager: {str(e)}")
                
            except ImportError:
                logger.debug("NodeConsciousnessManager not found, skipping that discovery method")
            
            # Method 3: Look for local node registry file
            registry_paths = [
                "node_registry.json",
                "data/node_registry.json",
                "config/node_registry.json",
                os.path.join(os.path.dirname(__file__), "node_registry.json")
            ]
            
            for registry_path in registry_paths:
                if os.path.exists(registry_path):
                    try:
                        with open(registry_path, 'r') as f:
                            registry = json.load(f)
                            
                        for node_data in registry.get("nodes", []):
                            node_id = node_data.get("id")
                            if node_id:
                                discovered_nodes.append(node_id)
                                logger.info(f"Discovered node via registry file: {node_id}")
                    except Exception as e:
                        logger.debug(f"Error reading node registry file {registry_path}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error during node discovery: {str(e)}")
        
        # Remove duplicates while preserving order
        unique_nodes = []
        for node_id in discovered_nodes:
            if node_id not in unique_nodes:
                unique_nodes.append(node_id)
        
        logger.info(f"Node discovery complete. Found {len(unique_nodes)} unique nodes.")
        return unique_nodes

# Test the FlexNode if run directly
if __name__ == "__main__":
    # Create a FlexNode
    node = FlexNode(embedding_dim=128, hidden_dims=[256, 128])
    
    # Test processing with different input types
    test_tensor = torch.randn(128)
    result_tensor = node.process(test_tensor)
    print(f"Processed tensor shape: {result_tensor.shape}")
    
    test_dict = {"value": 0.5, "text": "Test input"}
    result_dict = node.process(test_dict)
    print(f"Processed dict: {result_dict.keys()}")
    
    # Test connection functionality
    dummy_node = FlexNode(embedding_dim=64)
    node.connect_to_node("dummy", dummy_node, "test", 0.8, True)
    
    # Test metrics
    print(node.get_metrics())
    
    # Test optimization
    print(node.optimize_connections())
    
    # Test save/load state
    state_path = node.save_state()
    if state_path:
        new_node = FlexNode()
        new_node.load_state(state_path)
        print(f"State loaded successfully: {new_node.node_id}") 