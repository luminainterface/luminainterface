#!/usr/bin/env python3
"""
FlexNode - Simplified Implementation for Testing Neural Linguistic Bridge

This module provides a basic implementation of the FlexNode class for testing the
Neural Linguistic Flex Bridge. It simulates the core functionality needed for integration.
"""

import logging
import time
import threading
import numpy as np
from queue import Queue
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flex-node")

class NodeConnection:
    """Connection between nodes"""
    
    def __init__(self, source_id: str, target_id: str, connection_type: str = "default",
                weight: float = 0.5, bidirectional: bool = False):
        """Initialize connection"""
        self.source_id = source_id
        self.target_id = target_id
        self.connection_type = connection_type
        self.weight = weight
        self.bidirectional = bidirectional
        self.enabled = True
        self.last_used = time.time()
        self.metrics = {}

class FlexNode:
    """
    A flexible, adaptive neural network node that can dynamically connect 
    with other nodes and adjust its behavior based on the system's needs.
    
    This is a simplified implementation for testing the Neural Linguistic Bridge.
    """
    
    def __init__(self, embedding_dim: int = 256, hidden_dims: List[int] = [512, 256],
                 adaptation_rate: float = 0.01, max_connections: int = 10):
        """
        Initialize the FlexNode.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
            adaptation_rate: Rate at which the node adapts to new inputs
            max_connections: Maximum number of connections to maintain
        """
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.adaptation_rate = adaptation_rate
        self.max_connections = max_connections
        
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
        self.processing_thread = None
        
        logger.info(f"FlexNode initialized with ID: {self.node_id}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the FlexNode.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        # Simple transformation for testing
        return x * 1.2
    
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
            # Track processing
            self.processed_messages += 1
            self.last_activity = time.time()
            
            # Process based on data type
            if isinstance(data, dict):
                return self._process_dict_data(data)
            elif isinstance(data, (list, tuple)):
                return self._process_list_data(data)
            elif isinstance(data, np.ndarray):
                result = self.forward(data)
                return {"processed": True, "result": result}
            else:
                # Default simple response
                return {
                    "processed": True,
                    "result": f"Processed by {self.node_id}",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return {"error": str(e)}
        finally:
            # Update processing time metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            if self.processed_messages > 0:
                self.metrics["avg_latency"] = self.total_processing_time / self.processed_messages
    
    def _process_dict_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process dictionary data"""
        # Extract embeddings if available
        if "embedding" in data and isinstance(data["embedding"], np.ndarray):
            embedding = data["embedding"]
            # Apply simple transformation
            processed_embedding = embedding * 1.2 + 0.1
            
            result = {
                "processed": True,
                "source": self.node_id,
                "result": processed_embedding,
                "timestamp": time.time()
            }
            
            # If there was original data, include a reference
            if "original_data" in data:
                result["original_data_reference"] = "included"
                
            return result
            
        # General dict processing
        return {
            "processed": True,
            "source": self.node_id,
            "input_keys": list(data.keys()),
            "timestamp": time.time()
        }
    
    def _process_list_data(self, data: List[Any]) -> Dict[str, Any]:
        """Process list data"""
        return {
            "processed": True,
            "source": self.node_id,
            "list_length": len(data),
            "timestamp": time.time()
        }
    
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
            # Choose appropriate method to send data
            if hasattr(node, 'process'):
                node.process(data)
            elif callable(node):
                node(data)
            else:
                logger.warning(f"No suitable method found to send data to {node_id}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error sending data to node {node_id}: {str(e)}")
            return False
    
    def start(self) -> None:
        """Start the node's processing thread"""
        if self.is_active:
            logger.info("FlexNode already active")
            return
            
        self.stop_signal.clear()
        self.processing_thread = threading.Thread(
            target=self._process_queue,
            daemon=True,
            name=f"FlexNode_{self.node_id}_Thread"
        )
        self.processing_thread.start()
        self.is_active = True
        logger.info(f"FlexNode {self.node_id} started")
    
    def stop(self) -> None:
        """Stop the node's processing thread"""
        if not self.is_active:
            return
            
        self.stop_signal.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        self.is_active = False
        logger.info(f"FlexNode {self.node_id} stopped")
    
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
            "timestamp": time.time()
        }
            
        return metrics_report

def create_flex_node(embedding_dim: int = 256, hidden_dims: List[int] = [512, 256]) -> FlexNode:
    """
    Create and initialize a FlexNode instance.
    
    Args:
        embedding_dim: Dimension of input embeddings
        hidden_dims: Dimensions of hidden layers
        
    Returns:
        Initialized FlexNode instance
    """
    try:
        flex_node = FlexNode(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            adaptation_rate=0.01,
            max_connections=15
        )
        
        logger.info(f"Created FlexNode with ID: {flex_node.node_id}")
        return flex_node
    except Exception as e:
        logger.error(f"Failed to create FlexNode: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the FlexNode
    node = create_flex_node()
    node.start()
    
    # Process some data
    result = node.process({"test": "data", "embedding": np.random.randn(256)})
    print(f"Processing result: {result}")
    
    # Get metrics
    metrics = node.get_metrics()
    print(f"Metrics: {metrics}")
    
    node.stop() 