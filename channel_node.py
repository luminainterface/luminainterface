"""
Channel Node - Data Flow Optimization for Neural Networks
========================================================

The ChannelNode provides specialized data channels between neural network nodes,
optimizing data flow by type, priority, and transformation requirements.

Features:
1. Multiple specialized channels for different data types
2. Priority-based message routing
3. Data transformation and normalization
4. Channel monitoring and metrics
5. Adaptive throughput adjustment
"""

import os
import logging
import torch
import numpy as np
import json
import time
import threading
from queue import Queue, PriorityQueue
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("channel_node.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChannelNode")

class ChannelType(Enum):
    """Types of specialized channels"""
    DEFAULT = "default"
    TENSOR = "tensor"          # Raw tensor data
    EMBEDDING = "embedding"    # Vector embeddings
    TEXT = "text"              # Text data
    CONTROL = "control"        # Control signals
    FEEDBACK = "feedback"      # Feedback information
    QUANTUM = "quantum"        # Quantum-related data
    LOG = "log"                # Logging channel
    PRIORITY = "priority"      # High-priority messages

class MessagePriority(Enum):
    """Priority levels for messages"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Message:
    """Message passed between nodes through channels"""
    source_id: str
    target_id: str
    data: Any
    channel_type: ChannelType = ChannelType.DEFAULT
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "data_type": str(type(self.data)),
            "channel_type": self.channel_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

@dataclass
class ChannelMetrics:
    """Metrics for a specific channel"""
    message_count: int = 0
    bytes_transferred: int = 0
    avg_latency: float = 0.0
    peak_throughput: float = 0.0
    error_count: int = 0
    last_activity: float = field(default_factory=time.time)
    
    def update_latency(self, new_latency: float) -> None:
        """Update average latency with new measurement"""
        if self.message_count == 0:
            self.avg_latency = new_latency
        else:
            # Exponential moving average
            self.avg_latency = 0.9 * self.avg_latency + 0.1 * new_latency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "message_count": self.message_count,
            "bytes_transferred": self.bytes_transferred,
            "avg_latency": self.avg_latency,
            "peak_throughput": self.peak_throughput,
            "error_count": self.error_count,
            "last_activity": self.last_activity
        }

class Channel:
    """A specialized communication channel between nodes"""
    
    def __init__(self, channel_type: ChannelType):
        self.channel_type = channel_type
        self.queue = PriorityQueue()  # (priority, timestamp, message)
        self.metrics = ChannelMetrics()
        self.subscribers = set()  # Node IDs subscribed to this channel
        self.transformers = {}    # Node ID -> transformer function
        self.filters = {}         # Node ID -> filter function
        self.active = True
        self.lock = threading.Lock()
    
    def send(self, message: Message) -> bool:
        """Send a message through the channel"""
        if not self.active:
            logger.warning(f"Channel {self.channel_type.value} is inactive")
            return False
            
        try:
            with self.lock:
                # Prioritize by message priority, then timestamp (older first)
                priority_tuple = (
                    -message.priority.value,  # Negative so higher priority comes first
                    message.timestamp
                )
                self.queue.put((priority_tuple, message))
                
                # Update metrics
                self.metrics.message_count += 1
                try:
                    # Estimate size of data
                    if isinstance(message.data, (np.ndarray, torch.Tensor)):
                        self.metrics.bytes_transferred += message.data.nbytes
                    elif isinstance(message.data, (str, bytes)):
                        self.metrics.bytes_transferred += len(message.data)
                    elif isinstance(message.data, dict):
                        self.metrics.bytes_transferred += len(json.dumps(message.data))
                except:
                    # If we can't calculate size, just increment by 1
                    self.metrics.bytes_transferred += 1
                    
                self.metrics.last_activity = time.time()
                return True
        except Exception as e:
            logger.error(f"Error sending message through channel {self.channel_type.value}: {str(e)}")
            self.metrics.error_count += 1
            return False
    
    def receive(self, timeout: float = 0.1) -> Optional[Message]:
        """Receive the next message from the channel"""
        if not self.active:
            return None
            
        try:
            # Get next message without blocking for too long
            try:
                _, message = self.queue.get(timeout=timeout)
                return message
            except:
                return None
        except Exception as e:
            logger.error(f"Error receiving message from channel {self.channel_type.value}: {str(e)}")
            self.metrics.error_count += 1
            return None
    
    def subscribe(self, node_id: str) -> bool:
        """Subscribe a node to this channel"""
        with self.lock:
            self.subscribers.add(node_id)
            logger.info(f"Node {node_id} subscribed to channel {self.channel_type.value}")
            return True
    
    def unsubscribe(self, node_id: str) -> bool:
        """Unsubscribe a node from this channel"""
        with self.lock:
            if node_id in self.subscribers:
                self.subscribers.remove(node_id)
                logger.info(f"Node {node_id} unsubscribed from channel {self.channel_type.value}")
                return True
            return False
    
    def add_transformer(self, node_id: str, transformer_func: Callable) -> bool:
        """Add a transformer function for a specific node"""
        with self.lock:
            self.transformers[node_id] = transformer_func
            logger.info(f"Added transformer for node {node_id} on channel {self.channel_type.value}")
            return True
    
    def add_filter(self, node_id: str, filter_func: Callable) -> bool:
        """Add a filter function for a specific node"""
        with self.lock:
            self.filters[node_id] = filter_func
            logger.info(f"Added filter for node {node_id} on channel {self.channel_type.value}")
            return True
    
    def transform_for_node(self, node_id: str, data: Any) -> Any:
        """Apply node-specific transformation to data"""
        transformer = self.transformers.get(node_id)
        if transformer and callable(transformer):
            try:
                return transformer(data)
            except Exception as e:
                logger.error(f"Error in transformer for node {node_id}: {str(e)}")
        return data
    
    def should_send_to_node(self, node_id: str, message: Message) -> bool:
        """Check if message should be sent to the node based on filters"""
        # Check if node is subscribed
        if node_id not in self.subscribers:
            return False
            
        # Apply filter if it exists
        filter_func = self.filters.get(node_id)
        if filter_func and callable(filter_func):
            try:
                return filter_func(message)
            except Exception as e:
                logger.error(f"Error in filter for node {node_id}: {str(e)}")
                return False
                
        # By default, send to subscribers
        return True
    
    def get_queue_size(self) -> int:
        """Get the current number of messages in the queue"""
        return self.queue.qsize()
    
    def is_active(self) -> bool:
        """Check if the channel is active"""
        return self.active
    
    def activate(self) -> None:
        """Activate the channel"""
        with self.lock:
            self.active = True
            logger.info(f"Channel {self.channel_type.value} activated")
    
    def deactivate(self) -> None:
        """Deactivate the channel"""
        with self.lock:
            self.active = False
            logger.info(f"Channel {self.channel_type.value} deactivated")

class ChannelNode:
    """
    Node that manages multiple specialized channels for optimized data flow
    between neural network nodes.
    """
    
    def __init__(self, max_channels: int = 10):
        """
        Initialize the ChannelNode.
        
        Args:
            max_channels: Maximum number of channels to maintain
        """
        self.node_id = f"ChannelNode_{int(time.time())}"
        self.max_channels = max_channels
        
        # Initialize default channels
        self.channels: Dict[ChannelType, Channel] = {
            channel_type: Channel(channel_type)
            for channel_type in ChannelType
        }
        
        # Node registrations
        self.registered_nodes: Dict[str, Dict[str, Any]] = {}
        
        # Message routing rules
        self.routing_rules: Dict[str, Dict[str, ChannelType]] = {}
        
        # Channel processing threads
        self.processors: Dict[ChannelType, threading.Thread] = {}
        self.stop_event = threading.Event()
        
        # Node reference for central node if available
        self.central_node = None
        
        # Start processing threads for each channel
        self._start_processors()
        
        logger.info(f"ChannelNode initialized with {len(self.channels)} channels")
    
    def set_central_node(self, central_node) -> None:
        """Set reference to central node"""
        self.central_node = central_node
        logger.info(f"Connected to central node")
    
    def _start_processors(self) -> None:
        """Start processing threads for all channels"""
        for channel_type, channel in self.channels.items():
            processor = threading.Thread(
                target=self._process_channel,
                args=(channel_type,),
                daemon=True
            )
            processor.start()
            self.processors[channel_type] = processor
            logger.info(f"Started processor for channel {channel_type.value}")
    
    def _process_channel(self, channel_type: ChannelType) -> None:
        """Process messages from a specific channel"""
        channel = self.channels.get(channel_type)
        if not channel or not channel.is_active():
            return
        
        # Process messages in priority order
        message = channel.receive(timeout=0.01)
        if not message:
            return
        
        # Get start time for latency calculation
        start_time = time.time()
        
        # Check if target is registered
        if message.target_id == 'broadcast':
            # Handle broadcast messages
            delivery_count = 0
            error_count = 0
            
            # Build a list of recipients, excluding the source
            recipients = []
            for node_id in self.registered_nodes:
                if node_id != message.source_id and node_id in self.registered_nodes:
                    recipients.append((node_id, self.registered_nodes[node_id]))
                    
            # Sort recipients by priority
            # Priority is determined by:
            # 1. Critical nodes (marked in metadata)
            # 2. Nodes that specifically subscribe to this message type
            # 3. Nodes with active transformers for this channel
            def get_node_priority(recipient):
                node_id, node_instance = recipient
                
                # Start with base priority
                priority = 0
                
                # Check if node is marked critical
                if self.node_metadata.get(node_id, {}).get("critical", False):
                    priority += 100
                    
                # Check for message type subscription
                message_type = message.metadata.get("type", "default")
                if message_type in self.node_metadata.get(node_id, {}).get("subscribed_types", []):
                    priority += 50
                    
                # Check for active transformer
                if node_id in channel.transformers:
                    priority += 25
                    
                # Add priority from node metadata
                priority += self.node_metadata.get(node_id, {}).get("priority", 0)
                
                return priority
                
            # Sort recipients by priority (highest first)
            recipients.sort(key=get_node_priority, reverse=True)
            
            # Process each recipient
            for node_id, node_instance in recipients:
                if node_id in self.subscriptions.get(channel_type.value, []):
                    try:
                        # Check if node should receive this message
                        if channel.should_send_to_node(node_id, message):
                            # Deliver the message
                            if self._deliver_message_to_node(node_id, node_instance, message):
                                delivery_count += 1
                            else:
                                error_count += 1
                    except Exception as e:
                        logger.error(f"Error delivering broadcast message to {node_id}: {str(e)}")
                        error_count += 1
            
            # Log results
            if delivery_count > 0:
                logger.debug(f"Broadcast message delivered to {delivery_count} nodes (errors: {error_count})")
            else:
                logger.warning(f"Broadcast message not delivered to any nodes (errors: {error_count})")
        
        else:
            # Handle directed messages
            target_id = message.target_id
            
            # Check if the target is registered
            if target_id in self.registered_nodes:
                node_instance = self.registered_nodes[target_id]
                
                # Check if the target is subscribed
                if target_id in self.subscriptions.get(channel_type.value, []):
                    try:
                        # Check if node should receive this message
                        if channel.should_send_to_node(target_id, message):
                            # Apply priority-based thread allocation
                            if message.priority == MessagePriority.CRITICAL:
                                # Process critical messages immediately in the current thread
                                success = self._deliver_message_to_node(target_id, node_instance, message)
                                if not success:
                                    logger.error(f"Failed to deliver critical message to {target_id}")
                            
                            elif message.priority == MessagePriority.HIGH:
                                # Process high priority messages in a separate thread for quick handling
                                thread = threading.Thread(
                                    target=self._deliver_message_to_node,
                                    args=(target_id, node_instance, message),
                                    name=f"High-Priority-{target_id}"
                                )
                                thread.daemon = True
                                thread.start()
                                
                            else:
                                # Process normal and low priority messages normally
                                success = self._deliver_message_to_node(target_id, node_instance, message)
                                if not success and message.priority != MessagePriority.LOW:
                                    logger.warning(f"Failed to deliver message to {target_id}")
                                    
                    except Exception as e:
                        logger.error(f"Error delivering message to {target_id}: {str(e)}")
                        
                else:
                    logger.warning(f"Target {target_id} is not subscribed to channel {channel_type.value}")
                    
            else:
                # Target node not found - check for fallback routing
                if target_id in self.routing_rules:
                    new_target = self.routing_rules[target_id].get(channel_type.value)
                    if new_target:
                        # Reroute the message
                        logger.debug(f"Rerouting message from {target_id} to {new_target}")
                        message.target_id = new_target
                        channel.send(message)
                else:
                    logger.warning(f"Target node {target_id} not registered")
        
        # Calculate and update latency
        latency = time.time() - start_time
        channel.metrics.update_latency(latency)
    
    def _deliver_message_to_node(self, 
                             node_id: str, 
                             node_instance: Any, 
                             message: Message) -> bool:
        """
        Deliver a message to a specific node with priority handling
        
        Args:
            node_id: ID of the target node
            node_instance: Instance of the target node
            message: Message to deliver
            
        Returns:
            bool: True if message was delivered successfully
        """
        
        # Transform data if needed
        data = message.data
        try:
            # Get channel
            channel = self.channels.get(message.channel_type)
            if channel:
                # Apply transformation
                data = channel.transform_for_node(node_id, data)
        except Exception as e:
            logger.error(f"Error transforming data for node {node_id}: {str(e)}")
            
        # Attempt delivery based on node capabilities
        try:
            # Check delivery method based on node's reception capabilities
            if hasattr(node_instance, 'receive_channel_message'):
                # Full message reception capability
                return node_instance.receive_channel_message(message.source_id, data, message.channel_type.value, message)
            
            elif hasattr(node_instance, 'receive_message'):
                # Basic message reception
                return node_instance.receive_message(message.source_id, data)
                
            elif hasattr(node_instance, 'process'):
                # Process-style reception
                if message.priority in [MessagePriority.HIGH, MessagePriority.CRITICAL]:
                    # For high priority messages, wait for processing to complete
                    result = node_instance.process(data)
                    
                    # Store result if message has ID for later retrieval
                    if 'id' in message.metadata:
                        self.process_results[message.metadata['id']] = result
                        
                    return True
                else:
                    # For normal/low priority, don't wait for result
                    threading.Thread(
                        target=lambda: node_instance.process(data),
                        name=f"Process-{node_id}-{message.metadata.get('id', 'unknown')}"
                    ).start()
                    return True
                
            else:
                logger.warning(f"Node {node_id} has no compatible message reception method")
                return False
            
        except Exception as e:
            logger.error(f"Error delivering message to node {node_id}: {str(e)}")
            return False
    
    def register_node(self, 
                    node_id: str, 
                    node_instance: Any, 
                    channels: List[ChannelType] = None) -> bool:
        """
        Register a node with the ChannelNode.
        
        Args:
            node_id: Unique identifier for the node
            node_instance: The actual node object
            channels: List of channels to subscribe to (defaults to [ChannelType.DEFAULT])
            
        Returns:
            True if registration was successful, False otherwise
        """
        if node_id in self.registered_nodes:
            logger.warning(f"Node {node_id} already registered")
            return False
        
        # Default to DEFAULT channel if not specified
        if not channels:
            channels = [ChannelType.DEFAULT]
            
        # Register the node
        self.registered_nodes[node_id] = {
            "instance": node_instance,
            "registered_at": time.time(),
            "subscribed_channels": []
        }
        
        # Subscribe to channels
        for channel_type in channels:
            channel = self.channels.get(channel_type)
            if channel:
                channel.subscribe(node_id)
                self.registered_nodes[node_id]["subscribed_channels"].append(channel_type)
        
        logger.info(f"Registered node {node_id} with {len(channels)} channels")
        return True
    
    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the ChannelNode.
        
        Args:
            node_id: ID of the node to unregister
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        if node_id not in self.registered_nodes:
            logger.warning(f"Node {node_id} not registered")
            return False
            
        # Unsubscribe from all channels
        for channel in self.channels.values():
            channel.unsubscribe(node_id)
            
        # Remove routing rules
        if node_id in self.routing_rules:
            del self.routing_rules[node_id]
            
        # Remove node registration
        del self.registered_nodes[node_id]
        
        logger.info(f"Unregistered node {node_id}")
        return True
    
    def send_message(self, 
                   source_id: str, 
                   target_id: str, 
                   data: Any,
                   channel_type: Union[ChannelType, str] = ChannelType.DEFAULT,
                   priority: Union[MessagePriority, int] = MessagePriority.NORMAL,
                   metadata: Dict[str, Any] = None) -> bool:
        """
        Send a message from one node to another through appropriate channel.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            data: The data to send
            channel_type: The channel to use
            priority: Message priority
            metadata: Additional message metadata
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        # Convert string to enum if needed
        if isinstance(channel_type, str):
            try:
                channel_type = ChannelType(channel_type)
            except ValueError:
                logger.warning(f"Invalid channel type: {channel_type}, using DEFAULT")
                channel_type = ChannelType.DEFAULT
                
        # Convert int to enum if needed
        if isinstance(priority, int):
            priority_values = [p.value for p in MessagePriority]
            if priority in priority_values:
                priority = MessagePriority(priority)
            else:
                priority = MessagePriority.NORMAL
        
        # Check if we should override the channel based on routing rules
        if source_id in self.routing_rules and target_id in self.routing_rules[source_id]:
            channel_type = self.routing_rules[source_id][target_id]
            
        # Get the appropriate channel
        channel = self.channels.get(channel_type)
        if not channel:
            logger.error(f"Channel {channel_type.value} not found")
            return False
            
        # Create and send the message
        message = Message(
            source_id=source_id,
            target_id=target_id,
            data=data,
            channel_type=channel_type,
            priority=priority,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        return channel.send(message)
    
    def broadcast(self, 
                source_id: str, 
                data: Any,
                channel_type: ChannelType = ChannelType.DEFAULT,
                priority: MessagePriority = MessagePriority.NORMAL,
                metadata: Dict[str, Any] = None) -> int:
        """
        Broadcast a message to all subscribers of a channel.
        
        Args:
            source_id: ID of the source node
            data: The data to broadcast
            channel_type: The channel to use
            priority: Message priority
            metadata: Additional message metadata
            
        Returns:
            Number of nodes the message was sent to
        """
        channel = self.channels.get(channel_type)
        if not channel:
            logger.error(f"Channel {channel_type.value} not found")
            return 0
            
        sent_count = 0
        
        # Send to each subscriber
        for node_id in channel.subscribers:
            if self.send_message(
                source_id=source_id,
                target_id=node_id,
                data=data,
                channel_type=channel_type,
                priority=priority,
                metadata=metadata
            ):
                sent_count += 1
                
        logger.info(f"Broadcast from {source_id} to {sent_count} nodes on channel {channel_type.value}")
        return sent_count
    
    def add_routing_rule(self, 
                       source_id: str, 
                       target_id: str, 
                       channel_type: ChannelType) -> bool:
        """
        Add a routing rule for messages between specific nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            channel_type: The channel to use for messages between these nodes
            
        Returns:
            True if the rule was added successfully, False otherwise
        """
        if source_id not in self.routing_rules:
            self.routing_rules[source_id] = {}
            
        self.routing_rules[source_id][target_id] = channel_type
        logger.info(f"Added routing rule: {source_id} -> {target_id} via {channel_type.value}")
        return True
    
    def remove_routing_rule(self, source_id: str, target_id: str) -> bool:
        """
        Remove a routing rule.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            True if the rule was removed successfully, False otherwise
        """
        if source_id in self.routing_rules and target_id in self.routing_rules[source_id]:
            del self.routing_rules[source_id][target_id]
            logger.info(f"Removed routing rule: {source_id} -> {target_id}")
            
            # Clean up empty dictionaries
            if not self.routing_rules[source_id]:
                del self.routing_rules[source_id]
                
            return True
            
        return False
    
    def add_transformer(self, 
                      channel_type: ChannelType, 
                      node_id: str, 
                      transformer: Callable) -> bool:
        """
        Add a data transformer for a specific channel and node.
        
        Args:
            channel_type: The channel to add the transformer to
            node_id: ID of the node the transformer applies to
            transformer: Function that transforms data for the node
            
        Returns:
            True if the transformer was added successfully, False otherwise
        """
        channel = self.channels.get(channel_type)
        if not channel:
            logger.error(f"Channel {channel_type.value} not found")
            return False
            
        return channel.add_transformer(node_id, transformer)
    
    def add_filter(self, 
                 channel_type: ChannelType, 
                 node_id: str, 
                 filter_func: Callable) -> bool:
        """
        Add a message filter for a specific channel and node.
        
        Args:
            channel_type: The channel to add the filter to
            node_id: ID of the node the filter applies to
            filter_func: Function that determines which messages the node receives
            
        Returns:
            True if the filter was added successfully, False otherwise
        """
        channel = self.channels.get(channel_type)
        if not channel:
            logger.error(f"Channel {channel_type.value} not found")
            return False
            
        return channel.add_filter(node_id, filter_func)
    
    def get_channel_metrics(self, channel_type: ChannelType = None) -> Dict[str, Any]:
        """
        Get metrics for a specific channel or all channels.
        
        Args:
            channel_type: The specific channel to get metrics for, or None for all channels
            
        Returns:
            Dictionary of channel metrics
        """
        if channel_type:
            channel = self.channels.get(channel_type)
            if not channel:
                return {}
                
            return {
                "channel_type": channel_type.value,
                "metrics": channel.metrics.to_dict(),
                "subscribers": len(channel.subscribers),
                "queue_size": channel.get_queue_size(),
                "active": channel.is_active()
            }
            
        # Get metrics for all channels
        metrics = {}
        for ch_type, channel in self.channels.items():
            metrics[ch_type.value] = {
                "metrics": channel.metrics.to_dict(),
                "subscribers": len(channel.subscribers),
                "queue_size": channel.get_queue_size(),
                "active": channel.is_active()
            }
            
        return metrics
    
    def get_node_registrations(self) -> Dict[str, Any]:
        """
        Get information about registered nodes.
        
        Returns:
            Dictionary with node registration information
        """
        reg_info = {}
        
        for node_id, info in self.registered_nodes.items():
            reg_info[node_id] = {
                "registered_at": info["registered_at"],
                "subscribed_channels": [ch.value for ch in info["subscribed_channels"]],
                "has_transformer": any(
                    node_id in channel.transformers 
                    for channel in self.channels.values()
                ),
                "has_filter": any(
                    node_id in channel.filters 
                    for channel in self.channels.values()
                )
            }
            
        return reg_info
    
    def shutdown(self) -> None:
        """Shutdown the ChannelNode and stop all threads"""
        logger.info("Shutting down ChannelNode")
        
        # Set stop event to signal threads to exit
        self.stop_event.set()
        
        # Wait for processor threads to finish
        for channel_type, thread in self.processors.items():
            thread.join(timeout=1.0)
            logger.info(f"Processor for channel {channel_type.value} stopped")
            
        # Deactivate all channels
        for channel in self.channels.values():
            channel.deactivate()
            
        logger.info("ChannelNode shutdown complete")

    def process(self, data: Any) -> Dict[str, Any]:
        """
        Process arbitrary data input (compatibility with Node interface).
        
        Args:
            data: Input data to process
            
        Returns:
            Processing result
        """
        # This method exists for compatibility with the general Node interface
        try:
            if isinstance(data, dict):
                # Try to process structured data
                result = {
                    "status": "processed",
                    "channel_count": len(self.channels),
                    "node_count": len(self.registered_nodes),
                    "timestamp": time.time()
                }
                
                # Handle commands if present
                if "command" in data:
                    command = data["command"]
                    
                    if command == "get_metrics":
                        result["metrics"] = self.get_channel_metrics()
                    elif command == "get_nodes":
                        result["nodes"] = self.get_node_registrations()
                    elif command == "send_message" and all(k in data for k in ["source", "target", "message"]):
                        success = self.send_message(
                            source_id=data["source"],
                            target_id=data["target"],
                            data=data["message"],
                            channel_type=data.get("channel", ChannelType.DEFAULT),
                            priority=data.get("priority", MessagePriority.NORMAL)
                        )
                        result["success"] = success
                        
                return result
            else:
                # For simple data, just return basic info
                return {
                    "status": "received",
                    "message": "Unprocessable data format",
                    "channels": len(self.channels),
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"Error in process method: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }

# Test the ChannelNode if run directly
if __name__ == "__main__":
    print("Testing ChannelNode...")
    
    # Create a ChannelNode
    channel_node = ChannelNode()
    
    # Create a dummy node class for testing
    class DummyNode:
        def __init__(self, node_id):
            self.node_id = node_id
            self.received_messages = []
            
        def receive_message(self, source_id, data):
            print(f"Node {self.node_id} received message from {source_id}: {data}")
            self.received_messages.append((source_id, data))
            
        def process(self, data):
            print(f"Node {self.node_id} processing: {data}")
            return {"processed": True, "data": data}
    
    # Create dummy nodes
    node1 = DummyNode("node1")
    node2 = DummyNode("node2")
    node3 = DummyNode("node3")
    
    # Register nodes
    channel_node.register_node("node1", node1, [ChannelType.DEFAULT, ChannelType.TEXT])
    channel_node.register_node("node2", node2, [ChannelType.DEFAULT, ChannelType.TENSOR])
    channel_node.register_node("node3", node3, [ChannelType.TEXT])
    
    # Add a transformer for node2
    def double_values(data):
        if isinstance(data, (int, float)):
            return data * 2
        elif isinstance(data, dict) and "value" in data:
            result = data.copy()
            result["value"] *= 2
            return result
        return data
    
    channel_node.add_transformer(ChannelType.DEFAULT, "node2", double_values)
    
    # Send some messages
    channel_node.send_message("test", "node1", "Hello Node1", ChannelType.TEXT)
    channel_node.send_message("test", "node2", {"value": 10}, ChannelType.DEFAULT)
    
    # Broadcast a message
    channel_node.broadcast("test", "Broadcast message", ChannelType.TEXT)
    
    # Allow some time for processing
    time.sleep(0.5)
    
    # Print metrics
    print("\nChannel Metrics:")
    metrics = channel_node.get_channel_metrics()
    for channel_type, channel_metrics in metrics.items():
        print(f"  {channel_type}: {channel_metrics}")
    
    # Print node registrations
    print("\nNode Registrations:")
    registrations = channel_node.get_node_registrations()
    for node_id, reg_info in registrations.items():
        print(f"  {node_id}: {reg_info}")
    
    # Shutdown
    print("\nShutting down...")
    channel_node.shutdown()
    print("Done") 