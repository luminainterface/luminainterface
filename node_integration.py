"""
Node Integration System
=====================
Integrates various neural network nodes and manages their communication.
"""

import os
import logging
import torch
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import threading
from queue import Queue
import importlib.util
import glob
import psutil
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("node_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NodeInfo:
    """Information about a node in the system"""
    
    def __init__(self, node_id: str, node_instance: Any, node_type: str):
        self.node_id = node_id
        self.instance = node_instance
        self.node_type = node_type
        self.connections = {
            "incoming": [],
            "outgoing": []
        }
        self.status = "initialized"
        self.last_active = time.time()
        self.processing_count = 0
        self.error_count = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "status": self.status,
            "last_active": self.last_active,
            "processing_count": self.processing_count,
            "error_count": self.error_count,
            "connections": self.connections
        }

class NodeConnection:
    """Connection between two nodes"""
    
    def __init__(self, 
                source_id: str, 
                target_id: str, 
                connection_type: str = "default",
                weight: float = 1.0,
                bidirectional: bool = False):
        self.source_id = source_id
        self.target_id = target_id
        self.connection_type = connection_type
        self.weight = weight
        self.bidirectional = bidirectional
        self.status = "active"
        self.created_at = time.time()
        self.metadata = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "connection_type": self.connection_type,
            "weight": self.weight,
            "bidirectional": self.bidirectional,
            "status": self.status,
            "created_at": self.created_at,
            "metadata": self.metadata
        }

class NodeIntegrationSystem:
    """System for integrating and managing neural network nodes"""
    
    def __init__(self, config_path: str = "config/node_integration.json"):
        """Initialize the node integration system"""
        logger.info("Initializing Node Integration System")
        
        self.config = self._load_config(config_path)
        self.nodes = {}  # node_id -> NodeInfo
        self.connections = {}  # connection_key -> NodeConnection
        self.message_queue = Queue()
        self.result_cache = {}
        
        # Connection types with expanded patterns
        self.connection_types = {
            # Standard connection types
            "default": {"bidirectional": True, "weight_range": (0.0, 1.0), 
                        "description": "General purpose connection"},
            "data": {"bidirectional": False, "weight_range": (0.0, 1.0),
                     "description": "One-way data flow connection"}, 
            "control": {"bidirectional": False, "weight_range": (0.0, 1.0),
                        "description": "One-way control signal connection"},
            "feedback": {"bidirectional": True, "weight_range": (-1.0, 1.0),
                         "description": "Bidirectional feedback connection"},
            "resonance": {"bidirectional": True, "weight_range": (0.0, 1.0),
                          "description": "Bidirectional resonant connection"},
            
            # Advanced connection patterns
            "mirror": {"bidirectional": True, "weight_range": (0.5, 1.0),
                       "description": "Connection that mirrors state between nodes",
                       "sync_properties": True},
            "cascade": {"bidirectional": False, "weight_range": (0.5, 1.0),
                        "description": "Connection where output triggers downstream processing",
                        "auto_trigger": True},
            "conditional": {"bidirectional": False, "weight_range": (0.0, 1.0),
                            "description": "Connection active only when conditions are met",
                            "condition_function": None},
            "amplifier": {"bidirectional": False, "weight_range": (1.0, 5.0),
                          "description": "Connection that amplifies signals"},
            "filter": {"bidirectional": False, "weight_range": (0.0, 1.0),
                       "description": "Connection that filters signals",
                       "filter_function": None},
            "transformer": {"bidirectional": False, "weight_range": (0.0, 1.0),
                           "description": "Connection that transforms data",
                           "transform_function": None},
            
            # Neural-specific patterns
            "hebbian": {"bidirectional": True, "weight_range": (0.0, 1.0),
                        "description": "Connection that strengthens with use",
                        "learning_rate": 0.01,
                        "strengthens_with_use": True},
            "inhibitory": {"bidirectional": False, "weight_range": (-1.0, 0.0),
                          "description": "Connection that inhibits target activity"},
            "excitatory": {"bidirectional": False, "weight_range": (0.0, 1.0),
                          "description": "Connection that excites target activity"},
            "gated": {"bidirectional": False, "weight_range": (0.0, 1.0),
                     "description": "Connection controlled by a third gating node",
                     "gate_node_id": None},
            
            # Architecture patterns
            "bus": {"bidirectional": True, "weight_range": (0.0, 1.0),
                   "description": "Many-to-many shared connection",
                   "shared": True},
            "pipeline": {"bidirectional": False, "weight_range": (0.0, 1.0),
                        "description": "Sequential processing connection",
                        "sequential": True},
            "broadcast": {"bidirectional": False, "weight_range": (0.0, 1.0),
                         "description": "One-to-many message distribution"},
            "aggregator": {"bidirectional": False, "weight_range": (0.0, 1.0),
                          "description": "Many-to-one collection connection",
                          "aggregation_function": "sum"},  # sum, avg, max, etc.
        }
        
        # Connection pattern templates
        self.connection_patterns = {
            "star": {
                "description": "Central node connected to multiple satellite nodes",
                "implementation": self._create_star_pattern
            },
            "mesh": {
                "description": "All nodes connected to all other nodes",
                "implementation": self._create_mesh_pattern
            },
            "chain": {
                "description": "Nodes connected in a linear sequence",
                "implementation": self._create_chain_pattern
            },
            "ring": {
                "description": "Nodes connected in a circular pattern",
                "implementation": self._create_ring_pattern
            },
            "tree": {
                "description": "Hierarchical node connections",
                "implementation": self._create_tree_pattern
            },
            "layered": {
                "description": "Nodes connected in layers like a neural network",
                "implementation": self._create_layered_pattern
            },
            "bidirectional_pipeline": {
                "description": "Pipeline with feedback connections",
                "implementation": self._create_bidirectional_pipeline
            },
            "heterogeneous": {
                "description": "Different connection types based on node types",
                "implementation": self._create_heterogeneous_pattern
            }
        }
        
        # Performance metrics tracking
        self.metrics = {
            "system": {
                "start_time": time.time(),
                "uptime_seconds": 0,
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "thread_count": 0,
                "last_update": time.time()
            },
            "nodes": {
                "total": 0,
                "active": 0,
                "error": 0,
                "by_type": {}
            },
            "connections": {
                "total": 0,
                "active": 0,
                "message_throughput": 0.0,
                "by_type": {}
            },
            "message_processing": {
                "total_processed": 0,
                "success_rate": 1.0,
                "errors": 0,
                "avg_latency": 0.0,
                "avg_queue_size": 0
            },
            "discovery": {
                "last_scan": 0,
                "nodes_found": 0
            },
            "time_periods": {
                "last_minute": {},
                "last_hour": {},
                "last_day": {}
            }
        }
        
        # Metrics history for trending
        self.metrics_history = {
            "timestamps": [],
            "cpu_usage": [],
            "memory_usage": [],
            "active_nodes": [],
            "message_throughput": [],
            "success_rate": []
        }
        self.max_history_points = 100  # Maximum number of history points to keep
        
        # Start processing thread
        self.should_stop = False
        self.processing_thread = threading.Thread(target=self._message_processor)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start metrics collection thread
        self.metrics_thread = threading.Thread(target=self._metrics_collector)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        
        # Discover and load nodes
        self._discover_and_load_nodes()
        
        logger.info(f"Node Integration System initialized with {len(self.nodes)} nodes")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        config = {
            "node_dirs": ["."],
            "auto_connect": True,
            "default_connection_weight": 0.5,
            "default_connection_type": "default"
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            
        return config
        
    def _discover_and_load_nodes(self):
        """Discover and load all available nodes"""
        node_files = []
        for node_dir in self.config.get("node_dirs", ["."]):
            pattern = os.path.join(node_dir, "*node*.py")
            node_files.extend(glob.glob(pattern))
            
        # Exclude this integration file
        node_files = [f for f in node_files if os.path.basename(f) != "node_integration.py"]
        logger.info(f"Discovered {len(node_files)} potential node files")
        
        # Load each node file
        for file_path in node_files:
            try:
                # Load module
                module_name = os.path.splitext(os.path.basename(file_path))[0]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                
                if not spec or not spec.loader:
                    logger.warning(f"Could not load spec for {file_path}")
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for node classes in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    
                    # Check if it's a class with 'Node' in the name or that has node-like methods
                    if (isinstance(attr, type) and 
                        ("Node" in attr_name or "RSEN" in attr_name) and
                        (hasattr(attr, "process") or 
                         hasattr(attr, "forward") or 
                         hasattr(attr, "__call__"))):
                        
                        try:
                            # Create instance and register
                            instance = attr()
                            node_id = f"{module_name}.{attr_name}"
                            self.register_node(node_id, instance, attr_name)
                            logger.info(f"Loaded and registered node: {node_id}")
                        except Exception as e:
                            logger.error(f"Error instantiating node {attr_name} from {module_name}: {e}")
            
            except Exception as e:
                logger.error(f"Error loading module {file_path}: {e}")
                
        # Auto-connect nodes if enabled
        if self.config.get("auto_connect", True):
            self._auto_connect_nodes()
            
    def _auto_connect_nodes(self):
        """Automatically connect nodes based on their types"""
        # Define node types for potential connections
        node_types = {
            "hybrid": ["HybridNode"],
            "processing": ["ProcessingNode", "RSEN"],
            "web": ["WebServerNode"],
            "storage": ["NodeZero"]
        }
        
        # Map nodes to their types
        typed_nodes = {}
        for node_id, node_info in self.nodes.items():
            for type_name, class_names in node_types.items():
                if any(class_name in node_info.node_type for class_name in class_names):
                    if type_name not in typed_nodes:
                        typed_nodes[type_name] = []
                    typed_nodes[type_name].append(node_id)
                    break
                    
        # Define connection patterns
        connection_patterns = [
            # source_type, target_type, connection_type, weight, bidirectional
            ("hybrid", "processing", "data", 0.7, False),
            ("processing", "hybrid", "feedback", 0.5, False),
            ("hybrid", "storage", "data", 0.8, False),
            ("storage", "hybrid", "feedback", 0.3, False),
            ("processing", "storage", "data", 0.6, False),
            ("hybrid", "web", "data", 0.9, False),
            ("web", "hybrid", "control", 0.8, False)
        ]
        
        # Create connections based on patterns
        connections_created = 0
        for source_type, target_type, conn_type, weight, bidirectional in connection_patterns:
            if source_type in typed_nodes and target_type in typed_nodes:
                for source_id in typed_nodes[source_type]:
                    for target_id in typed_nodes[target_type]:
                        if source_id != target_id:
                            if self.connect_nodes(source_id, target_id, conn_type, weight, bidirectional):
                                connections_created += 1
                                
        logger.info(f"Auto-connected nodes: created {connections_created} connections")
        
    def register_node(self, node_id: str, node_instance: Any, node_type: str) -> bool:
        """Register a new node with the system"""
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already registered")
            return False
            
        # Create NodeInfo
        node_info = NodeInfo(node_id, node_instance, node_type)
        self.nodes[node_id] = node_info
        
        # Initialize if it has an initialization method
        if hasattr(node_instance, "initialize"):
            try:
                node_instance.initialize()
                logger.info(f"Initialized node {node_id}")
            except Exception as e:
                logger.error(f"Error initializing node {node_id}: {e}")
                
        # Activate if it has an activation method
        if hasattr(node_instance, "activate"):
            try:
                node_instance.activate()
                node_info.status = "active"
                logger.info(f"Activated node {node_id}")
            except Exception as e:
                logger.error(f"Error activating node {node_id}: {e}")
                
        return True
        
    def connect_nodes(self, 
                     source_id: str, 
                     target_id: str, 
                     connection_type: str = "default",
                     weight: float = None,
                     bidirectional: bool = None) -> bool:
        """Connect two nodes"""
        # Validate nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.error(f"Cannot connect: one or both nodes not found - {source_id}, {target_id}")
            return False
            
        # Validate connection type
        if connection_type not in self.connection_types:
            logger.error(f"Invalid connection type: {connection_type}")
            return False
            
        # Use defaults if not specified
        if weight is None:
            weight = self.config.get("default_connection_weight", 0.5)
            
        if bidirectional is None:
            bidirectional = self.connection_types[connection_type]["bidirectional"]
            
        # Enforce weight range
        weight_range = self.connection_types[connection_type]["weight_range"]
        weight = max(min(weight, weight_range[1]), weight_range[0])
        
        # Create connection key
        connection_key = f"{source_id}->{target_id}:{connection_type}"
        
        # Check if connection already exists
        if connection_key in self.connections:
            logger.warning(f"Connection {connection_key} already exists")
            return False
            
        # Create connection
        connection = NodeConnection(
            source_id=source_id,
            target_id=target_id,
            connection_type=connection_type,
            weight=weight,
            bidirectional=bidirectional
        )
        
        self.connections[connection_key] = connection
        
        # Update node connection lists
        self.nodes[source_id].connections["outgoing"].append({
            "target": target_id,
            "type": connection_type,
            "weight": weight
        })
        
        self.nodes[target_id].connections["incoming"].append({
            "source": source_id,
            "type": connection_type,
            "weight": weight
        })
        
        # Create reverse connection if bidirectional
        if bidirectional:
            reverse_key = f"{target_id}->{source_id}:{connection_type}"
            reverse_connection = NodeConnection(
                source_id=target_id,
                target_id=source_id,
                connection_type=connection_type,
                weight=weight,
                bidirectional=True
            )
            
            self.connections[reverse_key] = reverse_connection
            
            # Update node connection lists for reverse
            self.nodes[target_id].connections["outgoing"].append({
                "target": source_id,
                "type": connection_type,
                "weight": weight
            })
            
            self.nodes[source_id].connections["incoming"].append({
                "source": target_id,
                "type": connection_type,
                "weight": weight
            })
            
        logger.info(f"Created {'bidirectional' if bidirectional else 'one-way'} connection: {connection_key}")
        return True
        
    def send_message(self, 
                    source_id: str, 
                    target_id: str, 
                    message: Any,
                    connection_type: str = "default",
                    wait_for_result: bool = False,
                    timeout: float = 5.0) -> Optional[Any]:
        """Send a message from one node to another"""
        # Validate nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.error(f"Cannot send message: one or both nodes not found - {source_id}, {target_id}")
            return None
            
        # Check if connection exists
        connection_key = f"{source_id}->{target_id}:{connection_type}"
        if connection_key not in self.connections:
            logger.error(f"No connection found: {connection_key}")
            return None
            
        # Create message ID
        message_id = f"{source_id}->{target_id}:{time.time()}"
        
        # Put message in queue
        self.message_queue.put({
            "message_id": message_id,
            "source_id": source_id,
            "target_id": target_id,
            "connection_type": connection_type,
            "content": message,
            "timestamp": time.time()
        })
        
        # Wait for result if requested
        if wait_for_result:
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                if message_id in self.result_cache:
                    result = self.result_cache[message_id]
                    # Remove from cache
                    del self.result_cache[message_id]
                    return result
                time.sleep(0.01)
                
            logger.warning(f"Timeout waiting for result of message {message_id}")
            return None
            
        return None
        
    def _message_processor(self):
        """Background thread for processing messages"""
        while not self.should_stop:
            try:
                # Get message from queue
                try:
                    message_data = self.message_queue.get(timeout=1)
                except:
                    time.sleep(0.1)
                    continue
                    
                # Extract message info
                message_id = message_data["message_id"]
                source_id = message_data["source_id"]
                target_id = message_data["target_id"]
                connection_type = message_data["connection_type"]
                content = message_data["content"]
                
                # Process message
                try:
                    # Get target node
                    target_node = self.nodes[target_id].instance
                    
                    # Different nodes might have different processing methods
                    result = None
                    
                    if hasattr(target_node, "process"):
                        result = target_node.process(content)
                    elif hasattr(target_node, "process_data"):
                        result = target_node.process_data(content)
                    elif hasattr(target_node, "forward"):
                        result = target_node.forward(content)
                    elif hasattr(target_node, "__call__"):
                        result = target_node(content)
                    else:
                        logger.warning(f"Node {target_id} has no processing method")
                        
                    # Store result in cache
                    self.result_cache[message_id] = result
                    
                    # Update node stats
                    self.nodes[target_id].processing_count += 1
                    self.nodes[target_id].last_active = time.time()
                    self.nodes[target_id].status = "active"
                    
                except Exception as e:
                    logger.error(f"Error processing message {message_id}: {e}")
                    self.nodes[target_id].error_count += 1
                    self.result_cache[message_id] = {"error": str(e)}
                    
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                time.sleep(1)
                
    def process_input(self, input_data: Any, start_node_id: Optional[str] = None) -> Dict[str, Any]:
        """Process input data through the node network"""
        results = {}
        processed_nodes = set()
        
        # Determine start node
        if start_node_id and start_node_id in self.nodes:
            start_nodes = [start_node_id]
        else:
            # Find entry point nodes (nodes with no incoming connections)
            start_nodes = []
            for node_id, node_info in self.nodes.items():
                if not node_info.connections["incoming"]:
                    start_nodes.append(node_id)
                    
            # If no entry points found, use all nodes
            if not start_nodes:
                start_nodes = list(self.nodes.keys())
                
        # Create a queue of nodes to process
        nodes_to_process = [(node_id, input_data) for node_id in start_nodes]
        
        # Process nodes in topological order (based on connections)
        while nodes_to_process:
            node_id, node_input = nodes_to_process.pop(0)
            
            # Skip if already processed
            if node_id in processed_nodes:
                continue
                
            # Process node
            try:
                node_info = self.nodes[node_id]
                node_instance = node_info.instance
                
                # Different nodes might have different processing methods
                result = None
                
                if hasattr(node_instance, "process"):
                    result = node_instance.process(node_input)
                elif hasattr(node_instance, "process_data"):
                    result = node_instance.process_data(node_input)
                elif hasattr(node_instance, "forward"):
                    result = node_instance.forward(node_input)
                elif hasattr(node_instance, "__call__"):
                    result = node_instance(node_input)
                else:
                    logger.warning(f"Node {node_id} has no processing method")
                    
                # Store result
                results[node_id] = result
                processed_nodes.add(node_id)
                
                # Update node stats
                node_info.processing_count += 1
                node_info.last_active = time.time()
                node_info.status = "active"
                
                # Queue outgoing connections for processing
                for connection in node_info.connections["outgoing"]:
                    target_id = connection["target"]
                    if target_id not in processed_nodes:
                        nodes_to_process.append((target_id, result))
                        
            except Exception as e:
                logger.error(f"Error processing node {node_id}: {e}")
                results[node_id] = {"error": str(e)}
                self.nodes[node_id].error_count += 1
                
        return {
            "input": input_data,
            "node_results": results,
            "processed_nodes": list(processed_nodes)
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status.
        
        Returns:
            Dict containing system status information
        """
        # Update metrics before returning
        self._update_realtime_metrics()
        
        # Build status report
        status = {
            "system_info": {
                "timestamp": time.time(),
                "uptime": self.metrics["system"]["uptime_seconds"],
                "cpu_usage": self.metrics["system"]["cpu_usage"],
                "memory_usage": self.metrics["system"]["memory_usage"],
                "thread_count": self.metrics["system"]["thread_count"]
            },
            "nodes": {
                "total": len(self.nodes),
                "active": self.metrics["nodes"]["active"],
                "by_type": self.metrics["nodes"]["by_type"],
                "error_count": self.metrics["nodes"]["error"]
            },
                "connections": {
                "total": len(self.connections),
                "active": self.metrics["connections"]["active"],
                "by_type": self.metrics["connections"]["by_type"],
                "throughput": self.metrics["connections"]["message_throughput"]
            },
            "processing": {
                "queue_size": self.message_queue.qsize(),
                "success_rate": self.metrics["message_processing"]["success_rate"],
                "avg_latency": self.metrics["message_processing"]["avg_latency"]
            },
            "trends": {
                "usage_trend": self._calculate_trend("cpu_usage"),
                "memory_trend": self._calculate_trend("memory_usage"),
                "throughput_trend": self._calculate_trend("message_throughput"),
                "nodes_trend": self._calculate_trend("active_nodes")
            },
            "time_periods": self.metrics["time_periods"]
        }
        
        # Add node details
        node_details = {}
        for node_id, node_info in self.nodes.items():
            node_details[node_id] = node_info.to_dict()
        status["node_details"] = node_details
        
        # Add connections details
        connection_details = {}
        for conn_key, connection in self.connections.items():
            connection_details[conn_key] = connection.to_dict()
        status["connection_details"] = connection_details
        
        return status
        
    def _update_realtime_metrics(self):
        """Update realtime metrics before reporting"""
        # Update queue size
        self.metrics["message_processing"]["avg_queue_size"] = self.message_queue.qsize()
        
        # Update node counts in case they changed
        active_count = sum(1 for node in self.nodes.values() if node.status == "active")
        error_count = sum(1 for node in self.nodes.values() if node.status == "error")
        self.metrics["nodes"]["active"] = active_count
        self.metrics["nodes"]["error"] = error_count
        
    def _calculate_trend(self, metric_name: str) -> float:
        """
        Calculate trend for a specific metric.
        
        Args:
            metric_name: Name of the metric to calculate trend for
            
        Returns:
            float: Trend value (-1.0 to 1.0), where positive means increasing
        """
        # Need at least 2 data points for a trend
        if len(self.metrics_history[metric_name]) < 2:
            return 0.0
        
        # Get recent values (last 10 or fewer)
        values = self.metrics_history[metric_name][-10:]
        
        if len(values) < 2:
            return 0.0
        
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        
        # Calculate relative change and normalize to -1.0 to 1.0 range
        if first_avg == 0:
            return 0.0
        
        change = (second_avg - first_avg) / first_avg
        
        # Clamp to -1.0 to 1.0
        return max(-1.0, min(1.0, change))
        
    def _metrics_collector(self):
        """Background thread for collecting system metrics"""
        update_interval = self.config.get("metrics_interval", 5.0)  # seconds
        
        while not self.should_stop:
            try:
                # Calculate time-based metrics
                current_time = time.time()
                self.metrics["system"]["uptime_seconds"] = current_time - self.metrics["system"]["start_time"]
                
                # Get system metrics
                process = psutil.Process()
                self.metrics["system"]["cpu_usage"] = process.cpu_percent(interval=0.1)
                self.metrics["system"]["memory_usage"] = process.memory_info().rss / (1024 * 1024)  # MB
                self.metrics["system"]["thread_count"] = threading.active_count()
                self.metrics["system"]["last_update"] = current_time
                
                # Get node metrics
                active_nodes = 0
                error_nodes = 0
                node_type_counts = {}
                
                for node_id, node_info in self.nodes.items():
                    if node_info.status == "active":
                        active_nodes += 1
                    elif node_info.status == "error":
                        error_nodes += 1
                        
                    # Count by type
                    node_type = node_info.node_type
                    if node_type not in node_type_counts:
                        node_type_counts[node_type] = 0
                    node_type_counts[node_type] += 1
                
                self.metrics["nodes"]["total"] = len(self.nodes)
                self.metrics["nodes"]["active"] = active_nodes
                self.metrics["nodes"]["error"] = error_nodes
                self.metrics["nodes"]["by_type"] = node_type_counts
                
                # Get connection metrics
                active_connections = 0
                connection_type_counts = {}
                total_messages = 0
                
                for conn_key, connection in self.connections.items():
                    if connection.status == "active":
                        active_connections += 1
                        
                    conn_type = connection.connection_type
                    if conn_type not in connection_type_counts:
                        connection_type_counts[conn_type] = 0
                    connection_type_counts[conn_type] += 1
                    
                    # Get metadata about message counts if available
                    if "messages_sent" in connection.metadata:
                        total_messages += connection.metadata["messages_sent"]
                
                self.metrics["connections"]["total"] = len(self.connections)
                self.metrics["connections"]["active"] = active_connections
                self.metrics["connections"]["by_type"] = connection_type_counts
                
                # Calculate throughput (messages per second)
                elapsed = current_time - self.metrics["system"]["last_update"]
                if elapsed > 0 and hasattr(self, "_last_total_messages"):
                    new_messages = total_messages - self._last_total_messages
                    self.metrics["connections"]["message_throughput"] = new_messages / elapsed
                self._last_total_messages = total_messages
                
                # Add to history
                self.metrics_history["timestamps"].append(current_time)
                self.metrics_history["cpu_usage"].append(self.metrics["system"]["cpu_usage"])
                self.metrics_history["memory_usage"].append(self.metrics["system"]["memory_usage"])
                self.metrics_history["active_nodes"].append(active_nodes)
                self.metrics_history["message_throughput"].append(self.metrics["connections"]["message_throughput"])
                self.metrics_history["success_rate"].append(self.metrics["message_processing"]["success_rate"])
                
                # Trim history if needed
                if len(self.metrics_history["timestamps"]) > self.max_history_points:
                    for key in self.metrics_history:
                        self.metrics_history[key] = self.metrics_history[key][-self.max_history_points:]
                
                # Calculate time period metrics
                self._calculate_time_period_metrics()
                
                # Log metrics at a reduced interval to avoid flooding
                if current_time - self.metrics.get("last_logged", 0) > 60:  # Log only every 60 seconds
                    logger.info(f"System metrics: CPU {self.metrics['system']['cpu_usage']:.1f}%, " +
                               f"Memory {self.metrics['system']['memory_usage']:.1f}MB, " +
                               f"Nodes {active_nodes}/{len(self.nodes)}, " +
                               f"Throughput {self.metrics['connections']['message_throughput']:.2f} msg/s")
                    self.metrics["last_logged"] = current_time
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
            
            # Sleep until next update
            time.sleep(update_interval)

    def _calculate_time_period_metrics(self):
        """Calculate metrics for different time periods"""
        current_time = time.time()
        
        # Only calculate if we have enough history
        if not self.metrics_history["timestamps"]:
            return
        
        # Define time periods
        periods = {
            "last_minute": current_time - 60,
            "last_hour": current_time - 3600,
            "last_day": current_time - 86400
        }
        
        # Calculate for each period
        for period_name, start_time in periods.items():
            # Find indices within this time period
            indices = [i for i, ts in enumerate(self.metrics_history["timestamps"]) if ts >= start_time]
            
            if indices:
                # Calculate averages for the period
                self.metrics["time_periods"][period_name] = {
                    "avg_cpu": sum(self.metrics_history["cpu_usage"][i] for i in indices) / len(indices),
                    "avg_memory": sum(self.metrics_history["memory_usage"][i] for i in indices) / len(indices),
                    "avg_active_nodes": sum(self.metrics_history["active_nodes"][i] for i in indices) / len(indices),
                    "avg_throughput": sum(self.metrics_history["message_throughput"][i] for i in indices) / len(indices),
                    "avg_success_rate": sum(self.metrics_history["success_rate"][i] for i in indices) / len(indices),
                    "data_points": len(indices)
                }
            else:
                # No data for this period
                self.metrics["time_periods"][period_name] = {
                    "avg_cpu": 0,
                    "avg_memory": 0,
                    "avg_active_nodes": 0,
                    "avg_throughput": 0,
                    "avg_success_rate": 0,
                    "data_points": 0
        }
        
    def save_state(self, file_path: str) -> bool:
        """Save current system state to file"""
        try:
            # Prepare state data
            state = {
                "nodes": {},
                "connections": {},
                "config": self.config
            }
            
            # Save node data (excluding instance)
            for node_id, node_info in self.nodes.items():
                state["nodes"][node_id] = node_info.to_dict()
                
            # Save connection data
            for conn_key, connection in self.connections.items():
                state["connections"][conn_key] = connection.to_dict()
                
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved system state to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
            return False
            
    def load_state(self, file_path: str) -> bool:
        """Load system state from file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"State file not found: {file_path}")
                return False
                
            # Load state
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            # Load config
            if "config" in state:
                self.config.update(state["config"])
                
            # Load connections
            for conn_key, conn_data in state.get("connections", {}).items():
                # Only recreate connections if both nodes exist
                source_id = conn_data["source_id"]
                target_id = conn_data["target_id"]
                
                if source_id in self.nodes and target_id in self.nodes:
                    self.connect_nodes(
                        source_id=source_id,
                        target_id=target_id,
                        connection_type=conn_data["connection_type"],
                        weight=conn_data["weight"],
                        bidirectional=conn_data["bidirectional"]
                    )
                    
            logger.info(f"Loaded system state from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
            return False
            
    def shutdown(self):
        """Gracefully shut down the system"""
        logger.info("Shutting down Node Integration System")
        
        # Stop background thread
        self.should_stop = True
        if hasattr(self, "processing_thread"):
            try:
                self.processing_thread.join(timeout=2)
            except:
                pass
                
        # Deactivate nodes
        for node_id, node_info in self.nodes.items():
            try:
                if hasattr(node_info.instance, "deactivate"):
                    node_info.instance.deactivate()
                    logger.info(f"Deactivated node {node_id}")
            except Exception as e:
                logger.error(f"Error deactivating node {node_id}: {e}")
                
        logger.info("Node Integration System shutdown complete")

    def connect_nodes_with_pattern(self, pattern_name: str, node_ids: List[str], **kwargs) -> bool:
        """
        Connect nodes using a predefined connection pattern.
        
        Args:
            pattern_name: Name of the pattern to use
            node_ids: List of node IDs to connect
            **kwargs: Additional parameters for the pattern
            
        Returns:
            bool: True if successful, False otherwise
        """
        if pattern_name not in self.connection_patterns:
            logger.error(f"Unknown connection pattern: {pattern_name}")
            return False
        
        # Validate that all node IDs exist
        for node_id in node_ids:
            if node_id not in self.nodes:
                logger.error(f"Cannot apply pattern: Node {node_id} not found")
                return False
            
        # Apply the pattern
        try:
            pattern_func = self.connection_patterns[pattern_name]["implementation"]
            created_connections = pattern_func(node_ids, **kwargs)
            logger.info(f"Applied {pattern_name} connection pattern among {len(node_ids)} nodes, " +
                       f"created {created_connections} connections")
            return created_connections > 0
        except Exception as e:
            logger.error(f"Error applying connection pattern {pattern_name}: {str(e)}")
            return False

    def _create_star_pattern(self, node_ids: List[str], 
                           center_node_id: Optional[str] = None,
                           connection_type: str = "default",
                           bidirectional: bool = True) -> int:
        """
        Create a star pattern with one center node connected to all others.
        
        Args:
            node_ids: List of node IDs
            center_node_id: ID of the center node (first node if None)
            connection_type: Type of connection to create
            bidirectional: Whether connections should be bidirectional
            
        Returns:
            int: Number of connections created
        """
        if len(node_ids) < 2:
            return 0
        
        # Determine center node
        if center_node_id is None or center_node_id not in node_ids:
            center_node_id = node_ids[0]
        
        satellite_nodes = [n for n in node_ids if n != center_node_id]
        
        # Create connections
        connections_created = 0
        for satellite_id in satellite_nodes:
            if self.connect_nodes(center_node_id, satellite_id, connection_type, bidirectional=bidirectional):
                connections_created += 1
            
        return connections_created

    def _create_mesh_pattern(self, node_ids: List[str],
                           connection_type: str = "default",
                           bidirectional: bool = True) -> int:
        """
        Create a mesh pattern where every node connects to every other node.
        
        Args:
            node_ids: List of node IDs
            connection_type: Type of connection to create
            bidirectional: Whether connections should be bidirectional
            
        Returns:
            int: Number of connections created
        """
        if len(node_ids) < 2:
            return 0
        
        connections_created = 0
        
        for i, source_id in enumerate(node_ids):
            for target_id in node_ids[i+1:]:  # Connect to nodes after this one to avoid duplicates
                if self.connect_nodes(source_id, target_id, connection_type, bidirectional=bidirectional):
                    connections_created += 1
                
        return connections_created

    def _create_chain_pattern(self, node_ids: List[str],
                            connection_type: str = "default",
                            bidirectional: bool = False) -> int:
        """
        Create a chain pattern where nodes are connected in sequence.
        
        Args:
            node_ids: List of node IDs
            connection_type: Type of connection to create
            bidirectional: Whether connections should be bidirectional
            
        Returns:
            int: Number of connections created
        """
        if len(node_ids) < 2:
            return 0
        
        connections_created = 0
        
        for i in range(len(node_ids) - 1):
            if self.connect_nodes(node_ids[i], node_ids[i+1], connection_type, bidirectional=bidirectional):
                connections_created += 1
            
        return connections_created

    def _create_ring_pattern(self, node_ids: List[str],
                           connection_type: str = "default",
                           bidirectional: bool = False) -> int:
        """
        Create a ring pattern where nodes form a closed loop.
        
        Args:
            node_ids: List of node IDs
            connection_type: Type of connection to create
            bidirectional: Whether connections should be bidirectional
            
        Returns:
            int: Number of connections created
        """
        if len(node_ids) < 2:
            return 0
        
        # First create a chain
        connections_created = self._create_chain_pattern(node_ids, connection_type, bidirectional)
        
        # Connect the last node to the first to close the loop
        if self.connect_nodes(node_ids[-1], node_ids[0], connection_type, bidirectional=bidirectional):
            connections_created += 1
        
        return connections_created

    def _create_tree_pattern(self, node_ids: List[str],
                           branch_factor: int = 2,
                           connection_type: str = "default",
                           bidirectional: bool = False) -> int:
        """
        Create a tree pattern where each node connects to multiple child nodes.
        
        Args:
            node_ids: List of node IDs
            branch_factor: Maximum number of child nodes per parent
            connection_type: Type of connection to create
            bidirectional: Whether connections should be bidirectional
            
        Returns:
            int: Number of connections created
        """
        if len(node_ids) < 2:
            return 0
        
        connections_created = 0
        
        # For each parent node
        parent_count = max(1, len(node_ids) // branch_factor)
        for i in range(parent_count):
            parent_id = node_ids[i]
            
            # Connect to its children
            start_child = i * branch_factor + 1
            end_child = min(start_child + branch_factor, len(node_ids))
            
            for j in range(start_child, end_child):
                if j < len(node_ids):
                    child_id = node_ids[j]
                    if self.connect_nodes(parent_id, child_id, connection_type, bidirectional=bidirectional):
                        connections_created += 1
                    
        return connections_created

    def _create_layered_pattern(self, node_ids: List[str],
                             layer_sizes: List[int],
                             connection_type: str = "default",
                             full_connect_layers: bool = True) -> int:
        """
        Create a layered pattern like a neural network.
        
        Args:
            node_ids: List of node IDs
            layer_sizes: List containing the size of each layer
            connection_type: Type of connection to create
            full_connect_layers: Whether to fully connect adjacent layers
            
        Returns:
            int: Number of connections created
        """
        if len(node_ids) < 2 or not layer_sizes:
            return 0
        
        # Validate layer sizes
        total_nodes = sum(layer_sizes)
        if total_nodes > len(node_ids):
            logger.warning(f"Not enough nodes for specified layer sizes. Adjusting.")
            layer_sizes = self._adjust_layer_sizes(layer_sizes, len(node_ids))
        
        connections_created = 0
        node_index = 0
        
        # Get nodes for each layer
        layers = []
        for size in layer_sizes:
            layer_nodes = node_ids[node_index:node_index+size]
            layers.append(layer_nodes)
            node_index += size
        
        # Connect adjacent layers
        for i in range(len(layers) - 1):
            source_layer = layers[i]
            target_layer = layers[i+1]
            
            for source_id in source_layer:
                if full_connect_layers:
                    # Connect to all nodes in next layer
                    for target_id in target_layer:
                        if self.connect_nodes(source_id, target_id, connection_type, bidirectional=False):
                            connections_created += 1
                else:
                    # Connect to one random node in next layer
                    target_id = target_layer[hash(source_id) % len(target_layer)]
                    if self.connect_nodes(source_id, target_id, connection_type, bidirectional=False):
                        connections_created += 1
                    
        return connections_created

    def _adjust_layer_sizes(self, layer_sizes: List[int], max_nodes: int) -> List[int]:
        """Adjust layer sizes to fit within max_nodes"""
        total = sum(layer_sizes)
        ratio = max_nodes / total
        
        # Scale all layers proportionally
        adjusted = [max(1, int(size * ratio)) for size in layer_sizes]
        
        # Ensure we don't exceed max_nodes
        while sum(adjusted) > max_nodes:
            # Find the largest layer and reduce it
            largest_idx = adjusted.index(max(adjusted))
            if adjusted[largest_idx] > 1:
                adjusted[largest_idx] -= 1
            
        return adjusted

    def _create_bidirectional_pipeline(self, node_ids: List[str],
                                     forward_type: str = "data",
                                     feedback_type: str = "feedback") -> int:
        """
        Create a pipeline with forward and feedback connections.
        
        Args:
            node_ids: List of node IDs
            forward_type: Type for forward connections
            feedback_type: Type for feedback connections
            
        Returns:
            int: Number of connections created
        """
        if len(node_ids) < 2:
            return 0
        
        connections_created = 0
        
        # Create forward connections
        for i in range(len(node_ids) - 1):
            if self.connect_nodes(node_ids[i], node_ids[i+1], forward_type, bidirectional=False):
                connections_created += 1
            
        # Create feedback connections in reverse
        for i in range(len(node_ids) - 1, 0, -1):
            if self.connect_nodes(node_ids[i], node_ids[i-1], feedback_type, bidirectional=False):
                connections_created += 1
            
        return connections_created

    def _create_heterogeneous_pattern(self, node_ids: List[str],
                                    type_connections: Dict[str, Dict[str, str]] = None) -> int:
        """
        Create connections based on node types.
        
        Args:
            node_ids: List of node IDs
            type_connections: Dict mapping {source_type: {target_type: connection_type}}
            
        Returns:
            int: Number of connections created
        """
        if len(node_ids) < 2 or not type_connections:
            return 0
        
        # Default type connections if none provided
        if not type_connections:
            type_connections = {
                "processing": {"memory": "data", "control": "feedback"},
                "memory": {"output": "data"},
                "control": {"processing": "control"},
                "input": {"processing": "data"},
                "output": {"processing": "feedback"}
            }
        
        connections_created = 0
        
        # Get node types
        node_types = {}
        for node_id in node_ids:
            if node_id in self.nodes:
                node_types[node_id] = self.nodes[node_id].node_type
        
        # Create connections based on type rules
        for source_id in node_ids:
            source_type = node_types.get(source_id)
            if not source_type or source_type not in type_connections:
                continue
            
            type_rules = type_connections[source_type]
            
            for target_id in node_ids:
                if target_id == source_id:
                    continue
                
                target_type = node_types.get(target_id)
                if not target_type or target_type not in type_rules:
                    continue
                
                connection_type = type_rules[target_type]
                if self.connect_nodes(source_id, target_id, connection_type, bidirectional=False):
                    connections_created += 1
                
        return connections_created

    def get_available_connection_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available connection patterns.
        
        Returns:
            Dict mapping pattern names to their descriptions
        """
        return {name: {"description": pattern["description"]} 
                for name, pattern in self.connection_patterns.items()}
            
    def get_available_connection_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available connection types.
        
        Returns:
            Dict mapping type names to their properties
        """
        return {name: {"description": props.get("description", ""),
                      "bidirectional": props.get("bidirectional", False),
                      "weight_range": props.get("weight_range", (0.0, 1.0))}
                for name, props in self.connection_types.items()}

# Main entry point
if __name__ == "__main__":
    # Create the node integration system
    nis = NodeIntegrationSystem()
    
    # Print detected nodes
    print("Detected nodes:")
    for node_id, node_info in nis.nodes.items():
        print(f"  - {node_id} ({node_info.node_type}): {node_info.status}")
    
    # Process a test input
    test_input = "Test input for the node integration system"
    print(f"\nProcessing input: {test_input}")
    result = nis.process_input(test_input)
    
    # Print results
    print("\nProcessed nodes:")
    for node in result["processed_nodes"]:
        print(f"  - {node}")
    
    # Save system state
    nis.save_state("node_integration_state.json")
    
    # Shutdown
    nis.shutdown() 