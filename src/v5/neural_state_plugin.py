"""
Neural State Plugin for V5 Visualization System

This module provides the NeuralStatePlugin class that collects neural network
state information for visualization.
"""

import time
import logging
import threading
import numpy as np
from collections import deque
import uuid

from .node_socket import NodeSocket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the connection discovery service
try:
    from connection_discovery import ConnectionDiscovery, register_node
    HAS_DISCOVERY = True
except ImportError:
    logger.warning("ConnectionDiscovery not available. Limited plugin discovery will be used.")
    HAS_DISCOVERY = False
    # Define a dummy register_node function for when discovery is not available
    def register_node(node, **kwargs):
        logger.warning(f"Cannot register node: {node.node_id}")
        return None


class CircularBuffer:
    """Buffer for storing neural state data with a maximum size"""
    
    def __init__(self, max_size=1000):
        """
        Initialize the circular buffer
        
        Args:
            max_size: Maximum number of items to store
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        
    def append(self, item):
        """
        Add an item to the buffer
        
        Args:
            item: Item to add
        """
        self.buffer.append(item)
        
    def get_all(self):
        """Get all items in the buffer"""
        return list(self.buffer)
        
    def get_latest(self, n=1):
        """
        Get the latest n items from the buffer
        
        Args:
            n: Number of items to get
            
        Returns:
            List of the latest n items
        """
        return list(self.buffer)[-n:]
        
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        
    def __len__(self):
        """Get the current size of the buffer"""
        return len(self.buffer)


class NeuralStatePlugin:
    """Socket-ready plugin for providing neural network state"""
    
    def __init__(self, node_id="neural_state_provider"):
        """
        Initialize the neural state plugin
        
        Args:
            node_id: Unique identifier for this plugin
        """
        self.node_id = node_id
        self.socket = NodeSocket(node_id, "service")
        self.data_buffer = CircularBuffer(max_size=1000)
        self.mock_mode = False
        
        # Set up message handlers
        self.socket.register_message_handler("state_request", self._handle_state_request)
        
        # Register with discovery service if available
        if HAS_DISCOVERY:
            self.client = register_node(self)
            logger.info(f"Registered plugin with discovery service: {node_id}")
        else:
            self.client = None
            logger.warning(f"Running without discovery service: {node_id}")
        
        # Standard plugin interface attributes
        self.plugin_type = "v5_plugin"
        self.plugin_subtype = "data_provider"
        self.api_version = "v5.0"
        self.ui_requirements = ["neural_display", "activity_monitor"]
        
        # Periodic update thread
        self.update_interval = 0.5  # seconds
        self.update_thread = None
        self.running = False
        
        logger.info(f"Neural State Plugin initialized: {node_id}")
    
    def _handle_state_request(self, message):
        """
        Handle a request for neural state
        
        Args:
            message: Message containing the request
        """
        request_type = message.get("content", {}).get("request_type")
        request_id = message.get("request_id")
        
        if not request_id:
            logger.warning("Received state request without request_id")
            return
            
        if request_type == "latest":
            # Get the latest state
            latest_state = self.get_latest_state()
            
            # Send response
            self.socket.send_message({
                "type": "state_response",
                "request_id": request_id,
                "data": latest_state
            })
            
        elif request_type == "history":
            # Get state history
            count = message.get("content", {}).get("count", 10)
            history = self.get_state_history(count)
            
            # Send response
            self.socket.send_message({
                "type": "state_history_response",
                "request_id": request_id,
                "data": history
            })
            
        else:
            logger.warning(f"Unknown state request type: {request_type}")
    
    def collect_state(self, network):
        """
        Collect neural network state data
        
        Args:
            network: Neural network to collect state from
            
        Returns:
            Dictionary containing state data
        """
        try:
            # Collect real data if network is provided
            if network is not None:
                state_data = {
                    "timestamp": time.time(),
                    "id": str(uuid.uuid4()),
                    "layers": self._extract_layer_data(network),
                    "activations": self._extract_activations(network),
                    "weights": self._extract_weights(network),
                    "state_type": "real"
                }
            else:
                # Generate mock data if no network is provided
                state_data = self._generate_mock_state_data()
            
            # Store in buffer and broadcast to subscribers
            self.data_buffer.append(state_data)
            self.socket.send_message({
                "type": "neural_state_update",
                "data": state_data
            })
            
            logger.debug(f"Collected neural state: {len(state_data['layers'])} layers")
            return state_data
            
        except Exception as e:
            logger.error(f"Error collecting neural state: {str(e)}")
            # Return minimal data on error
            return {
                "timestamp": time.time(),
                "id": str(uuid.uuid4()),
                "layers": [],
                "activations": {},
                "weights": {},
                "state_type": "error",
                "error": str(e)
            }
    
    def _extract_layer_data(self, network):
        """
        Extract layer data from a neural network
        
        Args:
            network: Neural network to extract from
            
        Returns:
            List of layer data dictionaries
        """
        layers = []
        
        try:
            # Handle different types of neural networks
            if hasattr(network, "layers"):
                # Extract from network.layers
                for i, layer in enumerate(network.layers):
                    layer_info = {
                        "id": f"layer_{i}",
                        "name": getattr(layer, "name", f"Layer {i}"),
                        "type": layer.__class__.__name__,
                        "neurons": getattr(layer, "units", 0) or getattr(layer, "out_features", 0),
                        "activation": getattr(layer, "activation", "unknown")
                    }
                    layers.append(layer_info)
                    
            elif hasattr(network, "module"):
                # PyTorch model with nested modules
                for name, module in network.named_modules():
                    if len(list(module.children())) == 0:  # Only leaf modules
                        layer_info = {
                            "id": name or f"layer_{len(layers)}",
                            "name": name or f"Layer {len(layers)}",
                            "type": module.__class__.__name__,
                            "neurons": getattr(module, "out_features", 0),
                            "activation": "unknown"
                        }
                        layers.append(layer_info)
                        
            else:
                # Generic fallback
                logger.warning("Unknown network type, using generic layer extraction")
                layers.append({
                    "id": "layer_0",
                    "name": "Generic Layer",
                    "type": "unknown",
                    "neurons": 0,
                    "activation": "unknown"
                })
                
        except Exception as e:
            logger.error(f"Error extracting layer data: {str(e)}")
            
        return layers
    
    def _extract_activations(self, network):
        """
        Extract activation values from a neural network
        
        Args:
            network: Neural network to extract from
            
        Returns:
            Dictionary mapping layer IDs to activation arrays
        """
        activations = {}
        
        try:
            # This would need to be customized for the specific neural network framework
            # Here's a generic example
            if hasattr(network, "get_activations"):
                # If the network has a method to get activations
                raw_activations = network.get_activations()
                
                for layer_id, activation in raw_activations.items():
                    # Convert to list for JSON serialization
                    if isinstance(activation, np.ndarray):
                        activations[layer_id] = activation.tolist()
                    else:
                        activations[layer_id] = activation
                        
            else:
                # Generate some placeholder activations based on layers
                layers = self._extract_layer_data(network)
                
                for layer in layers:
                    if layer["neurons"] > 0:
                        # Generate random activations for demonstration
                        act_values = np.random.rand(min(layer["neurons"], 100)).tolist()
                        activations[layer["id"]] = act_values
                        
        except Exception as e:
            logger.error(f"Error extracting activations: {str(e)}")
            
        return activations
    
    def _extract_weights(self, network):
        """
        Extract weight values from a neural network
        
        Args:
            network: Neural network to extract from
            
        Returns:
            Dictionary mapping layer IDs to weight matrices
        """
        weights = {}
        
        try:
            # This would need to be customized for the specific neural network framework
            # Here's a generic example
            if hasattr(network, "get_weights"):
                # If the network has a method to get weights
                raw_weights = network.get_weights()
                
                for layer_id, weight in raw_weights.items():
                    # Convert to list for JSON serialization
                    if isinstance(weight, np.ndarray):
                        # Only include a sample of weights for large matrices
                        if weight.size > 1000:
                            # Sample weights to keep serialization manageable
                            if len(weight.shape) == 2:
                                # 2D matrix, sample rows and columns
                                rows = min(weight.shape[0], 20)
                                cols = min(weight.shape[1], 20)
                                sampled = weight[:rows, :cols].tolist()
                            else:
                                # 1D or other, just take first elements
                                sampled = weight.flatten()[:100].tolist()
                            weights[layer_id] = {
                                "sampled": True,
                                "shape": weight.shape,
                                "values": sampled
                            }
                        else:
                            weights[layer_id] = {
                                "sampled": False,
                                "shape": weight.shape,
                                "values": weight.tolist()
                            }
                    else:
                        weights[layer_id] = {
                            "sampled": False,
                            "shape": [len(weight)],
                            "values": weight
                        }
                        
            else:
                # Generate some placeholder weights based on layers
                layers = self._extract_layer_data(network)
                
                for i, layer in enumerate(layers):
                    if i > 0 and i < len(layers) and layer["neurons"] > 0 and layers[i-1]["neurons"] > 0:
                        # Generate random weights for demonstration
                        rows = min(layer["neurons"], 10)
                        cols = min(layers[i-1]["neurons"], 10)
                        
                        weight_values = np.random.randn(rows, cols).tolist()
                        weights[layer["id"]] = {
                            "sampled": True,
                            "shape": [layer["neurons"], layers[i-1]["neurons"]],
                            "values": weight_values
                        }
                        
        except Exception as e:
            logger.error(f"Error extracting weights: {str(e)}")
            
        return weights
    
    def _generate_mock_state_data(self):
        """
        Generate mock neural state data for testing
        
        Returns:
            Dictionary containing mock state data
        """
        # Generate mock layers
        layer_count = 5
        layers = []
        
        for i in range(layer_count):
            neurons = 64 if i < layer_count - 1 else 10  # Output layer has 10 neurons
            
            layer_info = {
                "id": f"layer_{i}",
                "name": f"Layer {i}",
                "type": "Dense" if i < layer_count - 1 else "Output",
                "neurons": neurons,
                "activation": "relu" if i < layer_count - 1 else "softmax"
            }
            layers.append(layer_info)
        
        # Generate mock activations
        activations = {}
        for layer in layers:
            # Generate random activations
            act_values = np.random.rand(layer["neurons"]).tolist()
            activations[layer["id"]] = act_values
        
        # Generate mock weights
        weights = {}
        for i, layer in enumerate(layers):
            if i > 0:  # Skip first layer (input layer)
                # Generate random weights
                prev_neurons = layers[i-1]["neurons"]
                curr_neurons = layer["neurons"]
                
                weight_values = np.random.randn(curr_neurons, prev_neurons).tolist()
                weights[layer["id"]] = {
                    "sampled": False,
                    "shape": [curr_neurons, prev_neurons],
                    "values": weight_values
                }
        
        # Create mock state data
        state_data = {
            "timestamp": time.time(),
            "id": str(uuid.uuid4()),
            "layers": layers,
            "activations": activations,
            "weights": weights,
            "state_type": "mock"
        }
        
        return state_data
    
    def start_periodic_updates(self, network=None):
        """
        Start periodic neural state updates
        
        Args:
            network: Neural network to collect state from (None for mock data)
        """
        if self.running:
            logger.warning("Periodic updates already running")
            return
            
        self.running = True
        
        # Set mock mode if no network provided
        if network is None:
            self.mock_mode = True
            logger.info("Starting periodic updates with mock data")
        else:
            self.mock_mode = False
            logger.info("Starting periodic updates with real network data")
        
        # Create and start update thread
        def update_thread():
            while self.running:
                try:
                    self.collect_state(network)
                except Exception as e:
                    logger.error(f"Error in periodic update: {str(e)}")
                finally:
                    time.sleep(self.update_interval)
        
        self.update_thread = threading.Thread(target=update_thread)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop_periodic_updates(self):
        """Stop periodic neural state updates"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2)
            self.update_thread = None
        logger.info("Stopped periodic updates")
    
    def get_latest_state(self):
        """
        Get the latest neural state
        
        Returns:
            Latest state data or None if no data available
        """
        if len(self.data_buffer) > 0:
            return self.data_buffer.get_latest(1)[0]
        return None
    
    def get_state_history(self, count=10):
        """
        Get historical neural state data
        
        Args:
            count: Number of historical states to get
            
        Returns:
            List of state data dictionaries
        """
        all_data = self.data_buffer.get_all()
        return all_data[-count:]
    
    def get_network_structure(self):
        """
        Get the structure of the neural network
        
        Returns:
            Dictionary containing network structure information
        """
        latest_state = self.get_latest_state()
        
        if not latest_state:
            # Return minimal structure if no state available
            return {
                "layers": [],
                "connections": []
            }
        
        # Extract layers from latest state
        layers = latest_state.get("layers", [])
        
        # Create connections between layers
        connections = []
        for i in range(len(layers) - 1):
            source_id = layers[i]["id"]
            target_id = layers[i + 1]["id"]
            
            connections.append({
                "source": source_id,
                "target": target_id,
                "strength": 1.0,
                "type": "forward"
            })
        
        # Create network structure
        structure = {
            "layers": layers,
            "connections": connections
        }
        
        return structure
    
    def get_socket_descriptor(self):
        """
        Return socket descriptor for frontend integration
        
        Returns:
            Socket descriptor dictionary
        """
        return {
            "plugin_id": self.node_id,
            "message_types": ["neural_state_update", "state_request", "state_response"],
            "data_format": "json",
            "subscription_mode": "push",
            "ui_components": self.ui_requirements
        } 