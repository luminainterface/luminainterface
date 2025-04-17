#!/usr/bin/env python3
"""
Neural Linguistic Bridge

This module provides a comprehensive bridge between neural network components and 
linguistic processing systems, enabling bidirectional communication and pattern exchange.

Key capabilities:
- Advanced pattern translation between neural and linguistic systems
- Asynchronous communication protocol with priority queuing
- Comprehensive monitoring and diagnostics
- Adaptive learning based on feedback loops
"""

import os
import sys
import logging
import time
import threading
import json
import numpy as np
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural-linguistic-bridge")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import required modules
try:
    from language.neural_linguistic_processor import NeuralLinguisticProcessor
    from language.neural_linguistic_flex_bridge import NeuralLinguisticFlexBridge
    # Import monitoring components
    from monitoring.metrics_system import track_metric, register_component
except ImportError as e:
    logger.warning(f"Some imports failed: {e}. Full functionality may not be available.")

class NeuralLinguisticBridge:
    """
    Advanced bridge between neural and linguistic systems.
    
    This bridge enables:
    1. Advanced pattern translation between neural and linguistic domains
    2. Asynchronous bidirectional communication with priority queuing
    3. Comprehensive monitoring and diagnostics
    4. Seamless integration with both neural and linguistic components
    5. Adaptive learning through feedback mechanisms
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Neural Linguistic Bridge
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.mock_mode = self.config.get("mock_mode", False)
        self.debug_mode = self.config.get("debug_mode", False)
        self.use_flex_bridge = self.config.get("use_flex_bridge", True)
        
        # Communication queues with priority support
        self.to_neural_queue = queue.PriorityQueue()
        self.to_linguistic_queue = queue.PriorityQueue()
        
        # Monitoring metrics
        self.metrics = {
            "neural_to_linguistic_messages": 0,
            "linguistic_to_neural_messages": 0,
            "pattern_translations": 0,
            "translation_errors": 0,
            "average_processing_time_ms": 0,
            "queue_backlog": 0,
            "last_activity_timestamp": time.time()
        }
        
        # Recent message history for diagnostics
        self.message_history = deque(maxlen=100)
        
        # Components
        self.nlp_processor = None
        self.flex_bridge = None
        
        # Processing
        self.processing_threads = []
        self.stop_event = threading.Event()
        self.is_running = False
        
        # Translation matrices (for direct pattern translation)
        self.linguistic_to_neural_matrix = None
        self.neural_to_linguistic_matrix = None
        self.embedding_dim = self.config.get("embedding_dim", 256)
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize neural and linguistic components"""
        # Create or connect to the Neural Linguistic Processor
        try:
            processor_config = {
                "mock_mode": self.mock_mode,
                "data_dir": self.config.get("data_dir", "data/neural_linguistic")
            }
            
            self.nlp_processor = NeuralLinguisticProcessor(config=processor_config)
            logger.info("Neural Linguistic Processor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neural Linguistic Processor: {e}")
            self.nlp_processor = None
        
        # Connect to FlexBridge if enabled
        if self.use_flex_bridge:
            try:
                flex_config = {
                    "mock_mode": self.mock_mode,
                    "embedding_dim": self.embedding_dim
                }
                
                self.flex_bridge = NeuralLinguisticFlexBridge(config=flex_config)
                logger.info("Connected to Neural Linguistic FlexBridge")
                
            except Exception as e:
                logger.error(f"Failed to connect to NeuralLinguisticFlexBridge: {e}")
                self.flex_bridge = None
        
        # Initialize translation matrices
        if not self.flex_bridge:
            self._initialize_translation_matrices()
            
        # Register with monitoring system if available
        try:
            register_component("neural_linguistic_bridge", self)
            logger.info("Registered with monitoring system")
        except:
            logger.debug("Monitoring system registration not available")
            
    def _initialize_translation_matrices(self):
        """Initialize matrices for pattern translation"""
        # For linguistic to neural transformation
        self.linguistic_to_neural_matrix = np.random.randn(8, self.embedding_dim) * 0.1
        
        # For neural to linguistic transformation
        self.neural_to_linguistic_matrix = np.random.randn(self.embedding_dim, 8) * 0.1
        
        logger.info(f"Initialized translation matrices with dimension {self.embedding_dim}")
    
    def start(self):
        """Start the bridge and processing threads"""
        if self.is_running:
            logger.info("Bridge is already running")
            return True
            
        logger.info("Starting Neural Linguistic Bridge")
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start the NLP processor
        if self.nlp_processor and hasattr(self.nlp_processor, "start_processor"):
            self.nlp_processor.start_processor()
        
        # Start FlexBridge if available
        if self.flex_bridge and hasattr(self.flex_bridge, "start"):
            self.flex_bridge.start()
        
        # Start processing threads
        self._start_processing_threads()
        
        self.is_running = True
        logger.info("Neural Linguistic Bridge started successfully")
        return True
        
    def stop(self):
        """Stop the bridge and processing threads"""
        if not self.is_running:
            return True
            
        logger.info("Stopping Neural Linguistic Bridge")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Stop FlexBridge if available
        if self.flex_bridge and hasattr(self.flex_bridge, "stop"):
            self.flex_bridge.stop()
        
        # Stop the NLP processor
        if self.nlp_processor and hasattr(self.nlp_processor, "stop_processor"):
            self.nlp_processor.stop_processor()
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        self.processing_threads = []
        self.is_running = False
        logger.info("Neural Linguistic Bridge stopped")
        return True
        
    def _start_processing_threads(self):
        """Start asynchronous processing threads"""
        # Thread for neural queue processing
        neural_thread = threading.Thread(
            target=self._process_neural_queue,
            daemon=True,
            name="NeuralQueueProcessor"
        )
        
        # Thread for linguistic queue processing
        linguistic_thread = threading.Thread(
            target=self._process_linguistic_queue,
            daemon=True,
            name="LinguisticQueueProcessor"
        )
        
        # Thread for monitoring
        monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="BridgeMonitoring"
        )
        
        # Start threads
        neural_thread.start()
        linguistic_thread.start()
        monitoring_thread.start()
        
        # Store thread references
        self.processing_threads = [neural_thread, linguistic_thread, monitoring_thread]
        
        logger.info("Started bridge processing threads")
        
    def _process_neural_queue(self):
        """Process messages from linguistic to neural systems"""
        while not self.stop_event.is_set():
            try:
                # Check if queue is empty
                if self.to_neural_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                
                # Get item from queue (with priority)
                priority, timestamp, item = self.to_neural_queue.get(block=False)
                
                # Extract data and callback
                data = item.get("data", {})
                callback = item.get("callback")
                context = item.get("context", {})
                
                # Process start time
                start_time = time.time()
                
                # Process using FlexBridge if available
                if self.flex_bridge:
                    # Use the FlexBridge for processing
                    result = self.flex_bridge.process_text(
                        text=data.get("text", ""),
                        context=context
                    )
                else:
                    # Use direct processing
                    result = self._process_linguistic_to_neural(data)
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Update metrics
                self._update_metrics("linguistic_to_neural", processing_time)
                
                # Add to message history
                self._add_to_history("linguistic_to_neural", data, result)
                
                # Handle callback if provided
                if callback and callable(callback):
                    callback(result)
                
                # Mark task as done
                self.to_neural_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing linguistic to neural: {e}")
                if self.debug_mode:
                    import traceback
                    logger.debug(traceback.format_exc())
                time.sleep(0.1)  # Wait before retrying
                
    def _process_linguistic_queue(self):
        """Process messages from neural to linguistic systems"""
        while not self.stop_event.is_set():
            try:
                # Check if queue is empty
                if self.to_linguistic_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                
                # Get item from queue (with priority)
                priority, timestamp, item = self.to_linguistic_queue.get(block=False)
                
                # Extract data and callback
                data = item.get("data", {})
                callback = item.get("callback")
                context = item.get("context", {})
                
                # Process start time
                start_time = time.time()
                
                # Process using NLP processor
                if self.nlp_processor:
                    result = self._process_neural_to_linguistic(data)
                else:
                    result = {"error": "No NLP processor available"}
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Update metrics
                self._update_metrics("neural_to_linguistic", processing_time)
                
                # Add to message history
                self._add_to_history("neural_to_linguistic", data, result)
                
                # Handle callback if provided
                if callback and callable(callback):
                    callback(result)
                
                # Mark task as done
                self.to_linguistic_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing neural to linguistic: {e}")
                if self.debug_mode:
                    import traceback
                    logger.debug(traceback.format_exc())
                time.sleep(0.1)  # Wait before retrying
    
    def _monitoring_loop(self):
        """Monitor bridge performance and health"""
        while not self.stop_event.is_set():
            try:
                # Update queue backlog metrics
                self.metrics["queue_backlog"] = (
                    self.to_neural_queue.qsize() + 
                    self.to_linguistic_queue.qsize()
                )
                
                # Push metrics to monitoring system if available
                try:
                    for metric, value in self.metrics.items():
                        track_metric(f"neural_linguistic_bridge.{metric}", value)
                except:
                    pass  # Monitoring not available
                
                # Check for stalled processing
                if (time.time() - self.metrics["last_activity_timestamp"] > 300 and  # 5 minutes
                    self.metrics["queue_backlog"] > 0):
                    logger.warning("Possible stalled processing detected - queues have pending items but no activity")
                
                # Periodic log of status
                if self.debug_mode:
                    logger.debug(f"Bridge status: {json.dumps(self.get_status())}")
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer after an error
    
    def _process_linguistic_to_neural(self, linguistic_data):
        """
        Process linguistic data to neural representation
        
        Args:
            linguistic_data: The linguistic data to process
            
        Returns:
            Neural representation
        """
        try:
            # Extract pattern parameters
            if isinstance(linguistic_data, dict):
                if "pattern_params" in linguistic_data:
                    pattern_params = linguistic_data["pattern_params"]
                elif "text" in linguistic_data:
                    # Process text to extract pattern parameters
                    if self.nlp_processor:
                        analysis = self.nlp_processor.analyze_linguistic_pattern(linguistic_data["text"])
                        pattern_params = analysis.get("pattern_params", {})
                    else:
                        pattern_params = {}
                else:
                    pattern_params = linguistic_data
            else:
                pattern_params = {}
            
            # Extract parameter values (with defaults)
            param_values = [
                pattern_params.get("resonance_factor", 0.75),
                pattern_params.get("recursion_depth", 3.0) / 5.0,  # Normalize
                pattern_params.get("semantic_density", 0.65),
                pattern_params.get("temporal_coherence", 0.7),
                pattern_params.get("pattern_complexity", 0.6),
                pattern_params.get("symmetry_factor", 0.5),
                pattern_params.get("harmony_index", 0.5),
                pattern_params.get("depth_of_context", 0.4)
            ]
            
            # Convert to vector
            param_vector = np.array(param_values)
            
            # Apply transformation to neural embedding
            neural_embedding = np.dot(param_vector, self.linguistic_to_neural_matrix)
            
            # Construct result
            result = {
                "embedding": neural_embedding,
                "source": "neural_linguistic_bridge",
                "timestamp": time.time(),
                "original_data": linguistic_data
            }
            
            # Update pattern translation count
            self.metrics["pattern_translations"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in linguistic to neural processing: {e}")
            self.metrics["translation_errors"] += 1
            return {"error": str(e), "timestamp": time.time()}
    
    def _process_neural_to_linguistic(self, neural_data):
        """
        Process neural data to linguistic representation
        
        Args:
            neural_data: The neural data to process
            
        Returns:
            Linguistic representation
        """
        try:
            # Extract neural embedding
            if isinstance(neural_data, dict):
                if "embedding" in neural_data and isinstance(neural_data["embedding"], np.ndarray):
                    embedding = neural_data["embedding"]
                elif "result" in neural_data and isinstance(neural_data["result"], np.ndarray):
                    embedding = neural_data["result"]
                else:
                    # No valid embedding found
                    raise ValueError("No valid neural embedding found in data")
            elif isinstance(neural_data, np.ndarray):
                embedding = neural_data
            else:
                raise ValueError(f"Unsupported neural data type: {type(neural_data)}")
            
            # Apply transformation to linguistic parameters
            param_vector = np.dot(embedding, self.neural_to_linguistic_matrix)
            
            # Normalize and constrain parameter values
            param_vector = np.clip(param_vector, 0.1, 0.95)
            
            # Convert to linguistic parameters
            linguistic_params = {
                "resonance_factor": float(param_vector[0]),
                "recursion_depth": max(1, min(5, int(round(param_vector[1] * 5)))),
                "semantic_density": float(param_vector[2]),
                "temporal_coherence": float(param_vector[3]),
                "pattern_complexity": float(param_vector[4]),
                "symmetry_factor": float(param_vector[5]),
                "harmony_index": float(param_vector[6]),
                "depth_of_context": float(param_vector[7])
            }
            
            # Apply to NLP processor if available
            if self.nlp_processor and hasattr(self.nlp_processor, "update_pattern_parameters"):
                self.nlp_processor.update_pattern_parameters(linguistic_params)
            
            # Construct result
            result = {
                "pattern_params": linguistic_params,
                "source": "neural_linguistic_bridge",
                "timestamp": time.time(),
                "original_data": neural_data if isinstance(neural_data, dict) else None
            }
            
            # Update pattern translation count
            self.metrics["pattern_translations"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in neural to linguistic processing: {e}")
            self.metrics["translation_errors"] += 1
            return {"error": str(e), "timestamp": time.time()}
    
    def send_to_neural(self, linguistic_data, callback=None, priority=1, context=None):
        """
        Send linguistic data to neural processing queue
        
        Args:
            linguistic_data: The linguistic data to process
            callback: Optional callback function for results
            priority: Priority level (lower numbers = higher priority)
            context: Optional context information
            
        Returns:
            bool: True if successfully queued
        """
        if not self.is_running:
            logger.warning("Cannot send data - bridge is not running")
            return False
        
        # Create queue item
        item = {
            "data": linguistic_data,
            "callback": callback,
            "context": context or {},
            "timestamp": time.time()
        }
        
        # Add to priority queue (priority, timestamp for FIFO within same priority, data)
        self.to_neural_queue.put((priority, time.time(), item))
        
        # Update metrics
        self.metrics["linguistic_to_neural_messages"] += 1
        self.metrics["last_activity_timestamp"] = time.time()
        
        return True
    
    def send_to_linguistic(self, neural_data, callback=None, priority=1, context=None):
        """
        Send neural data to linguistic processing queue
        
        Args:
            neural_data: The neural data to process
            callback: Optional callback function for results
            priority: Priority level (lower numbers = higher priority)
            context: Optional context information
            
        Returns:
            bool: True if successfully queued
        """
        if not self.is_running:
            logger.warning("Cannot send data - bridge is not running")
            return False
        
        # Create queue item
        item = {
            "data": neural_data,
            "callback": callback,
            "context": context or {},
            "timestamp": time.time()
        }
        
        # Add to priority queue (priority, timestamp for FIFO within same priority, data)
        self.to_linguistic_queue.put((priority, time.time(), item))
        
        # Update metrics
        self.metrics["neural_to_linguistic_messages"] += 1
        self.metrics["last_activity_timestamp"] = time.time()
        
        return True
    
    def process_text(self, text, context=None, callback=None):
        """
        Process text through the bridge (convenience method)
        
        Args:
            text: The text to process
            context: Optional context information
            callback: Optional callback for results
            
        Returns:
            Processing result or None if async
        """
        if not self.is_running:
            logger.warning("Cannot process text - bridge is not running")
            return {"error": "Bridge not running"}
        
        # If using FlexBridge, delegate to it
        if self.flex_bridge:
            return self.flex_bridge.process_text(text, context)
        
        # If NLP processor available, get linguistic analysis
        if self.nlp_processor:
            linguistic_analysis = self.nlp_processor.analyze_linguistic_pattern(text, context)
            
            if callback:
                # Async processing with callback
                self.send_to_neural(linguistic_analysis, callback=callback)
                return None
            else:
                # Synchronous processing
                neural_result = self._process_linguistic_to_neural(linguistic_analysis)
                linguistic_result = self._process_neural_to_linguistic(neural_result)
                
                # Combine results
                return {
                    "text": text,
                    "linguistic_analysis": linguistic_analysis,
                    "neural_result": neural_result,
                    "updated_params": linguistic_result.get("pattern_params"),
                    "timestamp": time.time()
                }
        else:
            return {"error": "No linguistic processor available"}
    
    def _update_metrics(self, direction, processing_time):
        """Update processing metrics"""
        # Update last activity timestamp
        self.metrics["last_activity_timestamp"] = time.time()
        
        # Update message counts
        if direction == "linguistic_to_neural":
            self.metrics["linguistic_to_neural_messages"] += 1
        else:
            self.metrics["neural_to_linguistic_messages"] += 1
        
        # Update average processing time (simple moving average)
        current_avg = self.metrics["average_processing_time_ms"]
        if current_avg == 0:
            self.metrics["average_processing_time_ms"] = processing_time
        else:
            self.metrics["average_processing_time_ms"] = (
                current_avg * 0.9 + processing_time * 0.1  # 90% old, 10% new
            )
    
    def _add_to_history(self, direction, input_data, output_data):
        """Add processing result to message history"""
        # Create history entry
        entry = {
            "direction": direction,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data
        }
        
        # Add to deque (automatically handles max length)
        self.message_history.append(entry)
    
    def get_status(self):
        """
        Get comprehensive bridge status
        
        Returns:
            Dict with status information
        """
        status = {
            "is_running": self.is_running,
            "metrics": self.metrics.copy(),
            "components": {
                "nlp_processor_available": self.nlp_processor is not None,
                "flex_bridge_available": self.flex_bridge is not None
            },
            "queues": {
                "to_neural_size": self.to_neural_queue.qsize(),
                "to_linguistic_size": self.to_linguistic_queue.qsize(),
            },
            "translation_matrices_initialized": (
                self.linguistic_to_neural_matrix is not None and 
                self.neural_to_linguistic_matrix is not None
            ),
            "message_history_length": len(self.message_history),
            "timestamp": time.time()
        }
        
        # Add FlexBridge status if available
        if self.flex_bridge and hasattr(self.flex_bridge, "get_status"):
            try:
                status["flex_bridge_status"] = self.flex_bridge.get_status()
            except Exception as e:
                status["flex_bridge_status_error"] = str(e)
        
        # Add NLP processor status if available
        if self.nlp_processor and hasattr(self.nlp_processor, "get_status"):
            try:
                status["nlp_processor_status"] = self.nlp_processor.get_status()
            except Exception as e:
                status["nlp_processor_status_error"] = str(e)
        
        return status

# Singleton instance
_bridge_instance = None

def get_neural_linguistic_bridge(config=None):
    """
    Get the singleton Neural Linguistic Bridge instance
    
    Args:
        config: Optional configuration
        
    Returns:
        The Neural Linguistic Bridge instance
    """
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = NeuralLinguisticBridge(config)
    
    return _bridge_instance


if __name__ == "__main__":
    # Simple test
    bridge = get_neural_linguistic_bridge({"mock_mode": True, "debug_mode": True})
    bridge.start()
    
    # Process sample text
    result = bridge.process_text("Neural networks can learn linguistic patterns through advanced pattern recognition algorithms.")
    print(f"Processing result: {json.dumps(result, indent=2)}")
    
    # Get status
    status = bridge.get_status()
    print(f"Bridge status: {json.dumps(status, indent=2)}")
    
    # Stop bridge
    bridge.stop() 
"""
Neural Linguistic Bridge

This module provides a comprehensive bridge between neural network components and 
linguistic processing systems, enabling bidirectional communication and pattern exchange.

Key capabilities:
- Advanced pattern translation between neural and linguistic systems
- Asynchronous communication protocol with priority queuing
- Comprehensive monitoring and diagnostics
- Adaptive learning based on feedback loops
"""

import os
import sys
import logging
import time
import threading
import json
import numpy as np
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural-linguistic-bridge")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import required modules
try:
    from language.neural_linguistic_processor import NeuralLinguisticProcessor
    from language.neural_linguistic_flex_bridge import NeuralLinguisticFlexBridge
    # Import monitoring components
    from monitoring.metrics_system import track_metric, register_component
except ImportError as e:
    logger.warning(f"Some imports failed: {e}. Full functionality may not be available.")

class NeuralLinguisticBridge:
    """
    Advanced bridge between neural and linguistic systems.
    
    This bridge enables:
    1. Advanced pattern translation between neural and linguistic domains
    2. Asynchronous bidirectional communication with priority queuing
    3. Comprehensive monitoring and diagnostics
    4. Seamless integration with both neural and linguistic components
    5. Adaptive learning through feedback mechanisms
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Neural Linguistic Bridge
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.mock_mode = self.config.get("mock_mode", False)
        self.debug_mode = self.config.get("debug_mode", False)
        self.use_flex_bridge = self.config.get("use_flex_bridge", True)
        
        # Communication queues with priority support
        self.to_neural_queue = queue.PriorityQueue()
        self.to_linguistic_queue = queue.PriorityQueue()
        
        # Monitoring metrics
        self.metrics = {
            "neural_to_linguistic_messages": 0,
            "linguistic_to_neural_messages": 0,
            "pattern_translations": 0,
            "translation_errors": 0,
            "average_processing_time_ms": 0,
            "queue_backlog": 0,
            "last_activity_timestamp": time.time()
        }
        
        # Recent message history for diagnostics
        self.message_history = deque(maxlen=100)
        
        # Components
        self.nlp_processor = None
        self.flex_bridge = None
        
        # Processing
        self.processing_threads = []
        self.stop_event = threading.Event()
        self.is_running = False
        
        # Translation matrices (for direct pattern translation)
        self.linguistic_to_neural_matrix = None
        self.neural_to_linguistic_matrix = None
        self.embedding_dim = self.config.get("embedding_dim", 256)
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize neural and linguistic components"""
        # Create or connect to the Neural Linguistic Processor
        try:
            processor_config = {
                "mock_mode": self.mock_mode,
                "data_dir": self.config.get("data_dir", "data/neural_linguistic")
            }
            
            self.nlp_processor = NeuralLinguisticProcessor(config=processor_config)
            logger.info("Neural Linguistic Processor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neural Linguistic Processor: {e}")
            self.nlp_processor = None
        
        # Connect to FlexBridge if enabled
        if self.use_flex_bridge:
            try:
                flex_config = {
                    "mock_mode": self.mock_mode,
                    "embedding_dim": self.embedding_dim
                }
                
                self.flex_bridge = NeuralLinguisticFlexBridge(config=flex_config)
                logger.info("Connected to Neural Linguistic FlexBridge")
                
            except Exception as e:
                logger.error(f"Failed to connect to NeuralLinguisticFlexBridge: {e}")
                self.flex_bridge = None
        
        # Initialize translation matrices
        if not self.flex_bridge:
            self._initialize_translation_matrices()
            
        # Register with monitoring system if available
        try:
            register_component("neural_linguistic_bridge", self)
            logger.info("Registered with monitoring system")
        except:
            logger.debug("Monitoring system registration not available")
            
    def _initialize_translation_matrices(self):
        """Initialize matrices for pattern translation"""
        # For linguistic to neural transformation
        self.linguistic_to_neural_matrix = np.random.randn(8, self.embedding_dim) * 0.1
        
        # For neural to linguistic transformation
        self.neural_to_linguistic_matrix = np.random.randn(self.embedding_dim, 8) * 0.1
        
        logger.info(f"Initialized translation matrices with dimension {self.embedding_dim}")
    
    def start(self):
        """Start the bridge and processing threads"""
        if self.is_running:
            logger.info("Bridge is already running")
            return True
            
        logger.info("Starting Neural Linguistic Bridge")
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start the NLP processor
        if self.nlp_processor and hasattr(self.nlp_processor, "start_processor"):
            self.nlp_processor.start_processor()
        
        # Start FlexBridge if available
        if self.flex_bridge and hasattr(self.flex_bridge, "start"):
            self.flex_bridge.start()
        
        # Start processing threads
        self._start_processing_threads()
        
        self.is_running = True
        logger.info("Neural Linguistic Bridge started successfully")
        return True
        
    def stop(self):
        """Stop the bridge and processing threads"""
        if not self.is_running:
            return True
            
        logger.info("Stopping Neural Linguistic Bridge")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Stop FlexBridge if available
        if self.flex_bridge and hasattr(self.flex_bridge, "stop"):
            self.flex_bridge.stop()
        
        # Stop the NLP processor
        if self.nlp_processor and hasattr(self.nlp_processor, "stop_processor"):
            self.nlp_processor.stop_processor()
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        self.processing_threads = []
        self.is_running = False
        logger.info("Neural Linguistic Bridge stopped")
        return True
        
    def _start_processing_threads(self):
        """Start asynchronous processing threads"""
        # Thread for neural queue processing
        neural_thread = threading.Thread(
            target=self._process_neural_queue,
            daemon=True,
            name="NeuralQueueProcessor"
        )
        
        # Thread for linguistic queue processing
        linguistic_thread = threading.Thread(
            target=self._process_linguistic_queue,
            daemon=True,
            name="LinguisticQueueProcessor"
        )
        
        # Thread for monitoring
        monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="BridgeMonitoring"
        )
        
        # Start threads
        neural_thread.start()
        linguistic_thread.start()
        monitoring_thread.start()
        
        # Store thread references
        self.processing_threads = [neural_thread, linguistic_thread, monitoring_thread]
        
        logger.info("Started bridge processing threads")
        
    def _process_neural_queue(self):
        """Process messages from linguistic to neural systems"""
        while not self.stop_event.is_set():
            try:
                # Check if queue is empty
                if self.to_neural_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                
                # Get item from queue (with priority)
                priority, timestamp, item = self.to_neural_queue.get(block=False)
                
                # Extract data and callback
                data = item.get("data", {})
                callback = item.get("callback")
                context = item.get("context", {})
                
                # Process start time
                start_time = time.time()
                
                # Process using FlexBridge if available
                if self.flex_bridge:
                    # Use the FlexBridge for processing
                    result = self.flex_bridge.process_text(
                        text=data.get("text", ""),
                        context=context
                    )
                else:
                    # Use direct processing
                    result = self._process_linguistic_to_neural(data)
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Update metrics
                self._update_metrics("linguistic_to_neural", processing_time)
                
                # Add to message history
                self._add_to_history("linguistic_to_neural", data, result)
                
                # Handle callback if provided
                if callback and callable(callback):
                    callback(result)
                
                # Mark task as done
                self.to_neural_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing linguistic to neural: {e}")
                if self.debug_mode:
                    import traceback
                    logger.debug(traceback.format_exc())
                time.sleep(0.1)  # Wait before retrying
                
    def _process_linguistic_queue(self):
        """Process messages from neural to linguistic systems"""
        while not self.stop_event.is_set():
            try:
                # Check if queue is empty
                if self.to_linguistic_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                
                # Get item from queue (with priority)
                priority, timestamp, item = self.to_linguistic_queue.get(block=False)
                
                # Extract data and callback
                data = item.get("data", {})
                callback = item.get("callback")
                context = item.get("context", {})
                
                # Process start time
                start_time = time.time()
                
                # Process using NLP processor
                if self.nlp_processor:
                    result = self._process_neural_to_linguistic(data)
                else:
                    result = {"error": "No NLP processor available"}
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Update metrics
                self._update_metrics("neural_to_linguistic", processing_time)
                
                # Add to message history
                self._add_to_history("neural_to_linguistic", data, result)
                
                # Handle callback if provided
                if callback and callable(callback):
                    callback(result)
                
                # Mark task as done
                self.to_linguistic_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing neural to linguistic: {e}")
                if self.debug_mode:
                    import traceback
                    logger.debug(traceback.format_exc())
                time.sleep(0.1)  # Wait before retrying
    
    def _monitoring_loop(self):
        """Monitor bridge performance and health"""
        while not self.stop_event.is_set():
            try:
                # Update queue backlog metrics
                self.metrics["queue_backlog"] = (
                    self.to_neural_queue.qsize() + 
                    self.to_linguistic_queue.qsize()
                )
                
                # Push metrics to monitoring system if available
                try:
                    for metric, value in self.metrics.items():
                        track_metric(f"neural_linguistic_bridge.{metric}", value)
                except:
                    pass  # Monitoring not available
                
                # Check for stalled processing
                if (time.time() - self.metrics["last_activity_timestamp"] > 300 and  # 5 minutes
                    self.metrics["queue_backlog"] > 0):
                    logger.warning("Possible stalled processing detected - queues have pending items but no activity")
                
                # Periodic log of status
                if self.debug_mode:
                    logger.debug(f"Bridge status: {json.dumps(self.get_status())}")
                
                # Wait before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer after an error
    
    def _process_linguistic_to_neural(self, linguistic_data):
        """
        Process linguistic data to neural representation
        
        Args:
            linguistic_data: The linguistic data to process
            
        Returns:
            Neural representation
        """
        try:
            # Extract pattern parameters
            if isinstance(linguistic_data, dict):
                if "pattern_params" in linguistic_data:
                    pattern_params = linguistic_data["pattern_params"]
                elif "text" in linguistic_data:
                    # Process text to extract pattern parameters
                    if self.nlp_processor:
                        analysis = self.nlp_processor.analyze_linguistic_pattern(linguistic_data["text"])
                        pattern_params = analysis.get("pattern_params", {})
                    else:
                        pattern_params = {}
                else:
                    pattern_params = linguistic_data
            else:
                pattern_params = {}
            
            # Extract parameter values (with defaults)
            param_values = [
                pattern_params.get("resonance_factor", 0.75),
                pattern_params.get("recursion_depth", 3.0) / 5.0,  # Normalize
                pattern_params.get("semantic_density", 0.65),
                pattern_params.get("temporal_coherence", 0.7),
                pattern_params.get("pattern_complexity", 0.6),
                pattern_params.get("symmetry_factor", 0.5),
                pattern_params.get("harmony_index", 0.5),
                pattern_params.get("depth_of_context", 0.4)
            ]
            
            # Convert to vector
            param_vector = np.array(param_values)
            
            # Apply transformation to neural embedding
            neural_embedding = np.dot(param_vector, self.linguistic_to_neural_matrix)
            
            # Construct result
            result = {
                "embedding": neural_embedding,
                "source": "neural_linguistic_bridge",
                "timestamp": time.time(),
                "original_data": linguistic_data
            }
            
            # Update pattern translation count
            self.metrics["pattern_translations"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in linguistic to neural processing: {e}")
            self.metrics["translation_errors"] += 1
            return {"error": str(e), "timestamp": time.time()}
    
    def _process_neural_to_linguistic(self, neural_data):
        """
        Process neural data to linguistic representation
        
        Args:
            neural_data: The neural data to process
            
        Returns:
            Linguistic representation
        """
        try:
            # Extract neural embedding
            if isinstance(neural_data, dict):
                if "embedding" in neural_data and isinstance(neural_data["embedding"], np.ndarray):
                    embedding = neural_data["embedding"]
                elif "result" in neural_data and isinstance(neural_data["result"], np.ndarray):
                    embedding = neural_data["result"]
                else:
                    # No valid embedding found
                    raise ValueError("No valid neural embedding found in data")
            elif isinstance(neural_data, np.ndarray):
                embedding = neural_data
            else:
                raise ValueError(f"Unsupported neural data type: {type(neural_data)}")
            
            # Apply transformation to linguistic parameters
            param_vector = np.dot(embedding, self.neural_to_linguistic_matrix)
            
            # Normalize and constrain parameter values
            param_vector = np.clip(param_vector, 0.1, 0.95)
            
            # Convert to linguistic parameters
            linguistic_params = {
                "resonance_factor": float(param_vector[0]),
                "recursion_depth": max(1, min(5, int(round(param_vector[1] * 5)))),
                "semantic_density": float(param_vector[2]),
                "temporal_coherence": float(param_vector[3]),
                "pattern_complexity": float(param_vector[4]),
                "symmetry_factor": float(param_vector[5]),
                "harmony_index": float(param_vector[6]),
                "depth_of_context": float(param_vector[7])
            }
            
            # Apply to NLP processor if available
            if self.nlp_processor and hasattr(self.nlp_processor, "update_pattern_parameters"):
                self.nlp_processor.update_pattern_parameters(linguistic_params)
            
            # Construct result
            result = {
                "pattern_params": linguistic_params,
                "source": "neural_linguistic_bridge",
                "timestamp": time.time(),
                "original_data": neural_data if isinstance(neural_data, dict) else None
            }
            
            # Update pattern translation count
            self.metrics["pattern_translations"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in neural to linguistic processing: {e}")
            self.metrics["translation_errors"] += 1
            return {"error": str(e), "timestamp": time.time()}
    
    def send_to_neural(self, linguistic_data, callback=None, priority=1, context=None):
        """
        Send linguistic data to neural processing queue
        
        Args:
            linguistic_data: The linguistic data to process
            callback: Optional callback function for results
            priority: Priority level (lower numbers = higher priority)
            context: Optional context information
            
        Returns:
            bool: True if successfully queued
        """
        if not self.is_running:
            logger.warning("Cannot send data - bridge is not running")
            return False
        
        # Create queue item
        item = {
            "data": linguistic_data,
            "callback": callback,
            "context": context or {},
            "timestamp": time.time()
        }
        
        # Add to priority queue (priority, timestamp for FIFO within same priority, data)
        self.to_neural_queue.put((priority, time.time(), item))
        
        # Update metrics
        self.metrics["linguistic_to_neural_messages"] += 1
        self.metrics["last_activity_timestamp"] = time.time()
        
        return True
    
    def send_to_linguistic(self, neural_data, callback=None, priority=1, context=None):
        """
        Send neural data to linguistic processing queue
        
        Args:
            neural_data: The neural data to process
            callback: Optional callback function for results
            priority: Priority level (lower numbers = higher priority)
            context: Optional context information
            
        Returns:
            bool: True if successfully queued
        """
        if not self.is_running:
            logger.warning("Cannot send data - bridge is not running")
            return False
        
        # Create queue item
        item = {
            "data": neural_data,
            "callback": callback,
            "context": context or {},
            "timestamp": time.time()
        }
        
        # Add to priority queue (priority, timestamp for FIFO within same priority, data)
        self.to_linguistic_queue.put((priority, time.time(), item))
        
        # Update metrics
        self.metrics["neural_to_linguistic_messages"] += 1
        self.metrics["last_activity_timestamp"] = time.time()
        
        return True
    
    def process_text(self, text, context=None, callback=None):
        """
        Process text through the bridge (convenience method)
        
        Args:
            text: The text to process
            context: Optional context information
            callback: Optional callback for results
            
        Returns:
            Processing result or None if async
        """
        if not self.is_running:
            logger.warning("Cannot process text - bridge is not running")
            return {"error": "Bridge not running"}
        
        # If using FlexBridge, delegate to it
        if self.flex_bridge:
            return self.flex_bridge.process_text(text, context)
        
        # If NLP processor available, get linguistic analysis
        if self.nlp_processor:
            linguistic_analysis = self.nlp_processor.analyze_linguistic_pattern(text, context)
            
            if callback:
                # Async processing with callback
                self.send_to_neural(linguistic_analysis, callback=callback)
                return None
            else:
                # Synchronous processing
                neural_result = self._process_linguistic_to_neural(linguistic_analysis)
                linguistic_result = self._process_neural_to_linguistic(neural_result)
                
                # Combine results
                return {
                    "text": text,
                    "linguistic_analysis": linguistic_analysis,
                    "neural_result": neural_result,
                    "updated_params": linguistic_result.get("pattern_params"),
                    "timestamp": time.time()
                }
        else:
            return {"error": "No linguistic processor available"}
    
    def _update_metrics(self, direction, processing_time):
        """Update processing metrics"""
        # Update last activity timestamp
        self.metrics["last_activity_timestamp"] = time.time()
        
        # Update message counts
        if direction == "linguistic_to_neural":
            self.metrics["linguistic_to_neural_messages"] += 1
        else:
            self.metrics["neural_to_linguistic_messages"] += 1
        
        # Update average processing time (simple moving average)
        current_avg = self.metrics["average_processing_time_ms"]
        if current_avg == 0:
            self.metrics["average_processing_time_ms"] = processing_time
        else:
            self.metrics["average_processing_time_ms"] = (
                current_avg * 0.9 + processing_time * 0.1  # 90% old, 10% new
            )
    
    def _add_to_history(self, direction, input_data, output_data):
        """Add processing result to message history"""
        # Create history entry
        entry = {
            "direction": direction,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data
        }
        
        # Add to deque (automatically handles max length)
        self.message_history.append(entry)
    
    def get_status(self):
        """
        Get comprehensive bridge status
        
        Returns:
            Dict with status information
        """
        status = {
            "is_running": self.is_running,
            "metrics": self.metrics.copy(),
            "components": {
                "nlp_processor_available": self.nlp_processor is not None,
                "flex_bridge_available": self.flex_bridge is not None
            },
            "queues": {
                "to_neural_size": self.to_neural_queue.qsize(),
                "to_linguistic_size": self.to_linguistic_queue.qsize(),
            },
            "translation_matrices_initialized": (
                self.linguistic_to_neural_matrix is not None and 
                self.neural_to_linguistic_matrix is not None
            ),
            "message_history_length": len(self.message_history),
            "timestamp": time.time()
        }
        
        # Add FlexBridge status if available
        if self.flex_bridge and hasattr(self.flex_bridge, "get_status"):
            try:
                status["flex_bridge_status"] = self.flex_bridge.get_status()
            except Exception as e:
                status["flex_bridge_status_error"] = str(e)
        
        # Add NLP processor status if available
        if self.nlp_processor and hasattr(self.nlp_processor, "get_status"):
            try:
                status["nlp_processor_status"] = self.nlp_processor.get_status()
            except Exception as e:
                status["nlp_processor_status_error"] = str(e)
        
        return status

# Singleton instance
_bridge_instance = None

def get_neural_linguistic_bridge(config=None):
    """
    Get the singleton Neural Linguistic Bridge instance
    
    Args:
        config: Optional configuration
        
    Returns:
        The Neural Linguistic Bridge instance
    """
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = NeuralLinguisticBridge(config)
    
    return _bridge_instance


if __name__ == "__main__":
    # Simple test
    bridge = get_neural_linguistic_bridge({"mock_mode": True, "debug_mode": True})
    bridge.start()
    
    # Process sample text
    result = bridge.process_text("Neural networks can learn linguistic patterns through advanced pattern recognition algorithms.")
    print(f"Processing result: {json.dumps(result, indent=2)}")
    
    # Get status
    status = bridge.get_status()
    print(f"Bridge status: {json.dumps(status, indent=2)}")
    
    # Stop bridge
    bridge.stop() 