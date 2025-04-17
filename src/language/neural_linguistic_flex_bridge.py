#!/usr/bin/env python3
"""
Neural Linguistic FlexNode Bridge

This module creates a bridge between the NeuralLinguisticProcessor and the FlexNode system,
enabling bidirectional communication between language processing and neural network components.

Key capabilities:
- Convert linguistic patterns to neural embeddings
- Send language analysis to FlexNode for processing
- Receive neural network feedback and integrate into language processing
- Create adaptive learning feedback loops between systems
"""

import os
import sys
import logging
import time
import threading
import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from queue import Queue
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural-linguistic-flex-bridge")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import required components
try:
    from language.flex_node import FlexNode, create_flex_node
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# Mock NeuralLinguisticProcessor for testing
class MockNeuralLinguisticProcessor:
    """Mock implementation of NeuralLinguisticProcessor for testing"""
    
    def __init__(self, llm_weight=0.5, nn_weight=0.5, config=None):
        """Initialize with basic configuration"""
        self.llm_weight = llm_weight
        self.nn_weight = nn_weight
        self.pattern_params = {
            "resonance_factor": 0.75,
            "recursion_depth": 3,
            "symmetry_type": "radial",
            "semantic_density": 0.65,
            "temporal_coherence": 0.7,
            "pattern_complexity": 0.6
        }
        self.is_processing = False
        self.processor_thread = None
        self.result_cache = {}
        logger.info(f"Initialized MockNeuralLinguisticProcessor with LLM weight {llm_weight}")
    
    def start_processor(self):
        """Start the processor (mock)"""
        self.is_processing = True
        logger.info("Mock neural linguistic processor started")
    
    def stop_processor(self):
        """Stop the processor (mock)"""
        self.is_processing = False
        logger.info("Mock neural linguistic processor stopped")
    
    def analyze_linguistic_pattern(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze linguistic patterns in text (mock)"""
        logger.info(f"Analyzing linguistic pattern: {text[:30]}...")
        
        # Create mock analysis
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        
        features = {
            "word_count": word_count,
            "unique_word_ratio": unique_words / max(1, word_count),
            "avg_word_length": sum(len(w) for w in words) / max(1, word_count),
            "complexity_score": 0.4 + (unique_words / max(1, word_count)) * 0.6
        }
        
        pattern_params = {
            "resonance_factor": min(0.3 + features["unique_word_ratio"] * 0.7, 0.95),
            "complexity": features["complexity_score"],
            "recursion_depth": max(2, min(5, int(features["complexity_score"] * 5))),
            "semantic_density": 0.3 + features["unique_word_ratio"] * 0.7,
            "pattern_complexity": features["complexity_score"],
            "temporal_coherence": 0.7,
            "symmetry_factor": 0.5
        }
        
        return {
            "text": text,
            "features": features,
            "key_phrases": [" ".join(words[i:i+3]) for i in range(min(5, len(words)-2))],
            "pattern_params": pattern_params,
            "timestamp": time.time()
        }
    
    def update_pattern_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update linguistic pattern parameters (mock)"""
        # Update parameters
        for param, value in parameters.items():
            if param in self.pattern_params:
                self.pattern_params[param] = value
        
        return self.pattern_params
    
    def generate_fractal_pattern(self, linguistic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fractal pattern from linguistic analysis (mock)"""
        params = linguistic_analysis.get("pattern_params", {})
        
        # Create a simple mock pattern
        pattern = {
            "pattern_type": "fractal",
            "core_frequency": params.get("resonance_factor", 0.75),
            "recursion_depth": params.get("recursion_depth", 3),
            "complexity": params.get("pattern_complexity", 0.6),
            "generated": True,
            "timestamp": time.time()
        }
        
        return pattern
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status (mock)"""
        return {
            "is_processing": self.is_processing,
            "pattern_parameters": self.pattern_params,
            "cache_size": len(self.result_cache),
            "mock_mode": True,
            "timestamp": time.time()
        }
    
    def set_llm_weight(self, weight: float) -> None:
        """Set LLM weight (mock)"""
        self.llm_weight = weight
    
    def set_nn_weight(self, weight: float) -> None:
        """Set neural network weight (mock)"""
        self.nn_weight = weight

class NeuralLinguisticFlexBridge:
    """
    Bridge between the NeuralLinguisticProcessor and FlexNode systems.
    
    This class enables:
    1. Converting linguistic patterns to neural embeddings
    2. Sending processed language data to the neural network
    3. Receiving feedback from neural processing for language adaptation
    4. Creating an adaptive learning loop between the systems
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Neural Linguistic FlexNode Bridge
        
        Args:
            config: Configuration parameters for the bridge (optional)
        """
        self.config = config or {}
        self.mock_mode = self.config.get("mock_mode", False)
        self.embedding_dim = self.config.get("embedding_dim", 256)
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.feedback_alpha = self.config.get("feedback_alpha", 0.3)
        self.pattern_weight = self.config.get("pattern_weight", 0.7)
        
        # Processing queues for asynchronous operations
        self.to_neural_queue = Queue()
        self.to_linguistic_queue = Queue()
        
        # Initialize components
        self.nlp = None
        self.flex_node = None
        self.initialized = False
        self.is_running = False
        self.processing_threads = []
        
        # Pattern transformation matrices
        self.linguistic_to_neural_matrix = None
        self.neural_to_linguistic_matrix = None
        
        # Statistics and metrics
        self.stats = {
            "linguistic_to_neural_count": 0,
            "neural_to_linguistic_count": 0,
            "adaptations_performed": 0,
            "processing_time_ms": 0,
            "last_activity": time.time()
        }
        
        # Initialize the bridge
        self._initialize_bridge()
        
    def _initialize_bridge(self) -> None:
        """Initialize the bridge components"""
        logger.info("Initializing Neural Linguistic FlexNode Bridge")
        
        # Initialize the NeuralLinguisticProcessor
        try:
            if not self.mock_mode:
                # Try to get existing processor
                try:
                    from neural_linguistic_processor import NeuralLinguisticProcessor
                    self.nlp = NeuralLinguisticProcessor(
                        llm_weight=self.config.get("llm_weight", 0.5),
                        nn_weight=self.config.get("nn_weight", 0.5)
                    )
                    logger.info("Using real NeuralLinguisticProcessor")
                except (ImportError, AttributeError, TypeError) as e:
                    logger.warning(f"Real NeuralLinguisticProcessor not available: {e}. Using mock instead.")
                    self.nlp = MockNeuralLinguisticProcessor()
            else:
                # Create a mock processor
                self.nlp = MockNeuralLinguisticProcessor(
                    llm_weight=self.config.get("llm_weight", 0.5),
                    nn_weight=self.config.get("nn_weight", 0.5)
                )
                logger.info("Created MockNeuralLinguisticProcessor")
                
        except Exception as e:
            logger.error(f"Failed to initialize NeuralLinguisticProcessor: {e}")
            self.nlp = None
        
        # Initialize FlexNode
        try:
            if not self.mock_mode:
                # Try to create a new FlexNode
                self.flex_node = create_flex_node(
                    embedding_dim=self.embedding_dim,
                    hidden_dims=[512, 256]
                )
                logger.info(f"Created FlexNode with ID: {self.flex_node.node_id}")
            else:
                # Create a minimal mock FlexNode for testing
                self.flex_node = self._create_mock_flex_node()
                logger.info("Created mock FlexNode for testing")
                
        except Exception as e:
            logger.error(f"Failed to initialize FlexNode: {e}")
            self.flex_node = None
            
        # Initialize transformation matrices
        self._initialize_transformation_matrices()
        
        # Set bridge status
        self.initialized = (self.nlp is not None and self.flex_node is not None)
        logger.info(f"Bridge initialization complete. Status: {'Success' if self.initialized else 'Failed'}")
    
    def _create_mock_flex_node(self) -> Any:
        """Create a mock FlexNode for testing"""
        # Simple object with the necessary methods
        class MockFlexNode:
            def __init__(self):
                self.node_id = f"MockFlexNode_{int(time.time())}"
                self.embedding_dim = 256
                self.processed_data = []
                self.is_active = False
                
            def process(self, data):
                self.processed_data.append(data)
                # Simple transformation for testing
                if isinstance(data, dict) and "embedding" in data:
                    return {"processed": True, "result": np.array(data["embedding"]) * 1.1}
                return {"processed": True, "result": "mock_result"}
                
            def connect_to_node(self, node_id, node_instance, connection_type="default", 
                               weight=0.5, bidirectional=False):
                return True
                
            def start(self):
                self.is_active = True
                
            def stop(self):
                self.is_active = False
                
            def get_metrics(self):
                return {"processed_messages": len(self.processed_data)}
        
        return MockFlexNode()
    
    def _initialize_transformation_matrices(self) -> None:
        """Initialize transformation matrices for pattern conversion"""
        # For transforming linguistic patterns to neural embeddings
        self.linguistic_to_neural_matrix = np.random.randn(6, self.embedding_dim) * 0.1
        
        # For transforming neural patterns back to linguistic parameters
        self.neural_to_linguistic_matrix = np.random.randn(self.embedding_dim, 6) * 0.1
        
        logger.info(f"Initialized transformation matrices with embedding dim {self.embedding_dim}")
    
    def connect_components(self) -> bool:
        """Connect the NLP and FlexNode components"""
        if not self.initialized:
            logger.error("Cannot connect components: Bridge not initialized")
            return False
            
        logger.info("Connecting NeuralLinguisticProcessor and FlexNode")
        
        # Connect FlexNode to NLP (for neural network to send feedback)
        if hasattr(self.flex_node, "connect_to_node"):
            try:
                self.flex_node.connect_to_node(
                    node_id="neural_linguistic_processor",
                    node_instance=self.nlp,
                    connection_type="language_processing",
                    weight=0.8,
                    bidirectional=True
                )
                logger.info("FlexNode connected to NeuralLinguisticProcessor")
            except Exception as e:
                logger.error(f"Failed to connect FlexNode to NLP: {e}")
                return False
        
        return True
    
    def start(self) -> bool:
        """Start the bridge processing"""
        if not self.initialized:
            logger.error("Cannot start: Bridge not initialized")
            return False
            
        if self.is_running:
            logger.info("Bridge is already running")
            return True
            
        logger.info("Starting Neural Linguistic FlexNode Bridge")
        
        # Start the NeuralLinguisticProcessor
        if hasattr(self.nlp, "start_processor"):
            self.nlp.start_processor()
        
        # Start the FlexNode
        if hasattr(self.flex_node, "start"):
            self.flex_node.start()
        
        # Start processing threads
        self._start_processing_threads()
        
        self.is_running = True
        logger.info("Neural Linguistic FlexNode Bridge started")
        return True
    
    def stop(self) -> bool:
        """Stop the bridge processing"""
        if not self.is_running:
            return True
            
        logger.info("Stopping Neural Linguistic FlexNode Bridge")
        
        # Stop the NeuralLinguisticProcessor
        if hasattr(self.nlp, "stop_processor"):
            self.nlp.stop_processor()
        
        # Stop the FlexNode
        if hasattr(self.flex_node, "stop"):
            self.flex_node.stop()
        
        # Stop processing threads
        self._stop_processing_threads()
        
        self.is_running = False
        logger.info("Neural Linguistic FlexNode Bridge stopped")
        return True
    
    def _start_processing_threads(self) -> None:
        """Start the background processing threads"""
        # Thread for processing linguistic->neural queue
        to_neural_thread = threading.Thread(
            target=self._process_to_neural_queue,
            daemon=True,
            name="ToNeuralThread"
        )
        
        # Thread for processing neural->linguistic queue
        to_linguistic_thread = threading.Thread(
            target=self._process_to_linguistic_queue,
            daemon=True,
            name="ToLinguisticThread"
        )
        
        # Start threads
        to_neural_thread.start()
        to_linguistic_thread.start()
        
        # Store thread references
        self.processing_threads = [to_neural_thread, to_linguistic_thread]
        
        logger.info("Started processing threads")
    
    def _stop_processing_threads(self) -> None:
        """Stop the background processing threads"""
        # Signal threads to stop (threads are daemon, so they'll exit when the program exits)
        self.is_running = False
        
        # Wait for threads to finish (with timeout)
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        self.processing_threads = []
        logger.info("Stopped processing threads")
    
    def _process_to_neural_queue(self) -> None:
        """Process items in the linguistic->neural queue"""
        while self.is_running:
            try:
                if self.to_neural_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                
                # Get item from queue
                item = self.to_neural_queue.get(block=False)
                linguistic_data = item.get("data", {})
                callback = item.get("callback")
                
                # Process the data
                neural_embedding = self._linguistic_to_neural(linguistic_data)
                
                # Send to FlexNode
                result = None
                if self.flex_node is not None:
                    start_time = time.time()
                    result = self.flex_node.process({
                        "source": "neural_linguistic_processor",
                        "embedding": neural_embedding,
                        "original_data": linguistic_data,
                        "timestamp": time.time()
                    })
                    processing_time = (time.time() - start_time) * 1000  # ms
                    self.stats["processing_time_ms"] = processing_time
                
                # Update stats
                self.stats["linguistic_to_neural_count"] += 1
                self.stats["last_activity"] = time.time()
                
                # Call callback if provided
                if callback and callable(callback):
                    callback(result)
                
                self.to_neural_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing linguistic->neural item: {e}")
                time.sleep(0.1)
    
    def _process_to_linguistic_queue(self) -> None:
        """Process items in the neural->linguistic queue"""
        while self.is_running:
            try:
                if self.to_linguistic_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                
                # Get item from queue
                item = self.to_linguistic_queue.get(block=False)
                neural_data = item.get("data", {})
                callback = item.get("callback")
                
                # Process the data
                linguistic_params = self._neural_to_linguistic(neural_data)
                
                # Send to NLP
                result = None
                if self.nlp is not None:
                    start_time = time.time()
                    result = self.nlp.update_pattern_parameters(linguistic_params)
                    processing_time = (time.time() - start_time) * 1000  # ms
                    self.stats["processing_time_ms"] = processing_time
                
                # Update stats
                self.stats["neural_to_linguistic_count"] += 1
                self.stats["last_activity"] = time.time()
                
                # Call callback if provided
                if callback and callable(callback):
                    callback(result)
                
                self.to_linguistic_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing neural->linguistic item: {e}")
                time.sleep(0.1)
    
    def _linguistic_to_neural(self, linguistic_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert linguistic patterns to neural embeddings
        
        Args:
            linguistic_data: Linguistic data with pattern parameters
            
        Returns:
            Neural embedding vector
        """
        # Extract pattern parameters
        pattern_params = linguistic_data.get("pattern_params", {})
        if not pattern_params:
            # If no pattern params, check if the data itself is the params
            if "resonance_factor" in linguistic_data:
                pattern_params = linguistic_data
        
        # Create parameter vector
        param_vector = np.array([
            pattern_params.get("resonance_factor", 0.75),
            pattern_params.get("recursion_depth", 3.0),
            pattern_params.get("semantic_density", 0.65),
            pattern_params.get("temporal_coherence", 0.7),
            pattern_params.get("pattern_complexity", 0.6),
            pattern_params.get("symmetry_factor", 0.5)
        ])
        
        # Apply transformation to get neural embedding
        embedding = np.dot(param_vector, self.linguistic_to_neural_matrix)
        
        # Add some randomness for exploration (reduced over time)
        if self.stats["adaptations_performed"] < 100:
            noise_level = max(0.1, 0.5 - self.stats["adaptations_performed"] / 200)
            embedding += np.random.randn(self.embedding_dim) * noise_level
        
        return embedding
    
    def _neural_to_linguistic(self, neural_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert neural output back to linguistic parameters
        
        Args:
            neural_data: Neural processing results
            
        Returns:
            Updated linguistic parameters
        """
        # Extract neural embedding
        if "result" in neural_data and isinstance(neural_data["result"], np.ndarray):
            embedding = neural_data["result"]
        elif "embedding" in neural_data and isinstance(neural_data["embedding"], np.ndarray):
            embedding = neural_data["embedding"]
        else:
            # Default to random embedding if none found
            embedding = np.random.randn(self.embedding_dim)
        
        # Apply transformation to get linguistic parameters
        param_vector = np.dot(embedding, self.neural_to_linguistic_matrix)
        
        # Normalize and constrain parameter values
        param_vector = np.clip(param_vector, 0.1, 0.95)
        
        # Convert to parameter dictionary
        linguistic_params = {
            "resonance_factor": float(param_vector[0]),
            "recursion_depth": max(1, min(5, int(round(param_vector[1] * 5)))),
            "semantic_density": float(param_vector[2]),
            "temporal_coherence": float(param_vector[3]),
            "pattern_complexity": float(param_vector[4]),
            "symmetry_factor": float(param_vector[5])
        }
        
        return linguistic_params
    
    def process_text(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process text through the neural-linguistic bridge
        
        Args:
            text: Text to process
            context: Additional context (optional)
            
        Returns:
            Processing results
        """
        if not self.initialized:
            return {"error": "Bridge not initialized"}
        
        logger.info(f"Processing text through neural-linguistic bridge: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Step 1: Process text through NLP
        linguistic_analysis = self.nlp.analyze_linguistic_pattern(text, context)
        
        # Step 2: Convert to neural embedding and process through FlexNode
        neural_embedding = self._linguistic_to_neural(linguistic_analysis)
        
        flex_result = None
        if self.flex_node is not None:
            flex_result = self.flex_node.process({
                "source": "neural_linguistic_processor",
                "text": text,
                "embedding": neural_embedding,
                "linguistic_analysis": linguistic_analysis,
                "timestamp": time.time()
            })
        
        # Step 3: Convert neural result back to linguistic parameters (for feedback)
        updated_params = None
        if flex_result is not None:
            updated_params = self._neural_to_linguistic(flex_result)
            
            # Apply feedback to NLP (adapt parameters)
            if self.nlp is not None and hasattr(self.nlp, "update_pattern_parameters"):
                self.nlp.update_pattern_parameters(updated_params)
                self.stats["adaptations_performed"] += 1
        
        # Step 4: Generate visualization pattern based on combined results
        pattern_result = None
        if self.nlp is not None and hasattr(self.nlp, "generate_fractal_pattern"):
            # If we got parameters back from FlexNode, use them to influence the pattern
            if updated_params:
                # Blend original analysis with flex node feedback
                blended_analysis = linguistic_analysis.copy()
                if "pattern_params" in blended_analysis:
                    # Update pattern params using weighted average
                    for param, value in updated_params.items():
                        if param in blended_analysis["pattern_params"]:
                            original = blended_analysis["pattern_params"][param]
                            blended_analysis["pattern_params"][param] = (
                                original * (1 - self.feedback_alpha) + 
                                value * self.feedback_alpha
                            )
                
                pattern_result = self.nlp.generate_fractal_pattern(blended_analysis)
            else:
                pattern_result = self.nlp.generate_fractal_pattern(linguistic_analysis)
        
        # Combine results
        return {
            "text": text,
            "linguistic_analysis": linguistic_analysis,
            "neural_result": flex_result,
            "updated_params": updated_params,
            "pattern_result": pattern_result,
            "processing_timestamp": time.time()
        }
    
    def send_to_neural(self, linguistic_data: Dict[str, Any], 
                      callback: callable = None) -> None:
        """
        Send linguistic data to neural processing (async)
        
        Args:
            linguistic_data: Linguistic data to process
            callback: Callback function for results (optional)
        """
        self.to_neural_queue.put({
            "data": linguistic_data,
            "callback": callback,
            "timestamp": time.time()
        })
    
    def send_to_linguistic(self, neural_data: Dict[str, Any],
                          callback: callable = None) -> None:
        """
        Send neural data to linguistic processing (async)
        
        Args:
            neural_data: Neural data to process
            callback: Callback function for results (optional)
        """
        self.to_linguistic_queue.put({
            "data": neural_data,
            "callback": callback,
            "timestamp": time.time()
        })
    
    def adapt_transformation_matrices(self, learning_samples: List[Tuple[Dict, np.ndarray]]) -> None:
        """
        Adapt transformation matrices based on learning samples
        
        Args:
            learning_samples: List of (linguistic_data, neural_embedding) pairs
        """
        if not learning_samples:
            return
        
        logger.info(f"Adapting transformation matrices with {len(learning_samples)} samples")
        
        # Extract training data
        X = []  # Linguistic parameters
        Y = []  # Neural embeddings
        
        for linguistic_data, neural_embedding in learning_samples:
            # Extract pattern parameters
            pattern_params = linguistic_data.get("pattern_params", {})
            if not pattern_params and "resonance_factor" in linguistic_data:
                pattern_params = linguistic_data
                
            # Skip if no valid pattern parameters
            if not pattern_params:
                continue
                
            param_vector = np.array([
                pattern_params.get("resonance_factor", 0.75),
                pattern_params.get("recursion_depth", 3.0) / 5.0,  # Normalize
                pattern_params.get("semantic_density", 0.65),
                pattern_params.get("temporal_coherence", 0.7),
                pattern_params.get("pattern_complexity", 0.6),
                pattern_params.get("symmetry_factor", 0.5)
            ])
            
            X.append(param_vector)
            Y.append(neural_embedding)
        
        if not X or not Y:
            return
            
        # Convert to numpy arrays
        X = np.array(X)
        Y = np.array(Y)
        
        # Simple gradient descent to update linguistic_to_neural_matrix
        for i in range(10):  # 10 iterations
            # Predict neural embeddings
            Y_pred = np.dot(X, self.linguistic_to_neural_matrix)
            
            # Calculate error
            error = Y - Y_pred
            
            # Update matrix
            gradient = np.dot(X.T, error) / len(X)
            self.linguistic_to_neural_matrix += self.learning_rate * gradient
        
        # Similarly update neural_to_linguistic_matrix
        for i in range(10):
            # Predict linguistic parameters
            X_pred = np.dot(Y, self.neural_to_linguistic_matrix)
            
            # Calculate error
            error = X - X_pred
            
            # Update matrix
            gradient = np.dot(Y.T, error) / len(Y)
            self.neural_to_linguistic_matrix += self.learning_rate * gradient
        
        logger.info("Transformation matrices adapted")
        self.stats["adaptations_performed"] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get bridge status and statistics
        
        Returns:
            Status information
        """
        status = {
            "initialized": self.initialized,
            "is_running": self.is_running,
            "mock_mode": self.mock_mode,
            "stats": self.stats,
            "components": {
                "nlp_available": self.nlp is not None,
                "flex_node_available": self.flex_node is not None
            },
            "queues": {
                "to_neural_size": self.to_neural_queue.qsize(),
                "to_linguistic_size": self.to_linguistic_queue.qsize()
            },
            "timestamp": time.time()
        }
        
        # Add FlexNode metrics if available
        if self.flex_node is not None and hasattr(self.flex_node, "get_metrics"):
            try:
                status["flex_node_metrics"] = self.flex_node.get_metrics()
            except Exception as e:
                status["flex_node_metrics_error"] = str(e)
        
        # Add NLP metrics if available
        if self.nlp is not None and hasattr(self.nlp, "get_status"):
            try:
                status["nlp_status"] = self.nlp.get_status()
            except Exception as e:
                status["nlp_status_error"] = str(e)
        
        return status
    
    def _process_linguistic_data(self, linguistic_data: Dict[str, Any], callback: callable = None) -> None:
        """
        Process linguistic data directly (for testing)
        
        Args:
            linguistic_data: Linguistic data to process
            callback: Callback function for result
        """
        # Process directly without using the queue
        try:
            # Extract pattern parameters
            pattern_params = linguistic_data.get("pattern_params", {})
            if not pattern_params and "resonance_factor" in linguistic_data:
                pattern_params = linguistic_data
                
            # Update NLP with the parameters
            result = None
            if self.nlp is not None:
                result = self.nlp.update_pattern_parameters(pattern_params)
                
            # Update stats
            self.stats["neural_to_linguistic_count"] += 1
            self.stats["last_activity"] = time.time()
                
            # Call callback if provided
            if callback and callable(callback):
                callback(result)
                
        except Exception as e:
            logger.error(f"Error processing linguistic data directly: {e}")
            if callback and callable(callback):
                callback({"error": str(e)})
    
    def _process_neural_data(self, neural_data: Dict[str, Any], callback: callable = None) -> None:
        """
        Process neural data directly (for testing)
        
        Args:
            neural_data: Neural data to process
            callback: Callback function for result
        """
        # Process directly without using the queue
        try:
            # Extract embedding
            embedding = None
            if "embedding" in neural_data and isinstance(neural_data["embedding"], np.ndarray):
                embedding = neural_data["embedding"]
            else:
                # Generate random embedding for testing
                embedding = np.random.randn(self.embedding_dim)
                
            # Process through FlexNode
            result = None
            if self.flex_node is not None:
                result = self.flex_node.process({
                    "source": "test_direct_processing",
                    "embedding": embedding,
                    "timestamp": time.time()
                })
                
            # Update stats
            self.stats["linguistic_to_neural_count"] += 1
            self.stats["last_activity"] = time.time()
                
            # Call callback if provided
            if callback and callable(callback):
                callback(result)
                
        except Exception as e:
            logger.error(f"Error processing neural data directly: {e}")
            if callback and callable(callback):
                callback({"error": str(e)})


def get_neural_linguistic_flex_bridge(config: Dict[str, Any] = None) -> NeuralLinguisticFlexBridge:
    """
    Get or create a NeuralLinguisticFlexBridge instance
    
    Args:
        config: Configuration parameters
        
    Returns:
        NeuralLinguisticFlexBridge instance
    """
    # Singleton instance
    if not hasattr(get_neural_linguistic_flex_bridge, "_instance") or \
       get_neural_linguistic_flex_bridge._instance is None:
        logger.info("Creating new NeuralLinguisticFlexBridge instance")
        get_neural_linguistic_flex_bridge._instance = NeuralLinguisticFlexBridge(config)
    
    return get_neural_linguistic_flex_bridge._instance


if __name__ == "__main__":
    # Test the bridge
    bridge = get_neural_linguistic_flex_bridge({"mock_mode": True})
    
    if bridge.initialized:
        bridge.start()
        
        # Test processing
        result = bridge.process_text("Neural networks learn patterns through adjusting weights based on error signals.")
        print(f"Processing result: {json.dumps(result, indent=2)}")
        
        # Test async processing
        def callback(result):
            print(f"Async callback result: {result}")
            
        bridge.send_to_neural({"pattern_params": {"resonance_factor": 0.8}}, callback)
        
        time.sleep(2)  # Wait for async processing
        
        # Print status
        status = bridge.get_status()
        print(f"Bridge status: {json.dumps(status, indent=2)}")
        
        bridge.stop()
    else:
        print("Failed to initialize bridge") 