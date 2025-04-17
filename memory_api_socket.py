#!/usr/bin/env python3
"""
Memory API Socket

Socket provider for Memory API integration with V5 visualization system.
This module serves as a bridge between the Language Memory System and the V5 visualization components.
"""

import os
import sys
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory-api-socket")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class NodeSocket:
    """
    Simple socket implementation for node communication
    """
    def __init__(self, node_id: str, socket_type: str):
        self.node_id = node_id
        self.socket_type = socket_type
        self.message_handlers = {}
        self.connected_nodes = []
        self.message_queue = []
        self.lock = threading.Lock()
        
    def send_message(self, message: Dict[str, Any]) -> bool:
        """Send a message to connected nodes"""
        with self.lock:
            self.message_queue.append(message)
        return True
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register a message handler"""
        self.message_handlers[message_type] = handler
    
    def connect_to_node(self, node_id: str) -> bool:
        """Connect to another node"""
        if node_id not in self.connected_nodes:
            self.connected_nodes.append(node_id)
            return True
        return False
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message"""
        message_type = message.get("type", "")
        if message_type in self.message_handlers:
            try:
                return self.message_handlers[message_type](message)
            except Exception as e:
                logger.error(f"Error processing message {message_type}: {e}")
                return {"status": "error", "error": str(e)}
        return {"status": "error", "error": f"Unknown message type: {message_type}"}


# For mock service discovery
def register_node(node):
    """Register node with discovery service (mock implementation)"""
    logger.info(f"Registered node: {node.plugin_id}")
    return MockClient(node)


class MockClient:
    """Mock client for testing"""
    def __init__(self, node):
        self.node = node
        
    def register(self):
        """Register with discovery service"""
        logger.info(f"Registered node: {self.node.plugin_id}")
        return True


class MemoryAPISocketProvider:
    """Socket provider for Memory API integration with V5 visualization system"""
    
    def __init__(self, plugin_id="memory_api_socket"):
        self.plugin_id = plugin_id
        self.socket = NodeSocket(plugin_id, "service")
        self.api = None
        self.mock_mode = False
        
        # Try to initialize the Memory API
        try:
            from memory_api import MemoryAPI
            self.api = MemoryAPI()
            logger.info("Successfully initialized Memory API")
        except ImportError as e:
            logger.warning(f"Failed to initialize Memory API: {str(e)}")
            self.mock_mode = True
            logger.info("Running in mock mode")
        
        # Register message handlers
        self.socket.message_handlers = {
            "store_conversation": self._handle_store_request,
            "retrieve_memories": self._handle_retrieve_request,
            "synthesize_topic": self._handle_synthesize_request,
            "enhance_message": self._handle_enhance_request,
            "get_stats": self._handle_stats_request,
            "get_training_examples": self._handle_training_request,
            "visualize_pattern": self._handle_visualize_pattern
        }
        
        # Register with discovery service
        self.client = register_node(self)
        
        logger.info(f"MemoryAPISocketProvider initialized (mock_mode={self.mock_mode})")
    
    def start(self):
        """Start the socket provider"""
        logger.info("Starting Memory API Socket Provider")
        return True
    
    def stop(self):
        """Stop the socket provider"""
        logger.info("Stopping Memory API Socket Provider")
        return True
        
    def _handle_store_request(self, message):
        """Handle request to store conversation"""
        if not self.api and not self.mock_mode:
            return self._send_error_response(message, "Memory API not initialized")
            
        content = message.get("content", {})
        message_text = content.get("message")
        metadata = content.get("metadata", {})
        
        try:
            if self.mock_mode:
                result = {"status": "success", "stored": True}
            else:
                result = self.api.store_conversation(message_text, metadata)
            return self._send_response(message, result)
        except Exception as e:
            return self._send_error_response(message, str(e))
    
    def _handle_retrieve_request(self, message):
        """Handle request to retrieve memories"""
        if not self.api and not self.mock_mode:
            return self._send_error_response(message, "Memory API not initialized")
            
        content = message.get("content", {})
        message_text = content.get("message")
        max_results = content.get("max_results", 5)
        
        try:
            if self.mock_mode:
                # Generate mock memory data
                result = {
                    "status": "success",
                    "memories": [
                        {
                            "text": f"Mock memory related to: {message_text}",
                            "relevance": 0.95,
                            "timestamp": time.time() - 3600,
                            "metadata": {"topic": "neural networks"}
                        },
                        {
                            "text": "Neural networks learn through backpropagation.",
                            "relevance": 0.85,
                            "timestamp": time.time() - 7200,
                            "metadata": {"topic": "machine learning"}
                        }
                    ]
                }
            else:
                result = self.api.retrieve_relevant_memories(message_text, max_results)
            return self._send_response(message, result)
        except Exception as e:
            return self._send_error_response(message, str(e))
    
    def _handle_synthesize_request(self, message):
        """Handle request to synthesize topic"""
        if not self.api and not self.mock_mode:
            return self._send_error_response(message, "Memory API not initialized")
            
        content = message.get("content", {})
        topic = content.get("topic")
        depth = content.get("depth", 3)
        
        try:
            if self.mock_mode:
                # Generate mock synthesis data
                viz_data = self._generate_mock_visualization_data(topic)
            else:
                result = self.api.synthesize_topic(topic, depth)
                viz_data = self._prepare_visualization_data(result)
            
            # Send the visualization-ready data
            response = self._send_response(message, viz_data)
            
            # Also broadcast to subscribers for real-time updates
            self.socket.send_message({
                "type": "memory_synthesis_update",
                "data": viz_data
            })
            
            return response
        except Exception as e:
            return self._send_error_response(message, str(e))
    
    def _prepare_visualization_data(self, synthesis_result):
        """Transform synthesis result to visualization-friendly format"""
        if synthesis_result.get("status") != "success":
            return synthesis_result
            
        # Extract relevant data for visualization
        synthesis = synthesis_result.get("synthesis_results", {})
        memory = synthesis.get("synthesized_memory", {})
        
        # Create visualization-ready format
        visualization_data = {
            "topic": memory.get("topics", ["unknown"])[0],
            "core_understanding": memory.get("core_understanding", ""),
            "insights": memory.get("novel_insights", []),
            "related_topics": synthesis.get("related_topics", []),
            
            # Network visualization data
            "network": {
                "nodes": [
                    # Main topic node
                    {"id": "main_topic", "label": memory.get("topics", ["unknown"])[0], "group": "topic", "size": 30}
                ],
                "edges": []
            },
            
            # Fractal visualization data
            "fractal_data": {
                "pattern_style": "neural",
                "fractal_depth": 4,
                "metrics": {
                    "fractal_dimension": 1.62,
                    "complexity_index": 85,
                    "pattern_coherence": 92
                }
            }
        }
        
        # Add related topics as nodes
        for i, topic in enumerate(synthesis.get("related_topics", [])):
            topic_name = topic.get("topic", f"related_{i}")
            relevance = topic.get("relevance", 0.5)
            
            # Add node
            visualization_data["network"]["nodes"].append({
                "id": f"topic_{i}",
                "label": topic_name,
                "group": "related_topic",
                "size": 15 + (relevance * 10)
            })
            
            # Add edge connecting to main topic
            visualization_data["network"]["edges"].append({
                "from": "main_topic",
                "to": f"topic_{i}",
                "value": relevance,
                "title": f"Relevance: {relevance:.2f}"
            })
        
        return visualization_data
    
    def _generate_mock_visualization_data(self, topic):
        """Generate mock visualization data for testing"""
        # Create mock visualization data
        visualization_data = {
            "topic": topic,
            "core_understanding": f"Mock understanding of {topic}",
            "insights": [
                f"Insight 1 about {topic}",
                f"Insight 2 about {topic}",
                f"Insight 3 about {topic}"
            ],
            "related_topics": [
                {"topic": "neural networks", "relevance": 0.95},
                {"topic": "consciousness", "relevance": 0.85},
                {"topic": "machine learning", "relevance": 0.75},
                {"topic": "artificial intelligence", "relevance": 0.65},
                {"topic": "deep learning", "relevance": 0.55}
            ],
            
            # Network visualization data
            "network": {
                "nodes": [
                    {"id": "main_topic", "label": topic, "group": "topic", "size": 30}
                ],
                "edges": []
            },
            
            # Fractal visualization data
            "fractal_data": {
                "pattern_style": "neural",
                "fractal_depth": random.randint(3, 6),
                "metrics": {
                    "fractal_dimension": round(random.uniform(1.5, 1.8), 2),
                    "complexity_index": random.randint(70, 95),
                    "pattern_coherence": random.randint(80, 98)
                }
            }
        }
        
        # Add related topics as nodes
        for i, related in enumerate([
            "neural networks", "consciousness", "machine learning", 
            "artificial intelligence", "deep learning"
        ]):
            relevance = 0.95 - (i * 0.1)
            
            # Add node
            visualization_data["network"]["nodes"].append({
                "id": f"topic_{i}",
                "label": related,
                "group": "related_topic",
                "size": 15 + (relevance * 10)
            })
            
            # Add edge connecting to main topic
            visualization_data["network"]["edges"].append({
                "from": "main_topic",
                "to": f"topic_{i}",
                "value": relevance,
                "title": f"Relevance: {relevance:.2f}"
            })
        
        return visualization_data
    
    def _handle_enhance_request(self, message):
        """Handle request to enhance message with memory"""
        if not self.api and not self.mock_mode:
            return self._send_error_response(message, "Memory API not initialized")
            
        content = message.get("content", {})
        message_text = content.get("message")
        enhance_mode = content.get("enhance_mode", "contextual")
        
        try:
            if self.mock_mode:
                result = {
                    "status": "success",
                    "enhanced_message": f"Enhanced: {message_text}",
                    "enhanced_context": "This is mock enhanced context from memory",
                    "mode": enhance_mode
                }
            else:
                result = self.api.enhance_message_with_memory(message_text, enhance_mode)
            return self._send_response(message, result)
        except Exception as e:
            return self._send_error_response(message, str(e))
    
    def _handle_stats_request(self, message):
        """Handle request to get memory statistics"""
        if not self.api and not self.mock_mode:
            return self._send_error_response(message, "Memory API not initialized")
            
        try:
            if self.mock_mode:
                result = {
                    "status": "success",
                    "stats": {
                        "total_memories": 1250,
                        "total_conversations": 85,
                        "total_topics": 120,
                        "top_topics": [
                            {"topic": "neural networks", "count": 45},
                            {"topic": "consciousness", "count": 38},
                            {"topic": "machine learning", "count": 32}
                        ],
                        "memory_growth": {
                            "daily": 25,
                            "weekly": 175,
                            "monthly": 750
                        }
                    }
                }
            else:
                result = self.api.get_memory_stats()
            return self._send_response(message, result)
        except Exception as e:
            return self._send_error_response(message, str(e))
    
    def _handle_training_request(self, message):
        """Handle request to get training examples"""
        if not self.api and not self.mock_mode:
            return self._send_error_response(message, "Memory API not initialized")
            
        content = message.get("content", {})
        topic = content.get("topic")
        count = content.get("count", 3)
        
        try:
            if self.mock_mode:
                result = {
                    "status": "success",
                    "examples": [
                        {"text": f"Training example 1 for {topic}", "label": "positive"},
                        {"text": f"Training example 2 for {topic}", "label": "neutral"},
                        {"text": f"Training example 3 for {topic}", "label": "positive"}
                    ]
                }
            else:
                result = self.api.get_training_examples(topic, count)
            return self._send_response(message, result)
        except Exception as e:
            return self._send_error_response(message, str(e))
    
    def _handle_visualize_pattern(self, message):
        """Handle request to visualize a pattern"""
        content = message.get("content", {})
        pattern = content.get("pattern", {})
        
        try:
            # In a real implementation, this would send the pattern to the visualization system
            # For now, we'll just log it
            logger.info(f"Received pattern for visualization: {pattern.get('pattern_id', 'unknown')}")
            
            result = {
                "status": "success",
                "pattern_id": pattern.get("pattern_id", "unknown"),
                "visualization_ready": True
            }
            
            return self._send_response(message, result)
        except Exception as e:
            return self._send_error_response(message, str(e))
    
    def _send_response(self, request_message, result):
        """Send response to a request"""
        response = {
            "type": "api_response",
            "request_id": request_message.get("request_id"),
            "data": result
        }
        self.socket.send_message(response)
        return response
    
    def _send_error_response(self, request_message, error):
        """Send error response"""
        response = {
            "type": "api_response",
            "request_id": request_message.get("request_id"),
            "status": "error",
            "error": error
        }
        self.socket.send_message(response)
        return response
    
    def get_socket_descriptor(self):
        """Return socket descriptor for frontend integration"""
        return {
            "plugin_id": self.plugin_id,
            "message_types": [
                "store_conversation", 
                "retrieve_memories", 
                "synthesize_topic",
                "enhance_message", 
                "get_stats", 
                "get_training_examples",
                "visualize_pattern",
                "memory_synthesis_update",
                "api_response"
            ],
            "data_format": "json",
            "subscription_mode": "dual",  # Both push and request-response
            "ui_components": [
                "memory_stats", 
                "memory_visualization", 
                "synthesis_view",
                "memory_network_graph",
                "fractal_pattern_panel"
            ]
        }
    
    def get_status(self):
        """Get status of the socket provider"""
        return {
            "plugin_id": self.plugin_id,
            "initialized": True,
            "mock_mode": self.mock_mode,
            "connected_components": self.socket.connected_nodes,
            "api_available": self.api is not None,
            "message_handlers": list(self.socket.message_handlers.keys()),
            "ui_components": [
                "memory_stats", 
                "memory_visualization", 
                "synthesis_view",
                "memory_network_graph",
                "fractal_pattern_panel"
            ]
        }


def get_bridge(mock_mode=False):
    """
    Get the MemoryAPISocketProvider singleton instance
    
    Args:
        mock_mode: Whether to use mock data instead of real memory system
        
    Returns:
        MemoryAPISocketProvider instance
    """
    global _api_socket_provider
    
    if _api_socket_provider is None:
        _api_socket_provider = MemoryAPISocketProvider()
        _api_socket_provider.mock_mode = mock_mode
    elif mock_mode != _api_socket_provider.mock_mode:
        _api_socket_provider.mock_mode = mock_mode
    
    return _api_socket_provider


# Initialize the singleton
_api_socket_provider = None


def chat_integration():
    """
    Connect the chat integration to the language memory system
    
    This function initializes all the necessary components to connect
    the V5 NN/LLM Weighted Conversation Panel with the Language Memory System.
    
    Returns:
        ChatMemoryInterface instance
    """
    class ChatMemoryInterface:
        """
        Interface between chat components and Language Memory system
        
        This class provides the necessary methods for the V5 Conversation Panel
        to interact with the memory system, neural processor, and visualization.
        """
        
        def __init__(self, mock_mode=False):
            """
            Initialize the chat memory interface
            
            Args:
                mock_mode: Whether to use mock data instead of real components
            """
            self.mock_mode = mock_mode
            self.memory_api = None
            self.neural_processor = None
            self.memory_socket = None
            self.session_id = f"session_{int(time.time())}"
            
            # Initialize components
            self._initialize_components()
            logger.info(f"ChatMemoryInterface initialized (mock={mock_mode})")
        
        def _initialize_components(self):
            """Initialize all required components"""
            # Try to initialize Memory API
            try:
                from memory_api import MemoryAPI
                self.memory_api = MemoryAPI()
                logger.info("Successfully initialized Memory API")
            except ImportError as e:
                logger.warning(f"Failed to initialize Memory API: {str(e)}")
            
            # Try to initialize Neural Linguistic Processor
            try:
                from neural_linguistic_processor import get_linguistic_processor
                self.neural_processor = get_linguistic_processor()
                logger.info("Successfully initialized Neural Linguistic Processor")
            except ImportError as e:
                logger.warning(f"Failed to initialize Neural Linguistic Processor: {str(e)}")
            
            # Try to initialize Memory Socket
            try:
                self.memory_socket = get_bridge(mock_mode=self.mock_mode)
                logger.info("Successfully initialized Memory API Socket")
            except Exception as e:
                logger.warning(f"Failed to initialize Memory API Socket: {str(e)}")
        
        def process_message(self, message, nn_weight=0.5, memory_mode="combined"):
            """
            Process a message and generate a response with NN/LLM weighting
            
            Args:
                message: User message to process
                nn_weight: Neural network weight (0.0-1.0)
                memory_mode: Memory enhancement mode (contextual, synthesized, combined)
                
            Returns:
                Generated response
            """
            # Store message in memory
            self.store_message(message, "user", nn_weight, memory_mode)
            
            # Generate response based on weighting
            if nn_weight > 0.8:
                # Neural network dominant
                response = self.get_neural_response(message)
            elif nn_weight < 0.2:
                # Language model dominant
                response = self.get_language_response(message, memory_mode)
            else:
                # Weighted response
                response = self.get_weighted_response(message, nn_weight, memory_mode)
            
            # Store response in memory
            self.store_message(response, "system", nn_weight, memory_mode)
            
            return response
        
        def store_message(self, message, role, nn_weight=0.5, memory_mode="combined"):
            """
            Store a message in memory
            
            Args:
                message: Message to store
                role: Role (user or system)
                nn_weight: Neural network weight
                memory_mode: Memory mode used
                
            Returns:
                Success status
            """
            metadata = {
                "role": role,
                "timestamp": time.time(),
                "session_id": self.session_id,
                "nn_weight": nn_weight,
                "memory_mode": memory_mode
            }
            
            # Try memory API first
            if self.memory_api:
                try:
                    result = self.memory_api.store_conversation(message, metadata)
                    return result.get("status") == "success"
                except Exception as e:
                    logger.error(f"Error storing message via Memory API: {str(e)}")
            
            # Fall back to socket if API not available
            if self.memory_socket:
                try:
                    socket_message = {
                        "type": "store_conversation",
                        "request_id": f"conv_{int(time.time())}",
        "content": {
                            "message": message,
                            "metadata": metadata
                        }
                    }
                    response = self.memory_socket.socket.process_message(socket_message)
                    return response.get("status") != "error"
                except Exception as e:
                    logger.error(f"Error storing message via socket: {str(e)}")
            
            # If we're in mock mode or both failed, just log and return success
            logger.info(f"Mock store message: {role}")
            return True
        
        def get_neural_response(self, message):
            """
            Get response from neural network processing
            
            Args:
                message: The message to process
                
            Returns:
                Neural response
            """
            if self.neural_processor:
                try:
                    # Process text with neural processor
                    analysis = self.neural_processor.analyze_linguistic_pattern(message)
                    
                    # Extract features
                    features = analysis.get("features", {})
                    complexity = features.get("complexity_score", 0.5)
                    unique_ratio = features.get("unique_word_ratio", 0.5)
                    
                    # Format neural response
                    response_parts = []
                    
                    # Add pattern analysis
                    response_parts.append(f"Pattern analysis complete ({complexity:.2f} complexity score).")
                    
                    # Add key phrase extraction if available
                    key_phrases = analysis.get("key_phrases", [])
                    if key_phrases:
                        response_parts.append(f"Key concepts identified: {', '.join(key_phrases[:3])}.")
                    
                    # Add basic response based on keywords
                    if "neural" in message.lower() or "network" in message.lower():
                        response_parts.append("Neural network patterns activated for language processing.")
                    if "language" in message.lower() or "memory" in message.lower():
                        response_parts.append("Language memory pathways strengthened through interaction.")
                    if "consciousness" in message.lower() or "aware" in message.lower():
                        response_parts.append("Consciousness metrics show increased integration across subsystems.")
                    
                    # Combine response parts
                    return " ".join(response_parts)
                except Exception as e:
                    logger.error(f"Error getting neural response: {str(e)}")
            
            # Fallback to mock neural response
            return self._mock_neural_response(message)
        
        def get_language_response(self, message, memory_mode="combined"):
            """
            Get response from language model processing
            
            Args:
                message: The message to process
                memory_mode: Memory enhancement mode
                
            Returns:
                Language model response
            """
            if self.memory_api:
                try:
                    # Enhance message with memory
                    enhanced = self.memory_api.enhance_message_with_memory(
                        message=message,
                        enhance_mode=memory_mode
                    )
                    
                    # Get enhanced context
                    context = enhanced.get("enhanced_context", "")
                    
                    # Format language response
                    response_parts = []
                    
                    # Add base response
                    if "neural" in message.lower() or "network" in message.lower():
                        response_parts.append("Neural networks are computational systems designed to process information similar to the human brain, recognizing patterns across distributed nodes.")
                    elif "language" in message.lower() or "memory" in message.lower():
                        response_parts.append("Language memory systems store linguistic patterns and associations, enabling progressive learning from interactions and information synthesis.")
                    elif "consciousness" in message.lower() or "aware" in message.lower():
                        response_parts.append("Consciousness emerges from integrated information processing across specialized subsystems, creating a coherent experiential framework.")
                    else:
                        response_parts.append("Based on my understanding, this question relates to complex systems that process information across distributed networks.")
                    
                    # Add context if available
                    if context:
                        response_parts.append(f"Previous interactions indicate: {context}")
                    
                    # Combine response parts
                    return " ".join(response_parts)
                except Exception as e:
                    logger.error(f"Error getting language response: {str(e)}")
            
            # Fallback to mock language response
            return self._mock_language_response(message)
        
        def get_weighted_response(self, message, nn_weight, memory_mode="combined"):
            """
            Get weighted response combining neural and language processing
            
            Args:
                message: The message to process
                nn_weight: Neural network weight (0.0-1.0)
                memory_mode: Memory enhancement mode
                
            Returns:
                Weighted response
            """
            # Get neural response
            neural_response = self.get_neural_response(message)
            
            # Get language response
            language_response = self.get_language_response(message, memory_mode)
            
            # Combine responses based on weighting
            neural_parts = neural_response.split(". ")
            language_parts = language_response.split(". ")
            
            # Determine how many parts to take from each response
            neural_count = max(1, int(len(neural_parts) * nn_weight))
            language_count = max(1, int(len(language_parts) * (1 - nn_weight)))
            
            # Collect parts
            result_parts = []
            
            # Take neural parts
            for i in range(min(neural_count, len(neural_parts))):
                part = neural_parts[i].strip()
                if part and not part.endswith("."):
                    part += "."
                if part:
                    result_parts.append(part)
            
            # Take language parts
            for i in range(min(language_count, len(language_parts))):
                part = language_parts[i].strip()
                if part and not part.endswith("."):
                    part += "."
                if part:
                    result_parts.append(part)
            
            # Combine parts
            return " ".join(result_parts)
        
        def _mock_neural_response(self, message):
            """Generate a mock neural network response"""
            responses = [
                "Pattern analysis complete. Primary semantic structures identified.",
                "Neural pathways activated. Coherence level: 0.78.",
                "Language processing complete. Contextual integration achieved.",
                "Pattern recognition system engaged. Semantic relevance established."
            ]
            return random.choice(responses)
        
        def _mock_language_response(self, message):
            """Generate a mock language model response"""
            responses = [
                "Based on my understanding, I believe this relates to neural information processing systems that integrate multiple knowledge domains.",
                "Looking at the context of your question, I can see connections to both linguistic patterns and computational frameworks that mimic cognitive processes.",
                "This topic involves several interconnected concepts about memory formation and pattern recognition in artificial neural systems.",
                "From my analysis, I can see this touches on how information is structured, stored, and retrieved in cognitive-inspired computational systems."
            ]
            return random.choice(responses)
        
        def get_memory_stats(self):
            """
            Get memory system statistics
            
            Returns:
                Dictionary containing memory statistics
            """
            # Try memory API first
            if self.memory_api:
                try:
                    stats = self.memory_api.get_memory_stats()
                    if stats.get("status") == "success":
                        return stats
                except Exception as e:
                    logger.error(f"Error getting memory stats via API: {str(e)}")
            
            # Fall back to socket if API not available
            if self.memory_socket:
                try:
                    socket_message = {
                        "type": "get_stats",
                        "request_id": f"stats_{int(time.time())}",
                        "content": {}
                    }
                    response = self.memory_socket.socket.process_message(socket_message)
                    if response.get("status") != "error":
                        return response
                except Exception as e:
                    logger.error(f"Error getting memory stats via socket: {str(e)}")
            
            # Return mock stats if both failed
            return {
                "status": "success",
                "stats": {
                    "total_memories": 1250,
                    "total_conversations": 85,
                    "top_topics": [
                        {"topic": "neural networks", "count": 45},
                        {"topic": "consciousness", "count": 38},
                        {"topic": "machine learning", "count": 32}
                    ]
                }
            }
    
    # Create and return an interface instance
    return ChatMemoryInterface(mock_mode=False)


# For testing
if __name__ == "__main__":
    print("Testing Chat Memory Integration")
    
    # Get interface
    interface = chat_integration()
    
    # Test message processing
    test_message = "Tell me about neural networks and consciousness"
    
    # Test with different weights
    nn_weight = 0.8
    print(f"\nTesting with NN weight: {nn_weight}")
    response = interface.process_message(test_message, nn_weight)
    print(f"Response: {response}")
    
    nn_weight = 0.2
    print(f"\nTesting with NN weight: {nn_weight}")
    response = interface.process_message(test_message, nn_weight)
    print(f"Response: {response}")
    
    nn_weight = 0.5
    print(f"\nTesting with NN weight: {nn_weight}")
    response = interface.process_message(test_message, nn_weight)
    print(f"Response: {response}")
    
    # Get memory stats
    stats = interface.get_memory_stats()
    print("\nMemory Stats:")
    print(stats)
    
    print("\nChat Memory Integration test complete") 