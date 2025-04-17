#!/usr/bin/env python3
"""
Language Modulation Bridge

This module serves as the key connection point between V5 Fractal Echo Visualization
and V10 Language Memory Systems, enabling advanced language modulation capabilities.
It facilitates the bidirectional flow of language patterns, memory structures, and
neural processing results between visualization components and memory systems.
"""

import os
import sys
import json
import logging
import threading
import time
from queue import Queue
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("language-modulation-bridge")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class LanguageModulationBridge:
    """
    Bridge between V5 and V10 systems focusing on language modulation capabilities.
    
    This class provides:
    - Bidirectional communication between V5 Visualization and Language Memory
    - Translation of semantic patterns to visual representations
    - Neural activation mapping to language structures
    - Real-time modulation of language parameters
    - Compatibility with both the V5 plugin socket architecture and V10 consciousness components
    """
    
    def __init__(self, mock_mode: bool = False):
        """
        Initialize the Language Modulation Bridge
        
        Args:
            mock_mode: If True, use simulated data instead of actual components
        """
        self.mock_mode = mock_mode
        self.is_connected = False
        self.message_queue = Queue()
        self.components = {}
        self.modulation_parameters = {
            "resonance_factor": 0.75,
            "pattern_complexity": 0.65,
            "semantic_depth": 3,
            "consciousness_influence": 0.5,
            "memory_retention": 0.8,
            "language_flexibility": 0.7
        }
        self.active_modulations = {}
        self.processing_thread = None
        self.is_running = False
        
        # Attempt to load required components
        self._load_components()
        
        logger.info(f"Language Modulation Bridge initialized (mock_mode={mock_mode})")
    
    def _load_components(self):
        """Load required components based on availability"""
        try:
            # Try to import V5 components
            if not self.mock_mode:
                try:
                    from language_memory_v5_bridge import get_bridge as get_v5_bridge
                    self.components["v5_bridge"] = get_v5_bridge(mock_mode=self.mock_mode)
                    logger.info("Successfully loaded V5 bridge component")
                except ImportError:
                    logger.warning("V5 bridge not available, some functionality will be limited")
            
            # Try to import central language node (V10)
            if not self.mock_mode:
                try:
                    from src.central_language_node import CentralLanguageNode
                    self.components["language_node"] = CentralLanguageNode()
                    logger.info("Successfully loaded Central Language Node component")
                except ImportError:
                    logger.warning("Central Language Node not available, some functionality will be limited")
            
            # Try to import language memory synthesis integration
            if not self.mock_mode:
                try:
                    from language_memory_synthesis_integration import LanguageMemorySynthesisIntegration
                    self.components["synthesis"] = LanguageMemorySynthesisIntegration()
                    logger.info("Successfully loaded Language Memory Synthesis component")
                except ImportError:
                    logger.warning("Language Memory Synthesis not available, some functionality will be limited")
            
            # Try to import the language processor
            if not self.mock_mode:
                try:
                    from language_processor import LanguageProcessor
                    self.components["processor"] = LanguageProcessor()
                    logger.info("Successfully loaded Language Processor component")
                except ImportError:
                    logger.warning("Language Processor not available, some functionality will be limited")
                    
            # At least some components need to be available
            if not self.components and not self.mock_mode:
                logger.warning("No required components available, switching to mock mode")
                self.mock_mode = True
                
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            if not self.mock_mode:
                logger.warning("Switching to mock mode due to component loading errors")
                self.mock_mode = True
    
    def connect(self) -> bool:
        """
        Connect to required systems
        
        Returns:
            bool: Success status
        """
        if self.is_connected:
            logger.info("Already connected")
            return True
        
        try:
            # Connect to V5 bridge if available
            if "v5_bridge" in self.components:
                v5_success = self.components["v5_bridge"].connect()
                logger.info(f"V5 bridge connection: {'successful' if v5_success else 'failed'}")
            else:
                v5_success = self.mock_mode
            
            # Connect to central language node if available
            language_node_success = "language_node" in self.components
            
            # Consider connected if either component is available or in mock mode
            self.is_connected = v5_success or language_node_success or self.mock_mode
            
            if self.is_connected:
                # Start message processing thread
                self.is_running = True
                self.processing_thread = threading.Thread(
                    target=self._process_messages,
                    daemon=True,
                    name="LanguageModulationThread"
                )
                self.processing_thread.start()
                logger.info("Started message processing thread")
            
            logger.info(f"Connection {'successful' if self.is_connected else 'failed'}")
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Error connecting: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from all systems"""
        if not self.is_connected:
            return
        
        # Stop processing thread
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Disconnect V5 bridge if available
        if "v5_bridge" in self.components:
            try:
                self.components["v5_bridge"].disconnect()
            except:
                pass
        
        self.is_connected = False
        logger.info("Disconnected from all systems")
    
    def _process_messages(self):
        """Process messages from the queue"""
        while self.is_running:
            try:
                if self.message_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                
                message = self.message_queue.get(block=False)
                self._handle_message(message)
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    def _handle_message(self, message: Dict[str, Any]):
        """
        Handle messages from either system
        
        Args:
            message: Message dictionary
        """
        message_type = message.get("type", "unknown")
        source = message.get("source", "unknown")
        content = message.get("content", {})
        
        logger.debug(f"Handling message: {message_type} from {source}")
        
        # Router for different message types
        if message_type == "modulate_language":
            self._handle_modulation_request(content)
        elif message_type == "visualize_pattern":
            self._handle_visualization_request(content)
        elif message_type == "memory_query":
            self._handle_memory_query(content)
        elif message_type == "fractal_pattern":
            self._handle_fractal_pattern(content)
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    def _handle_modulation_request(self, content: Dict[str, Any]):
        """
        Handle language modulation requests
        
        Args:
            content: Modulation parameters
        """
        logger.info(f"Handling modulation request: {content}")
        
        # Extract parameters
        modulation_id = content.get("modulation_id", f"mod_{int(time.time())}")
        parameters = content.get("parameters", {})
        
        # Update modulation parameters
        for param, value in parameters.items():
            if param in self.modulation_parameters:
                self.modulation_parameters[param] = float(value)
        
        # Store active modulation
        self.active_modulations[modulation_id] = {
            "parameters": parameters,
            "timestamp": time.time(),
            "status": "active"
        }
        
        # Apply modulation through appropriate components
        self._apply_modulation(modulation_id, parameters)
    
    def _apply_modulation(self, modulation_id: str, parameters: Dict[str, float]):
        """
        Apply language modulation to relevant components
        
        Args:
            modulation_id: Unique identifier for the modulation
            parameters: Modulation parameters
        """
        logger.info(f"Applying modulation {modulation_id} with parameters: {parameters}")
        
        # Modulate language processor if available
        if "processor" in self.components and hasattr(self.components["processor"], "set_parameters"):
            try:
                processor_params = {
                    "embedding_dim": int(300 * parameters.get("pattern_complexity", 1.0)),
                    "context_window": int(5 * parameters.get("semantic_depth", 1.0))
                }
                self.components["processor"].set_parameters(**processor_params)
                logger.info(f"Applied modulation to language processor: {processor_params}")
            except Exception as e:
                logger.error(f"Error modulating language processor: {e}")
        
        # Modulate visualization if V5 bridge is available
        if "v5_bridge" in self.components:
            try:
                fractal_params = {
                    "core_frequency": parameters.get("resonance_factor", 0.75),
                    "complexity": parameters.get("pattern_complexity", 0.65),
                    "recursion_depth": int(parameters.get("semantic_depth", 3.0)),
                }
                
                # Send modulation parameters to visualization system
                if hasattr(self.components["v5_bridge"], "set_fractal_parameters"):
                    self.components["v5_bridge"].set_fractal_parameters(fractal_params)
                    logger.info(f"Applied modulation to V5 visualization: {fractal_params}")
            except Exception as e:
                logger.error(f"Error modulating V5 visualization: {e}")
        
        # Update modulation status
        self.active_modulations[modulation_id]["status"] = "applied"
    
    def _handle_visualization_request(self, content: Dict[str, Any]):
        """
        Handle visualization requests
        
        Args:
            content: Visualization parameters
        """
        logger.info(f"Handling visualization request: {content}")
        
        topic = content.get("topic", "")
        if not topic:
            logger.warning("Visualization request missing topic")
            return
        
        depth = content.get("depth", 3)
        
        # Fetch memory synthesis for the topic
        synthesis = self._get_topic_synthesis(topic, depth)
        
        # Convert to visualization pattern
        pattern = self._synthesized_memory_to_pattern(synthesis, content.get("parameters", {}))
        
        # Send pattern to V5 visualization system
        if "v5_bridge" in self.components and hasattr(self.components["v5_bridge"], "visualize_pattern"):
            try:
                self.components["v5_bridge"].visualize_pattern(pattern)
                logger.info(f"Sent pattern for topic '{topic}' to V5 visualization")
            except Exception as e:
                logger.error(f"Error sending pattern to V5 visualization: {e}")
    
    def _handle_memory_query(self, content: Dict[str, Any]):
        """
        Handle memory query requests
        
        Args:
            content: Query parameters
        """
        logger.info(f"Handling memory query: {content}")
        
        query = content.get("query", "")
        if not query:
            logger.warning("Memory query missing query string")
            return
        
        query_type = content.get("query_type", "topic")
        limit = content.get("limit", 5)
        
        # Route to appropriate handler based on query type
        if query_type == "topic":
            results = self._get_topic_synthesis(query, depth=content.get("depth", 3))
        elif query_type == "search":
            results = self._search_memories(query, limit=limit)
        else:
            results = {"error": f"Unknown query type: {query_type}"}
        
        # Send response to caller
        if "callback" in content and callable(content["callback"]):
            try:
                content["callback"](results)
            except Exception as e:
                logger.error(f"Error in query callback: {e}")
    
    def _handle_fractal_pattern(self, content: Dict[str, Any]):
        """
        Handle incoming fractal pattern data
        
        Args:
            content: Fractal pattern data
        """
        logger.info(f"Handling fractal pattern: {content.get('pattern_id', 'unknown')}")
        
        pattern = content.get("pattern", {})
        if not pattern:
            logger.warning("Empty fractal pattern received")
            return
        
        # Extract language parameters from pattern
        language_parameters = self._extract_language_parameters(pattern)
        
        # Apply these parameters to language components
        self.set_language_parameters(language_parameters)
    
    def _extract_language_parameters(self, pattern: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract language parameters from fractal pattern
        
        Args:
            pattern: Fractal pattern data
            
        Returns:
            Dictionary of language parameters
        """
        parameters = {}
        
        # Map fractal parameters to language parameters
        if "core_frequency" in pattern:
            parameters["resonance_factor"] = pattern["core_frequency"]
        
        if "complexity" in pattern:
            parameters["pattern_complexity"] = pattern["complexity"]
        
        if "recursion_depth" in pattern:
            parameters["semantic_depth"] = float(pattern["recursion_depth"]) / 5.0
        
        return parameters
    
    def _get_topic_synthesis(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """
        Get synthetic memory for a topic
        
        Args:
            topic: Topic to synthesize
            depth: Search depth
            
        Returns:
            Synthesis results
        """
        logger.info(f"Getting topic synthesis for '{topic}' (depth={depth})")
        
        # Try central language node first
        if "language_node" in self.components:
            try:
                return self.components["language_node"].synthesize_topic(topic, depth)
            except Exception as e:
                logger.error(f"Error using central language node: {e}")
        
        # Try language memory synthesis
        if "synthesis" in self.components:
            try:
                return self.components["synthesis"].synthesize_topic(topic, depth)
            except Exception as e:
                logger.error(f"Error using language memory synthesis: {e}")
        
        # If in mock mode or previous attempts failed, generate mock data
        if self.mock_mode:
            return self._generate_mock_synthesis(topic, depth)
        
        return {"error": "No synthesis component available"}
    
    def _search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories by query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        logger.info(f"Searching memories for '{query}' (limit={limit})")
        
        # Try central language node first
        if "language_node" in self.components:
            try:
                return self.components["language_node"].retrieve_memories(query, "text")
            except Exception as e:
                logger.error(f"Error searching with central language node: {e}")
        
        # Try V5 bridge
        if "v5_bridge" in self.components and hasattr(self.components["v5_bridge"], "search"):
            try:
                return self.components["v5_bridge"].search(query, limit)
            except Exception as e:
                logger.error(f"Error searching with V5 bridge: {e}")
        
        # If in mock mode or previous attempts failed, generate mock data
        if self.mock_mode:
            return self._generate_mock_search_results(query, limit)
        
        return [{"error": "No search component available"}]
    
    def _generate_mock_synthesis(self, topic: str, depth: int) -> Dict[str, Any]:
        """
        Generate mock synthesis data for testing
        
        Args:
            topic: Topic
            depth: Depth
            
        Returns:
            Mock synthesis data
        """
        import random
        
        # Create insights based on topic
        insights = [
            f"The concept of {topic} involves recursive pattern recognition",
            f"{topic} demonstrates connections to language understanding",
            f"{topic} shows correlations with consciousness metrics",
            f"Neural patterns related to {topic} exhibit high coherence",
            f"Processing {topic} activates both abstract and concrete reasoning pathways"
        ]
        
        # Create related topics
        related_topics = []
        for i in range(3 + depth):
            related_topics.append({
                "topic": f"{topic}_related_{i}",
                "relevance": round(random.uniform(0.5, 0.95), 2)
            })
        
        return {
            "topic": topic,
            "depth": depth,
            "synthesis_results": {
                "synthesized_memory": {
                    "core_understanding": f"{topic} is a fundamental concept connected to neural processing",
                    "novel_insights": insights[:depth+1],
                    "topics": [topic],
                    "source_count": random.randint(5, 20)
                },
                "related_topics": related_topics,
                "metrics": {
                    "synthesis_time": round(random.uniform(0.1, 1.5), 3),
                    "memory_sources": random.randint(3, 12),
                    "confidence": round(random.uniform(0.7, 0.95), 2)
                }
            },
            "mock": True
        }
    
    def _generate_mock_search_results(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Generate mock search results for testing
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            Mock search results
        """
        import random
        
        results = []
        for i in range(min(5, limit)):
            results.append({
                "id": f"mock_{i}_{int(time.time())}",
                "content": f"Mock memory result {i+1} for query: {query}",
                "relevance": round(1.0 - (i * 0.15), 2),
                "timestamp": time.time() - (i * 3600),
                "mock": True
            })
        
        return results
    
    def _synthesized_memory_to_pattern(self, synthesis: Dict[str, Any], 
                                      parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert synthesized memory to visualization pattern
        
        Args:
            synthesis: Synthesized memory
            parameters: Visualization parameters
            
        Returns:
            Pattern data for visualization
        """
        import random
        
        try:
            # Extract synthesis data
            synthesized = synthesis.get("synthesis_results", {}).get("synthesized_memory", {})
            related_topics = synthesis.get("synthesis_results", {}).get("related_topics", [])
            
            # Apply modulation parameters
            complexity = parameters.get("pattern_complexity", 
                                      self.modulation_parameters.get("pattern_complexity", 0.65))
            resonance = parameters.get("resonance_factor", 
                                     self.modulation_parameters.get("resonance_factor", 0.75))
            depth = parameters.get("semantic_depth", 
                                 self.modulation_parameters.get("semantic_depth", 3.0))
            
            # Create base pattern
            pattern = {
                "source_id": f"lm_{int(time.time())}",
                "pattern_type": "language_fractal",
                "core_frequency": resonance,
                "amplitude": random.uniform(0.6, 0.9),
                "complexity": complexity,
                "color_palette": [
                    f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
                    for _ in range(5)
                ],
                "recursion_depth": int(depth),
                "symmetry_type": random.choice(["radial", "bilateral", "spiral"]),
                "nodes": []
            }
            
            # Create nodes from insights
            insights = synthesized.get("novel_insights", [])
            for i, insight in enumerate(insights):
                words = insight.split()
                for j, word in enumerate(words[:min(10, len(words))]):
                    node_id = f"node_{i}_{j}"
                    pattern["nodes"].append({
                        "id": node_id,
                        "label": word,
                        "size": len(word) / 5.0,
                        "position": {
                            "x": random.uniform(-1, 1),
                            "y": random.uniform(-1, 1),
                            "z": random.uniform(-0.5, 0.5)
                        },
                        "color": pattern["color_palette"][i % len(pattern["color_palette"])],
                        "connections": []
                    })
            
            # Add connections between nodes
            for i, node in enumerate(pattern["nodes"]):
                # Connect to 2-4 other nodes
                for _ in range(random.randint(2, min(4, len(pattern["nodes"])))):
                    target_idx = random.randint(0, len(pattern["nodes"]) - 1)
                    if target_idx != i:
                        target_id = pattern["nodes"][target_idx]["id"]
                        node["connections"].append({
                            "target": target_id,
                            "strength": random.uniform(0.3, 0.9),
                            "type": random.choice(["semantic", "temporal", "causal"])
                        })
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error converting synthesis to pattern: {e}")
            return {"error": str(e)}
    
    def set_language_parameters(self, parameters: Dict[str, float]):
        """
        Set language modulation parameters
        
        Args:
            parameters: Modulation parameters
        """
        logger.info(f"Setting language parameters: {parameters}")
        
        # Update modulation parameters
        for param, value in parameters.items():
            if param in self.modulation_parameters:
                self.modulation_parameters[param] = float(value)
                logger.debug(f"Updated {param} to {value}")
        
        # Create modulation request
        modulation_id = f"mod_{int(time.time())}"
        modulation_request = {
            "type": "modulate_language",
            "source": "language_modulation_bridge",
            "content": {
                "modulation_id": modulation_id,
                "parameters": parameters
            }
        }
        
        # Process through message queue
        self.message_queue.put(modulation_request)
    
    def visualize_topic(self, topic: str, depth: int = 3, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate visualization for a topic
        
        Args:
            topic: Topic to visualize
            depth: Depth of synthesis
            parameters: Optional visualization parameters
            
        Returns:
            Visualization data
        """
        logger.info(f"Visualizing topic: {topic} (depth={depth})")
        
        if not self.is_connected:
            success = self.connect()
            if not success:
                return {"error": "Not connected to required systems"}
        
        # Create visualization request
        visualization_request = {
            "type": "visualize_pattern",
            "source": "language_modulation_bridge",
            "content": {
                "topic": topic,
                "depth": depth,
                "parameters": parameters or {}
            }
        }
        
        # Process through message queue (async)
        self.message_queue.put(visualization_request)
        
        # Also directly get synthesis for immediate return
        synthesis = self._get_topic_synthesis(topic, depth)
        pattern = self._synthesized_memory_to_pattern(synthesis, parameters or {})
        
        return {
            "topic": topic,
            "pattern": pattern,
            "timestamp": time.time()
        }
    
    def search_memories(self, query: str, limit: int = 5, callback: Callable = None) -> List[Dict[str, Any]]:
        """
        Search memories
        
        Args:
            query: Search query
            limit: Maximum results
            callback: Optional callback for async results
            
        Returns:
            Search results (if callback is None)
        """
        if callback:
            # Async mode
            memory_request = {
                "type": "memory_query",
                "source": "language_modulation_bridge",
                "content": {
                    "query": query,
                    "query_type": "search",
                    "limit": limit,
                    "callback": callback
                }
            }
            self.message_queue.put(memory_request)
            return None
        else:
            # Sync mode
            return self._search_memories(query, limit)
    
    def get_modulation_status(self) -> Dict[str, Any]:
        """
        Get current modulation status
        
        Returns:
            Status information
        """
        return {
            "current_parameters": self.modulation_parameters,
            "active_modulations": self.active_modulations,
            "connected": self.is_connected,
            "components_available": list(self.components.keys()),
            "mock_mode": self.mock_mode
        }


# Singleton instance for easy access
_bridge_instance = None

def get_modulation_bridge(mock_mode: bool = False) -> LanguageModulationBridge:
    """
    Get the singleton bridge instance
    
    Args:
        mock_mode: Whether to use simulated data
        
    Returns:
        Bridge instance
    """
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = LanguageModulationBridge(mock_mode=mock_mode)
    return _bridge_instance


# When run directly, perform a simple test
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    print("Testing Language Modulation Bridge...")
    
    bridge = LanguageModulationBridge(mock_mode=True)
    connected = bridge.connect()
    print(f"Connected: {connected}")
    
    # Test language parameter setting
    bridge.set_language_parameters({
        "resonance_factor": 0.8,
        "pattern_complexity": 0.7,
        "semantic_depth": 4.0
    })
    
    # Test topic visualization
    result = bridge.visualize_topic("neural networks", depth=3)
    print(f"Visualization contains {len(result['pattern']['nodes'])} nodes")
    
    # Test memory search
    results = bridge.search_memories("consciousness", limit=3)
    print(f"Search returned {len(results)} results")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.get('content', '')[:50]}... (relevance: {result.get('relevance', 0)})")
    
    # Print current modulation status
    status = bridge.get_modulation_status()
    print(f"Current modulation parameters: {status['current_parameters']}")
    
    # Cleanup
    bridge.disconnect()
    print("Test complete") 