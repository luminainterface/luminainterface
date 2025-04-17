#!/usr/bin/env python3
"""
Neural Linguistic Processor

This module provides the core processing capabilities for linguistic pattern analysis
and modulation between the V5 Fractal Echo Visualization system and the Language Memory System.
It forms a key component of the neural pattern modulation infrastructure and enables
the translation of linguistic structures into visual representations.
"""

import os
import sys
import json
import logging
import threading
import time
import random
from queue import Queue
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural-linguistic-processor")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


class NeuralLinguisticProcessor:
    """
    Processes neural linguistic patterns and facilitates the bidirectional
    flow between language structures and visualization components.
    
    Core responsibilities:
    - Analyzing linguistic patterns for visualization
    - Mapping visual patterns back to linguistic structures
    - Modulating pattern parameters based on linguistic context
    - Coordinating with both V5 and Language Memory systems
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Neural Linguistic Processor
        
        Args:
            config: Configuration parameters (optional)
        """
        self.config = config or {}
        self.mock_mode = self.config.get("mock_mode", False)
        self.processing_queue = Queue()
        self.result_cache = {}
        self.is_processing = False
        self.processor_thread = None
        
        # Linguistic pattern parameters
        self.pattern_params = {
            "resonance_factor": 0.75,
            "recursion_depth": 3,
            "symmetry_type": "radial",
            "semantic_density": 0.65,
            "temporal_coherence": 0.7,
            "pattern_complexity": 0.6
        }
        
        # Initialize components
        self._init_components()
        
        logger.info(f"Neural Linguistic Processor initialized (mock_mode={self.mock_mode})")

    def _init_components(self):
        """Initialize required components"""
        self.components = {}
        
        # Load Language Modulation Bridge
        if not self.mock_mode:
            try:
                from language_modulation_bridge import get_modulation_bridge
                self.components["modulation_bridge"] = get_modulation_bridge(mock_mode=self.mock_mode)
                logger.info("Successfully loaded Language Modulation Bridge")
            except ImportError:
                logger.warning("Language Modulation Bridge not available, functionality will be limited")
        
        # Load Language Memory V5 Bridge
        if not self.mock_mode:
            try:
                from language_memory_v5_bridge import get_bridge
                self.components["memory_bridge"] = get_bridge(mock_mode=self.mock_mode)
                logger.info("Successfully loaded Language Memory V5 Bridge")
            except ImportError:
                logger.warning("Language Memory V5 Bridge not available, functionality will be limited")
        
        # If no components are available and not in mock mode, switch to mock mode
        if not self.components and not self.mock_mode:
            logger.warning("No required components available, switching to mock mode")
            self.mock_mode = True
    
    def start_processor(self):
        """Start the background processing thread"""
        if self.is_processing:
            logger.info("Processor is already running")
            return
        
        self.is_processing = True
        self.processor_thread = threading.Thread(
            target=self._process_queue,
            daemon=True,
            name="NeuralLinguisticProcessorThread"
        )
        self.processor_thread.start()
        logger.info("Neural linguistic processor thread started")
    
    def stop_processor(self):
        """Stop the background processing thread"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        logger.info("Neural linguistic processor thread stopped")
    
    def _process_queue(self):
        """Process items from the queue"""
        while self.is_processing:
            try:
                if self.processing_queue.empty():
                    time.sleep(0.05)  # Prevent CPU spinning
                    continue
                
                item = self.processing_queue.get(block=False)
                function_name = item.get("function", "")
                args = item.get("args", [])
                kwargs = item.get("kwargs", {})
                callback = item.get("callback")
                
                # Call the appropriate function
                if hasattr(self, function_name) and callable(getattr(self, function_name)):
                    try:
                        result = getattr(self, function_name)(*args, **kwargs)
                        if callback and callable(callback):
                            callback(result)
                    except Exception as e:
                        logger.error(f"Error processing {function_name}: {e}")
                        if callback and callable(callback):
                            callback({"error": str(e)})
                else:
                    logger.error(f"Unknown function: {function_name}")
                
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processor thread: {e}")
    
    def analyze_linguistic_pattern(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze linguistic patterns in text
        
        Args:
            text: Text to analyze
            context: Additional context (optional)
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing linguistic pattern: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Use modulation bridge if available
        if "modulation_bridge" in self.components:
            try:
                # Search memories related to text
                memories = self.components["modulation_bridge"].search_memories(text, limit=5)
                
                # Combine with direct analysis
                return self._combine_analysis_results(
                    self._analyze_text_direct(text, context),
                    self._analyze_memories(memories)
                )
            except Exception as e:
                logger.error(f"Error using modulation bridge: {e}")
        
        # Fallback to direct analysis
        return self._analyze_text_direct(text, context)
    
    def _analyze_text_direct(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Directly analyze text without using external components
        
        Args:
            text: Text to analyze
            context: Additional context
            
        Returns:
            Analysis results
        """
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        
        # Basic linguistic features
        features = {
            "word_count": word_count,
            "unique_word_ratio": unique_words / max(1, word_count),
            "avg_word_length": sum(len(w) for w in words) / max(1, word_count),
            "complexity_score": 0.4 + (unique_words / max(1, word_count)) * 0.6
        }
        
        # Extract key phrases (simple approach)
        key_phrases = []
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if len(phrase) > 8:  # Arbitrary threshold
                key_phrases.append(phrase)
        
        # Limit to top 5 phrases
        key_phrases = key_phrases[:5]
        
        # Generate pattern parameters
        pattern_params = {
            "resonance_factor": min(0.3 + features["unique_word_ratio"] * 0.7, 0.95),
            "complexity": features["complexity_score"],
            "recursion_depth": max(2, min(5, int(features["complexity_score"] * 5))),
            "semantic_density": 0.3 + features["unique_word_ratio"] * 0.7
        }
        
        return {
            "text": text,
            "features": features,
            "key_phrases": key_phrases,
            "pattern_params": pattern_params,
            "timestamp": time.time()
        }
    
    def _analyze_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze memory results
        
        Args:
            memories: Memory search results
            
        Returns:
            Analysis of memory patterns
        """
        if not memories:
            return {"error": "No memories provided"}
        
        # Aggregate relevance scores
        relevance_sum = sum(memory.get("relevance", 0) for memory in memories)
        avg_relevance = relevance_sum / len(memories) if memories else 0
        
        # Generate pattern parameters from memories
        pattern_params = {
            "resonance_factor": min(0.5 + avg_relevance * 0.5, 0.95),
            "complexity": 0.4 + avg_relevance * 0.5,
            "recursion_depth": max(2, min(5, int(avg_relevance * 5))),
            "semantic_density": avg_relevance
        }
        
        return {
            "memories_analyzed": len(memories),
            "avg_relevance": avg_relevance,
            "pattern_params": pattern_params,
            "timestamp": time.time()
        }
    
    def _combine_analysis_results(self, direct_analysis: Dict[str, Any], 
                                 memory_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine results from direct text analysis and memory analysis
        
        Args:
            direct_analysis: Results from direct text analysis
            memory_analysis: Results from memory analysis
            
        Returns:
            Combined analysis results
        """
        # Combine pattern parameters with weighted average
        direct_weight = 0.6  # Weight for direct analysis
        memory_weight = 0.4  # Weight for memory-based analysis
        
        combined_params = {}
        
        # Direct analysis params
        direct_params = direct_analysis.get("pattern_params", {})
        
        # Memory analysis params
        memory_params = memory_analysis.get("pattern_params", {})
        
        # Combine parameters that exist in both
        for param in set(direct_params.keys()) & set(memory_params.keys()):
            combined_params[param] = (
                direct_params[param] * direct_weight + 
                memory_params[param] * memory_weight
            )
        
        # Add parameters that exist only in direct analysis
        for param in set(direct_params.keys()) - set(memory_params.keys()):
            combined_params[param] = direct_params[param]
        
        # Add parameters that exist only in memory analysis
        for param in set(memory_params.keys()) - set(direct_params.keys()):
            combined_params[param] = memory_params[param]
        
        # Create combined result
        combined_result = {
            "text": direct_analysis.get("text", ""),
            "features": direct_analysis.get("features", {}),
            "key_phrases": direct_analysis.get("key_phrases", []),
            "memory_relevance": memory_analysis.get("avg_relevance", 0),
            "memories_analyzed": memory_analysis.get("memories_analyzed", 0),
            "pattern_params": combined_params,
            "timestamp": time.time()
        }
        
        return combined_result
    
    def generate_fractal_pattern(self, linguistic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fractal visualization pattern from linguistic analysis
        
        Args:
            linguistic_analysis: Results from linguistic analysis
            
        Returns:
            Fractal pattern data
        """
        logger.info("Generating fractal pattern from linguistic analysis")
        
        # Extract pattern parameters
        pattern_params = linguistic_analysis.get("pattern_params", {})
        resonance = pattern_params.get("resonance_factor", self.pattern_params["resonance_factor"])
        complexity = pattern_params.get("complexity", self.pattern_params["pattern_complexity"])
        recursion_depth = pattern_params.get("recursion_depth", self.pattern_params["recursion_depth"])
        
        # Create base pattern
        pattern = {
            "pattern_id": f"nlp_{int(time.time())}",
            "source": "neural_linguistic_processor",
            "pattern_type": "linguistic_fractal",
            "core_frequency": resonance,
            "complexity": complexity,
            "recursion_depth": recursion_depth,
            "symmetry_type": self.pattern_params["symmetry_type"],
            "color_palette": self._generate_color_palette(linguistic_analysis),
            "nodes": []
        }
        
        # Create nodes from key phrases
        key_phrases = linguistic_analysis.get("key_phrases", [])
        for i, phrase in enumerate(key_phrases):
            node_id = f"node_{i}"
            pattern["nodes"].append({
                "id": node_id,
                "label": phrase,
                "size": len(phrase) / 10.0,
                "position": self._generate_node_position(i, len(key_phrases)),
                "color": pattern["color_palette"][i % len(pattern["color_palette"])],
                "connections": []
            })
        
        # Add connections between nodes
        for i, node in enumerate(pattern["nodes"]):
            # Connect to 2-3 other nodes
            for _ in range(min(2, len(pattern["nodes"]))):
                target_idx = (i + 1) % len(pattern["nodes"])
                target_id = pattern["nodes"][target_idx]["id"]
                node["connections"].append({
                    "target": target_id,
                    "strength": random.uniform(0.5, 0.9),
                    "type": "semantic"
                })
        
        # Add linguistic metadata
        pattern["metadata"] = {
            "source_text": linguistic_analysis.get("text", "")[:100],
            "word_count": linguistic_analysis.get("features", {}).get("word_count", 0),
            "complexity_score": linguistic_analysis.get("features", {}).get("complexity_score", 0),
            "timestamp": time.time()
        }
        
        return pattern
    
    def _generate_color_palette(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate a color palette based on linguistic analysis
        
        Args:
            analysis: Linguistic analysis results
            
        Returns:
            List of hex color codes
        """
        # Base hue derived from complexity
        complexity = analysis.get("features", {}).get("complexity_score", 0.5)
        base_hue = int(complexity * 360) % 360
        
        palette = []
        for i in range(5):
            # Vary hue around base
            hue = (base_hue + i * 30) % 360
            
            # Convert HSL to RGB (simplified)
            h = hue / 360
            s = 0.7
            l = 0.5
            
            if s == 0:
                r = g = b = l
            else:
                def hue_to_rgb(p, q, t):
                    if t < 0: t += 1
                    if t > 1: t -= 1
                    if t < 1/6: return p + (q - p) * 6 * t
                    if t < 1/2: return q
                    if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                    return p
                
                q = l * (1 + s) if l < 0.5 else l + s - l * s
                p = 2 * l - q
                r = hue_to_rgb(p, q, h + 1/3)
                g = hue_to_rgb(p, q, h)
                b = hue_to_rgb(p, q, h - 1/3)
            
            # Convert to hex
            hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            palette.append(hex_color)
        
        return palette
    
    def _generate_node_position(self, index: int, total: int) -> Dict[str, float]:
        """
        Generate a position for a node in 3D space
        
        Args:
            index: Node index
            total: Total number of nodes
            
        Returns:
            Position coordinates
        """
        # Generate positions in a circle
        angle = (index / max(1, total)) * 2 * 3.14159
        radius = 0.8
        
        return {
            "x": radius * math.cos(angle),
            "y": radius * math.sin(angle),
            "z": 0.0
        }
    
    def process_text(self, text: str, context: Dict[str, Any] = None, 
                    callback: Callable = None) -> Dict[str, Any]:
        """
        Process text to generate visualization pattern
        
        Args:
            text: Text to process
            context: Additional context (optional)
            callback: Optional callback for async results
            
        Returns:
            Processing results (if callback is None)
        """
        logger.info(f"Processing text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        if callback:
            # Async processing
            item = {
                "function": "_process_text_async",
                "args": [text],
                "kwargs": {"context": context},
                "callback": callback
            }
            self.processing_queue.put(item)
            # Make sure processing thread is running
            if not self.is_processing:
                self.start_processor()
            return None
        else:
            # Sync processing
            return self._process_text_sync(text, context)
    
    def _process_text_sync(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Synchronously process text
        
        Args:
            text: Text to process
            context: Additional context
            
        Returns:
            Processing results
        """
        # Analyze linguistic patterns
        analysis = self.analyze_linguistic_pattern(text, context)
        
        # Generate fractal pattern
        pattern = self.generate_fractal_pattern(analysis)
        
        # Cache results
        cache_key = f"text_{hash(text)}"
        self.result_cache[cache_key] = {
            "analysis": analysis,
            "pattern": pattern,
            "timestamp": time.time()
        }
        
        # Send to V5 visualization if bridge is available
        self._send_to_visualization(pattern)
        
        return {
            "text": text,
            "analysis": analysis,
            "pattern": pattern
        }
    
    def _process_text_async(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Asynchronously process text (called by processor thread)
        
        Args:
            text: Text to process
            context: Additional context
            
        Returns:
            Processing results
        """
        return self._process_text_sync(text, context)
    
    def _send_to_visualization(self, pattern: Dict[str, Any]) -> bool:
        """
        Send pattern to V5 visualization system
        
        Args:
            pattern: Fractal pattern data
            
        Returns:
            Success status
        """
        # Try using modulation bridge
        if "modulation_bridge" in self.components:
            try:
                # Set modulation parameters from pattern
                self.components["modulation_bridge"].set_language_parameters({
                    "resonance_factor": pattern.get("core_frequency", 0.75),
                    "pattern_complexity": pattern.get("complexity", 0.65),
                    "semantic_depth": float(pattern.get("recursion_depth", 3))
                })
                logger.info("Sent pattern parameters to modulation bridge")
                return True
            except Exception as e:
                logger.error(f"Error sending to modulation bridge: {e}")
        
        # Try using memory bridge
        if "memory_bridge" in self.components and hasattr(self.components["memory_bridge"], "visualize_pattern"):
            try:
                self.components["memory_bridge"].visualize_pattern(pattern)
                logger.info("Sent pattern to memory bridge")
                return True
            except Exception as e:
                logger.error(f"Error sending to memory bridge: {e}")
        
        if self.mock_mode:
            logger.info("[MOCK] Pattern would be sent to visualization system")
            return True
        
        logger.warning("No visualization component available")
        return False
    
    def visualize_linguistic_structure(self, structure_type: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Visualize a specific linguistic structure
        
        Args:
            structure_type: Type of structure ("topic", "concept", "conversation")
            content: Structure content
            
        Returns:
            Visualization data
        """
        logger.info(f"Visualizing linguistic structure: {structure_type}")
        
        if structure_type == "topic":
            return self._visualize_topic(content)
        elif structure_type == "concept":
            return self._visualize_concept(content)
        elif structure_type == "conversation":
            return self._visualize_conversation(content)
        else:
            return {"error": f"Unknown structure type: {structure_type}"}
    
    def _visualize_topic(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Visualize a topic
        
        Args:
            content: Topic data
            
        Returns:
            Visualization data
        """
        topic = content.get("topic", "")
        if not topic:
            return {"error": "No topic specified"}
        
        # Try using modulation bridge
        if "modulation_bridge" in self.components:
            try:
                result = self.components["modulation_bridge"].visualize_topic(
                    topic, 
                    depth=content.get("depth", 3),
                    parameters=content.get("parameters", {})
                )
                return result
            except Exception as e:
                logger.error(f"Error using modulation bridge: {e}")
        
        # Fallback to text processing
        return self.process_text(f"Topic analysis: {topic}")
    
    def _visualize_concept(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Visualize a concept
        
        Args:
            content: Concept data
            
        Returns:
            Visualization data
        """
        concept = content.get("concept", "")
        if not concept:
            return {"error": "No concept specified"}
        
        # Process as text with concept-specific context
        return self.process_text(
            f"Concept: {concept}",
            context={"analysis_type": "concept"}
        )
    
    def _visualize_conversation(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Visualize a conversation
        
        Args:
            content: Conversation data
            
        Returns:
            Visualization data
        """
        messages = content.get("messages", [])
        if not messages:
            return {"error": "No messages in conversation"}
        
        # Concatenate messages with context
        combined_text = "\n".join(
            f"{msg.get('sender', 'unknown')}: {msg.get('text', '')}"
            for msg in messages[-5:]  # Only use last 5 messages
        )
        
        # Process with conversation-specific context
        return self.process_text(
            combined_text,
            context={"analysis_type": "conversation"}
        )
    
    def update_pattern_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update linguistic pattern parameters
        
        Args:
            parameters: New parameters
            
        Returns:
            Updated parameters
        """
        logger.info(f"Updating pattern parameters: {parameters}")
        
        # Update parameters
        for param, value in parameters.items():
            if param in self.pattern_params:
                self.pattern_params[param] = value
        
        # Forward to modulation bridge if available
        if "modulation_bridge" in self.components:
            try:
                self.components["modulation_bridge"].set_language_parameters(parameters)
            except Exception as e:
                logger.error(f"Error updating modulation bridge parameters: {e}")
        
        return self.pattern_params
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get processor status
        
        Returns:
            Status information
        """
        return {
            "is_processing": self.is_processing,
            "pattern_parameters": self.pattern_params,
            "cache_size": len(self.result_cache),
            "components_available": list(self.components.keys()),
            "mock_mode": self.mock_mode,
            "timestamp": time.time()
        }


# Helper functions
import math

def get_linguistic_processor(config: Dict[str, Any] = None) -> NeuralLinguisticProcessor:
    """
    Get or create a NeuralLinguisticProcessor instance
    
    Args:
        config: Optional configuration
        
    Returns:
        Processor instance
    """
    global _processor_instance
    if '_processor_instance' not in globals() or _processor_instance is None:
        _processor_instance = NeuralLinguisticProcessor(config)
    return _processor_instance


# When run directly, perform a simple test
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    print("Testing Neural Linguistic Processor...")
    
    processor = NeuralLinguisticProcessor({"mock_mode": True})
    processor.start_processor()
    
    # Test text processing
    text = "The neural networks create fractal patterns that represent consciousness and language understanding."
    result = processor.process_text(text)
    
    print(f"Processed text: {text}")
    print(f"Analysis features: {result['analysis']['features']}")
    print(f"Generated pattern with {len(result['pattern']['nodes'])} nodes")
    print(f"Pattern parameters: {result['pattern']['core_frequency']}, {result['pattern']['complexity']}")
    
    # Test topic visualization
    topic_result = processor.visualize_linguistic_structure("topic", {"topic": "neural linguistics"})
    print(f"Topic visualization contains {len(topic_result['pattern']['nodes'])} nodes")
    
    # Clean up
    processor.stop_processor()
    print("Test complete") 