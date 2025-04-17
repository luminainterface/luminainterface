#!/usr/bin/env python
"""
Neural-Linguistic Bridge for V7 System

This module provides bidirectional communication between neural network and language systems,
enabling advanced pattern exchange and cross-domain learning.

Key capabilities:
- Pattern translation between neural and linguistic systems
- Bidirectional data transformation
- Asynchronous communication protocol
- Neural-linguistic resonance detection
"""

import os
import sys
import time
import threading
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from queue import Queue
from datetime import datetime

# Add parent directory to path if needed
current_dir = Path(__file__).parent
if current_dir.parent not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

# Import needed components
try:
    from v7.memory import OnsiteMemory
    from v7.neural_network import get_neural_network, NeuralNetworkProcessor
    from v7.enhanced_language_integration import EnhancedLanguageIntegration
    HAS_V7_CORE = True
except ImportError as e:
    print(f"Warning: V7 core components not fully available: {e}")
    HAS_V7_CORE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neural-linguistic-bridge")

class PatternTranslator:
    """
    Translates patterns between neural and linguistic domains
    
    This class converts neural network patterns to linguistic representations
    and vice versa, enabling cross-domain communication.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            "neural_threshold": 0.65,
            "linguistic_threshold": 0.70,
            "resonance_factor": 0.85,
            "dimensionality_reduction": True,
            "use_transformer_embeddings": True
        }
        if config:
            self.config.update(config)
        
        self.pattern_cache = {}
        logger.info("PatternTranslator initialized with config: %s", self.config)
    
    def neural_to_linguistic(self, neural_pattern: Dict) -> Dict:
        """Convert neural pattern to linguistic representation"""
        # Extract key components from neural pattern
        pattern_id = neural_pattern.get("id", f"pattern_{time.time()}")
        activation_values = neural_pattern.get("activations", {})
        connections = neural_pattern.get("connections", [])
        
        # Create linguistic representation
        linguistic_pattern = {
            "id": pattern_id,
            "source": "neural",
            "timestamp": datetime.now().isoformat(),
            "concepts": self._extract_concepts(activation_values),
            "relationships": self._extract_relationships(connections),
            "meta": {
                "confidence": self._calculate_confidence(activation_values),
                "complexity": len(activation_values),
                "neural_signature": self._create_signature(neural_pattern)
            }
        }
        
        # Cache pattern for future reference
        self.pattern_cache[pattern_id] = {
            "neural": neural_pattern,
            "linguistic": linguistic_pattern,
            "timestamp": time.time()
        }
        
        return linguistic_pattern
    
    def linguistic_to_neural(self, linguistic_pattern: Dict) -> Dict:
        """Convert linguistic pattern to neural representation"""
        # Extract key components from linguistic pattern
        pattern_id = linguistic_pattern.get("id", f"pattern_{time.time()}")
        concepts = linguistic_pattern.get("concepts", {})
        relationships = linguistic_pattern.get("relationships", [])
        
        # Create neural representation
        neural_pattern = {
            "id": pattern_id,
            "source": "linguistic",
            "timestamp": datetime.now().isoformat(),
            "activations": self._create_activations(concepts),
            "connections": self._create_connections(relationships),
            "meta": {
                "confidence": self._calculate_linguistic_confidence(concepts),
                "complexity": len(concepts),
                "linguistic_signature": self._create_linguistic_signature(linguistic_pattern)
            }
        }
        
        # Cache pattern for future reference
        self.pattern_cache[pattern_id] = {
            "neural": neural_pattern,
            "linguistic": linguistic_pattern,
            "timestamp": time.time()
        }
        
        return neural_pattern
    
    def _extract_concepts(self, activations: Dict) -> Dict:
        """Extract concepts from neural activations"""
        concepts = {}
        threshold = self.config["neural_threshold"]
        
        for node_id, activation in activations.items():
            if activation > threshold:
                # Convert neural node to concept
                concepts[node_id] = {
                    "weight": activation,
                    "type": self._determine_concept_type(node_id, activation),
                    "related_nodes": [n for n in activations if activations[n] > threshold * 0.8][:5]
                }
        
        return concepts
    
    def _determine_concept_type(self, node_id: str, activation: float) -> str:
        """Determine concept type based on node and activation"""
        if "pattern" in node_id:
            return "pattern"
        elif "memory" in node_id:
            return "memory"
        elif activation > 0.85:
            return "core"
        elif activation > 0.75:
            return "supporting"
        else:
            return "peripheral"
    
    def _extract_relationships(self, connections: List) -> List:
        """Extract relationships from neural connections"""
        relationships = []
        
        for connection in connections:
            source = connection.get("source")
            target = connection.get("target")
            weight = connection.get("weight", 0.5)
            
            if weight > self.config["neural_threshold"]:
                relationship = {
                    "source": source,
                    "target": target,
                    "type": self._determine_relationship_type(weight),
                    "strength": weight
                }
                relationships.append(relationship)
        
        return relationships
    
    def _determine_relationship_type(self, weight: float) -> str:
        """Determine relationship type based on connection weight"""
        if weight > 0.9:
            return "causal"
        elif weight > 0.8:
            return "correlative"
        elif weight > 0.7:
            return "associative"
        else:
            return "weak_association"
    
    def _calculate_confidence(self, activations: Dict) -> float:
        """Calculate confidence level of neural pattern"""
        if not activations:
            return 0.0
        
        top_activations = sorted(activations.values(), reverse=True)[:5]
        return sum(top_activations) / len(top_activations) if top_activations else 0.0
    
    def _create_signature(self, neural_pattern: Dict) -> str:
        """Create unique signature for neural pattern"""
        key_elements = {
            "top_nodes": sorted([(k, v) for k, v in neural_pattern.get("activations", {}).items()], 
                               key=lambda x: x[1], reverse=True)[:3],
            "connection_count": len(neural_pattern.get("connections", [])),
            "timestamp": neural_pattern.get("timestamp", "")
        }
        return json.dumps(key_elements)
    
    def _create_activations(self, concepts: Dict) -> Dict:
        """Create neural activations from linguistic concepts"""
        activations = {}
        
        for concept_id, concept in concepts.items():
            node_id = self._concept_to_node_id(concept_id)
            weight = concept.get("weight", 0.7)
            
            # Adjust weight based on concept type
            if concept.get("type") == "core":
                weight = min(weight * 1.2, 1.0)
            elif concept.get("type") == "peripheral":
                weight = weight * 0.9
                
            activations[node_id] = weight
            
            # Add related nodes with lower activation
            for related in concept.get("related_nodes", []):
                related_node = self._concept_to_node_id(related)
                if related_node not in activations:
                    activations[related_node] = weight * 0.7
        
        return activations
    
    def _concept_to_node_id(self, concept_id: str) -> str:
        """Convert concept ID to neural node ID"""
        if concept_id.startswith("neural_"):
            return concept_id
        return f"neural_{concept_id}"
    
    def _create_connections(self, relationships: List) -> List:
        """Create neural connections from linguistic relationships"""
        connections = []
        
        for rel in relationships:
            source = self._concept_to_node_id(rel.get("source", ""))
            target = self._concept_to_node_id(rel.get("target", ""))
            strength = rel.get("strength", 0.5)
            
            # Adjust strength based on relationship type
            if rel.get("type") == "causal":
                strength = min(strength * 1.2, 1.0)
            elif rel.get("type") == "weak_association":
                strength = strength * 0.8
                
            connection = {
                "source": source,
                "target": target,
                "weight": strength,
                "type": rel.get("type", "associative")
            }
            connections.append(connection)
        
        return connections
    
    def _calculate_linguistic_confidence(self, concepts: Dict) -> float:
        """Calculate confidence level of linguistic pattern"""
        if not concepts:
            return 0.0
        
        weights = [c.get("weight", 0.0) for c in concepts.values()]
        return sum(weights) / len(weights) if weights else 0.0
    
    def _create_linguistic_signature(self, linguistic_pattern: Dict) -> str:
        """Create unique signature for linguistic pattern"""
        concepts = linguistic_pattern.get("concepts", {})
        key_concepts = sorted([(k, c.get("weight", 0)) for k, c in concepts.items()], 
                             key=lambda x: x[1], reverse=True)[:3]
        
        key_elements = {
            "top_concepts": key_concepts,
            "relationship_count": len(linguistic_pattern.get("relationships", [])),
            "timestamp": linguistic_pattern.get("timestamp", "")
        }
        return json.dumps(key_elements)


class NeuralLinguisticBridge:
    """
    Facilitates bidirectional communication between neural and language systems
    
    This bridge enables the exchange of patterns, concepts, and knowledge between
    the neural network and language processing systems, allowing for enhanced
    cross-domain learning and information integration.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Neural-Linguistic Bridge"""
        self.config = {
            "mock_mode": not HAS_V7_CORE,
            "async_communication": True,
            "queue_size": 100,
            "processing_threads": 2,
            "processing_interval": 0.5,  # seconds
            "save_patterns": True,
            "patterns_path": "data/neural_linguistic/patterns",
            "logging_level": "INFO"
        }
        if config:
            self.config.update(config)
        
        # Set logging level
        logger.setLevel(getattr(logging, self.config["logging_level"]))
        
        # Initialize components
        self.translator = PatternTranslator()
        self.neural_processor = None
        self.language_system = None
        self.memory_system = None
        
        # Create message queues
        self.neural_to_lang_queue = Queue(maxsize=self.config["queue_size"])
        self.lang_to_neural_queue = Queue(maxsize=self.config["queue_size"])
        
        # Initialize processing threads
        self.processing_threads = []
        self.running = False
        
        # Create pattern storage directory
        if self.config["save_patterns"]:
            os.makedirs(self.config["patterns_path"], exist_ok=True)
        
        # Initialize V7 components if available
        self._initialize_v7_components()
        
        logger.info("Neural-Linguistic Bridge initialized with config: %s", self.config)
    
    def _initialize_v7_components(self):
        """Initialize V7 components if available"""
        if not self.config["mock_mode"] and HAS_V7_CORE:
            try:
                # Initialize neural network
                neural_network = get_neural_network()
                self.neural_processor = NeuralNetworkProcessor(neural_network)
                logger.info("Neural Network Processor initialized")
                
                # Initialize language system
                self.language_system = EnhancedLanguageIntegration(mock_mode=False)
                logger.info("Enhanced Language Integration initialized")
                
                # Initialize memory system
                self.memory_system = OnsiteMemory()
                logger.info("Onsite Memory system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize V7 components: {e}")
                logger.info("Falling back to mock mode")
                self.config["mock_mode"] = True
    
    def start(self):
        """Start the bridge processing threads"""
        if self.running:
            logger.warning("Bridge is already running")
            return
        
        self.running = True
        
        # Start neural-to-language processing thread
        n2l_thread = threading.Thread(
            target=self._process_neural_to_language_queue,
            daemon=True,
            name="NeuralToLanguageThread"
        )
        self.processing_threads.append(n2l_thread)
        n2l_thread.start()
        
        # Start language-to-neural processing thread
        l2n_thread = threading.Thread(
            target=self._process_language_to_neural_queue,
            daemon=True,
            name="LanguageToNeuralThread"
        )
        self.processing_threads.append(l2n_thread)
        l2n_thread.start()
        
        logger.info("Neural-Linguistic Bridge started")
    
    def stop(self):
        """Stop the bridge processing threads"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        self.processing_threads = []
        logger.info("Neural-Linguistic Bridge stopped")
    
    def send_neural_pattern(self, pattern: Dict):
        """Send neural pattern to language system"""
        if not self.running:
            self.start()
        
        try:
            self.neural_to_lang_queue.put(pattern, block=False)
            logger.debug(f"Neural pattern queued: {pattern.get('id', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to queue neural pattern: {e}")
            return False
    
    def send_linguistic_pattern(self, pattern: Dict):
        """Send linguistic pattern to neural system"""
        if not self.running:
            self.start()
        
        try:
            self.lang_to_neural_queue.put(pattern, block=False)
            logger.debug(f"Linguistic pattern queued: {pattern.get('id', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to queue linguistic pattern: {e}")
            return False
    
    def _process_neural_to_language_queue(self):
        """Process neural patterns and send to language system"""
        logger.info("Starting neural-to-language processing thread")
        
        while self.running:
            try:
                # Get neural pattern from queue
                if self.neural_to_lang_queue.empty():
                    time.sleep(self.config["processing_interval"])
                    continue
                
                neural_pattern = self.neural_to_lang_queue.get(block=False)
                
                # Translate neural pattern to linguistic representation
                linguistic_pattern = self.translator.neural_to_linguistic(neural_pattern)
                
                # Process the linguistic pattern
                self._process_linguistic_pattern(linguistic_pattern)
                
                # Save pattern if configured
                if self.config["save_patterns"]:
                    self._save_pattern(linguistic_pattern, "linguistic")
                
                self.neural_to_lang_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in neural-to-language processing: {e}")
                time.sleep(1.0)  # Pause on error
    
    def _process_language_to_neural_queue(self):
        """Process linguistic patterns and send to neural system"""
        logger.info("Starting language-to-neural processing thread")
        
        while self.running:
            try:
                # Get linguistic pattern from queue
                if self.lang_to_neural_queue.empty():
                    time.sleep(self.config["processing_interval"])
                    continue
                
                linguistic_pattern = self.lang_to_neural_queue.get(block=False)
                
                # Translate linguistic pattern to neural representation
                neural_pattern = self.translator.linguistic_to_neural(linguistic_pattern)
                
                # Process the neural pattern
                self._process_neural_pattern(neural_pattern)
                
                # Save pattern if configured
                if self.config["save_patterns"]:
                    self._save_pattern(neural_pattern, "neural")
                
                self.lang_to_neural_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in language-to-neural processing: {e}")
                time.sleep(1.0)  # Pause on error
    
    def _process_linguistic_pattern(self, pattern: Dict):
        """Process linguistic pattern with language system"""
        logger.debug(f"Processing linguistic pattern: {pattern.get('id', 'unknown')}")
        
        if self.config["mock_mode"] or not self.language_system:
            logger.debug("Mock processing linguistic pattern")
            return
        
        try:
            # Convert pattern to format expected by language system
            language_input = {
                "text": self._pattern_to_text(pattern),
                "pattern_data": pattern,
                "source": "neural_bridge",
                "type": "pattern_integration"
            }
            
            # Process with language system
            result = self.language_system.process_text(language_input["text"], 
                                                     context={"pattern": pattern})
            
            logger.debug(f"Language system processed pattern with result: {result}")
            
            # Store in memory if available
            if self.memory_system:
                memory_entry = {
                    "type": "neural_linguistic_pattern",
                    "content": pattern,
                    "timestamp": datetime.now().isoformat(),
                    "processing_result": result
                }
                self.memory_system.update_memory(pattern["id"], memory_entry)
                
        except Exception as e:
            logger.error(f"Error processing linguistic pattern: {e}")
    
    def _pattern_to_text(self, pattern: Dict) -> str:
        """Convert pattern to natural language text"""
        # Extract concepts and relationships
        concepts = pattern.get("concepts", {})
        relationships = pattern.get("relationships", [])
        
        # Create text representation
        text_parts = ["Pattern Analysis:"]
        
        # Add concepts
        if concepts:
            text_parts.append("\nKey concepts:")
            for concept_id, concept in concepts.items():
                weight = concept.get("weight", 0.0)
                concept_type = concept.get("type", "unknown")
                text_parts.append(f"- {concept_id} ({concept_type}, weight: {weight:.2f})")
        
        # Add relationships
        if relationships:
            text_parts.append("\nRelationships:")
            for rel in relationships:
                source = rel.get("source", "")
                target = rel.get("target", "")
                rel_type = rel.get("type", "unknown")
                strength = rel.get("strength", 0.0)
                text_parts.append(f"- {source} → {target} ({rel_type}, strength: {strength:.2f})")
        
        # Add metadata
        meta = pattern.get("meta", {})
        if meta:
            text_parts.append("\nMetadata:")
            for key, value in meta.items():
                if isinstance(value, (int, float, str, bool)):
                    text_parts.append(f"- {key}: {value}")
        
        return "\n".join(text_parts)
    
    def _process_neural_pattern(self, pattern: Dict):
        """Process neural pattern with neural system"""
        logger.debug(f"Processing neural pattern: {pattern.get('id', 'unknown')}")
        
        if self.config["mock_mode"] or not self.neural_processor:
            logger.debug("Mock processing neural pattern")
            return
        
        try:
            # Extract activations and connections
            activations = pattern.get("activations", {})
            connections = pattern.get("connections", [])
            
            # Process with neural processor
            self.neural_processor.activate_nodes(activations)
            
            # Enhance connections based on pattern
            for connection in connections:
                source = connection.get("source")
                target = connection.get("target")
                weight = connection.get("weight", 0.5)
                
                if source and target:
                    self.neural_processor.enhance_connection(source, target, weight)
            
            logger.debug(f"Neural processor integrated pattern")
            
        except Exception as e:
            logger.error(f"Error processing neural pattern: {e}")
    
    def _save_pattern(self, pattern: Dict, pattern_type: str):
        """Save pattern to disk"""
        if not self.config["save_patterns"]:
            return
        
        try:
            pattern_id = pattern.get("id", f"pattern_{time.time()}")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{pattern_type}_{pattern_id}_{timestamp}.json"
            filepath = os.path.join(self.config["patterns_path"], filename)
            
            with open(filepath, 'w') as f:
                json.dump(pattern, f, indent=2)
                
            logger.debug(f"Saved {pattern_type} pattern to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save pattern: {e}")
    
    def get_status(self) -> Dict:
        """Get current status of the bridge"""
        return {
            "running": self.running,
            "neural_to_lang_queue_size": self.neural_to_lang_queue.qsize(),
            "lang_to_neural_queue_size": self.lang_to_neural_queue.qsize(),
            "mock_mode": self.config["mock_mode"],
            "translator_config": self.translator.config,
            "neural_processor_available": self.neural_processor is not None,
            "language_system_available": self.language_system is not None,
            "memory_system_available": self.memory_system is not None,
            "processing_threads": [t.name for t in self.processing_threads if t.is_alive()],
            "timestamp": datetime.now().isoformat()
        }


# Global instance for module-level access
_bridge_instance = None

def get_neural_linguistic_bridge(config: Optional[Dict] = None) -> NeuralLinguisticBridge:
    """Get the global Neural-Linguistic Bridge instance"""
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = NeuralLinguisticBridge(config)
    elif config:
        logger.warning("Bridge already initialized, config update ignored")
        
    return _bridge_instance

# Example usage when run directly
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create bridge instance
    bridge = get_neural_linguistic_bridge({
        "mock_mode": True,  # Use mock mode for testing
        "processing_threads": 2,
        "save_patterns": True
    })
    
    # Start the bridge
    bridge.start()
    
    # Create a sample neural pattern
    sample_neural_pattern = {
        "id": "test_pattern_1",
        "activations": {
            "node_1": 0.85,
            "node_2": 0.75,
            "node_3": 0.65,
            "node_pattern_4": 0.90
        },
        "connections": [
            {"source": "node_1", "target": "node_2", "weight": 0.8},
            {"source": "node_2", "target": "node_3", "weight": 0.7},
            {"source": "node_1", "target": "node_pattern_4", "weight": 0.85}
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    # Send the pattern through the bridge
    bridge.send_neural_pattern(sample_neural_pattern)
    
    # Wait for processing
    time.sleep(2)
    
    # Create a sample linguistic pattern
    sample_linguistic_pattern = {
        "id": "test_concept_1",
        "concepts": {
            "consciousness": {"weight": 0.85, "type": "core"},
            "learning": {"weight": 0.75, "type": "supporting"},
            "neural_network": {"weight": 0.80, "type": "core"},
            "memory": {"weight": 0.70, "type": "supporting"}
        },
        "relationships": [
            {"source": "consciousness", "target": "learning", "type": "causal", "strength": 0.8},
            {"source": "neural_network", "target": "memory", "type": "associative", "strength": 0.75},
            {"source": "learning", "target": "memory", "type": "causal", "strength": 0.85}
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    # Send the linguistic pattern through the bridge
    bridge.send_linguistic_pattern(sample_linguistic_pattern)
    
    # Wait for processing
    time.sleep(2)
    
    # Print status
    print(json.dumps(bridge.get_status(), indent=2))
    
    # Stop the bridge
    bridge.stop()
    
    print("Neural-Linguistic Bridge test completed") 