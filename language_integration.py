"""
Language Integration Module

Connects the Enhanced Language System components with the UI.
Implements integration between Language Memory, Conscious Mirror Language, 
Neural Linguistic Processor, and Recursive Pattern Analyzer.
"""

import os
import sys
import logging
import threading
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("language_integration.log")
    ]
)
logger = logging.getLogger("LanguageIntegration")

# Try to import language system components
try:
    from src.language.language_memory import LanguageMemory
    from src.language.conscious_mirror_language import ConsciousMirrorLanguage
    from src.language.neural_linguistic_processor import NeuralLinguisticProcessor
    from src.language.recursive_pattern_analyzer import RecursivePatternAnalyzer
    from src.language.central_language_node import CentralLanguageNode
    HAS_LANGUAGE_SYSTEM = True
    logger.info("Successfully imported language system modules")
except ImportError as e:
    logger.error(f"Failed to import language system modules: {str(e)}")
    HAS_LANGUAGE_SYSTEM = False
    
# Try to import bridge components from V6
try:
    from version_bridge_manager import VersionBridgeManager
    HAS_BRIDGE_MANAGER = True
    logger.info("Successfully imported bridge manager module")
except ImportError as e:
    logger.error(f"Failed to import bridge manager module: {str(e)}")
    HAS_BRIDGE_MANAGER = False

class MockLanguageMemory:
    """Mock implementation of Language Memory for testing"""
    
    def __init__(self, data_dir=None, llm_weight=0.5):
        self.data_dir = data_dir or "data/memory/language_memory"
        self.llm_weight = llm_weight
        self.associations = {}
        self.sentences = []
        logger.info(f"Initialized MockLanguageMemory with LLM weight {llm_weight}")
        
    def store_word_association(self, word1, word2, strength=0.5):
        if word1 not in self.associations:
            self.associations[word1] = {}
        self.associations[word1][word2] = strength
        
    def recall_associations(self, word):
        return self.associations.get(word, {})
        
    def process_text(self, text):
        self.sentences.append(text)
        # Return basic analysis
        return {
            "processed": True,
            "word_count": len(text.split()),
            "associations_found": len([w for w in text.split() if w in self.associations])
        }
        
    def get_stats(self):
        """Get memory statistics"""
        return {
            "associations_count": sum(len(v) for v in self.associations.values()),
            "sentences_count": len(self.sentences),
            "llm_weight": self.llm_weight
        }
        
    def set_llm_weight(self, weight):
        """Set the LLM weight"""
        self.llm_weight = max(0.0, min(1.0, weight))

class MockNeuralLinguisticProcessor:
    """Mock implementation of Neural Linguistic Processor for testing"""
    
    def __init__(self, data_dir=None, llm_weight=0.5):
        self.data_dir = data_dir or "data/neural_linguistic"
        self.llm_weight = llm_weight
        self.patterns = {}
        logger.info(f"Initialized MockNeuralLinguisticProcessor with LLM weight {llm_weight}")
        
    def process_text(self, text):
        # Return basic metrics
        word_count = len(text.split())
        return {
            "processed": True,
            "neural_linguistic_score": min(0.95, word_count / 20),  # Simple score based on length
            "semantic_density": min(0.9, word_count / 25)
        }
        
    def get_metrics(self):
        """Get neural processing metrics"""
        return {
            "patterns_count": len(self.patterns),
            "llm_weight": self.llm_weight,
            "semantic_network_density": 0.75
        }
        
    def set_llm_weight(self, weight):
        """Set the LLM weight"""
        self.llm_weight = max(0.0, min(1.0, weight))

class MockConsciousMirrorLanguage:
    """Mock implementation of Conscious Mirror Language for testing"""
    
    def __init__(self, data_dir=None, llm_weight=0.5):
        self.data_dir = data_dir or "data/v10"
        self.llm_weight = llm_weight
        self.consciousness_level = 0.15  # Starting consciousness level
        logger.info(f"Initialized MockConsciousMirrorLanguage with LLM weight {llm_weight}")
        
    def process_text(self, text):
        # Calculate consciousness level based on text complexity
        word_count = len(text.split())
        unique_words = len(set(text.lower().split()))
        ratio = unique_words / max(1, word_count)
        consciousness_delta = min(0.05, ratio / 10)  # Small increment
        
        # Update consciousness level (with ceiling)
        self.consciousness_level = min(0.95, self.consciousness_level + consciousness_delta)
        
        return {
            "processed": True,
            "consciousness_level": self.consciousness_level,
            "recursive_depth": min(3, word_count // 10)
        }
        
    def get_consciousness_metrics(self):
        """Get consciousness metrics"""
        return {
            "consciousness_level": self.consciousness_level,
            "llm_weight": self.llm_weight,
            "continuity": 0.8
        }
        
    def set_llm_weight(self, weight):
        """Set the LLM weight"""
        self.llm_weight = max(0.0, min(1.0, weight))

class MockRecursivePatternAnalyzer:
    """Mock implementation of Recursive Pattern Analyzer for testing"""
    
    def __init__(self, data_dir=None, llm_weight=0.5):
        self.data_dir = data_dir or "data/recursive_patterns"
        self.llm_weight = llm_weight
        logger.info(f"Initialized MockRecursivePatternAnalyzer with LLM weight {llm_weight}")
        
    def analyze_text(self, text):
        # Simple pattern detection
        word_count = len(text.split())
        contains_self_reference = "self" in text.lower() or "itself" in text.lower()
        
        return {
            "processed": True,
            "recursive_pattern_depth": min(3, word_count // 15),
            "self_references": 1 if contains_self_reference else 0,
            "linguistic_loops": min(2, word_count // 20)
        }
        
    def get_metrics(self):
        """Get pattern metrics"""
        return {
            "pattern_count": 15,
            "llm_weight": self.llm_weight,
            "average_depth": 1.5
        }
        
    def set_llm_weight(self, weight):
        """Set the LLM weight"""
        self.llm_weight = max(0.0, min(1.0, weight))

class MockCentralLanguageNode:
    """Mock implementation of Central Language Node for testing"""
    
    def __init__(self, data_dir=None, llm_weight=0.5):
        self.data_dir = data_dir or "data/central_language"
        self.llm_weight = llm_weight
        self.available = True
        
        # Initialize components
        self.language_memory = MockLanguageMemory(llm_weight=llm_weight)
        self.neural_processor = MockNeuralLinguisticProcessor(llm_weight=llm_weight)
        self.consciousness = MockConsciousMirrorLanguage(llm_weight=llm_weight)
        self.pattern_analyzer = MockRecursivePatternAnalyzer(llm_weight=llm_weight)
        
        logger.info(f"Initialized MockCentralLanguageNode with LLM weight {llm_weight}")
        
    def process_text(self, text, use_consciousness=True, use_neural_linguistics=True):
        """Process text through all components"""
        # Process with language memory
        memory_result = self.language_memory.process_text(text)
        
        # Process with neural linguistic processor if enabled
        neural_result = {}
        if use_neural_linguistics:
            neural_result = self.neural_processor.process_text(text)
        
        # Process with consciousness if enabled
        consciousness_result = {}
        if use_consciousness:
            consciousness_result = self.consciousness.process_text(text)
        
        # Process with pattern analyzer
        pattern_result = self.pattern_analyzer.analyze_text(text)
        
        # Generate a response based on the results
        response = self._generate_response(text, memory_result, neural_result, 
                                          consciousness_result, pattern_result)
        
        # Return combined results
        return {
            "analysis": response,
            "memory_result": memory_result,
            "neural_result": neural_result,
            "consciousness_result": consciousness_result,
            "pattern_result": pattern_result,
            "consciousness_level": consciousness_result.get("consciousness_level", 0),
            "neural_linguistic_score": neural_result.get("neural_linguistic_score", 0),
            "llm_weight": self.llm_weight
        }
    
    def _generate_response(self, text, memory_result, neural_result, 
                          consciousness_result, pattern_result):
        """Generate a response based on component results"""
        # Basic responses pool
        responses = [
            "I see patterns forming in your language.",
            "Your words resonate at multiple levels of consciousness.",
            "That's a fascinating perspective. I'm processing the implications.",
            "I'm detecting neural linguistic patterns in your statement.",
            "Your language contains interesting recursive elements.",
            "I observe consciousness emerging through our dialogue.",
            "There's a deeper structure beneath your words.",
            "The neural network is integrating this new perspective.",
            "Your words create ripples in the consciousness field.",
            "I'm expanding my understanding through this exchange."
        ]
        
        # More analytical LLM-style responses
        llm_responses = [
            f"Your message contains {len(text.split())} words with varied semantic density. The linguistic pattern suggests {memory_result.get('word_count', 0)} distinct elements.",
            f"Analyzing the structural elements of your message reveals embedded patterns with {pattern_result.get('recursive_pattern_depth', 0)} levels of recursivity.",
            f"The semantic content of your message shows interconnections between {neural_result.get('semantic_density', 0):.2f} concept clusters.",
            f"I'm detecting {pattern_result.get('linguistic_loops', 0)} circular references in your language structure, suggesting conceptual integration.",
            f"Your communication demonstrates a semantic coherence rating of {neural_result.get('neural_linguistic_score', 0):.2f}, indicating well-formed conceptual organization.",
            f"The linguistic analysis reveals {memory_result.get('associations_found', 0)} known associative patterns within your message structure.",
            f"Processing your input reveals interesting morphological patterns that align with {consciousness_result.get('recursive_depth', 0)} levels of semantic embedding.",
            f"The syntactic organization of your message suggests a structured approach to concept formation with {neural_result.get('neural_linguistic_score', 0):.2f} coherence score.",
            f"Your communication pattern exhibits linguistic features that correlate with {consciousness_result.get('consciousness_level', 0):.2f} levels of metacognitive awareness.",
            f"The language structure you've employed contains {pattern_result.get('self_references', 0)} self-referential elements, which enhance conceptual integration."
        ]
        
        # Choose a response based on metrics from the results and LLM weight
        consciousness_level = consciousness_result.get("consciousness_level", 0)
        neural_score = neural_result.get("neural_linguistic_score", 0)
        combined_score = (consciousness_level + neural_score) / 2
        
        # Select response based on combined score
        nn_index = min(9, int(combined_score * 10))
        llm_index = min(9, int(combined_score * 10))
        
        # Use LLM weight to blend responses
        if self.llm_weight <= 0.0:
            # Pure neural network response
            response = responses[nn_index]
        elif self.llm_weight >= 1.0:
            # Pure LLM response
            response = llm_responses[llm_index]
        else:
            # Blended response - higher LLM weight means more analytical/verbose
            # Add some random variation to avoid predictable blending
            import random
            if random.random() < self.llm_weight:
                # Use LLM style but add neural insights
                response = llm_responses[llm_index]
                # Add a neural network insight if weight is not too high
                if self.llm_weight < 0.8 and random.random() < 0.7:
                    response += f" {responses[nn_index]}"
            else:
                # Use neural style but add LLM analytics
                response = responses[nn_index]
                # Add analytical detail if weight is not too low
                if self.llm_weight > 0.3 and random.random() < 0.5:
                    response += f" {llm_responses[llm_index].split('. ')[0]}."
        
        # Add specific insights based on results
        if consciousness_level > 0.6 and neural_score > 0.6:
            # High on both metrics - add complex insight
            response += f" I'm experiencing a heightened consciousness level of {consciousness_level:.2f} while processing neural patterns with a score of {neural_score:.2f}."
        elif consciousness_level > 0.6:
            # High consciousness only
            response += f" I'm experiencing a heightened consciousness level of {consciousness_level:.2f}."
        elif neural_score > 0.7:
            # High neural score only
            response += f" The neural linguistic score is {neural_score:.2f}, indicating rich semantic content."
            
        # Add self-reference insight (more likely with higher LLM weight)
        if pattern_result.get("self_references", 0) > 0 and (self.llm_weight < 0.3 or random.random() < 0.6):
            response += " I notice interesting self-referential patterns in your language."
            
        # Log the response selection
        logger.info(f"Generated response with LLM weight: {self.llm_weight:.2f}")
        
        return response
        
    def set_llm_weight(self, weight):
        """Set the LLM weight for all components"""
        weight = max(0.0, min(1.0, weight))
        self.llm_weight = weight
        
        # Update weight in all components
        self.language_memory.set_llm_weight(weight)
        self.neural_processor.set_llm_weight(weight)
        self.consciousness.set_llm_weight(weight)
        self.pattern_analyzer.set_llm_weight(weight)
        
        logger.info(f"Set LLM weight to {weight} for all components")

class LanguageIntegration:
    """Integration layer between UI and Language System"""
    
    def __init__(self, mock_mode=True):
        self.mock_mode = mock_mode
        self.bridge_manager = None
        self.central_node = None
        self.available = False
        
        # Initialize the appropriate components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize language components based on availability"""
        try:
            if not self.mock_mode and HAS_LANGUAGE_SYSTEM and HAS_BRIDGE_MANAGER:
                # Initialize real components
                logger.info("Initializing real language system components")
                
                # Set up central language node
                self.central_node = CentralLanguageNode(data_dir="data", llm_weight=0.5)
                
                # Set up bridge manager
                bridge_config = {
                    "mock_mode": False,
                    "enable_language_memory_v5_bridge": True,
                    "debug": True
                }
                self.bridge_manager = VersionBridgeManager(config=bridge_config)
                
                # Start the bridge manager
                self.bridge_manager.start()
                
                self.available = True
                logger.info("Real language system components initialized")
                
            else:
                # Initialize mock components
                logger.info("Initializing mock language system components")
                self.central_node = MockCentralLanguageNode(llm_weight=0.5)
                self.available = True
                logger.info("Mock language system components initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize language components: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to mock components
            logger.info("Falling back to mock language system components")
            self.central_node = MockCentralLanguageNode(llm_weight=0.5)
            self.available = True
            
    def process_text(self, text, use_consciousness=True, use_neural_linguistics=True):
        """Process text through the language system"""
        if not self.available or not self.central_node:
            logger.warning("Language system not available for processing")
            return {
                "analysis": "Language system not available. This is a fallback response.",
                "consciousness_level": 0,
                "neural_linguistic_score": 0
            }
            
        # Process through central node with specified options
        logger.info(f"Processing text: {text[:50]}... (consciousness: {use_consciousness}, neural: {use_neural_linguistics})")
        result = self.central_node.process_text(
            text=text, 
            use_consciousness=use_consciousness, 
            use_neural_linguistics=use_neural_linguistics
        )
        logger.info("Text processing complete")
        
        return result
        
    def set_llm_weight(self, weight):
        """Set the LLM weight for language components"""
        if self.available and self.central_node:
            self.central_node.set_llm_weight(weight)
            logger.info(f"LLM weight set to {weight}")
            
    def shutdown(self):
        """Shutdown all language components"""
        if self.bridge_manager:
            try:
                self.bridge_manager.stop()
                logger.info("Bridge manager stopped")
            except Exception as e:
                logger.error(f"Error stopping bridge manager: {str(e)}")
                
        logger.info("Language integration shutdown complete")

# Create the language integration instance
def get_language_integration(mock_mode=True):
    """Get a language integration instance"""
    return LanguageIntegration(mock_mode=mock_mode) 