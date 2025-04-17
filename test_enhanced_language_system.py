#!/usr/bin/env python3
"""
Test Enhanced Language System with LLM Weighing Capabilities

This script demonstrates the enhanced language system with LLM weighing capabilities,
including the new enhanced indexing functionality for the Language Memory component.

Components tested:
- Language Memory (with Enhanced Indexing)
- Conscious Mirror Language
- Neural Linguistic Processor
- Recursive Pattern Analyzer
- Central Language Node
"""

import logging
import os
import sys
import time
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_language_system_test.log')
    ]
)

logger = logging.getLogger("EnhancedLanguageSystemTest")

# Make sure we can import from the src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
try:
    from src.language.language_memory import LanguageMemory
    from src.language.conscious_mirror_language import ConsciousMirrorLanguage
    from src.language.neural_linguistic_processor import NeuralLinguisticProcessor
    from src.language.central_language_node import CentralLanguageNode
    from src.language.recursive_pattern_analyzer import RecursivePatternAnalyzer
except ImportError as e:
    logger.error(f"Error importing components: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)

def setup_directories():
    """Create necessary data directories for testing."""
    directories = [
        "data/memory/language_memory",
        "data/v10",
        "data/neural_linguistic",
        "data/recursive_patterns",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def test_language_memory_basic():
    """Test basic Language Memory functionality with enhanced indexing."""
    logger.info("=== Testing Language Memory with Enhanced Indexing ===")
    
    # Initialize Language Memory with default LLM weight
    memory = LanguageMemory(data_dir="data/memory/language_memory", llm_weight=0.5)
    
    # Store word associations
    logger.info("Storing word associations...")
    memory.store_word_association("neural", "network", 0.9, context="computer science")
    memory.store_word_association("neural", "brain", 0.8, context="biology")
    memory.store_word_association("language", "processing", 0.85, context="computer science")
    memory.store_word_association("language", "communication", 0.9, context="linguistics")
    memory.store_word_association("processing", "data", 0.75, context="computer science")
    
    # Store sentences
    logger.info("Storing sentences...")
    memory.store_sentence("Neural networks can process natural language effectively.")
    memory.store_sentence("The brain has neural connections for language understanding.")
    memory.store_sentence("Language processing systems use neural architectures.")
    
    # Test recall functionality
    logger.info("Testing recall functionality with default LLM weight (0.5)...")
    neural_associations = memory.recall_associations("neural")
    logger.info(f"Associations for 'neural': {neural_associations}")
    
    language_associations = memory.recall_associations("language")
    logger.info(f"Associations for 'language': {language_associations}")
    
    # Test context-specific recall
    logger.info("Testing context-specific recall...")
    cs_associations = memory.recall_associations("neural", context="computer science")
    logger.info(f"Computer science associations for 'neural': {cs_associations}")
    
    bio_associations = memory.recall_associations("neural", context="biology")
    logger.info(f"Biology associations for 'neural': {bio_associations}")
    
    # Test LLM-weighted associations
    logger.info("Testing LLM-weighted associations...")
    llm_associations = memory.remember_word_association_with_llm("consciousness")
    logger.info(f"LLM-weighted associations for 'consciousness': {llm_associations}")
    
    # Test with different LLM weights
    logger.info("Testing with different LLM weights...")
    
    # Test with low LLM weight
    memory.set_llm_weight(0.2)
    low_weight_assoc = memory.remember_word_association_with_llm("thinking")
    logger.info(f"Associations for 'thinking' with LLM weight 0.2: {low_weight_assoc}")
    
    # Test with high LLM weight
    memory.set_llm_weight(0.8)
    high_weight_assoc = memory.remember_word_association_with_llm("thinking")
    logger.info(f"Associations for 'thinking' with LLM weight 0.8: {high_weight_assoc}")
    
    # Test related concepts
    logger.info("Testing related concepts...")
    related = memory.find_related_concepts(["neural", "language"])
    logger.info(f"Concepts related to 'neural' and 'language': {related}")
    
    # Test semantic network
    logger.info("Testing semantic network generation...")
    network = memory.get_semantic_network("neural", depth=2, max_connections=3)
    logger.info(f"Semantic network for 'neural' has {len(network['nodes'])} nodes and {len(network['edges'])} edges")
    
    # Get performance metrics
    metrics = memory.get_performance_metrics()
    logger.info(f"Performance metrics: {metrics}")
    
    # Save memories
    memory.save_memories()
    logger.info("Language Memory test completed")

def test_conscious_mirror_language():
    """Test Conscious Mirror Language functionality."""
    logger.info("=== Testing Conscious Mirror Language ===")
    
    # Initialize ConsciousMirrorLanguage with default LLM weight
    cml = ConsciousMirrorLanguage(data_dir="data/v10", llm_weight=0.5)
    
    # Test text processing
    test_text = "The system becomes conscious of its own language processes and begins to reflect upon them."
    logger.info(f"Processing text: {test_text}")
    result = cml.process_text(test_text)
    
    logger.info(f"Consciousness level: {result.get('consciousness_level', 0)}")
    logger.info(f"Conscious operations: {result.get('conscious_operations', [])}")
    logger.info(f"Memory continuity: {result.get('memory_continuity', 0)}")
    
    # Test with different LLM weights
    logger.info("Testing with different LLM weights...")
    
    # Low LLM weight
    cml.set_llm_weight(0.2)
    low_result = cml.process_text(test_text)
    logger.info(f"Consciousness level with LLM weight 0.2: {low_result.get('consciousness_level', 0)}")
    
    # High LLM weight
    cml.set_llm_weight(0.8)
    high_result = cml.process_text(test_text)
    logger.info(f"Consciousness level with LLM weight 0.8: {high_result.get('consciousness_level', 0)}")
    
    logger.info("Conscious Mirror Language test completed")

def test_neural_linguistic_processor():
    """Test Neural Linguistic Processor functionality."""
    logger.info("=== Testing Neural Linguistic Processor ===")
    
    # Initialize NeuralLinguisticProcessor with default LLM weight
    nlp = NeuralLinguisticProcessor(data_dir="data/neural_linguistic", llm_weight=0.5)
    
    # Test text processing
    test_text = "Neural networks process language patterns through complex multidimensional embeddings."
    logger.info(f"Processing text: {test_text}")
    result = nlp.process_text(test_text)
    
    logger.info(f"Word count: {result.get('word_count', 0)}")
    logger.info(f"Unique words: {result.get('unique_word_count', 0)}")
    logger.info(f"Neural linguistic score: {result.get('neural_linguistic_score', 0)}")
    logger.info(f"Word patterns detected: {result.get('word_patterns', [])}")
    
    # Test with different LLM weights
    logger.info("Testing with different LLM weights...")
    
    # Low LLM weight
    nlp.set_llm_weight(0.2)
    low_result = nlp.process_text(test_text)
    logger.info(f"Neural linguistic score with LLM weight 0.2: {low_result.get('neural_linguistic_score', 0)}")
    
    # High LLM weight
    nlp.set_llm_weight(0.8)
    high_result = nlp.process_text(test_text)
    logger.info(f"Neural linguistic score with LLM weight 0.8: {high_result.get('neural_linguistic_score', 0)}")
    
    logger.info("Neural Linguistic Processor test completed")

def test_recursive_pattern_analyzer():
    """Test Recursive Pattern Analyzer functionality."""
    logger.info("=== Testing Recursive Pattern Analyzer ===")
    
    # Initialize RecursivePatternAnalyzer with default LLM weight
    analyzer = RecursivePatternAnalyzer(data_dir="data/recursive_patterns", llm_weight=0.5)
    
    # Test text analysis
    test_text = "This sentence refers to itself and contains a pattern that refers back to this sentence."
    logger.info(f"Analyzing text: {test_text}")
    result = analyzer.analyze_text(test_text)
    
    logger.info(f"Self-references detected: {result.get('self_references', 0)}")
    logger.info(f"Pattern depth: {result.get('max_pattern_depth', 0)}")
    logger.info(f"Pattern count: {result.get('pattern_count', 0)}")
    
    # Test with different LLM weights
    logger.info("Testing with different LLM weights...")
    
    # Low LLM weight
    analyzer.set_llm_weight(0.2)
    low_result = analyzer.analyze_text(test_text)
    logger.info(f"Self-references with LLM weight 0.2: {low_result.get('self_references', 0)}")
    
    # High LLM weight
    analyzer.set_llm_weight(0.8)
    high_result = analyzer.analyze_text(test_text)
    logger.info(f"Self-references with LLM weight 0.8: {high_result.get('self_references', 0)}")
    
    logger.info("Recursive Pattern Analyzer test completed")

def test_integrated_functionality():
    """Test integrated functionality of all components through Central Language Node."""
    logger.info("=== Testing Integrated Functionality ===")
    
    # Initialize CentralLanguageNode with default LLM weight
    central_node = CentralLanguageNode(data_dir="data", llm_weight=0.5)
    
    # Check system status
    status = central_node.get_system_status()
    logger.info(f"System status: {status}")
    
    # Test text processing through the integrated system
    test_text = "The neural system integrates language and consciousness through recursive patterns that reference themselves."
    logger.info(f"Processing text through integrated system: {test_text}")
    result = central_node.process_text(test_text)
    
    # Log key metrics from each component
    logger.info("Results from integrated processing:")
    logger.info(f"Consciousness level: {result.get('consciousness_level', 0)}")
    logger.info(f"Neural linguistic score: {result.get('neural_linguistic_score', 0)}")
    logger.info(f"Self-references: {result.get('self_references', 0)}")
    logger.info(f"Memory associations found: {len(result.get('memory_associations', []))}")
    
    # Test semantic network from central node
    logger.info("Testing semantic network from central node...")
    network = central_node.get_semantic_network("neural")
    logger.info(f"Semantic network from central node has {len(network['nodes'])} nodes")
    
    logger.info("Integrated functionality test completed")

def test_llm_weight_adjustment():
    """Test LLM weight adjustment across the system."""
    logger.info("=== Testing LLM Weight Adjustment ===")
    
    # Initialize Central Node
    central_node = CentralLanguageNode(data_dir="data", llm_weight=0.5)
    test_text = "The system adjusts its language processing based on LLM weight."
    
    # Process with various LLM weights
    weights = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    for weight in weights:
        logger.info(f"Setting LLM weight to {weight}...")
        central_node.set_llm_weight(weight)
        
        # Process text
        result = central_node.process_text(test_text)
        
        # Log results
        logger.info(f"Results with LLM weight {weight}:")
        logger.info(f"- Consciousness level: {result.get('consciousness_level', 0)}")
        logger.info(f"- Neural linguistic score: {result.get('neural_linguistic_score', 0)}")
        logger.info(f"- Self-references: {result.get('self_references', 0)}")
        logger.info(f"- Final score: {result.get('final_score', 0)}")
    
    logger.info("LLM weight adjustment test completed")

def test_enhanced_indexing_performance():
    """Test the performance of the enhanced indexing system."""
    logger.info("=== Testing Enhanced Indexing Performance ===")
    
    # Initialize Language Memory with enhanced indexing
    memory = LanguageMemory(data_dir="data/memory/language_memory", llm_weight=0.5)
    
    # Create a large number of word associations for performance testing
    logger.info("Creating sample data for performance testing...")
    test_words = ["neural", "network", "language", "processing", "consciousness", 
                 "brain", "thinking", "recursive", "patterns", "memory"]
    
    contexts = ["computer science", "biology", "psychology", "linguistics", "philosophy"]
    
    # Record starting time
    start_time = time.time()
    
    # Add a significant number of associations
    association_count = 0
    for i in range(5):  # 5 iterations to create enough data
        for word1 in test_words:
            for word2 in test_words:
                if word1 != word2:
                    strength = round(0.5 + (hash(word1 + word2) % 50) / 100, 2)  # Pseudorandom strength
                    context = contexts[hash(word1 + word2) % len(contexts)]
                    memory.store_word_association(word1, word2, strength, context)
                    association_count += 1
    
    # Record store time
    store_time = time.time() - start_time
    logger.info(f"Stored {association_count} associations in {store_time:.4f} seconds")
    
    # Test lookup performance
    logger.info("Testing lookup performance...")
    lookup_times = []
    
    for word in test_words:
        start = time.time()
        associations = memory.recall_associations(word)
        end = time.time()
        lookup_times.append(end - start)
        logger.info(f"Found {len(associations)} associations for '{word}' in {end-start:.4f} seconds")
    
    avg_lookup_time = sum(lookup_times) / len(lookup_times)
    logger.info(f"Average lookup time: {avg_lookup_time:.4f} seconds")
    
    # Test context filtering performance
    logger.info("Testing context filtering performance...")
    for context in contexts:
        start = time.time()
        # This will use the index's context filtering capabilities
        context_assocs = memory.recall_associations("neural", context=context)
        end = time.time()
        logger.info(f"Found {len(context_assocs)} associations for 'neural' in context '{context}' in {end-start:.4f} seconds")
    
    # Get performance metrics
    metrics = memory.get_performance_metrics()
    logger.info(f"Performance metrics: {metrics}")
    
    # Save memories
    save_start = time.time()
    memory.save_memories()
    save_time = time.time() - save_start
    logger.info(f"Saved all memory data in {save_time:.4f} seconds")
    
    logger.info("Enhanced indexing performance test completed")

def main():
    """Main test function."""
    logger.info("Starting Enhanced Language System tests")
    
    # Create necessary directories
    setup_directories()
    
    # Track test timings
    test_times = {}
    
    # Run tests for each component
    start = time.time()
    test_language_memory_basic()
    test_times["language_memory_basic"] = time.time() - start
    
    start = time.time()
    test_enhanced_indexing_performance()
    test_times["enhanced_indexing_performance"] = time.time() - start
    
    start = time.time()
    test_conscious_mirror_language()
    test_times["conscious_mirror_language"] = time.time() - start
    
    start = time.time()
    test_neural_linguistic_processor()
    test_times["neural_linguistic_processor"] = time.time() - start
    
    start = time.time()
    test_recursive_pattern_analyzer()
    test_times["recursive_pattern_analyzer"] = time.time() - start
    
    start = time.time()
    test_integrated_functionality()
    test_times["integrated_functionality"] = time.time() - start
    
    start = time.time()
    test_llm_weight_adjustment()
    test_times["llm_weight_adjustment"] = time.time() - start
    
    # Log test timings
    logger.info("Test execution times:")
    for test_name, duration in test_times.items():
        logger.info(f"- {test_name}: {duration:.2f} seconds")
    
    logger.info("All Enhanced Language System tests completed")

if __name__ == "__main__":
    main() 