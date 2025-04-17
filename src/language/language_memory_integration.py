#!/usr/bin/env python
"""
Language Memory Integration Module

Demonstrates the enhanced Language Memory system with improved indexing capabilities
for Phase 1 of the Lumina Neural Network Project roadmap.

This module provides usage examples for optimized Language Memory persistence with
enhanced indexing, featuring batch processing, vector embedding, and performance optimization.
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LanguageMemoryIntegration")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import language memory components
from language.language_memory import LanguageMemory
from language.enhanced_indexing import EnhancedIndex

class LanguageMemoryIntegration:
    """
    Integration example for the enhanced Language Memory system.
    Demonstrates optimized persistence with enhanced indexing.
    """
    
    def __init__(self, data_dir: str, llm_weight: float = 0.5, nn_weight: float = 0.5):
        """
        Initialize the Language Memory Integration.
        
        Args:
            data_dir: Directory for data storage
            llm_weight: Weight for LLM integration (0.0-1.0)
            nn_weight: Weight for neural network processing (0.0-1.0)
        """
        logger.info("Initializing Language Memory Integration")
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize Language Memory with enhanced indexing
        self.memory = LanguageMemory(
            data_dir=os.path.join(data_dir, "language_memory"),
            llm_weight=llm_weight,
            nn_weight=nn_weight
        )
        
        # Track performance metrics
        self.performance_metrics = {
            "initialization_time": 0,
            "vector_embedding_time": 0,
            "batch_processing_time": 0,
            "optimization_time": 0,
            "query_times": []
        }
        
        # Initialize timer
        start_time = time.time()
        
        # Generate simple vector embeddings for demonstration
        self._initialize_demo_embeddings()
        
        # Record initialization time
        self.performance_metrics["initialization_time"] = time.time() - start_time
        
        logger.info(f"Language Memory Integration initialized in {self.performance_metrics['initialization_time']:.3f}s")
    
    def _initialize_demo_embeddings(self):
        """Initialize demo vector embeddings for semantic search demonstration."""
        # Skip if we've already initialized embeddings in a previous run
        if self.memory.index.vector_enabled and len(self.memory.index.vector_index) > 0:
            logger.info(f"Using existing vector embeddings ({len(self.memory.index.vector_index)} words)")
            return
            
        # Enable vector indexing in the enhanced index
        self.memory.index.vector_enabled = True
        
        # Create simple word embeddings for demonstration
        # In a real system, these would come from a proper embedding model
        logger.info("Generating demo word embeddings for semantic search")
        start_time = time.time()
        
        # Word categories for demo embeddings
        categories = {
            "animals": ["cat", "dog", "bird", "fish", "elephant", "tiger", "lion"],
            "colors": ["red", "blue", "green", "yellow", "purple", "orange", "brown"],
            "emotions": ["happy", "sad", "angry", "excited", "calm", "nervous", "relaxed"],
            "technology": ["computer", "phone", "internet", "software", "network", "database", "algorithm"],
            "nature": ["tree", "mountain", "river", "ocean", "forest", "plant", "flower"]
        }
        
        # Generate embeddings with category-based similarities
        vector_dimension = self.memory.index.vector_dimension
        category_vectors = {}
        
        # Create a base vector for each category
        for category in categories:
            # Create a somewhat random but consistent vector for each category
            np.random.seed(hash(category) % 10000)
            category_vectors[category] = np.random.rand(vector_dimension)
        
        # Create word embeddings based on their category
        for category, words in categories.items():
            for word in words:
                # Start with the category vector
                word_vector = category_vectors[category].copy()
                
                # Add some random noise to make each word unique but similar to category
                np.random.seed(hash(word) % 10000)
                noise = np.random.rand(vector_dimension) * 0.3
                word_vector += noise
                
                # Add the vector embedding
                self.memory.index.add_vector_embedding(word, word_vector)
                
                # Also add some associations between words in the same category
                for other_word in words:
                    if word != other_word:
                        # Stronger associations within the same category
                        strength = 0.7 + (np.random.rand() * 0.2)  # 0.7-0.9
                        self.memory.remember_word_association(
                            word, other_word, strength, context=category
                        )
        
        # Add some cross-category associations
        cross_category_pairs = [
            ("red", "angry", "color_emotion"),
            ("blue", "calm", "color_emotion"),
            ("green", "nature", "color_context"),
            ("computer", "algorithm", "tech_related"),
            ("nervous", "cat", "animal_emotion"),
            ("river", "blue", "nature_color"),
            ("elephant", "tree", "nature_animal"),
            ("network", "internet", "tech_connection")
        ]
        
        for word1, word2, context in cross_category_pairs:
            strength = 0.5 + (np.random.rand() * 0.3)  # 0.5-0.8
            self.memory.remember_word_association(word1, word2, strength, context=context)
        
        # Record time
        self.performance_metrics["vector_embedding_time"] = time.time() - start_time
        logger.info(f"Generated demo embeddings for {sum(len(words) for words in categories.values())} words in {self.performance_metrics['vector_embedding_time']:.3f}s")
    
    def demonstrate_batch_processing(self, batch_size: int = 100):
        """
        Demonstrate batch processing capabilities.
        
        Args:
            batch_size: Number of associations to process in batch
        """
        logger.info(f"Demonstrating batch processing with {batch_size} items")
        start_time = time.time()
        
        # Set batch size for memory index
        self.memory.index.batch_size_limit = batch_size
        
        # Generate test data
        test_data = []
        for i in range(batch_size):
            source = f"batch_source_{i % 10}"
            target = f"batch_target_{i}"
            strength = 0.5 + (i / batch_size / 2)  # 0.5-1.0
            context = f"batch_context_{i % 5}"
            test_data.append((source, target, strength, context))
        
        # Process batch
        for source, target, strength, context in test_data:
            self.memory.remember_word_association(source, target, strength, context)
        
        # Force processing any remaining items in batch
        if hasattr(self.memory.index, '_process_batch_updates'):
            self.memory.index._process_batch_updates()
        
        # Record time
        elapsed = time.time() - start_time
        self.performance_metrics["batch_processing_time"] = elapsed
        
        # Calculate items per second
        items_per_second = batch_size / elapsed if elapsed > 0 else 0
        
        logger.info(f"Batch processing completed: {batch_size} items in {elapsed:.3f}s ({items_per_second:.1f} items/sec)")
        return items_per_second
    
    def demonstrate_optimization(self):
        """
        Demonstrate memory optimization capabilities.
        
        Returns:
            dict: Optimization statistics
        """
        logger.info("Demonstrating memory optimization")
        start_time = time.time()
        
        # Run optimization
        stats = self.memory.optimize_memory()
        
        # Record time
        elapsed = time.time() - start_time
        self.performance_metrics["optimization_time"] = elapsed
        
        logger.info(f"Memory optimization completed in {elapsed:.3f}s")
        return stats
    
    def demonstrate_semantic_search(self, query_word: str, limit: int = 5):
        """
        Demonstrate semantic search capabilities using vector embeddings.
        
        Args:
            query_word: Word to search for
            limit: Maximum number of results to return
            
        Returns:
            list: Semantically similar words with similarity scores
        """
        if not self.memory.index.vector_enabled:
            logger.warning("Vector search not enabled")
            return []
            
        logger.info(f"Demonstrating semantic search for '{query_word}'")
        
        # Check if query word has a vector
        if query_word not in self.memory.index.vector_index:
            logger.warning(f"No vector embedding found for '{query_word}'")
            return []
        
        # Perform semantic search
        start_time = time.time()
        query_vector = self.memory.index.vector_index[query_word]
        results = self.memory.index.semantic_search(query_vector, limit=limit)
        
        # Record query time
        elapsed = time.time() - start_time
        self.performance_metrics["query_times"].append(elapsed)
        
        # Format results
        formatted_results = []
        for word, similarity in results:
            if word != query_word:  # Skip the query word itself
                formatted_results.append({
                    "word": word,
                    "similarity": similarity,
                    "query": query_word
                })
        
        logger.info(f"Semantic search completed in {elapsed:.3f}s with {len(formatted_results)} results")
        return formatted_results
    
    def compare_memory_performance(self, test_queries: List[str] = None):
        """
        Compare memory performance with and without optimization.
        
        Args:
            test_queries: List of words to test querying
            
        Returns:
            dict: Performance comparison results
        """
        if not test_queries:
            test_queries = ["computer", "happy", "blue", "network", "tree"]
            
        logger.info(f"Comparing memory performance with {len(test_queries)} test queries")
        
        results = {
            "before_optimization": {
                "query_times": [],
                "average_query_time": 0
            },
            "after_optimization": {
                "query_times": [],
                "average_query_time": 0
            },
            "improvement_percentage": 0
        }
        
        # Test queries before optimization
        for query in test_queries:
            start_time = time.time()
            self.memory.recall_associations(query, limit=10)
            elapsed = time.time() - start_time
            results["before_optimization"]["query_times"].append(elapsed)
        
        avg_before = sum(results["before_optimization"]["query_times"]) / len(test_queries)
        results["before_optimization"]["average_query_time"] = avg_before
        
        # Run optimization
        self.demonstrate_optimization()
        
        # Clear cache to ensure fair comparison
        self.memory.association_cache = {}
        
        # Test queries after optimization
        for query in test_queries:
            start_time = time.time()
            self.memory.recall_associations(query, limit=10)
            elapsed = time.time() - start_time
            results["after_optimization"]["query_times"].append(elapsed)
        
        avg_after = sum(results["after_optimization"]["query_times"]) / len(test_queries)
        results["after_optimization"]["average_query_time"] = avg_after
        
        # Calculate improvement
        if avg_before > 0:
            improvement = (avg_before - avg_after) / avg_before * 100
            results["improvement_percentage"] = improvement
        
        logger.info(f"Performance comparison completed: {results['improvement_percentage']:.1f}% improvement")
        return results
    
    def get_memory_statistics(self):
        """
        Get comprehensive statistics about the language memory system.
        
        Returns:
            dict: Memory statistics
        """
        stats = self.memory.get_memory_statistics()
        stats.update({
            "integration_performance": self.performance_metrics,
            "avg_query_time": sum(self.performance_metrics["query_times"]) / len(self.performance_metrics["query_times"]) 
                              if self.performance_metrics["query_times"] else 0
        })
        return stats

def main():
    """Main demonstration function."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "language_memory_integration")
    
    logger.info("Starting Language Memory Integration demonstration")
    
    # Initialize integration with enhanced indexing
    integration = LanguageMemoryIntegration(data_dir, llm_weight=0.6, nn_weight=0.7)
    
    # Demonstrate batch processing
    batch_performance = integration.demonstrate_batch_processing(batch_size=200)
    
    # Demonstrate semantic search
    for query in ["computer", "happy", "blue"]:
        semantic_results = integration.demonstrate_semantic_search(query)
        logger.info(f"Semantic search results for '{query}':")
        for result in semantic_results:
            logger.info(f"  {result['word']} (similarity: {result['similarity']:.3f})")
    
    # Compare performance before and after optimization
    performance_comparison = integration.compare_memory_performance()
    
    # Get memory statistics
    stats = integration.get_memory_statistics()
    
    # Display summary statistics
    logger.info("\nMemory Statistics Summary:")
    logger.info(f"Total words: {stats['total_words']}")
    logger.info(f"Total associations: {stats['total_associations']}")
    logger.info(f"Average associations per word: {stats['index_metrics']['average_associations_per_word']:.2f}")
    logger.info(f"Vector index size: {stats['index_metrics']['vector_index_size']} words")
    logger.info(f"Memory usage estimate: {stats['index_metrics']['memory_usage_estimate_mb']:.2f} MB")
    logger.info(f"Performance improvement after optimization: {performance_comparison['improvement_percentage']:.1f}%")
    
    logger.info("Language Memory Integration demonstration completed")
        
if __name__ == "__main__":
    main() 