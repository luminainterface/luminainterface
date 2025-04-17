"""
Enhanced Indexing System for Language Memory Component

This module implements advanced indexing capabilities for the Language Memory component,
providing faster association lookups, optimized storage, and support for more complex
query patterns. The indexing system integrates with LLM weighing capabilities.

Part of Phase 1 improvements for the Lumina Neural Network Project.
"""

import os
import json
import pickle
import logging
import time
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/enhanced_indexing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedIndexing")

class EnhancedIndex:
    """
    Enhanced indexing system for language memory with optimized lookup performance
    and LLM weight integration.
    """
    
    def __init__(self, data_dir: str, llm_weight: float = 0.5):
        """
        Initialize the enhanced indexing system.
        
        Args:
            data_dir: Directory to store index data
            llm_weight: Weight to apply to LLM suggestions (0.0-1.0)
        """
        self.data_dir = data_dir
        self.llm_weight = llm_weight
        self.primary_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.inverse_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.metadata_index: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.frequency_index: Dict[str, int] = defaultdict(int)
        self.context_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.temporal_index: Dict[str, List[Tuple[float, str, float]]] = defaultdict(list)
        
        # New vector storage for semantic search
        self.vector_dimension = 100  # Default vector dimension
        self.vector_index: Dict[str, np.ndarray] = {}
        self.vector_enabled = False
        
        # Optimization settings
        self.index_compression_enabled = True
        self.pruning_threshold = 0.1  # Prune associations below this strength
        self.cache_limit = 1000  # Maximum cache size
        self.batch_updates = []  # Queue for batch processing
        self.batch_size_limit = 100  # Process in batches of this size
        
        # LLM integration tracking
        self.llm_suggestion_cache: Dict[str, List[Tuple[str, float]]] = {}
        self.llm_integration_stats = {
            "suggestions_used": 0,
            "suggestion_strength_avg": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(data_dir, "enhanced_index"), exist_ok=True)
        
        # Load existing index if available
        self._load_indices()
        
        logger.info(f"Enhanced indexing system initialized with LLM weight: {llm_weight}")
    
    def add_association(self, source: str, target: str, strength: float, 
                       context: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
        """
        Add a word association to the index with optimized storage.
        
        Args:
            source: Source word or phrase
            target: Target word or phrase
            strength: Association strength (0.0-1.0)
            context: Optional context where this association appears
            metadata: Optional metadata for this association
        """
        # Prune weak associations if enabled
        if self.index_compression_enabled and strength < self.pruning_threshold:
            logger.debug(f"Skipping weak association: {source} -> {target} (strength: {strength})")
            return
        
        # Add to batch update queue if batch processing is enabled
        if self.batch_size_limit > 0:
            self.batch_updates.append((source, target, strength, context, metadata))
            
            # Process batch if reached limit
            if len(self.batch_updates) >= self.batch_size_limit:
                self._process_batch_updates()
                return
        
        # Update primary and inverse indices
        self.primary_index[source][target] = strength
        self.inverse_index[target][source] = strength
        
        # Update frequency index
        self.frequency_index[source] += 1
        self.frequency_index[target] += 1
        
        # Update context index if context provided
        if context:
            self.context_index[source][context].add(target)
            self.context_index[target][context].add(source)
        
        # Update temporal index with timestamp
        timestamp = time.time()
        self.temporal_index[source].append((timestamp, target, strength))
        self.temporal_index[target].append((timestamp, source, strength))
        
        # Store metadata if provided
        if metadata:
            if source not in self.metadata_index:
                self.metadata_index[source] = {}
            if target not in self.metadata_index[source]:
                self.metadata_index[source][target] = {}
            
            # Update with new metadata
            self.metadata_index[source][target].update(metadata)
        
        logger.debug(f"Added association: {source} -> {target} (strength: {strength})")
    
    def _process_batch_updates(self) -> None:
        """Process queued batch updates for better efficiency"""
        if not self.batch_updates:
            return
            
        logger.info(f"Processing batch of {len(self.batch_updates)} updates")
        start_time = time.time()
        
        # Group updates by source for efficiency
        updates_by_source = defaultdict(list)
        for source, target, strength, context, metadata in self.batch_updates:
            updates_by_source[source].append((target, strength, context, metadata))
        
        # Process grouped updates
        for source, targets in updates_by_source.items():
            # Pre-check if indices exist
            if source not in self.primary_index:
                self.primary_index[source] = {}
            
            # Process each target
            for target, strength, context, metadata in targets:
                # Update indices
                self.primary_index[source][target] = strength
                
                if target not in self.inverse_index:
                    self.inverse_index[target] = {}
                self.inverse_index[target][source] = strength
                
                # Update frequency
                self.frequency_index[source] += 1
                self.frequency_index[target] += 1
                
                # Add context if provided
                if context:
                    self.context_index[source][context].add(target)
                    self.context_index[target][context].add(source)
                
                # Add timestamp
                timestamp = time.time()
                self.temporal_index[source].append((timestamp, target, strength))
                self.temporal_index[target].append((timestamp, source, strength))
                
                # Add metadata if provided
                if metadata:
                    if source not in self.metadata_index:
                        self.metadata_index[source] = {}
                    if target not in self.metadata_index[source]:
                        self.metadata_index[source][target] = {}
                    self.metadata_index[source][target].update(metadata)
        
        # Clear batch queue
        self.batch_updates = []
        
        logger.info(f"Batch processing completed in {time.time() - start_time:.4f}s")
    
    def lookup_associations(self, word: str, 
                          limit: int = 10, 
                          min_strength: float = 0.0,
                          context: Optional[str] = None,
                          include_llm_suggestions: bool = True) -> List[Tuple[str, float]]:
        """
        Lookup associations for a word with enhanced retrieval features.
        
        Args:
            word: The word to find associations for
            limit: Maximum number of associations to return
            min_strength: Minimum association strength
            context: Optional context filter
            include_llm_suggestions: Whether to include LLM-weighted suggestions
            
        Returns:
            List of (associated_word, strength) tuples
        """
        start_time = time.time()
        results = []
        
        # Process any pending batch updates first
        if self.batch_updates:
            self._process_batch_updates()
        
        # Get direct associations from primary index
        if word in self.primary_index:
            for target, strength in self.primary_index[word].items():
                if strength >= min_strength:
                    # Apply context filtering if specified
                    if context and context in self.context_index[word]:
                        if target not in self.context_index[word][context]:
                            continue
                    
                    results.append((target, strength))
        
        # Sort by strength (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # If we need to include LLM suggestions and don't have enough results
        if include_llm_suggestions and len(results) < limit:
            suggestions = self._get_llm_suggestions(word, limit - len(results))
            
            # Merge direct associations with LLM suggestions
            for suggestion, sugg_strength in suggestions:
                # Apply LLM weight
                weighted_strength = sugg_strength * self.llm_weight
                
                # Only include if it meets minimum strength and isn't already in results
                if weighted_strength >= min_strength and suggestion not in [r[0] for r in results]:
                    results.append((suggestion, weighted_strength))
            
            # Re-sort after adding suggestions
            results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply limit
        results = results[:limit]
        
        logger.debug(f"Lookup for '{word}' returned {len(results)} associations (time: {time.time() - start_time:.4f}s)")
        
        return results
    
    def find_related_terms(self, words: List[str], limit: int = 10) -> List[Tuple[str, float]]:
        """
        Find terms related to multiple input words using optimized indexing.
        
        Args:
            words: List of input words to find relations for
            limit: Maximum number of results to return
            
        Returns:
            List of (related_term, combined_strength) tuples
        """
        # Process any pending batch updates first
        if self.batch_updates:
            self._process_batch_updates()
            
        # Dictionary to accumulate combined strengths
        combined_strengths = defaultdict(float)
        word_count = defaultdict(int)
        
        # Process each input word
        for word in words:
            if word in self.primary_index:
                # Get associations for this word
                for target, strength in self.primary_index[word].items():
                    combined_strengths[target] += strength
                    word_count[target] += 1
        
        # Calculate average strength for terms that connect to multiple input words
        results = []
        for term, total_strength in combined_strengths.items():
            # Skip if the term is one of our input words
            if term in words:
                continue
                
            # Higher weight for terms that connect to multiple input words
            count = word_count[term]
            connectivity_boost = min(count / len(words), 1.0)
            avg_strength = (total_strength / count) * (1.0 + connectivity_boost)
            
            results.append((term, avg_strength))
        
        # Sort by strength and apply limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_context_associations(self, context: str, limit: int = 20) -> List[Tuple[str, str, float]]:
        """
        Get word associations that appear in a specific context.
        
        Args:
            context: The context to filter by
            limit: Maximum number of associations to return
            
        Returns:
            List of (word1, word2, strength) tuples
        """
        # Process any pending batch updates first
        if self.batch_updates:
            self._process_batch_updates()
            
        results = []
        
        # Find all words that have associations in this context
        context_words = set()
        for word, contexts in self.context_index.items():
            if context in contexts:
                context_words.add(word)
        
        # Get associations between these words
        for word in context_words:
            for target in self.context_index[word][context]:
                if target in self.primary_index[word]:
                    strength = self.primary_index[word][target]
                    results.append((word, target, strength))
        
        # Sort by strength
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]
    
    def add_vector_embedding(self, word: str, vector: np.ndarray) -> None:
        """
        Add or update a vector embedding for a word to enable semantic search.
        
        Args:
            word: The word to add a vector for
            vector: The embedding vector
        """
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            normalized_vector = vector / norm
        else:
            normalized_vector = vector
            
        # Store the vector
        self.vector_index[word] = normalized_vector
        self.vector_enabled = True
        
        logger.debug(f"Added vector embedding for '{word}'")
    
    def semantic_search(self, query_vector: np.ndarray, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if not self.vector_enabled or not self.vector_index:
            logger.warning("Vector search attempted but vector index is empty")
            return []
            
        # Normalize query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
            
        # Calculate similarities
        similarities = []
        for word, vector in self.vector_index.items():
            similarity = np.dot(query_vector, vector)
            similarities.append((word, similarity))
            
        # Sort by similarity (descending) and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def optimize_indices(self) -> Dict[str, Any]:
        """
        Optimize indices for better performance and memory usage.
        
        Returns:
            Dictionary with optimization statistics
        """
        start_time = time.time()
        stats = {
            "pruned_associations": 0,
            "pruned_words": 0,
            "compressed_bytes": 0,
            "time_taken": 0
        }
        
        # Process any pending batch updates first
        if self.batch_updates:
            self._process_batch_updates()
        
        # 1. Prune weak associations
        for word in list(self.primary_index.keys()):
            initial_assoc_count = len(self.primary_index[word])
            self.primary_index[word] = {
                target: strength for target, strength in self.primary_index[word].items()
                if strength >= self.pruning_threshold
            }
            
            pruned_count = initial_assoc_count - len(self.primary_index[word])
            stats["pruned_associations"] += pruned_count
            
            # Remove the word entirely if it has no associations left
            if not self.primary_index[word]:
                del self.primary_index[word]
                stats["pruned_words"] += 1
        
        # 2. Rebuild inverse index to match pruned primary index
        self.inverse_index = defaultdict(dict)
        for source, targets in self.primary_index.items():
            for target, strength in targets.items():
                self.inverse_index[target][source] = strength
        
        # 3. Trim temporal indices to keep only the most recent N entries
        max_temporal_entries = 100
        for word in self.temporal_index:
            if len(self.temporal_index[word]) > max_temporal_entries:
                # Sort by timestamp (newest first) and keep only recent ones
                self.temporal_index[word].sort(reverse=True)
                self.temporal_index[word] = self.temporal_index[word][:max_temporal_entries]
                
        # 4. Clean up LLM suggestion cache if it's too large
        if len(self.llm_suggestion_cache) > self.cache_limit:
            # Keep only the most frequently used entries
            # For simplicity, just clear the cache completely
            self.llm_suggestion_cache = {}
            
        # Calculate statistics
        stats["time_taken"] = time.time() - start_time
        logger.info(f"Index optimization completed: {stats['pruned_associations']} associations pruned in {stats['time_taken']:.4f}s")
        
        return stats
    
    def set_llm_weight(self, weight: float) -> None:
        """
        Set the weight for LLM-based suggestions.
        
        Args:
            weight: New weight value (0.0-1.0)
        """
        if 0.0 <= weight <= 1.0:
            self.llm_weight = weight
            logger.info(f"LLM weight updated to: {weight}")
        else:
            logger.warning(f"Invalid LLM weight: {weight}. Must be between 0.0 and 1.0")
    
    def get_llm_integration_stats(self) -> Dict[str, Any]:
        """
        Get statistics about LLM integration in the indexing system.
        
        Returns:
            Dictionary of statistics
        """
        # Update with current llm_weight
        stats = self.llm_integration_stats.copy()
        stats["current_llm_weight"] = self.llm_weight
        stats["cache_size"] = len(self.llm_suggestion_cache)
        
        return stats
    
    def save_indices(self) -> None:
        """Save all indices to disk with optimized serialization."""
        # Process any pending batch updates first
        if self.batch_updates:
            self._process_batch_updates()
            
        index_dir = os.path.join(self.data_dir, "enhanced_index")
        
        # Save primary index
        with open(os.path.join(index_dir, "primary_index.json"), 'w') as f:
            json.dump(self.primary_index, f)
        
        # Save inverse index
        with open(os.path.join(index_dir, "inverse_index.json"), 'w') as f:
            json.dump(self.inverse_index, f)
        
        # Use pickle for more complex data structures
        with open(os.path.join(index_dir, "metadata_index.pkl"), 'wb') as f:
            pickle.dump(self.metadata_index, f)
        
        with open(os.path.join(index_dir, "context_index.pkl"), 'wb') as f:
            pickle.dump(self.context_index, f)
        
        with open(os.path.join(index_dir, "temporal_index.pkl"), 'wb') as f:
            pickle.dump(self.temporal_index, f)
        
        # Save frequency index (simple format)
        with open(os.path.join(index_dir, "frequency_index.json"), 'w') as f:
            json.dump(self.frequency_index, f)
        
        # Save LLM integration stats
        with open(os.path.join(index_dir, "llm_integration_stats.json"), 'w') as f:
            json.dump(self.llm_integration_stats, f)
            
        # Save vector index if enabled
        if self.vector_enabled and self.vector_index:
            vector_data = {word: vector.tolist() for word, vector in self.vector_index.items()}
            with open(os.path.join(index_dir, "vector_index.json"), 'w') as f:
                json.dump(vector_data, f)
            
        # Save optimization settings
        with open(os.path.join(index_dir, "optimization_settings.json"), 'w') as f:
            json.dump({
                "index_compression_enabled": self.index_compression_enabled,
                "pruning_threshold": self.pruning_threshold,
                "cache_limit": self.cache_limit,
                "batch_size_limit": self.batch_size_limit,
                "vector_dimension": self.vector_dimension,
                "vector_enabled": self.vector_enabled
            }, f)
        
        logger.info(f"All indices saved to {index_dir}")
    
    def _load_indices(self) -> None:
        """Load indices from disk if they exist."""
        index_dir = os.path.join(self.data_dir, "enhanced_index")
        
        # Load optimization settings first if they exist
        settings_path = os.path.join(index_dir, "optimization_settings.json")
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    self.index_compression_enabled = settings.get("index_compression_enabled", True)
                    self.pruning_threshold = settings.get("pruning_threshold", 0.1)
                    self.cache_limit = settings.get("cache_limit", 1000)
                    self.batch_size_limit = settings.get("batch_size_limit", 100)
                    self.vector_dimension = settings.get("vector_dimension", 100)
                    self.vector_enabled = settings.get("vector_enabled", False)
                    logger.info(f"Loaded optimization settings")
            except Exception as e:
                logger.error(f"Error loading optimization settings: {e}")
        
        # Check for primary index
        primary_path = os.path.join(index_dir, "primary_index.json")
        if os.path.exists(primary_path):
            try:
                with open(primary_path, 'r') as f:
                    # Use defaultdict for loading
                    self.primary_index = defaultdict(dict, json.load(f))
                
                # Load other indices if primary exists
                
                # Inverse index
                inverse_path = os.path.join(index_dir, "inverse_index.json")
                if os.path.exists(inverse_path):
                    with open(inverse_path, 'r') as f:
                        self.inverse_index = defaultdict(dict, json.load(f))
                
                # Frequency index
                freq_path = os.path.join(index_dir, "frequency_index.json")
                if os.path.exists(freq_path):
                    with open(freq_path, 'r') as f:
                        self.frequency_index = defaultdict(int, json.load(f))
                
                # Complex indices with pickle
                metadata_path = os.path.join(index_dir, "metadata_index.pkl")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        self.metadata_index = pickle.load(f)
                
                context_path = os.path.join(index_dir, "context_index.pkl")
                if os.path.exists(context_path):
                    with open(context_path, 'rb') as f:
                        self.context_index = pickle.load(f)
                
                temporal_path = os.path.join(index_dir, "temporal_index.pkl")
                if os.path.exists(temporal_path):
                    with open(temporal_path, 'rb') as f:
                        self.temporal_index = pickle.load(f)
                
                # LLM stats
                llm_stats_path = os.path.join(index_dir, "llm_integration_stats.json")
                if os.path.exists(llm_stats_path):
                    with open(llm_stats_path, 'r') as f:
                        self.llm_integration_stats = json.load(f)
                        
                # Load vector index if enabled
                vector_path = os.path.join(index_dir, "vector_index.json")
                if os.path.exists(vector_path) and self.vector_enabled:
                    with open(vector_path, 'r') as f:
                        vector_data = json.load(f)
                        self.vector_index = {word: np.array(vector) for word, vector in vector_data.items()}
                
                logger.info(f"Loaded indices with {len(self.primary_index)} words and {sum(len(targets) for targets in self.primary_index.values())} associations")
            except Exception as e:
                logger.error(f"Error loading indices: {e}")
                self._initialize_empty_indices()
        else:
            logger.info("No existing indices found, initializing empty indices")
            self._initialize_empty_indices()
    
    def _initialize_empty_indices(self) -> None:
        """Initialize empty indices when loading fails or no indices exist."""
        self.primary_index = defaultdict(dict)
        self.inverse_index = defaultdict(dict)
        self.metadata_index = defaultdict(dict)
        self.frequency_index = defaultdict(int)
        self.context_index = defaultdict(lambda: defaultdict(set))
        self.temporal_index = defaultdict(list)
        self.vector_index = {}
        self.llm_suggestion_cache = {}
        
        logger.info("Initialized empty indices")
    
    def _get_llm_suggestions(self, word: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Get word association suggestions from LLM integration.
        
        This is a simulated implementation that would be replaced with
        actual LLM API calls in production.
        
        Args:
            word: Word to get suggestions for
            limit: Maximum number of suggestions to return
            
        Returns:
            List of (suggested_word, strength) tuples
        """
        # Check cache first
        if word in self.llm_suggestion_cache:
            self.llm_integration_stats["cache_hits"] += 1
            return self.llm_suggestion_cache[word][:limit]
        
        self.llm_integration_stats["cache_misses"] += 1
        
        # In a real implementation, this would call an LLM API
        # Here we simulate suggestions based on existing indices
        
        # Simulated association generator based on frequency
        suggestions = []
        
        # 1. Look for words that share connections with our target word
        shared_connections = defaultdict(int)
        
        # If the word is in our index, use its connections
        if word in self.primary_index:
            # Get words directly connected to our target
            direct_connections = set(self.primary_index[word].keys())
            
            # For each direct connection, find their connections
            for connected_word in direct_connections:
                if connected_word in self.primary_index:
                    for second_degree, strength in self.primary_index[connected_word].items():
                        # Don't include the original word or direct connections
                        if second_degree != word and second_degree not in direct_connections:
                            shared_connections[second_degree] += 1
        
        # Convert shared connections to suggestions with strength
        for second_degree, count in shared_connections.items():
            # Higher count means more shared connections
            strength = min(0.9, count * 0.1)  # Scale up to max 0.9
            suggestions.append((second_degree, strength))
        
        # 2. If we have vector embeddings, use them for additional suggestions
        if self.vector_enabled and word in self.vector_index:
            vector_suggestions = self.semantic_search(self.vector_index[word], limit=limit)
            
            # Add any new suggestions not already in the list
            existing_words = {s[0] for s in suggestions}
            for sugg_word, similarity in vector_suggestions:
                if sugg_word != word and sugg_word not in existing_words:
                    suggestions.append((sugg_word, similarity * 0.8))  # Scale to max 0.8
        
        # 3. Add some frequency-based suggestions
        if len(suggestions) < limit * 2:
            # Get top words by frequency
            top_words = sorted(self.frequency_index.items(), key=lambda x: x[1], reverse=True)
            
            # Add some high-frequency words that aren't already in suggestions
            existing_words = {s[0] for s in suggestions}
            for freq_word, freq in top_words[:20]:
                if freq_word != word and freq_word not in existing_words:
                    # Normalize frequency to strength between 0.3-0.7
                    max_freq = max(self.frequency_index.values()) if self.frequency_index else 1
                    norm_freq = freq / max_freq if max_freq > 0 else 0
                    strength = 0.3 + (norm_freq * 0.4)
                    suggestions.append((freq_word, strength))
                    
                    if len(suggestions) >= limit * 2:
                        break
        
        # Sort by strength and limit
        suggestions.sort(key=lambda x: x[1], reverse=True)
        suggestions = suggestions[:limit]
        
        # Update stats
        if suggestions:
            self.llm_integration_stats["suggestions_used"] += len(suggestions)
            avg_strength = sum(s[1] for s in suggestions) / len(suggestions)
            self.llm_integration_stats["suggestion_strength_avg"] = (
                (self.llm_integration_stats["suggestion_strength_avg"] * 
                 (self.llm_integration_stats["suggestions_used"] - len(suggestions)) +
                 avg_strength * len(suggestions)) / 
                self.llm_integration_stats["suggestions_used"]
            )
        
        # Cache suggestions for future use
        self.llm_suggestion_cache[word] = suggestions
        
        return suggestions

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the enhanced indexing system.
        
        Returns:
            Dictionary of performance metrics
        """
        # Process any pending batch updates first
        if self.batch_updates:
            self._process_batch_updates()
            
        metrics = {
            "total_words": len(self.primary_index),
            "total_associations": sum(len(assocs) for assocs in self.primary_index.values()),
            "average_associations_per_word": 0,
            "words_with_metadata": len(self.metadata_index),
            "words_with_context": len(self.context_index),
            "memory_usage_estimate_mb": 0,
            "vector_index_size": len(self.vector_index),
            "batch_queue_size": len(self.batch_updates),
            "optimization_enabled": self.index_compression_enabled,
            "vector_enabled": self.vector_enabled
        }
        
        # Calculate average associations per word
        if metrics["total_words"] > 0:
            metrics["average_associations_per_word"] = metrics["total_associations"] / metrics["total_words"]
        
        # Estimate memory usage (very rough approximation)
        total_bytes = 0
        # Primary index
        total_bytes += sum(len(word) * 2 + len(assocs) * 20 for word, assocs in self.primary_index.items())
        # Inverse index - similar structure to primary
        total_bytes += sum(len(word) * 2 + len(assocs) * 20 for word, assocs in self.inverse_index.items())
        # Frequency index - word + integer
        total_bytes += sum(len(word) * 2 + 8 for word in self.frequency_index)
        # Context index - more complex
        total_bytes += sum(len(word) * 2 + sum(len(context) * 2 + len(targets) * 20 
                                             for context, targets in contexts.items())
                          for word, contexts in self.context_index.items())
        # Vector index
        if self.vector_enabled:
            total_bytes += sum(len(word) * 2 + (self.vector_dimension * 4) for word in self.vector_index)
        
        metrics["memory_usage_estimate_mb"] = total_bytes / (1024 * 1024)
        
        return metrics

def test_enhanced_indexing():
    """Basic test function for the enhanced indexing system."""
    import tempfile
    import shutil
    
    # Create temp directory for testing
    test_dir = tempfile.mkdtemp()
    try:
        logger.info(f"Testing enhanced indexing in {test_dir}")
        
        # Initialize index
        index = EnhancedIndex(test_dir, llm_weight=0.7)
        
        # Add some associations
        index.add_association("neural", "network", 0.9, context="AI")
        index.add_association("neural", "brain", 0.8, context="biology")
        index.add_association("network", "connection", 0.7, context="AI")
        
        # Test lookups
        results = index.lookup_associations("neural")
        logger.info(f"Lookup results for 'neural': {results}")
        
        # Test related terms
        related = index.find_related_terms(["neural", "network"])
        logger.info(f"Related terms: {related}")
        
        # Test context associations
        context_assocs = index.get_context_associations("AI")
        logger.info(f"Context associations for 'AI': {context_assocs}")
        
        # Test batch processing
        for i in range(50):
            index.add_association(f"test{i}", f"batch{i}", 0.5 + (i/100))
        
        # Test optimization
        stats = index.optimize_indices()
        logger.info(f"Optimization stats: {stats}")
        
        # Add some vector data
        if index.vector_enabled:
            index.add_vector_embedding("neural", np.random.rand(index.vector_dimension))
            index.add_vector_embedding("network", np.random.rand(index.vector_dimension))
            
            # Test semantic search
            if "neural" in index.vector_index:
                sem_results = index.semantic_search(index.vector_index["neural"])
                logger.info(f"Semantic search results: {sem_results}")
        
        # Get performance metrics
        metrics = index.get_performance_metrics()
        logger.info(f"Performance metrics: {metrics}")
        
        # Save and reload
        index.save_indices()
        new_index = EnhancedIndex(test_dir, llm_weight=0.7)
        
        # Verify data was loaded
        results2 = new_index.lookup_associations("neural")
        logger.info(f"Lookup results after reload: {results2}")
        
        logger.info("Enhanced indexing test completed successfully")
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)
        logger.info(f"Cleaned up test directory {test_dir}")

if __name__ == "__main__":
    test_enhanced_indexing() 