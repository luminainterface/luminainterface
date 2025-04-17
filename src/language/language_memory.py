#!/usr/bin/env python3
"""
Language Memory Module

This module provides language-specific memory capabilities for neural language systems,
storing and retrieving language patterns, vocabulary, and linguistic structures
for improved language processing and generation.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
import re
import random
import numpy as np
import time
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict
from .enhanced_indexing import EnhancedIndex
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/language_memory.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LanguageMemory")

class LanguageMemory:
    """
    Language-specific memory system that stores and retrieves language patterns,
    vocabulary usage, and linguistic structures for improved language processing.
    
    Provides:
    - Memory of word associations and semantic relationships
    - Storage of grammatical patterns and their usage frequencies
    - Tracking of vocabulary usage and mastery levels
    - Memory integration with language training systems
    - LLM weight integration for enhanced capabilities
    """
    
    def __init__(self, data_dir: str = None, llm_weight: float = 0.5, nn_weight: float = 0.5):
        if data_dir is None:
            data_dir = "data/memory/language_memory"

        """
        Initialize a Language Memory system.
        
        Args:
            data_dir: Directory to store memory files
            llm_weight: Weight for LLM integration (0.0-1.0)
            nn_weight: Weight for neural vs symbolic processing (0.0-1.0)
        """
        logger.info("Initializing Language Memory system")
        
        self.data_dir = data_dir
        self.llm_weight = llm_weight
        self.nn_weight = nn_weight
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize enhanced indexing system
        self.index = EnhancedIndex(data_dir, llm_weight)
        
        # Initialize memory components
        self.word_associations = {}
        self.grammar_patterns = {}
        self.word_usage = {}
        self.sentences = {}
        self.sentence_metadata = {}
        self.language_fragments = []
        
        # LLM integration parameters
        self.llm_confidence = 0.5
        self.llm_suggestions = 0
        self.llm_memory_accesses = 0
        
        # Version tracking
        self.memory_version = "1.2.0"  # Updated for enhanced indexing
        
        # Memory stats
        self.access_count = 0
        self.store_count = 0
        self.last_access = None
        self.last_store = None
        
        # Initialize indexes for faster retrieval
        self.word_index = {}
        self.pattern_index = {}
        self.linguistic_features = {}
        
        # Performance optimization settings
        self.cache_enabled = True
        self.cache_size_limit = 1000
        self.association_cache = {}
        self.batch_processing = True
        self.auto_optimization_interval = 1000  # Auto-optimize after this many operations
        self.operations_since_optimization = 0
        
        # Load existing memories
        self._load_memories()
        
        logger.info(f"Language Memory initialized with LLM weight: {llm_weight}, enhanced indexing enabled")
    
    def _load_memories(self):
        """Load existing memories from disk."""
        # Load word associations
        self.word_associations = {}
        assoc_file = os.path.join(self.data_dir, "word_associations.json")
        if os.path.exists(assoc_file):
            try:
                with open(assoc_file, 'r') as f:
                    self.word_associations = json.load(f)
                logger.info(f"Loaded {len(self.word_associations)} word associations")
            except Exception as e:
                logger.error(f"Error loading word associations: {e}")
        
        # Load grammar patterns
        self.grammar_patterns = {}
        patterns_file = os.path.join(self.data_dir, "grammar_patterns.json")
        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r') as f:
                    self.grammar_patterns = json.load(f)
                logger.info(f"Loaded {len(self.grammar_patterns)} grammar patterns")
            except Exception as e:
                logger.error(f"Error loading grammar patterns: {e}")
        
        # Load word usage statistics
        self.word_usage = {}
        usage_file = os.path.join(self.data_dir, "word_usage.json")
        if os.path.exists(usage_file):
            try:
                with open(usage_file, 'r') as f:
                    self.word_usage = json.load(f)
                logger.info(f"Loaded usage data for {len(self.word_usage)} words")
            except Exception as e:
                logger.error(f"Error loading word usage: {e}")
        
        # Load sentences with proper handling of formats
        self.sentences = {}
        self.sentence_metadata = {}
        sentences_file = os.path.join(self.data_dir, "sentences.json")
        if os.path.exists(sentences_file):
            try:
                with open(sentences_file, 'r') as f:
                    sentences_data = json.load(f)
                    
                # Handle different formats of the sentences file
                if isinstance(sentences_data, dict):
                    if "sentences" in sentences_data and "metadata" in sentences_data:
                        # Old format with separate sentences and metadata
                        self.sentences = sentences_data.get("sentences", {})
                        self.sentence_metadata = sentences_data.get("metadata", {})
                    else:
                        # New format with combined structure
                        for sentence_id, sentence_obj in sentences_data.items():
                            if isinstance(sentence_obj, dict) and "text" in sentence_obj:
                                self.sentences[sentence_id] = sentence_obj["text"]
                                self.sentence_metadata[sentence_id] = sentence_obj.get("metadata", {})
                            else:
                                # Simple format where value is just the text
                                self.sentences[sentence_id] = sentence_obj
                                
                logger.info(f"Loaded {len(self.sentences)} sentences")
            except Exception as e:
                logger.error(f"Error loading sentences: {e}")
        
        # Load LLM integration
        llm_file = os.path.join(self.data_dir, "llm_integration.json")
        if os.path.exists(llm_file):
            try:
                with open(llm_file, 'r') as f:
                    llm_data = json.load(f)
                    self.llm_weight = llm_data.get("llm_weight", self.llm_weight)
                    self.nn_weight = llm_data.get("nn_weight", self.nn_weight)
                    self.llm_confidence = llm_data.get("llm_confidence", 0.5)
                    self.llm_suggestions = llm_data.get("llm_suggestions", 0)
                    self.llm_memory_accesses = llm_data.get("llm_memory_accesses", 0)
                logger.info(f"Loaded LLM integration data with weight {self.llm_weight}")
            except Exception as e:
                logger.error(f"Error loading LLM integration data: {e}")
        
        return True
    
    def _build_indexes(self):
        """Build indexes for faster memory retrieval"""
        # Word index
        self.word_index = {}
        for word, associations in self.word_associations.items():
            self.word_index[word] = {
                "associations": associations,
                "usage_count": self.word_usage.get(word, {}).get("count", 0)
            }
        
        # Pattern index
        self.pattern_index = {}
        for pattern_id, pattern in self.grammar_patterns.items():
            pattern_key = pattern.get("pattern", "")
            if pattern_key not in self.pattern_index:
                self.pattern_index[pattern_key] = []
            self.pattern_index[pattern_key].append(pattern_id)
        
        # Extract linguistic features from sentences
        self.linguistic_features = {}
        for sentence in self.sentences.values():
            features = sentence.get("features", {})
            for feature, value in features.items():
                if feature not in self.linguistic_features:
                    self.linguistic_features[feature] = set()
                self.linguistic_features[feature].add(value)
        
        logger.info("Memory indexes built successfully")
    
    def save_memories(self):
        """Save all memories to disk."""
        try:
            # Save word associations
            assoc_file = os.path.join(self.data_dir, "word_associations.json")
            with open(assoc_file, 'w') as f:
                json.dump(self.word_associations, f, indent=2)
            
            # Save grammar patterns
            patterns_file = os.path.join(self.data_dir, "grammar_patterns.json")
            with open(patterns_file, 'w') as f:
                json.dump(self.grammar_patterns, f, indent=2)
            
            # Save word usage statistics
            usage_file = os.path.join(self.data_dir, "word_usage.json")
            with open(usage_file, 'w') as f:
                json.dump(self.word_usage, f, indent=2)
            
            # Save sentences with proper handling for metadata
            sentences_file = os.path.join(self.data_dir, "sentences.json")
            # Initialize sentence metadata if it doesn't exist
            if not hasattr(self, 'sentence_metadata'):
                self.sentence_metadata = {}
                
            # Create serializable sentences dictionary
            serializable_sentences = {}
            for sentence_id, text in self.sentences.items():
                metadata = self.sentence_metadata.get(sentence_id, {})
                serializable_sentences[sentence_id] = {
                    "text": text,
                    "metadata": metadata
                }
                
            with open(sentences_file, 'w') as f:
                json.dump(serializable_sentences, f, indent=2)
            
            # Save LLM integration data
            llm_file = os.path.join(self.data_dir, "llm_integration.json")
            with open(llm_file, 'w') as f:
                json.dump({
                    "llm_weight": self.llm_weight,
                    "nn_weight": self.nn_weight,
                    "llm_confidence": self.llm_confidence,
                    "llm_suggestions": self.llm_suggestions,
                    "llm_memory_accesses": self.llm_memory_accesses
                }, f, indent=2)
            
            logger.info("Language memories saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving language memories: {e}")
            return False
    
    def remember_word_association(self, word, associated_word, strength=1.0, context=None):
        """
        Store association between two words
        
        Args:
            word: Primary word
            associated_word: Associated word
            strength: Association strength (0.0-1.0)
            context: Optional context where association was observed
            
        Returns:
            bool: Success status
        """
        self.last_store = datetime.now()
        self.store_count += 1
        
        # Initialize word if not exist
        if word not in self.word_associations:
            self.word_associations[word] = {}
        
        # Update word association
        self.word_associations[word][associated_word] = strength
        
        # Also store in enhanced index
        metadata = None
        if context:
            metadata = {"context": context, "timestamp": datetime.now().timestamp()}
        
        self.index.add_association(word, associated_word, strength, context, metadata)
        
        # Update word usage statistics
        for w in [word, associated_word]:
            if w not in self.word_usage:
                self.word_usage[w] = {"count": 0, "last_seen": None, "contexts": {}}
            
            self.word_usage[w]["count"] += 1
            self.word_usage[w]["last_seen"] = datetime.now().isoformat()
            
            if context:
                if context not in self.word_usage[w]["contexts"]:
                    self.word_usage[w]["contexts"][context] = 0
                self.word_usage[w]["contexts"][context] += 1
        
        # Increment operations counter and check if optimization is needed
        self.operations_since_optimization += 1
        if self.auto_optimization_interval > 0 and self.operations_since_optimization >= self.auto_optimization_interval:
            self.optimize_memory()
        
        return True
    
    def remember_word_association_with_llm(self, word, associated_word, strength=1.0, context=None, llm_suggestion=False):
        """
        Store association between two words with LLM integration
        
        Args:
            word: Primary word
            associated_word: Associated word
            strength: Association strength (0.0-1.0)
            context: Optional context where association was observed
            llm_suggestion: Whether this was suggested by LLM
            
        Returns:
            bool: Success status
        """
        # Record LLM suggestion if applicable
        if llm_suggestion:
            if not isinstance(self.llm_suggestions, list):
                self.llm_suggestions = []
                
            self.llm_suggestions.append({
                "word": word,
                "associated_word": associated_word,
                "suggested_strength": strength,
                "timestamp": datetime.now().isoformat(),
                "context": context
            })
        
        # Apply LLM weight to the strength if it's an LLM suggestion
        if llm_suggestion:
            # Weight the suggestion based on LLM confidence
            weighted_strength = strength * self.llm_weight
            self.llm_confidence = max(0.5, min(0.9, self.llm_confidence + 0.01))  # Slight confidence boost
        else:
            weighted_strength = strength
        
        # Store the weighted association
        return self.remember_word_association(word, associated_word, weighted_strength, context)
    
    def remember_pattern(self, pattern, example, source=None):
        """
        Store a grammatical pattern in memory
        
        Args:
            pattern: Grammatical pattern string
            example: Example sentence using the pattern
            source: Source of the pattern observation
            
        Returns:
            str: Pattern ID
        """
        self.last_store = datetime.now()
        self.store_count += 1
        
        # Generate pattern ID
        pattern_id = f"pattern_{len(self.grammar_patterns) + 1}"
        
        # Store pattern
        self.grammar_patterns[pattern_id] = {
            "pattern": pattern,
            "example": example,
            "source": source,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "count": 1
        }
        
        # Update pattern index
        if pattern not in self.pattern_index:
            self.pattern_index[pattern] = []
        self.pattern_index[pattern].append(pattern_id)
        
        logger.debug(f"Stored grammar pattern: {pattern}")
        
        # Periodically save memories
        if self.store_count % 50 == 0:
            self.save_memories()
            
        return pattern_id
    
    def store_sentence(self, sentence, features=None, metadata=None):
        """
        Store a sentence in memory with linguistic features
        
        Args:
            sentence: The sentence text
            features: Dictionary of linguistic features of the sentence
            metadata: Optional metadata about the sentence
            
        Returns:
            str: Sentence ID
        """
        self.last_store = datetime.now()
        self.store_count += 1
        
        # Generate a sentence ID
        sentence_id = str(uuid.uuid4())
        
        # Use current time as timestamp
        timestamp = datetime.now().isoformat()
        
        # Create sentence entry
        sentence_data = {
            "text": sentence,
            "timestamp": timestamp,
            "features": features or {}
        }
        
        # Store in sentences dictionary
        self.sentences[sentence_id] = sentence_data
        
        # Store metadata separately if provided
        if metadata:
            self.sentence_metadata[sentence_id] = metadata
        
        # Extract and store word associations from the sentence
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        # Store associations between adjacent words
        for i in range(len(words) - 1):
            if len(words[i]) > 1 and len(words[i+1]) > 1:  # Skip single-letter words
                # Determine association strength - could be more sophisticated
                strength = 0.5  # Default association strength
                self.remember_word_association(words[i], words[i+1], strength, context="sentence")
        
        # Extract features for linguistic index
        if features:
            for feature, value in features.items():
                if feature not in self.linguistic_features:
                    self.linguistic_features[feature] = set()
                self.linguistic_features[feature].add(value)
        
        # Increment operations counter and check if optimization is needed
        self.operations_since_optimization += 1
        if self.auto_optimization_interval > 0 and self.operations_since_optimization >= self.auto_optimization_interval:
            self.optimize_memory()
        
        return sentence_id
    
    def _update_word_usage(self, word):
        """Update usage statistics for a word"""
        if word not in self.word_usage:
            self.word_usage[word] = {
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }
        else:
            self.word_usage[word]["count"] += 1
            self.word_usage[word]["last_seen"] = datetime.now().isoformat()
        
        # Update word index
        if word not in self.word_index:
            self.word_index[word] = {"associations": {}, "usage_count": 0}
        self.word_index[word]["usage_count"] = self.word_usage[word]["count"]
    
    def recall_associations(self, word, limit=10, min_strength=0.0, context=None):
        """
        Recall words associated with a given word
        
        Args:
            word: Word to find associations for
            limit: Maximum number of associations to return
            min_strength: Minimum association strength
            context: Optional context to filter by
            
        Returns:
            list: Associated words with strengths
        """
        self.last_access = datetime.now()
        self.access_count += 1
        self.llm_memory_accesses += 1
        
        # Check cache first if enabled
        cache_key = f"{word}:{limit}:{min_strength}:{context}"
        if self.cache_enabled and cache_key in self.association_cache:
            logger.debug(f"Cache hit for '{word}'")
            return self.association_cache[cache_key]
        
        # Use the enhanced index for faster lookups
        raw_associations = self.index.lookup_associations(
            word, 
            limit=limit, 
            min_strength=min_strength, 
            context=context,
            include_llm_suggestions=(self.llm_weight > 0.0)
        )
        
        # Format results
        results = []
        for associated_word, strength in raw_associations:
            result = {
                "word": associated_word,
                "strength": strength,
                "contexts": {}
            }
            
            # Add context information if available
            if context:
                result["current_context"] = context
            
            # Try to get all contexts for this association
            if word in self.word_usage and "contexts" in self.word_usage[word]:
                for ctx, count in self.word_usage[word]["contexts"].items():
                    result["contexts"][ctx] = count
            
            results.append(result)
        
        # Update cache if enabled
        if self.cache_enabled:
            if len(self.association_cache) >= self.cache_size_limit:
                # Simple cache clearing - just clear the whole cache when it gets too large
                self.association_cache = {}
            self.association_cache[cache_key] = results
        
        return results
    
    def adjust_llm_weight(self, new_weight):
        """
        Manually adjust the LLM weight
        
        Args:
            new_weight: New weight (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        if 0 <= new_weight <= 1:
            self.llm_weight = new_weight
            # Update the weight in the enhanced index as well
            self.index.set_llm_weight(new_weight)
            logger.info(f"LLM weight manually adjusted to {self.llm_weight}")
            return True
        else:
            logger.warning(f"Invalid LLM weight: {new_weight} (must be between 0.0 and 1.0)")
            return False
    
    def get_llm_integration_stats(self):
        """
        Get stats about the LLM integration
        
        Returns:
            dict: Stats about LLM integration
        """
        # Get enhanced index stats
        index_stats = self.index.get_llm_integration_stats()
        
        # Combine with language memory specific stats
        stats = {
            "llm_weight": self.llm_weight,
            "llm_confidence": self.llm_confidence,
            "llm_memory_accesses": self.llm_memory_accesses,
            "llm_suggestion_count": len(self.llm_suggestions),
            "index_stats": index_stats
        }
        
        return stats
    
    def recall_patterns(self, pattern=None, limit=5):
        """
        Recall grammatical patterns
        
        Args:
            pattern: Optional pattern to search for (exact match)
            limit: Maximum number of patterns to return
            
        Returns:
            list: Matching patterns
        """
        self.last_access = datetime.now()
        self.access_count += 1
        
        if pattern and pattern in self.pattern_index:
            # Return exact pattern matches
            pattern_ids = self.pattern_index[pattern][:limit]
            return [self.grammar_patterns[pid] for pid in pattern_ids]
        
        if pattern:
            # Search for pattern substrings
            matching_patterns = []
            for p_key, p_ids in self.pattern_index.items():
                if pattern in p_key:
                    for pid in p_ids:
                        matching_patterns.append(self.grammar_patterns[pid])
                        if len(matching_patterns) >= limit:
                            break
                if len(matching_patterns) >= limit:
                    break
            return matching_patterns
        
        # Return most frequently used patterns
        sorted_patterns = sorted(
            self.grammar_patterns.values(),
            key=lambda p: p.get("count", 0),
            reverse=True
        )
        
        return sorted_patterns[:limit]
    
    def recall_sentences_by_feature(self, feature, value, limit=5):
        """
        Recall sentences that have a specific linguistic feature
        
        Args:
            feature: Feature name
            value: Feature value
            limit: Maximum number of sentences to return
            
        Returns:
            list: Matching sentences
        """
        self.last_access = datetime.now()
        self.access_count += 1
        
        matching_sentences = []
        
        for sentence in self.sentences.values():
            features = sentence.get("features", {})
            if feature in features and features[feature] == value:
                matching_sentences.append(sentence)
                if len(matching_sentences) >= limit:
                    break
        
        return matching_sentences
    
    def recall_sentences_with_word(self, word, limit=5):
        """
        Recall sentences containing a specific word
        
        Args:
            word: Word to search for
            limit: Maximum number of sentences to return
            
        Returns:
            list: Matching sentences
        """
        self.last_access = datetime.now()
        self.access_count += 1
        
        word = word.lower()
        matching_sentences = []
        
        for sentence in self.sentences.values():
            words = sentence.get("words", [])
            if word in words:
                matching_sentences.append(sentence)
                if len(matching_sentences) >= limit:
                    break
        
        return matching_sentences
    
    def get_word_usage_stats(self, word):
        """
        Get usage statistics for a specific word
        
        Args:
            word: Word to get stats for
            
        Returns:
            dict: Word usage statistics
        """
        self.last_access = datetime.now()
        self.access_count += 1
        
        if word not in self.word_usage:
            return {"word": word, "found": False}
        
        stats = self.word_usage[word].copy()
        stats["word"] = word
        stats["found"] = True
        
        # Add associations
        if word in self.word_associations:
            stats["association_count"] = len(self.word_associations[word])
            # Get top 3 strongest associations
            associations = self.recall_associations(word, limit=3)
            stats["top_associations"] = [a["word"] for a in associations]
        
        return stats
    
    def get_memory_statistics(self):
        """
        Get statistics about the language memory system
        
        Returns:
            dict: Memory statistics
        """
        # Get basic statistics
        stats = {
            "total_words": len(self.word_usage),
            "total_associations": sum(len(assocs) for assocs in self.word_associations.values()),
            "total_patterns": len(self.grammar_patterns),
            "total_sentences": len(self.sentences),
            "access_count": self.access_count,
            "store_count": self.store_count,
            "last_access": self.last_access.isoformat() if self.last_access else None,
            "last_store": self.last_store.isoformat() if self.last_store else None,
            "memory_path": str(self.data_dir),
            "memory_version": self.memory_version,
            "feature_types": list(self.linguistic_features.keys()),
            "llm_weight": self.llm_weight,
            "nn_weight": self.nn_weight,
            "llm_suggestions": self.llm_suggestions,
            "llm_memory_accesses": self.llm_memory_accesses,
        }
        
        # Add enhanced index statistics
        index_metrics = self.index.get_performance_metrics()
        stats.update({
            "index_metrics": index_metrics,
            "operations_since_optimization": self.operations_since_optimization,
            "auto_optimization_interval": self.auto_optimization_interval,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.association_cache)
        })
        
        return stats
    
    def clear_memories(self, memory_type=None):
        """
        Clear memories of a specific type or all
        
        Args:
            memory_type: Type of memory to clear ('words', 'patterns', 'sentences', or None for all)
            
        Returns:
            bool: Success status
        """
        if memory_type == 'words' or memory_type is None:
            self.word_associations = {}
            self.word_usage = {}
            self.word_index = {}
            logger.info("Cleared word memories")
        
        if memory_type == 'patterns' or memory_type is None:
            self.grammar_patterns = {}
            self.pattern_index = {}
            logger.info("Cleared pattern memories")
        
        if memory_type == 'sentences' or memory_type is None:
            self.sentences = {}
            self.sentence_metadata = {}
            self.linguistic_features = {}
            logger.info("Cleared sentence memories")
        
        # Save empty files
        self.save_memories()
        
        return True
    
    def analyze_language_fragment(self, text):
        """
        Analyze and store a language fragment
        
        Args:
            text: Text fragment to analyze
            
        Returns:
            dict: Analysis results
        """
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract words
        all_words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(all_words)
        
        # Look for patterns we know
        detected_patterns = []
        for pattern_id, pattern in self.grammar_patterns.items():
            pattern_example = pattern.get("example", "").lower()
            pattern_text = pattern.get("pattern", "").lower()
            
            # Simple detection
            if any(pattern_text in s.lower() for s in sentences) or any(pattern_example in s.lower() for s in sentences):
                detected_patterns.append(pattern_id)
        
        # Find word associations in the text
        detected_associations = []
        for i, word in enumerate(all_words[:-1]):
            next_word = all_words[i+1]
            if word in self.word_associations and next_word in self.word_associations[word]:
                detected_associations.append((word, next_word))
        
        # Store each sentence
        sentence_ids = []
        for sentence in sentences:
            if sentence:
                features = {
                    "word_count": len(re.findall(r'\b\w+\b', sentence)),
                    "has_question": "?" in sentence,
                    "has_exclamation": "!" in sentence
                }
                
                sentence_id = self.store_sentence(sentence, features)
                sentence_ids.append(sentence_id)
        
        # Store word associations found in the text
        for i, word in enumerate(all_words[:-1]):
            for j in range(1, min(4, len(all_words) - i)):
                associated_word = all_words[i+j]
                context = " ".join(all_words[max(0, i-2):min(len(all_words), i+j+2)])
                self.remember_word_association(word, associated_word, strength=0.5, context=context)
        
        analysis = {
            "sentence_count": len(sentences),
            "word_count": len(all_words),
            "unique_word_count": len(unique_words),
            "stored_sentence_ids": sentence_ids,
            "detected_patterns": detected_patterns,
            "detected_associations": detected_associations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store the fragment and analysis
        self.language_fragments.append({
            "text": text,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
        
        return analysis
    
    def generate_from_memory(self, seed_word=None, pattern_id=None, length=10):
        """
        Generate text using language memories
        
        Args:
            seed_word: Optional word to start with
            pattern_id: Optional pattern ID to use
            length: Target number of words
            
        Returns:
            str: Generated text
        """
        # Start with a seed word or pick a random frequent word
        if not seed_word:
            # Get a random word from the top 100 most used
            frequent_words = sorted(
                self.word_usage.items(),
                key=lambda x: x[1].get("count", 0),
                reverse=True
            )[:100]
            
            if frequent_words:
                seed_word = random.choice(frequent_words)[0]
            else:
                return "No vocabulary available for generation."
        
        # If a pattern is specified, try to use it
        if pattern_id and pattern_id in self.grammar_patterns:
            pattern = self.grammar_patterns[pattern_id]
            example = pattern.get("example", "")
            
            # Very simple substitution
            words = example.split()
            for i, word in enumerate(words):
                if random.random() < 0.5:  # 50% chance to substitute
                    associations = self.recall_associations(word)
                    if associations:
                        words[i] = random.choice(associations)["word"]
            
            return " ".join(words)
        
        # Generate by word association
        current_word = seed_word
        generated_words = [current_word]
        
        for _ in range(length - 1):
            associations = self.recall_associations(current_word)
            
            if not associations:
                # No associations, pick random word
                if self.word_usage:
                    current_word = random.choice(list(self.word_usage.keys()))
                else:
                    break
            else:
                # Weighted choice based on association strength
                weights = [a["strength"] for a in associations]
                words = [a["word"] for a in associations]
                current_word = random.choices(words, weights=weights, k=1)[0]
            
            generated_words.append(current_word)
        
        # Capitalize first word and add period
        text = " ".join(generated_words)
        text = text[0].upper() + text[1:] + "."
        
        return text
    
    def set_llm_weight(self, weight):
        """
        Set the weight for LLM integration
        
        Args:
            weight: The new weight value (0.0-1.0)
            
        Returns:
            bool: Success status
        """
        if 0.0 <= weight <= 1.0:
            self.llm_weight = weight
            # Also update the enhanced index
            self.index.set_llm_weight(weight)
            logger.info(f"LLM weight updated to {weight}")
            return True
        else:
            logger.error(f"Invalid LLM weight: {weight}. Must be between 0.0 and 1.0")
            return False
    
    def set_nn_weight(self, weight: float) -> None:
        """
        Set the neural network weight for the language memory system.
        
        Args:
            weight: New neural network weight (0.0-1.0)
        """
        self.nn_weight = max(0.0, min(1.0, weight))
        if hasattr(self, 'index'):
            self.index.nn_weight = weight
        logger.info(f"Language Memory neural network weight set to {weight}")

    def optimize_memory(self):
        """
        Optimize memory storage and retrieval efficiency
        
        Returns:
            dict: Optimization statistics
        """
        logger.info("Starting language memory optimization")
        start_time = time.time()
        
        # Clear the association cache
        self.association_cache = {}
        
        # Optimize the enhanced index
        index_stats = self.index.optimize_indices()
        
        # Prune empty entries in word associations
        words_to_remove = []
        for word, associations in self.word_associations.items():
            if not associations:
                words_to_remove.append(word)
        
        for word in words_to_remove:
            del self.word_associations[word]
        
        # Save memories after optimization
        self.save_memories()
        
        # Reset operations counter
        self.operations_since_optimization = 0
        
        # Calculate statistics
        optimization_stats = {
            "time_taken": time.time() - start_time,
            "empty_words_removed": len(words_to_remove),
            "index_stats": index_stats,
            "cache_cleared": True
        }
        
        logger.info(f"Memory optimization completed in {optimization_stats['time_taken']:.4f}s")
        
        return optimization_stats

# For testing
if __name__ == "__main__":
    memory = LanguageMemory("data/memory/language_memory", llm_weight=0.5, nn_weight=0.5)
    
    # Store some associations
    memory.remember_word_association("neural", "network", 0.9, "AI discussion")
    memory.remember_word_association("neural", "processing", 0.7, "Brain function")
    memory.remember_word_association("network", "connection", 0.8, "Network topology")
    
    # Store a grammar pattern
    memory.remember_pattern(
        "DET ADJ NOUN VERB DET NOUN", 
        "The neural network processes the data."
    )
    
    # Store a sentence
    memory.store_sentence(
        "Neural networks can process complex data structures efficiently.", 
        {"tense": "present", "topic": "ai", "complexity": "medium"}
    )
    
    # Analyze a text fragment
    text = "Neural networks are computational systems. They process data through layers of nodes. Modern neural architectures can solve complex problems."
    analysis = memory.analyze_language_fragment(text)
    print("Text Analysis:", analysis)
    
    # Recall associations for a word
    associations = memory.recall_associations("neural")
    print("\nAssociations for 'neural':")
    for assoc in associations:
        print(f"- {assoc['word']} (strength: {assoc['strength']:.2f})")
    
    # Generate text
    generated = memory.generate_from_memory(seed_word="neural")
    print(f"\nGenerated text: {generated}")
    
    # Get memory statistics
    stats = memory.get_memory_statistics()
    print("\nMemory Statistics:")
    print(f"- Words: {stats['total_words']}")
    print(f"- Associations: {stats['total_associations']}")
    print(f"- Patterns: {stats['total_patterns']}")
    print(f"- Sentences: {stats['total_sentences']}")
    
    # Save memories
    memory.save_memories()
    print("Language Memory test completed") 