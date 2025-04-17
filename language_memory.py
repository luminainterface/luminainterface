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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("language_memory")

class LanguageMemory:
    """
    Language-specific memory system that stores and retrieves language patterns,
    vocabulary usage, and linguistic structures for improved language processing.
    
    Provides:
    - Memory of word associations and semantic relationships
    - Storage of grammatical patterns and their usage frequencies
    - Tracking of vocabulary usage and mastery levels
    - Memory integration with language training systems
    """
    
    def __init__(self, memory_path="data/memory/language_memory"):
        """
        Initialize language memory system
        
        Args:
            memory_path: Path to store language memory data
        """
        logger.info("Initializing Language Memory system")
        
        # Setup memory storage
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory components
        self.word_associations = {}
        self.grammar_patterns = {}
        self.word_usage = {}
        self.sentences = []
        self.language_fragments = []
        
        # Version tracking
        self.memory_version = "1.0.0"
        
        # Memory stats
        self.access_count = 0
        self.store_count = 0
        self.last_access = None
        self.last_store = None
        
        # Initialize indexes for faster retrieval
        self.word_index = {}
        self.pattern_index = {}
        self.linguistic_features = {}
        
        # Load existing memories
        self._load_memories()
        
        logger.info(f"Language Memory initialized with {len(self.word_associations)} word associations")
    
    def _load_memories(self):
        """Load all existing language memories from files"""
        # Load word associations
        word_assoc_file = self.memory_path / "word_associations.json"
        if word_assoc_file.exists():
            try:
                with open(word_assoc_file, 'r', encoding='utf-8') as f:
                    self.word_associations = json.load(f)
                logger.info(f"Loaded {len(self.word_associations)} word associations")
            except Exception as e:
                logger.error(f"Error loading word associations: {str(e)}")
        
        # Load grammar patterns
        grammar_file = self.memory_path / "grammar_patterns.json"
        if grammar_file.exists():
            try:
                with open(grammar_file, 'r', encoding='utf-8') as f:
                    self.grammar_patterns = json.load(f)
                logger.info(f"Loaded {len(self.grammar_patterns)} grammar patterns")
            except Exception as e:
                logger.error(f"Error loading grammar patterns: {str(e)}")
        
        # Load word usage
        usage_file = self.memory_path / "word_usage.json"
        if usage_file.exists():
            try:
                with open(usage_file, 'r', encoding='utf-8') as f:
                    self.word_usage = json.load(f)
                logger.info(f"Loaded usage data for {len(self.word_usage)} words")
            except Exception as e:
                logger.error(f"Error loading word usage: {str(e)}")
        
        # Load sentences
        sentences_file = self.memory_path / "sentences.jsonl"
        if sentences_file.exists():
            try:
                self.sentences = []
                with open(sentences_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.sentences.append(json.loads(line))
                logger.info(f"Loaded {len(self.sentences)} sentences")
            except Exception as e:
                logger.error(f"Error loading sentences: {str(e)}")
        
        # Build indexes
        self._build_indexes()
    
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
        for sentence in self.sentences:
            features = sentence.get("features", {})
            for feature, value in features.items():
                if feature not in self.linguistic_features:
                    self.linguistic_features[feature] = set()
                self.linguistic_features[feature].add(value)
        
        logger.info("Memory indexes built successfully")
    
    def save_memories(self):
        """Save all language memories to disk"""
        try:
            # Save word associations
            with open(self.memory_path / "word_associations.json", 'w', encoding='utf-8') as f:
                json.dump(self.word_associations, f, indent=2)
            
            # Save grammar patterns
            with open(self.memory_path / "grammar_patterns.json", 'w', encoding='utf-8') as f:
                json.dump(self.grammar_patterns, f, indent=2)
            
            # Save word usage
            with open(self.memory_path / "word_usage.json", 'w', encoding='utf-8') as f:
                json.dump(self.word_usage, f, indent=2)
            
            # Save sentences (append-only)
            with open(self.memory_path / "sentences.jsonl", 'w', encoding='utf-8') as f:
                for sentence in self.sentences:
                    f.write(json.dumps(sentence) + "\n")
            
            logger.info("Language memories saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving language memories: {str(e)}")
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
        
        # Update association
        if associated_word in self.word_associations[word]:
            # Strengthen existing association
            current = self.word_associations[word][associated_word]
            # Average the strength, slightly favoring new observations
            new_strength = (current["strength"] + strength) / 2
            self.word_associations[word][associated_word] = {
                "strength": new_strength,
                "count": current.get("count", 1) + 1,
                "last_seen": datetime.now().isoformat(),
                "contexts": current.get("contexts", []) + ([context] if context else [])
            }
        else:
            # New association
            self.word_associations[word][associated_word] = {
                "strength": strength,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "contexts": [context] if context else []
            }
        
        # Update the word index
        if word not in self.word_index:
            self.word_index[word] = {"associations": {}, "usage_count": 0}
        self.word_index[word]["associations"] = self.word_associations[word]
        
        logger.debug(f"Stored association between '{word}' and '{associated_word}' (strength: {strength})")
        
        # Periodically save memories
        if self.store_count % 100 == 0:
            self.save_memories()
            
        return True
    
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
    
    def remember_sentence(self, text, features=None, source=None):
        """
        Remember a sentence and its linguistic features
        
        Args:
            text: Sentence text
            features: Dict of linguistic features
            source: Source of the sentence
            
        Returns:
            int: Sentence ID (index)
        """
        self.last_store = datetime.now()
        self.store_count += 1
        
        features = features or {}
        
        # Extract words for word usage tracking
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            self._update_word_usage(word)
        
        # Store sentence
        sentence_id = len(self.sentences)
        sentence_entry = {
            "id": sentence_id,
            "text": text,
            "features": features,
            "source": source,
            "words": words,
            "timestamp": datetime.now().isoformat()
        }
        
        self.sentences.append(sentence_entry)
        
        # Update linguistic features index
        for feature, value in features.items():
            if feature not in self.linguistic_features:
                self.linguistic_features[feature] = set()
            self.linguistic_features[feature].add(value)
        
        logger.debug(f"Stored sentence: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Append to sentences file for persistence (append-only)
        try:
            with open(self.memory_path / "sentences.jsonl", 'a', encoding='utf-8') as f:
                f.write(json.dumps(sentence_entry) + "\n")
        except Exception as e:
            logger.error(f"Error appending sentence to file: {str(e)}")
        
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
    
    def recall_associations(self, word, min_strength=0.0, limit=10):
        """
        Recall words associated with the given word
        
        Args:
            word: Word to find associations for
            min_strength: Minimum association strength
            limit: Maximum number of associations to return
            
        Returns:
            list: Associated words sorted by strength
        """
        self.last_access = datetime.now()
        self.access_count += 1
        
        if word not in self.word_associations:
            return []
        
        # Get associations above minimum strength
        associations = []
        for assoc_word, details in self.word_associations[word].items():
            if details["strength"] >= min_strength:
                associations.append({
                    "word": assoc_word,
                    "strength": details["strength"],
                    "count": details.get("count", 1),
                    "last_seen": details.get("last_seen"),
                })
        
        # Sort by strength (descending)
        sorted_associations = sorted(associations, key=lambda x: x["strength"], reverse=True)
        
        return sorted_associations[:limit]
    
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
        
        for sentence in self.sentences:
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
        
        for sentence in self.sentences:
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
        return {
            "total_words": len(self.word_usage),
            "total_associations": sum(len(assocs) for assocs in self.word_associations.values()),
            "total_patterns": len(self.grammar_patterns),
            "total_sentences": len(self.sentences),
            "access_count": self.access_count,
            "store_count": self.store_count,
            "last_access": self.last_access.isoformat() if self.last_access else None,
            "last_store": self.last_store.isoformat() if self.last_store else None,
            "memory_path": str(self.memory_path),
            "memory_version": self.memory_version,
            "feature_types": list(self.linguistic_features.keys())
        }
    
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
            self.sentences = []
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
                
                sentence_id = self.remember_sentence(sentence, features)
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

# For testing
if __name__ == "__main__":
    memory = LanguageMemory()
    
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
    memory.remember_sentence(
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