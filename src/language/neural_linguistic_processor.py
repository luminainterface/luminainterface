#!/usr/bin/env python3
"""
Neural Linguistic Processor Module

This module enhances the Language Memory System with advanced capabilities for detecting
complex linguistic patterns, analyzing semantic relationships, and integrating different
processing approaches (neural, symbolic, and LLM).
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
import threading
import time
import uuid
import re
import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, Counter

# Default weights for different processing components
DEFAULT_LLM_WEIGHT = 0.65  # Weight given to LLM inputs/outputs
DEFAULT_NN_WEIGHT = 0.5    # Neural network processing weight
DEFAULT_NEURAL_WEIGHT = 0.6  # Weight for neural vs. symbolic processing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger("neural_linguistic_processor")

class NeuralLinguisticProcessor:
    """
    Neural Linguistic Processor for advanced language analysis.
    
    Combines neural network, symbolic, and LLM processing approaches
    to identify patterns, relationships, and structures in text.
    """
    
    def __init__(self, language_memory=None, conscious_mirror_language=None):
        """
        Initialize a new Neural Linguistic Processor instance.
        
        Args:
            language_memory: Optional Language Memory System reference
            conscious_mirror_language: Optional Conscious Mirror Language reference
        """
        # Configure logging
        self.logger = logging.getLogger('neural_linguistic_processor')
        self.logger.info(f"NeuralLinguisticProcessor initialized with LLM weight: {DEFAULT_LLM_WEIGHT}, NN weight: {DEFAULT_NN_WEIGHT}")
        
        # Track basic metrics
        self.processed_text_count = 0
        self.total_words_processed = 0
        self.last_neural_linguistic_score = 0.0
        
        # External system references
        self.language_memory = language_memory
        self.conscious_mirror_language = conscious_mirror_language
        
        # Processing weights
        self.llm_weight = DEFAULT_LLM_WEIGHT  # Default 0.65
        self.nn_weight = DEFAULT_NN_WEIGHT  # Default 0.5
        self.neural_weight = DEFAULT_NEURAL_WEIGHT  # Default 0.6
        
        # Initialize processing data structures
        self.pattern_recognition = {}
        self.recursive_patterns = []
        self.word_frequencies = Counter()
        
        # Initialize semantic network as a defaultdict of dicts
        self.semantic_network = defaultdict(lambda: defaultdict(float))
        
        # LLM-specific tracking
        self.llm_pattern_recognition = {}
        self.llm_semantic_suggestions = defaultdict(dict)
        self.llm_confidence_history = []
        
        # Load saved state if available
        self._load_state()
        
        # Initialize semantic network with some basic relationships if empty
        if not self.semantic_network:
            self._initialize_semantic_network()
    
    def _load_state(self):
        """Load processor state from disk."""
        try:
            # Create a data directory path in the current directory
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'language')
            
            # Path to state file
            state_file = os.path.join(data_dir, 'neural_linguistic_state.json')
            
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                    # Load basic metrics
                    self.processed_text_count = state.get('processed_text_count', 0)
                    self.total_words_processed = state.get('total_words_processed', 0)
                    self.last_neural_linguistic_score = state.get('last_neural_linguistic_score', 0.0)
                
                # Load processing weights
                self.llm_weight = state.get('llm_weight', DEFAULT_LLM_WEIGHT)
                self.nn_weight = state.get('nn_weight', DEFAULT_NN_WEIGHT)
                self.neural_weight = state.get('neural_weight', DEFAULT_NEURAL_WEIGHT)
                
                # Load word frequencies (convert back from list of pairs)
                word_freq = state.get('word_frequencies', [])
                self.word_frequencies = Counter(dict(word_freq))
                
                # Load pattern recognition data
                self.pattern_recognition = state.get('pattern_recognition', {})
                
                # Load recursive patterns
                self.recursive_patterns = state.get('recursive_patterns', [])
                
                # Load semantic network (convert from list format)
                semantic_net = state.get('semantic_network', {})
                self.semantic_network = defaultdict(lambda: defaultdict(float))
                for word, connections in semantic_net.items():
                    for related_word, strength in connections.items():
                        self.semantic_network[word][related_word] = float(strength)
                
                # Load LLM-specific data
                self.llm_pattern_recognition = state.get('llm_pattern_recognition', {})
                
                # Convert llm semantic suggestions from list format
                llm_semantic = state.get('llm_semantic_suggestions', {})
                self.llm_semantic_suggestions = defaultdict(dict)
                for word, suggestions in llm_semantic.items():
                    self.llm_semantic_suggestions[word] = suggestions
                
                self.llm_confidence_history = state.get('llm_confidence_history', [])
                
                self.logger.info(f"Loaded neural linguistic processor state from {state_file}")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error loading neural linguistic processor state: {e}")
            return False
    
    def _save_state(self):
        """Save processor state to disk."""
        try:
            # Create a data directory path in the current directory
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'language')
            os.makedirs(data_dir, exist_ok=True)
            
            # Path to state file
            state_file = os.path.join(data_dir, 'neural_linguistic_state.json')
            
            # Prepare state dictionary
            state = {
                # Basic metrics
                'processed_text_count': self.processed_text_count,
                'total_words_processed': self.total_words_processed,
                'last_neural_linguistic_score': self.last_neural_linguistic_score,
                
                # Processing weights
                'llm_weight': self.llm_weight,
                'nn_weight': self.nn_weight,
                'neural_weight': self.neural_weight,
                
                # Pattern data
                'pattern_recognition': self.pattern_recognition,
                'recursive_patterns': self.recursive_patterns,
                
                # Vocabulary data (convert Counter to list of pairs)
                'word_frequencies': list(self.word_frequencies.items()),
                
                # Semantic network
                'semantic_network': dict(self.semantic_network),
                
                # LLM data
                'llm_pattern_recognition': self.llm_pattern_recognition,
                'llm_semantic_suggestions': dict(self.llm_semantic_suggestions),
                'llm_confidence_history': self.llm_confidence_history,
                
                # Save timestamp
                'last_saved': datetime.now().isoformat()
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Saved neural linguistic processor state to {state_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving neural linguistic processor state: {e}")
            return False
    
    def process_text(self, text):
        """
        Process text through neural linguistic analysis with enhanced pattern recognition.
        Returns analysis results and neural linguistic score.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis results and score
        """
        # Process with neural network methods
        neural_results = self._neural_process_text(text)
        
        # Process with symbolic methods (with enhanced pattern recognition)
        symbolic_results = self._symbolic_process_text(text)
        
        # Process with LLM methods (or simulated LLM for now)
        llm_results = self._llm_process_text(text)
        
        # Combine neural and symbolic results based on current weights
        internal_results = self._combine_neural_symbolic(neural_results, symbolic_results)
        
        # Then combine with LLM results
        combined_results = self._combine_results(internal_results, llm_results)
        
        # Track metrics for future improvement
        self.processed_text_count += 1
        self.total_words_processed += len(text.split())
        
        # Update neural linguistic score history
        self.last_neural_linguistic_score = combined_results['neural_linguistic_score']
        self.neural_linguistic_score_history.append({
            'timestamp': datetime.now().isoformat(),
            'score': self.last_neural_linguistic_score,
            'text_length': len(text),
            'pattern_count': sum(p.get('count', 1) for p in symbolic_results.get('detected_patterns', [])),
            'recursive_count': len(symbolic_results.get('recursive_patterns', []))
        })
        
        # Keep history at a reasonable size
        if len(self.neural_linguistic_score_history) > 100:
            self.neural_linguistic_score_history = self.neural_linguistic_score_history[-100:]
        
        # Periodically save state (every 5 processes)
        if self.processed_text_count % 5 == 0:
            self._save_state()
        
        return combined_results
    
    def _neural_process_text(self, text):
        """Process text using neural network methods."""
        # This would call an actual neural model in a real system
        # Here, we'll simulate neural processing
        
        # Simulate text embedding by converting to lowercase and extracting features
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Simulated topic detection
        topics = self._simulate_topic_detection(words)
        
        # Simulated sentiment analysis
        sentiment = self._simulate_sentiment(words)
        
        # Simulated emotional tone detection
        emotions = self._simulate_emotions(words)
        
        # Simulated vector representation (simplified)
        # In a real system, this would be an actual embedding
        vector_representation = self._simulate_vector_embedding(words)
        
        # Enhanced pattern extraction using neural techniques
        neural_patterns = self._extract_neural_patterns(words, text)
        
        # Generate a neural score based on these simulated analyses
        neural_score = (
            (topics['confidence'] * 0.3) +
            (sentiment['confidence'] * 0.2) +
            (emotions['confidence'] * 0.2) +
            (0.3 * (sum(p['confidence'] for p in neural_patterns) / max(1, len(neural_patterns))))
        )
        
        return {
            'neural_score': neural_score,
            'confidence': min(0.95, topics['confidence'] * 0.3 + sentiment['confidence'] * 0.3 + 0.4),
            'primary_topics': topics['topics'],
            'sentiment': sentiment,
            'emotions': emotions,
            'vector_representation': vector_representation,
            'neural_patterns': neural_patterns
        }
    
    def _extract_neural_patterns(self, words, text):
        """
        Extract patterns using neural-inspired techniques.
        
        Args:
            words: List of words from the text
            
        Returns:
            List of patterns detected using neural methods
        """
        patterns = []
        
        # 1. Cluster related words using embedding similarity (simulated)
        clusters = []
        unique_words = list(set(words))
        
        # In a real system, this would use actual word embeddings
        # For simulation, we'll use our semantic network as a proxy
        for word in unique_words:
            if word in self.semantic_network:
                related = self.semantic_network[word]
                
                # Check which related words are in the text
                related_in_text = [w for w in related if w in unique_words]
                
                if len(related_in_text) >= 2:
                    clusters.append({
                        'central_word': word,
                        'related': related_in_text,
                        'strength': len(related_in_text) / len(related) if related else 0
                    })
        
        if clusters:
            # Sort by strength
            clusters.sort(key=lambda x: x['strength'], reverse=True)
            
            patterns.append({
                'type': 'semantic_embeddings',
                'clusters': clusters[:3],  # Top 3 clusters
                'confidence': min(0.9, 0.5 + (len(clusters) * 0.1))
            })
        
        # 2. Simulated sequence pattern detection (would use RNN-like approach in real system)
        if len(words) >= 10:
            sequence_patterns = []
            
            # Simplified approach: look for repeating n-grams that might indicate a pattern
            for n in range(2, 4):
                ngrams = {}
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    if ngram in ngrams:
                        ngrams[ngram].append(i)
                    else:
                        ngrams[ngram] = [i]
                
                # Analyze distribution of repeated n-grams
                for ngram, positions in ngrams.items():
                    if len(positions) >= 2:
                        # Check if n-grams appear in regular intervals
                        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                        avg_interval = sum(intervals) / len(intervals) if intervals else 0
                        
                        # Calculate variance of intervals
                        variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals) if intervals else 0
                        
                        # Low variance suggests regular pattern
                        if variance < 5 and avg_interval > 0:
                            sequence_patterns.append({
                                'ngram': ngram,
                                'positions': positions,
                                'avg_interval': avg_interval,
                                'variance': variance,
                                'regularity': 1.0 / (1.0 + variance)  # Higher for more regular patterns
                            })
            
            if sequence_patterns:
                # Sort by regularity
                sequence_patterns.sort(key=lambda x: x['regularity'], reverse=True)
                
                patterns.append({
                    'type': 'sequence_pattern',
                    'patterns': sequence_patterns[:3],  # Top 3 most regular patterns
                    'confidence': min(0.85, 0.5 + (len(sequence_patterns) * 0.05))
                })
        
        # 3. Simulated attention-like focus (words that stand out in context)
        if len(words) >= 15:
            attention_scores = {}
            
            # In a real system, this would use attention mechanisms
            # Here we'll use heuristics like position and relationship to other words
            for i, word in enumerate(words):
                # Words at beginning/end get higher attention
                position_score = 0.3 if i < len(words) * 0.1 or i > len(words) * 0.9 else 0.0
                
                # Words with many connections get higher attention
                connection_score = 0.0
                if word in self.semantic_network:
                    connection_score = min(0.4, len(self.semantic_network[word]) * 0.05)
                
                # Words that are rare in general vocabulary get higher attention
                frequency = max(1, self.word_frequencies.get(word, 0))
                rarity_score = min(0.3, 0.3 / (math.log2(frequency + 2)))
                
                # Calculate final attention score
                attention_scores[word] = position_score + connection_score + rarity_score
            
            # Get top attention words
            top_attention = sorted([(word, score) for word, score in attention_scores.items()], 
                                   key=lambda x: x[1], reverse=True)[:5]
            
            if top_attention:
                patterns.append({
                    'type': 'attention_focus',
                    'words': [{'word': word, 'score': score} for word, score in top_attention],
                    'confidence': min(0.8, 0.5 + (sum(score for _, score in top_attention) * 0.2))
                })
        
        return patterns
    
    def _simulate_topic_detection(self, words):
        """Simulate topic detection (would use a real model in production)."""
        # Filter common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        filtered_words = [w for w in words if w not in common_words and len(w) > 2]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Get top words
        top_words = word_counts.most_common(5)
        
        # Simple topic assignment based on semantic relationships
        topics = []
        confidence = 0.0
        
        if top_words:
            # Try to group related top words into topics
            for word, count in top_words:
                related_topic = None
                
                # Check if word is related to any existing topic
                for topic in topics:
                    if word in self.semantic_network and topic['keyword'] in self.semantic_network[word]:
                        related_topic = topic
                        topic['related_words'].append(word)
                        topic['count'] += count
                        break
                
                if not related_topic:
                    # Create new topic
                    topics.append({
                        'keyword': word,
                        'related_words': [],
                        'count': count
                    })
            
            # Calculate confidence based on topic coherence
            topic_coherence = sum(topic['count'] for topic in topics) / sum(count for _, count in top_words)
            confidence = min(0.9, 0.4 + (topic_coherence * 0.5))
        
        return {
            'topics': sorted(topics, key=lambda x: x['count'], reverse=True),
            'confidence': confidence if topics else 0.4
        }
    
    def _simulate_sentiment(self, words):
        """Simulate sentiment analysis (would use a real model in production)."""
        # Simple positive/negative word lists
        positive_words = {'good', 'great', 'excellent', 'best', 'love', 'happy', 'wonderful', 'beautiful', 'perfect'}
        negative_words = {'bad', 'worst', 'terrible', 'awful', 'hate', 'sad', 'horrible', 'poor', 'negative'}
        
        # Count positive and negative words
        pos_count = sum(1 for word in words if word.lower() in positive_words)
        neg_count = sum(1 for word in words if word.lower() in negative_words)
        
        # Calculate sentiment score and label
        if pos_count > neg_count:
            sentiment = 'positive'
            score = min(0.95, 0.5 + ((pos_count - neg_count) / max(1, len(words)) * 5))
        elif neg_count > pos_count:
            sentiment = 'negative'
            score = min(0.95, 0.5 + ((neg_count - pos_count) / max(1, len(words)) * 5))
        else:
            sentiment = 'neutral'
            score = 0.5
        
        # Calculate confidence
        confidence = min(0.9, 0.5 + ((pos_count + neg_count) / max(1, len(words)) * 2))
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence
        }
    
    def _simulate_emotions(self, words):
        """Simulate emotion detection (would use a real model in production)."""
        # Simple emotion word lists
        emotion_words = {
            'joy': {'happy', 'joy', 'delighted', 'pleased', 'glad', 'cheerful'},
            'sadness': {'sad', 'unhappy', 'depressed', 'gloomy', 'miserable'},
            'anger': {'angry', 'mad', 'furious', 'outraged', 'irritated'},
            'fear': {'afraid', 'scared', 'frightened', 'terrified', 'anxious'},
            'surprise': {'surprised', 'amazed', 'astonished', 'shocked'},
            'disgust': {'disgusted', 'revolted', 'appalled'}
        }
        
        # Count emotion words
        emotion_counts = {emotion: 0 for emotion in emotion_words}
        for word in words:
            word_lower = word.lower()
            for emotion, words_list in emotion_words.items():
                if word_lower in words_list:
                    emotion_counts[emotion] += 1
        
        # Calculate primary emotions
        total_emotion_words = sum(emotion_counts.values())
        emotions = []
        
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                emotions.append({
                    'emotion': emotion,
                    'intensity': min(0.95, count / max(1, len(words)) * 10),
                    'count': count
                })
        
        # Calculate confidence
        confidence = min(0.9, 0.4 + (total_emotion_words / max(1, len(words)) * 3))
        
        return {
            'emotions': emotions,
            'primary_emotion': emotions[0]['emotion'] if emotions else 'neutral',
            'confidence': confidence
        }
    
    def _simulate_vector_embedding(self, words):
        """
        Simulate vector embedding (would use actual embeddings in production).
        Returns a simple vector representation for demonstration purposes.
        """
        # Create a pseudo-random vector based on word characteristics
        vector_length = 10  # Simplified vector
        vector = [0.0] * vector_length
        
        # Create a deterministic but simplified embedding
        for word in words:
            # Use character codes to generate components
            for i, char in enumerate(word):
                idx = i % vector_length
                vector[idx] += ord(char) / 1000.0
        
        # Normalize the vector
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector
    
    def _symbolic_process_text(self, text):
        """Process text using symbolic (rule-based) methods."""
        # This would be more rule-based and structured in a real system
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Update word frequencies
        for word in words:
            self.word_frequencies[word] += 1
        
        # Build/update semantic network
        self._update_semantic_network(words)
        
        # Detect patterns using enhanced rule-based methods
        patterns = self._detect_patterns(words, text)
        
        # We can't directly call the recursive patterns detection method since it operates on 
        # the results of _detect_patterns, so we'll use a different approach for this
        local_patterns = {}
        for pattern in patterns:
            if pattern.get('type') == 'repeated_2gram' or pattern.get('type') == 'repeated_3gram':
                for ngram in pattern.get('ngrams', []):
                    pattern_key = ngram.get('text', '')
                    if pattern_key:
                        local_patterns[pattern_key] = [{'context': [], 'score': 0.5}]
        
        # Look for patterns with enhanced detection if we found any n-grams
        if local_patterns:
            self._detect_recursive_patterns(local_patterns)
            
        # Get recursive patterns from the processor's state
        recursive_patterns = self.get_recursive_patterns()
        
        return {
            'word_count': len(words),
            'unique_words': len(set(words)),
            'detected_patterns': patterns,
            'recursive_patterns': recursive_patterns,
            'symbolic_score': self._calculate_neural_linguistic_score(words, patterns, recursive_patterns),
            'top_words': self._get_top_words(words, 5)
        }
    
    def _llm_process_text(self, text):
        """
        Process text using LLM capabilities (simulated for now).
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with LLM processing results
        """
        # This would call the actual LLM API in a real system
        # For now, simulate LLM processing
        
        # Simulate simple analysis
        sentiment = 'neutral'
        if len(text) > 200:
            sentiment = 'positive' if text.count('good') > text.count('bad') else 'negative'
        
        # Simulate complexity evaluation
        avg_word_length = sum(len(word) for word in text.split()) / max(1, len(text.split()))
        complexity = min(1.0, max(0.1, avg_word_length / 10))
        
        # Simulate LLM score - weighted toward complexity and coherence
        llm_score = min(1.0, 0.4 + (complexity * 0.6))
        
        # Create simulated insights
        insights = [
            {
                'type': 'sentiment',
                'value': sentiment,
                'confidence': 0.85
            },
            {
                'type': 'complexity',
                'value': 'high' if complexity > 0.7 else 'medium' if complexity > 0.4 else 'low',
                'confidence': 0.9
            },
            {
                'type': 'coherence',
                'value': 'high' if llm_score > 0.7 else 'medium' if llm_score > 0.5 else 'low',
                'confidence': 0.8
            }
        ]
        
        return {
            'llm_score': llm_score,
            'insights': insights,
            'confidence': 0.85,
            'model': 'mistral-simulation'  # In a real system, this would be the actual model name
        }
    
    def _combine_neural_symbolic(self, neural_results, symbolic_results):
        """
        Combine neural and symbolic results based on configured weights.
        
        Args:
            neural_results: Results from neural processing
            symbolic_results: Results from symbolic processing
            
        Returns:
            Combined results dictionary
        """
        # Determine weights (could be dynamic based on text characteristics)
        neural_weight = self.neural_weight  # Typically 0.6
        symbolic_weight = 1.0 - neural_weight  # Typically 0.4
        
        # Calculate combined score
        neural_score = neural_results.get('neural_score', 0.5)
        symbolic_score = symbolic_results.get('symbolic_score', 0.5)
        combined_score = (neural_score * neural_weight) + (symbolic_score * symbolic_weight)
        
        # Integrate pattern recognition from both approaches
        combined_patterns = {}
        
        # Add symbolic patterns with their source
        for pattern in symbolic_results.get('detected_patterns', []):
            pattern_key = f"symbolic_{pattern['type']}"
            pattern['source'] = 'symbolic'
            combined_patterns[pattern_key] = pattern
        
        # Add neural patterns with their source
        for pattern in neural_results.get('neural_patterns', []):
            pattern_key = f"neural_{pattern['type']}"
            pattern['source'] = 'neural'
            combined_patterns[pattern_key] = pattern
        
        # Identify cross-validated patterns (recognized by both methods)
        cross_validated = []
        neural_semantic = next((p for p in neural_results.get('neural_patterns', []) 
                                if p['type'] == 'semantic_embeddings'), None)
        symbolic_semantic = next((p for p in symbolic_results.get('detected_patterns', []) 
                                  if p['type'] == 'semantic_cluster'), None)
        
        if neural_semantic and symbolic_semantic:
            # Try to find overlapping clusters
            for neural_cluster in neural_semantic.get('clusters', []):
                neural_words = set([neural_cluster['central_word']] + neural_cluster.get('related', []))
                
                for symbolic_cluster in symbolic_semantic.get('clusters', []):
                    symbolic_words = set([symbolic_cluster['central_word']] + 
                                         [w['word'] for w in symbolic_cluster.get('related_words', [])])
                    
                    # Calculate overlap
                    overlap = neural_words.intersection(symbolic_words)
                    if len(overlap) >= 2:
                        cross_validated.append({
                            'type': 'cross_validated_semantic',
                            'words': list(overlap),
                            'neural_source': neural_cluster,
                            'symbolic_source': symbolic_cluster,
                            'confidence': min(0.98, 
                                          neural_semantic['confidence'] * neural_weight + 
                                          symbolic_semantic['confidence'] * symbolic_weight)
                        })
        
        # Calculate confidence adjustments for cross-validated patterns
        for pattern in cross_validated:
            # Boost confidence for cross-validated patterns
            pattern['confidence'] = min(0.99, pattern['confidence'] * 1.2) 
        
        # Combine results
        combined = {
            'neural_linguistic_score': combined_score,
            'confidence': min(0.95, (neural_results.get('confidence', 0.5) * neural_weight + 
                                   symbolic_weight * 0.8)),
            'analysis': {
                'neural': {
                    'score': neural_score,
                    'topics': neural_results.get('primary_topics', []),
                    'vector_sample': neural_results.get('vector_representation', [])[:5],
                    'sentiment': neural_results.get('sentiment', {})
                },
                'symbolic': {
                    'score': symbolic_score,
                    'patterns': symbolic_results.get('detected_patterns', []),
                    'recursive_patterns': symbolic_results.get('recursive_patterns', [])
                },
                'cross_validated': cross_validated
            },
            'combined_patterns': combined_patterns,
            'word_metrics': {
                'count': symbolic_results.get('word_count', 0),
                'unique': symbolic_results.get('unique_words', 0),
                'top_words': symbolic_results.get('top_words', [])
            }
        }
        
        return combined
    
    def _combine_results(self, internal_results, llm_results):
        """
        Combine internal results with LLM processing results.
        
        Args:
            internal_results: Combined neural and symbolic results
            llm_results: Results from LLM processing
            
        Returns:
            Final combined results dictionary
        """
        # Determine weights for internal vs LLM results
        internal_weight = 1.0 - self.llm_weight  # Typically 0.7
        llm_weight = self.llm_weight  # Typically 0.3
        
        # Calculate blended neural linguistic score
        internal_score = internal_results.get('neural_linguistic_score', 0.5)
        llm_score = llm_results.get('llm_score', 0.5)
        final_score = (internal_score * internal_weight) + (llm_score * llm_weight)
        
        # Structure final results
        final_results = {
            'neural_linguistic_score': final_score,
            'confidence': min(0.98, (internal_results.get('confidence', 0.7) * internal_weight + 
                                  llm_results.get('confidence', 0.8) * llm_weight)),
            'analysis': {
                'internal': internal_results.get('analysis', {}),
                'llm': {
                    'score': llm_score,
                    'insights': llm_results.get('insights', []),
                    'model': llm_results.get('model', 'simulated')
                }
            },
            'word_metrics': internal_results.get('word_metrics', {}),
            'processing_timestamp': datetime.now().isoformat(),
            'processing_id': str(uuid.uuid4())
        }
        
        return final_results
    
    def _update_semantic_network(self, words):
        """Update the semantic network based on word co-occurrence."""
        # Simple sliding window for co-occurrence
        window_size = 5
        for i in range(len(words)):
            current_word = words[i]
            
            # Look at words within the window
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:  # Don't connect word to itself
                    related_word = words[j]
                    
                    # Calculate relationship strength based on proximity
                    # Words closer together have stronger connections
                    strength = 1.0 / (abs(i - j) + 1)
                    
                    # Update connection (will automatically create if needed due to defaultdict)
                    self.semantic_network[current_word][related_word] += strength
                    self.semantic_network[related_word][current_word] += strength
    
    def _detect_patterns(self, words, text=None):
        """
        Detect linguistic patterns in the text with enhanced pattern recognition.
        
        Args:
            words: List of words from the text
            text: Optional original text for more complex pattern analysis
            
        Returns:
            List of detected pattern objects
        """
        patterns = []
        
        # 1. Enhanced repetition patterns - look for words and phrases with more nuanced repetition
        word_counts = Counter(words)
        repetitions = [word for word, count in word_counts.items() if count > 1]
        
        if repetitions:
            # Calculate repetition significance based on frequency and word rarity
            repetition_scores = {}
            for word in repetitions:
                # Words that are less common in general but repeated in this text are more significant
                # Make sure global_frequency is always a positive value
                global_frequency = max(1, self.word_frequencies.get(word, 0) - word_counts[word] + 1)
                repetition_score = word_counts[word] * (1.0 / max(1, math.log2(global_frequency + 1)))
                # Add context awareness - words appearing in key positions are more significant
                position_boost = 0
                for i, w in enumerate(words):
                    if w == word:
                        # Words at beginning/end of text are typically more significant
                        if i < len(words) * 0.1 or i > len(words) * 0.9:
                            position_boost += 0.2
                repetition_score *= (1 + position_boost)
                repetition_scores[word] = repetition_score
            
            # Sort by significance score
            significant_repetitions = sorted(
                [(word, score) for word, score in repetition_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Keep top 5 most significant
            
            patterns.append({
                'type': 'repetition',
                'words': [word for word, _ in significant_repetitions],
                'count': len(significant_repetitions),
                'significance_scores': {word: score for word, score in significant_repetitions},
                'confidence': min(0.9, 0.5 + (len(significant_repetitions) * 0.1))
            })
        
        # 2. Enhanced N-gram pattern detection (sequences of 2-4 words)
        for n in range(2, min(5, len(words))):
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            ngram_counts = Counter(ngrams)
            repeated_ngrams = {ng: count for ng, count in ngram_counts.items() if count > 1}
            
            if repeated_ngrams:
                # Calculate significance scores for each ngram based on uniqueness and frequency
                ngram_scores = {}
                for ngram, count in repeated_ngrams.items():
                    # Base score from frequency
                    base_score = count / len(words)
                    
                    # Bonus for rarer ngrams (those not seen much before)
                    ngram_key = f"ngram_{ngram}"
                    historical_count = max(0, self.pattern_recognition.get(ngram_key, 0))
                    rarity_bonus = 1.0 / max(1, math.log2(historical_count + 1))
                    
                    # Calculate final score
                    ngram_scores[ngram] = base_score * rarity_bonus
                
                # Sort by significance score
                significant_ngrams = [
                    {'text': ng, 'count': repeated_ngrams[ng], 'score': score}
                    for ng, score in sorted(ngram_scores.items(), key=lambda x: x[1], reverse=True)
                ]
                
            patterns.append({
                    'type': f'repeated_{n}gram',
                    'ngrams': significant_ngrams,
                    'count': len(repeated_ngrams),
                    'confidence': min(0.95, 0.6 + (len(repeated_ngrams) * 0.1))
                })
        
        # 3. Syntactic patterns using simplified POS-like tagging
        if text:
            # Identify potential parts of speech with simple heuristics
            pos_patterns = self._identify_syntactic_patterns(words, text)
            if pos_patterns:
                patterns.append({
                    'type': 'syntactic',
                    'patterns': pos_patterns,
                    'count': len(pos_patterns),
                    'confidence': min(0.9, 0.5 + (len(pos_patterns) * 0.08))
                })
        
        # 4. Semantic patterns based on word relationships in our semantic network
        semantic_clusters = self._identify_semantic_clusters(words)
        if semantic_clusters:
            patterns.append({
                'type': 'semantic_cluster',
                'clusters': semantic_clusters,
                'count': len(semantic_clusters),
                'confidence': min(0.85, 0.4 + (len(semantic_clusters) * 0.15))
            })
        
        # 5. Identify parallelism (similar grammatical structures)
        if text:
            parallel_structures = self._detect_parallel_structures(text)
            if parallel_structures:
                patterns.append({
                    'type': 'parallelism',
                    'structures': parallel_structures,
                    'count': len(parallel_structures),
                    'confidence': min(0.9, 0.6 + (len(parallel_structures) * 0.1))
                })
        
        # 6. NEW: Identify chiasmus patterns (A-B-B-A structure)
        if text:
            chiasmus_patterns = self._detect_chiasmus(words, text)
            if chiasmus_patterns:
                patterns.append({
                    'type': 'chiasmus',
                    'instances': chiasmus_patterns,
                    'count': len(chiasmus_patterns),
                    'confidence': min(0.95, 0.7 + (len(chiasmus_patterns) * 0.1))
                })
        
        # 7. NEW: Identify rhetorical patterns
        if text:
            rhetorical_patterns = self._detect_rhetorical_patterns(text)
            if rhetorical_patterns:
                patterns.append({
                    'type': 'rhetorical',
                    'instances': rhetorical_patterns,
                    'count': len(rhetorical_patterns),
                    'confidence': min(0.92, 0.6 + (len(rhetorical_patterns) * 0.08))
                })
        
        # 8. NEW: Identify thematic clusters using topic modeling techniques
        thematic_clusters = self._identify_thematic_clusters(words)
        if thematic_clusters:
            patterns.append({
                'type': 'thematic_cluster',
                'clusters': thematic_clusters,
                'count': len(thematic_clusters),
                'confidence': min(0.88, 0.5 + (len(thematic_clusters) * 0.1))
            })
        
        # Update global pattern recognition database
        for pattern in patterns:
            pattern_key = f"{pattern['type']}_{'-'.join(pattern.get('words', []) or [p.get('text', 'unknown') for p in pattern.get('ngrams', [])[:2]])}"
            if pattern_key in self.pattern_recognition:
                self.pattern_recognition[pattern_key] += 1
            else:
                self.pattern_recognition[pattern_key] = 1
        
        return patterns
    
    def _identify_syntactic_patterns(self, words, text):
        """
        Identify syntactic patterns using simple heuristics.
        
        Args:
            words: List of words
            text: Original text
            
        Returns:
            List of identified syntactic patterns
        """
        patterns = []
        
        # Common syntactic pattern markers
        sentence_starters = ["the", "a", "this", "these", "those", "if", "when", "while", "because"]
        conjunctions = ["and", "but", "or", "yet", "so", "for", "nor", "although", "since", "unless"]
        prepositions = ["in", "on", "at", "by", "with", "from", "to", "for", "of", "about", "through"]
        
        # Split into sentences (simple approach)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # Pattern: Sentences starting with the same word/phrase
        sentence_start_counts = Counter()
        for sentence in sentences:
            if sentence:
                words = sentence.lower().split()
                if words:
                    sentence_start_counts[words[0]] += 1
                    if len(words) > 1:
                        sentence_start_counts[f"{words[0]} {words[1]}"] += 1
        
        common_starts = [start for start, count in sentence_start_counts.items() if count > 1]
        if common_starts:
            patterns.append({
                'name': 'repeated_sentence_start',
                'elements': common_starts,
                'count': len(common_starts)
            })
        
        # Pattern: Conjunction usage at beginning of sentences
        sentence_conj_starts = [s for s in sentences if any(s.lower().startswith(conj) for conj in conjunctions)]
        if len(sentence_conj_starts) > 1:
            patterns.append({
                'name': 'conjunction_sentence_start',
                'count': len(sentence_conj_starts),
                'examples': sentence_conj_starts[:2]  # Include a couple examples
            })
        
        # Pattern: Prepositional phrase chains (multiple prepositions in sequence)
        prep_chains = []
        for i, word in enumerate(words[:-2]):
            if word.lower() in prepositions and words[i+2].lower() in prepositions:
                # Found potential prepositional chain: "in the on..."
                chain = ' '.join(words[i:i+4])
                prep_chains.append(chain)
        
        if prep_chains:
            patterns.append({
                'name': 'prepositional_chain',
                'count': len(prep_chains),
                'examples': prep_chains[:2]
            })
        
        return patterns
    
    def _identify_semantic_clusters(self, words):
        """
        Identify semantically related word clusters based on the semantic network.
        
        Args:
            words: List of words from the text
            
        Returns:
            List of semantic clusters found in the text
        """
        clusters = []
        
        # Get unique words in the text
        unique_words = set(words)
        
        # Find words that have connections in our semantic network
        connected_words = [word for word in unique_words if word in self.semantic_network]
        
        # For each connected word, check if any of its related words are also in the text
        for word in connected_words:
            related_in_text = []
            for related_word, strength in self.semantic_network[word].items():
                if related_word in unique_words and strength > 0.2:  # Use a minimum strength threshold
                    related_in_text.append({'word': related_word, 'strength': strength})
            
            # If we have enough related words, consider it a cluster
            if len(related_in_text) >= 2:
                clusters.append({
                    'central_word': word,
                    'related_words': sorted(related_in_text, key=lambda x: x['strength'], reverse=True),
                    'size': len(related_in_text) + 1,  # Include the central word
                    'average_strength': sum(item['strength'] for item in related_in_text) / len(related_in_text)
                })
        
        # Sort clusters by size and average strength
        return sorted(clusters, key=lambda x: (x['size'], x['average_strength']), reverse=True)
    
    def _detect_parallel_structures(self, text):
        """
        Detect parallel grammatical structures in the text.
        
        Args:
            text: Original text
            
        Returns:
            List of parallel structures detected
        """
        parallel_structures = []
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # Common parallel structure markers
        parallel_markers = [
            r'not only .* but also',
            r'either .* or',
            r'neither .* nor',
            r'both .* and',
            r'first .* second',
            r'(less|more|better) .* than',
            r'as .* as',
        ]
        
        # Check for common parallel structure patterns
        for sentence in sentences:
            for marker in parallel_markers:
                if re.search(marker, sentence.lower()):
                    parallel_structures.append({
                        'type': 'marked_parallel',
                        'text': sentence,
                        'marker': marker
                    })
        
        # Check for sentences with similar structure (same starting words) - more advanced
        if len(sentences) >= 2:
            for i in range(len(sentences)-1):
                for j in range(i+1, len(sentences)):
                    words_i = sentences[i].lower().split()
                    words_j = sentences[j].lower().split()
                    
                    # Check if first two words match
                    if len(words_i) >= 2 and len(words_j) >= 2:
                        if words_i[0] == words_j[0] and words_i[1] == words_j[1]:
                            parallel_structures.append({
                                'type': 'similar_start',
                                'text1': sentences[i],
                                'text2': sentences[j],
                                'common_start': f"{words_i[0]} {words_i[1]}"
                            })
        
        return parallel_structures
    
    def _detect_chiasmus(self, words, text):
        """
        Detect chiasmus patterns in the text (A-B-B-A structure).
        This is a rhetorical device where words/ideas are repeated in reverse order.
        
        Args:
            words: List of words from the text
            text: Original text
            
        Returns:
            List of chiasmus patterns detected
        """
        chiasmus_patterns = []
        
        # Split into sentences for analysis
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            # Convert to lowercase and split into words
            sentence_words = re.findall(r'\b\w+\b', sentence.lower())
            
            # Need at least 4 words for a simple chiasmus
            if len(sentence_words) < 4:
                continue
            
            # 1. Look for simple A-B-B-A patterns (exact word repetition)
            for i in range(len(sentence_words) - 3):
                if (sentence_words[i] == sentence_words[i+3] and 
                    sentence_words[i+1] == sentence_words[i+2] and
                    sentence_words[i] != sentence_words[i+1]):
                    chiasmus_patterns.append({
                        'type': 'simple_chiasmus',
                        'text': sentence,
                        'pattern': f"{sentence_words[i]}-{sentence_words[i+1]}-{sentence_words[i+2]}-{sentence_words[i+3]}",
                        'confidence': 0.9
                    })
            
            # 2. Look for more complex chiasmus (A-B-C-B-A)
            for i in range(len(sentence_words) - 4):
                if (sentence_words[i] == sentence_words[i+4] and 
                    sentence_words[i+1] == sentence_words[i+3] and
                    sentence_words[i] != sentence_words[i+1]):
                    chiasmus_patterns.append({
                        'type': 'complex_chiasmus',
                        'text': sentence,
                        'pattern': f"{sentence_words[i]}-{sentence_words[i+1]}-{sentence_words[i+2]}-{sentence_words[i+3]}-{sentence_words[i+4]}",
                        'confidence': 0.95
                    })
            
            # 3. Look for semantic chiasmus (not exact word repetition but similar meaning)
            for i in range(len(sentence_words) - 3):
                word_a = sentence_words[i]
                word_d = sentence_words[i+3]
                
                # Skip common words that would create false positives
                if word_a in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with'} or \
                   word_d in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with'}:
                    continue
                
                # If not exact match, check if semantically related
                if word_a != word_d and word_a in self.semantic_network and word_d in self.semantic_network[word_a]:
                    word_b = sentence_words[i+1]
                    word_c = sentence_words[i+2]
                    
                    if word_b != word_c and word_b in self.semantic_network and word_c in self.semantic_network[word_b]:
                        chiasmus_patterns.append({
                            'type': 'semantic_chiasmus',
                            'text': sentence,
                            'pattern': f"{word_a}-{word_b}-{word_c}-{word_d}",
                            'confidence': 0.7,
                            'semantic_relation': True
                        })
            
            # 4. NEW: Look for A-B-A patterns (partial chiasmus)
            for i in range(len(sentence_words) - 2):
                word_a1 = sentence_words[i]
                word_b = sentence_words[i+1]
                word_a2 = sentence_words[i+2]
                
                # Skip common words and ensure the words are meaningful
                if word_a1 in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with'} or \
                   len(word_a1) < 3 or len(word_a2) < 3:
                    continue
                
                if word_a1 == word_a2 and word_a1 != word_b:
                    chiasmus_patterns.append({
                        'type': 'partial_chiasmus',
                        'text': sentence,
                        'pattern': f"{word_a1}-{word_b}-{word_a2}",
                        'confidence': 0.8
                    })
            
            # 5. NEW: Look for "X for Y, Y for X" patterns (common in chiasmus)
            prep_pattern = re.compile(r'\b(\w+)\s+for\s+(\w+).*\b\2\s+for\s+\1\b', re.IGNORECASE)
            match = prep_pattern.search(sentence)
            if match:
                chiasmus_patterns.append({
                    'type': 'prepositional_chiasmus',
                    'text': sentence,
                    'pattern': f"{match.group(1)}-for-{match.group(2)}-{match.group(2)}-for-{match.group(1)}",
                    'confidence': 0.95
                })
            
            # 6. NEW: Look for reversed phrase patterns, allowing intervening words
            # This is more complex but can catch patterns like "... first ... last ... last ... first ..."
            for i in range(len(sentence_words) - 6):  # Need at least 7 words for this pattern with intervening words
                for j in range(i + 2, len(sentence_words) - 3):  # Look for the second pair
                    if (sentence_words[i] == sentence_words[j+2] and 
                        sentence_words[i+1] == sentence_words[j+1] and
                        sentence_words[i] != sentence_words[i+1] and
                        abs(j - i) < 10):  # Limit the distance to avoid false positives
                        
                        # Skip common words
                        if sentence_words[i] in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}:
                            continue
                            
                        chiasmus_patterns.append({
                            'type': 'separated_chiasmus',
                            'text': sentence,
                            'pattern': f"{sentence_words[i]}-{sentence_words[i+1]}...{sentence_words[j+1]}-{sentence_words[j+2]}",
                            'confidence': 0.85
                        })
        
        return chiasmus_patterns
    
    def _detect_rhetorical_patterns(self, text):
        """
        Detect various rhetorical patterns in the text.
        
        Args:
            text: Original text
            
        Returns:
            List of rhetorical patterns detected
        """
        rhetorical_patterns = []
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # 1. Anaphora (repeated beginning)
        anaphora_candidates = {}
        
        for sentence in sentences:
            words = sentence.lower().split()
            if len(words) >= 2:
                start_phrase = ' '.join(words[:2])  # Consider first two words
                if start_phrase in anaphora_candidates:
                    anaphora_candidates[start_phrase].append(sentence)
                else:
                    anaphora_candidates[start_phrase] = [sentence]
        
        # Only consider it anaphora if the phrase starts at least 2 sentences
        for phrase, instances in anaphora_candidates.items():
            if len(instances) >= 2:
                rhetorical_patterns.append({
                    'type': 'anaphora',
                    'phrase': phrase,
                    'count': len(instances),
                    'examples': instances[:2],  # Just include first two examples
                    'confidence': min(0.9, 0.6 + (len(instances) * 0.1))
                })
        
        # 2. Epistrophe (repeated ending)
        epistrophe_candidates = {}
        
        for sentence in sentences:
            words = sentence.lower().split()
            if len(words) >= 2:
                end_phrase = ' '.join(words[-2:])  # Consider last two words
                if end_phrase in epistrophe_candidates:
                    epistrophe_candidates[end_phrase].append(sentence)
                else:
                    epistrophe_candidates[end_phrase] = [sentence]
        
        # Only consider it epistrophe if the phrase ends at least 2 sentences
        for phrase, instances in epistrophe_candidates.items():
            if len(instances) >= 2:
                rhetorical_patterns.append({
                    'type': 'epistrophe',
                    'phrase': phrase,
                    'count': len(instances),
                    'examples': instances[:2],
                    'confidence': min(0.9, 0.6 + (len(instances) * 0.1))
                })
        
        # 3. Rhetorical questions
        question_markers = [
            r'(why|how|what|when|where|who)\s+\w+\s+\w+',
            r'(is|are|was|were|do|does|did|have|has|had)\s+\w+',
            r'(can|could|should|would|will|may|might)'
        ]
        
        for sentence in sentences:
            if sentence.endswith('?'):
                # Check if it matches a rhetorical question pattern
                is_rhetorical = False
                for marker in question_markers:
                    if re.search(marker, sentence.lower()):
                        is_rhetorical = True
                        break
                
                if is_rhetorical:
                    rhetorical_patterns.append({
                        'type': 'rhetorical_question',
                        'text': sentence,
                        'confidence': 0.7
                    })
        
        # 4. Alliteration (repeated consonant sounds)
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            
            if len(words) >= 3:
                # Group words by first letter
                first_letters = {}
                for word in words:
                    if word and word[0] not in first_letters:
                        first_letters[word[0]] = 1
                    elif word:
                        first_letters[word[0]] += 1
                
                # Check for alliteration (at least 3 words starting with same letter)
                for letter, count in first_letters.items():
                    if count >= 3:
                        # Find examples
                        examples = [word for word in words if word and word[0] == letter][:5]
                        
                        rhetorical_patterns.append({
                            'type': 'alliteration',
                            'letter': letter,
                            'count': count,
                            'examples': examples,
                            'text': sentence,
                            'confidence': min(0.85, 0.5 + (count * 0.1))
                        })
        
        return rhetorical_patterns
    
    def _identify_thematic_clusters(self, words):
        """
        Identify thematic clusters in the text using simple topic modeling techniques.
        
        Args:
            words: List of words from the text
            
        Returns:
            List of thematic clusters found in the text
        """
        # Simple stop words list (would be more comprehensive in production)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'with', 'by', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'this', 'that', 'these', 'those', 'it', 'its', 'we', 'i', 'you', 'they'}
        
        # Filter stop words and keep meaningful words
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 2]
        
        # Count word occurrences
        word_counts = Counter(filtered_words)
        
        # Need enough meaningful words to identify themes
        if len(word_counts) < 5:
            return []
        
        # Get top words by frequency (potential theme indicators)
        top_words = word_counts.most_common(10)
        
        # Initialize thematic clusters around top words
        themes = []
        for core_word, count in top_words:
            # Skip if count is too low
            if count < 2:
                continue
                
            # Find related words in our semantic network
            related_words = []
            if core_word in self.semantic_network:
                # Get directly related words that also appear in the text
                for related, strength in self.semantic_network[core_word].items():
                    if related in word_counts and strength > 0.2:
                        related_words.append({
                            'word': related,
                            'count': word_counts[related],
                            'strength': strength
                        })
            
            # Find co-occurring words (words that appear near the core word)
            co_occurring = {}
            for i, word in enumerate(words):
                if word.lower() == core_word:
                    # Look at surrounding words
                    window_start = max(0, i - 5)
                    window_end = min(len(words), i + 6)
                    
                    for j in range(window_start, window_end):
                        if i != j:
                            context_word = words[j].lower()
                            if context_word not in stop_words and len(context_word) > 2:
                                if context_word in co_occurring:
                                    co_occurring[context_word] += 1 / max(1, abs(i - j))
                                else:
                                    co_occurring[context_word] = 1 / max(1, abs(i - j))
            
            # Only consider co-occurring words that appear at least twice
            significant_co_occurring = {word: score for word, score in co_occurring.items() 
                                       if word_counts.get(word, 0) >= 2}
            
            # Only create a thematic cluster if we have enough related/co-occurring words
            combined_related = related_words + [
                {'word': word, 'count': word_counts.get(word, 0), 'strength': score}
                for word, score in significant_co_occurring.items()
                if word not in [r['word'] for r in related_words]
            ]
            
            if len(combined_related) >= 2:
                # Sort by relevance (combination of frequency and relationship strength)
                sorted_related = sorted(
                    combined_related,
                    key=lambda x: x['count'] * x['strength'],
                    reverse=True
                )
                
                themes.append({
                    'core_word': core_word,
                    'core_count': count,
                    'related_words': sorted_related[:7],  # Limit to top 7 related words
                    'theme_strength': count * sum(item['strength'] for item in sorted_related[:7]),
                    'size': len(sorted_related) + 1  # Count the core word too
                })
        
        # Sort by theme strength
        return sorted(themes, key=lambda x: x['theme_strength'], reverse=True)
    
    def _calculate_neural_linguistic_score(self, words, patterns, recursive_patterns):
        """Calculate a neural linguistic score with enhanced metrics."""
        # Base score from lexical diversity
        unique_ratio = len(set(words)) / max(1, len(words))
        base_score = unique_ratio * 0.5  # Max 0.5 from lexical diversity
        
        # Add points for patterns with weighted importance by type
        pattern_weights = {
            'repetition': 0.05,
            'repeated_2gram': 0.06,
            'repeated_3gram': 0.08,
            'repeated_4gram': 0.10,
            'syntactic': 0.07,
            'semantic_cluster': 0.09,
            'parallelism': 0.08
        }
        
        # Calculate weighted pattern score
        pattern_score = 0
        for pattern in patterns:
            pattern_type = pattern['type']
            weight = pattern_weights.get(pattern_type, 0.05)
            confidence = pattern.get('confidence', 0.5)
            
            # Each pattern contributes a weighted score based on type and confidence
            pattern_score += weight * confidence
        
        # Cap pattern score
        pattern_score = min(0.3, pattern_score)
        
        # Add points for recursive patterns with enhanced scoring
        recursive_weights = {
            'self_reference': 0.05,
            'meta_reference': 0.07,
            'self_modifying': 0.08,
            'logical_recursion': 0.10
        }
        
        recursive_score = 0
        for pattern in recursive_patterns:
            pattern_type = pattern.get('type', '')
            weight = recursive_weights.get(pattern_type, 0.05)
            confidence = pattern.get('confidence', 0.5)
            count = pattern.get('count', 1)
            
            # Higher count patterns get more weight, but with diminishing returns
            count_factor = min(1.0, math.log2(max(1, count) + 1) / 2)
            
            # Combine factors for the recursive score
            recursive_score += weight * confidence * count_factor
        
        # Cap recursive score
        recursive_score = min(0.2, recursive_score)
        
        # Calculate final score
        final_score = base_score + pattern_score + recursive_score
        
        return final_score
    
    def _get_top_words(self, words, limit=5):
        """Get the most significant words from the text."""
        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'are', 'was', 'were'}
        filtered_words = [word for word in words if word.lower() not in common_words and len(word) > 2]
        
        # Count occurrences
        word_counts = Counter(filtered_words)
        
        # Weight by global rarity
        word_scores = {}
        for word, count in word_counts.items():
            # Words that are less common in general language get higher weight
            global_frequency = max(1, self.word_frequencies.get(word.lower(), 0) - count + 1)
            significance = count * (1.0 / math.log2(global_frequency + 1))
            word_scores[word] = significance
        
        # Get top words by significance score
        top_words = sorted(
            [(word, score) for word, score in word_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [{'word': word, 'score': round(score, 3)} for word, score in top_words]
    
    def get_neural_linguistic_metrics(self):
        """Return current neural linguistic processing metrics."""
        return {
            'processed_text_count': self.processed_text_count,
            'total_words_processed': self.total_words_processed,
            'vocabulary_size': len(self.word_frequencies),
            'semantic_network_size': sum(len(connections) for connections in self.semantic_network.values()),
            'pattern_types_recognized': len(self.pattern_recognition),
            'recursive_patterns_found': len(self.recursive_patterns),
            'last_neural_linguistic_score': self.last_neural_linguistic_score,
            'llm_weight': self.llm_weight,
            'average_llm_confidence': np.mean(self.llm_confidence_history) if self.llm_confidence_history else 0
        }
    
    def get_semantic_network(self, word=None, depth=1):
        """Get the semantic network for a word, or the whole network if no word specified."""
        if word is None:
            # Return summary of entire network (could be large)
            return {
                'network_size': len(self.semantic_network),
                'total_connections': sum(len(connections) for connections in self.semantic_network.values()),
                'top_connected_words': sorted(
                    [(w, len(connections)) for w, connections in self.semantic_network.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10 most connected words
            }
        
        # Return network for specific word
        if word not in self.semantic_network:
            return {'word': word, 'connections': [], 'found': False}
        
        # Get direct connections
        direct_connections = [
            {'word': related, 'strength': strength}
            for related, strength in sorted(
                self.semantic_network[word].items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
        result = {
            'word': word,
            'connections': direct_connections,
            'found': True
        }
        
        # Add deeper connections if requested
        if depth > 1 and direct_connections:
            result['deeper_connections'] = {}
            # Limit to top 3 connections to avoid explosion
            for connection in direct_connections[:3]:
                related_word = connection['word']
                result['deeper_connections'][related_word] = self.get_semantic_network(related_word, depth - 1)
        
        return result
    
    def adjust_llm_weight(self, new_weight):
        """Manually adjust the LLM weight."""
        if 0 <= new_weight <= 1:
            self.llm_weight = new_weight
            self.logger.info(f"LLM weight manually adjusted to {self.llm_weight}")
            self._save_state()
            return True
        else:
            self.logger.warning(f"Invalid LLM weight value: {new_weight}, must be between 0 and 1")
            return False
    
    def get_recursive_patterns(self, limit=10):
        """Get the most recent recursive patterns detected."""
        return self.recursive_patterns[-limit:] if self.recursive_patterns else []
    
    def reset_processor(self, keep_semantic_network=False):
        """Reset the processor to initial state, optionally keeping the semantic network."""
        self.pattern_recognition = {}
        self.recursive_patterns = []
        self.word_frequencies = Counter()
        self.processed_text_count = 0
        self.total_words_processed = 0
        self.last_neural_linguistic_score = 0.0
        self.llm_pattern_recognition = {}
        self.llm_confidence_history = []
        
        if not keep_semantic_network:
            self.semantic_network = defaultdict(lambda: defaultdict(float))
            self.llm_semantic_suggestions = defaultdict(dict)
        
        self.logger.info(f"Neural linguistic processor reset (semantic network kept: {keep_semantic_network})")
        self._save_state()
        return True
    
    def set_llm_weight(self, weight: float) -> None:
        """
        Set the LLM weight for the neural linguistic processor.
        
        Args:
            weight: New LLM weight (0.0-1.0)
        """
        self.llm_weight = max(0.0, min(1.0, weight))
        self.logger.info(f"NeuralLinguisticProcessor LLM weight set to {weight}")
    
    def set_nn_weight(self, weight: float) -> None:
        """
        Set the neural network weight for the neural linguistic processor.
        
        Args:
            weight: New neural network weight (0.0-1.0)
        """
        self.nn_weight = max(0.0, min(1.0, weight))
        self.logger.info(f"NeuralLinguisticProcessor neural network weight set to {weight}")

    def _initialize_semantic_network(self):
        """Initialize a basic semantic network with common word relationships."""
        # Define common word relationships as a starting point
        initial_semantic_relationships = {
            'neural': {'network': 0.8, 'brain': 0.7, 'learning': 0.6, 'artificial': 0.6, 'intelligence': 0.7},
            'network': {'neural': 0.8, 'connection': 0.7, 'system': 0.6, 'graph': 0.5, 'linked': 0.6},
            'language': {'text': 0.7, 'word': 0.8, 'speech': 0.7, 'communication': 0.8, 'linguistic': 0.9},
            'pattern': {'recognition': 0.9, 'repeat': 0.6, 'structure': 0.7, 'recurring': 0.8, 'template': 0.7},
            'learning': {'education': 0.6, 'training': 0.7, 'knowledge': 0.8, 'understand': 0.7, 'algorithm': 0.6},
            'data': {'information': 0.8, 'statistics': 0.7, 'analysis': 0.7, 'collection': 0.6, 'processing': 0.7},
            'model': {'representation': 0.7, 'simulation': 0.8, 'structure': 0.6, 'framework': 0.7, 'system': 0.6},
            'algorithm': {'procedure': 0.7, 'process': 0.8, 'method': 0.8, 'computation': 0.7, 'steps': 0.6},
            'system': {'organization': 0.6, 'structure': 0.7, 'framework': 0.7, 'network': 0.6, 'process': 0.6},
            'analysis': {'examination': 0.7, 'study': 0.7, 'investigation': 0.8, 'evaluation': 0.8, 'assessment': 0.8}
        }
        
        # Load relationships into the semantic network
        for word, related_words in initial_semantic_relationships.items():
            for related_word, strength in related_words.items():
                self.semantic_network[word][related_word] = strength
                # Ensure bidirectional connections
                if related_word not in initial_semantic_relationships:
                    self.semantic_network[related_word][word] = strength
        
        self.logger.info(f"Initialized semantic network with {len(initial_semantic_relationships)} base concepts")

# Factory function to get a configured processor
def get_neural_linguistic_processor(language_memory=None, conscious_mirror_language=None):
    """Get a configured Neural Linguistic Processor instance"""
    def recognize_patterns(self, text):
        """
        Recognize patterns in text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of patterns detected
        """
        result = self.process_text(text)
        patterns = []
        
        # Extract patterns from the result
        if isinstance(result, dict):
            if "patterns" in result:
                patterns = result["patterns"]
            elif "neural_patterns" in result:
                patterns = result["neural_patterns"]
            
        # Ensure each pattern has a confidence score
        for pattern in patterns:
            if "confidence" not in pattern:
                pattern["confidence"] = 0.75
                
        return patterns

    def filter_patterns_by_confidence(self, patterns, threshold=0.7):
        """
        Filter patterns by confidence score
        
        Args:
            patterns: List of patterns to filter
            threshold: Confidence threshold
            
        Returns:
            List of patterns with confidence >= threshold
        """
        return [p for p in patterns if p.get("confidence", 0) >= threshold]

    return NeuralLinguisticProcessor(language_memory, conscious_mirror_language) 