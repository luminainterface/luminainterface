"""
Auto Learning Engine for AutoWiki System
Provides pattern recognition and learning capabilities
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AutoLearningEngine:
    def __init__(self):
        self.logger = logger
        self._initialized = False
        self.vectorizer = None
        self.patterns = {}
        self.learning_history = {}
        self._initialize()
        
    def _initialize(self):
        """Initialize learning engine"""
        try:
            # Initialize text vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english'
            )
            
            self._initialized = True
            logger.info("Learning engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize learning engine: {str(e)}")
            raise
            
    def learn_patterns(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Learn patterns from content"""
        try:
            # Vectorize content
            content_vector = self.vectorizer.fit_transform([content])
            
            # Extract key terms
            feature_names = self.vectorizer.get_feature_names_out()
            scores = content_vector.toarray()[0]
            key_terms = [
                (term, score) for term, score in zip(feature_names, scores)
                if score > 0
            ]
            key_terms.sort(key=lambda x: x[1], reverse=True)
            
            # Store pattern
            pattern_id = f"pat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            pattern = {
                'vector': content_vector,
                'key_terms': key_terms[:20],  # Top 20 terms
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            self.patterns[pattern_id] = pattern
            
            # Store in learning history
            self.learning_history[pattern_id] = {
                'content_length': len(content.split()),
                'unique_terms': len(key_terms),
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'id': pattern_id,
                'key_terms': key_terms[:20],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to learn patterns: {str(e)}")
            raise
            
    def find_similar_patterns(self, content: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find patterns similar to given content"""
        try:
            # Vectorize input content
            content_vector = self.vectorizer.transform([content])
            
            # Compare with stored patterns
            similar_patterns = []
            for pattern_id, pattern in self.patterns.items():
                similarity = float(cosine_similarity(content_vector, pattern['vector'])[0][0])
                if similarity >= threshold:
                    similar_patterns.append({
                        'id': pattern_id,
                        'similarity': similarity,
                        'key_terms': pattern['key_terms'],
                        'metadata': pattern['metadata'],
                        'timestamp': pattern['timestamp']
                    })
            
            # Sort by similarity
            similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_patterns
            
        except Exception as e:
            logger.error(f"Failed to find similar patterns: {str(e)}")
            raise
            
    def analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure and patterns"""
        try:
            # Split into sections
            sections = [s.strip() for s in content.split('\n\n') if s.strip()]
            
            # Analyze each section
            section_analysis = []
            for i, section in enumerate(sections):
                # Vectorize section
                section_vector = self.vectorizer.transform([section])
                
                # Get key terms
                feature_names = self.vectorizer.get_feature_names_out()
                scores = section_vector.toarray()[0]
                key_terms = [
                    (term, score) for term, score in zip(feature_names, scores)
                    if score > 0
                ]
                key_terms.sort(key=lambda x: x[1], reverse=True)
                
                # Calculate metrics
                section_analysis.append({
                    'index': i,
                    'length': len(section.split()),
                    'key_terms': key_terms[:10],  # Top 10 terms
                    'complexity': self._calculate_complexity(section),
                    'coherence': self._calculate_coherence(section)
                })
            
            return {
                'section_count': len(sections),
                'total_length': sum(s['length'] for s in section_analysis),
                'section_analysis': section_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze content structure: {str(e)}")
            raise
            
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        try:
            words = text.split()
            if not words:
                return 0.0
                
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Sentence length
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            avg_sentence_length = len(words) / max(len(sentences), 1)
            
            # Normalize scores
            word_score = min(avg_word_length / 10, 1.0)  # Max word length of 10
            sentence_score = min(avg_sentence_length / 30, 1.0)  # Max sentence length of 30
            
            return (word_score + sentence_score) / 2
            
        except Exception as e:
            logger.error(f"Failed to calculate complexity: {str(e)}")
            return 0.0
            
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score"""
        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) < 2:
                return 1.0  # Single sentence is coherent
                
            # Vectorize sentences
            sentence_vectors = self.vectorizer.fit_transform(sentences)
            
            # Calculate similarity between consecutive sentences
            similarities = []
            for i in range(len(sentences) - 1):
                similarity = cosine_similarity(
                    sentence_vectors[i:i+1],
                    sentence_vectors[i+1:i+2]
                )[0][0]
                similarities.append(similarity)
                
            # Average similarity
            return float(np.mean(similarities))
            
        except Exception as e:
            logger.error(f"Failed to calculate coherence: {str(e)}")
            return 0.0
            
    def get_learning_history(self) -> List[Dict[str, Any]]:
        """Get history of learning events"""
        return [
            {
                'id': pattern_id,
                **history_data
            }
            for pattern_id, history_data in self.learning_history.items()
        ]
        
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get specific pattern by ID"""
        return self.patterns.get(pattern_id)
        
    def clear_patterns(self):
        """Clear learned patterns"""
        self.patterns = {}
        self.learning_history = {} 