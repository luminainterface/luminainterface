"""
Suggestion Engine for AutoWiki System
Provides content analysis and improvement suggestions
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class SuggestionEngine:
    def __init__(self):
        self.logger = logger
        self._initialized = False
        self.model = None
        self.tokenizer = None
        self.suggestions = {}
        self._initialize()
        
    def _initialize(self):
        """Initialize suggestion engine"""
        try:
            # Initialize transformers model for content quality analysis
            model_name = "bert-base-uncased"  # Using BERT as base model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self._initialized = True
            logger.info("Suggestion engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize suggestion engine: {str(e)}")
            raise
            
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content and generate quality metrics"""
        try:
            # Basic metrics
            word_count = len(content.split())
            sentence_count = len([s for s in content.split('.') if s.strip()])
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Readability score (simplified Flesch-Kincaid)
            readability = self._calculate_readability(content)
            
            # Sentiment analysis using transformer model
            sentiment = self._analyze_sentiment(content)
            
            return {
                'metrics': {
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'avg_sentence_length': avg_sentence_length,
                    'readability_score': readability,
                    'sentiment_score': sentiment
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze content: {str(e)}")
            raise
            
    def generate_suggestions(self, content: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on content analysis"""
        try:
            suggestions = []
            
            # Check word count
            if metrics['metrics']['word_count'] < 100:
                suggestions.append({
                    'type': 'length',
                    'severity': 'high',
                    'message': 'Content is too short. Consider adding more detail.',
                    'context': None
                })
                
            # Check average sentence length
            if metrics['metrics']['avg_sentence_length'] > 25:
                suggestions.append({
                    'type': 'readability',
                    'severity': 'medium',
                    'message': 'Sentences are too long. Consider breaking them down.',
                    'context': None
                })
                
            # Check readability score
            if metrics['metrics']['readability_score'] < 60:
                suggestions.append({
                    'type': 'complexity',
                    'severity': 'medium',
                    'message': 'Content may be too complex. Consider simplifying.',
                    'context': None
                })
                
            # Check sentiment
            if metrics['metrics']['sentiment_score'] < 0:
                suggestions.append({
                    'type': 'tone',
                    'severity': 'low',
                    'message': 'Content tone is negative. Consider revising if inappropriate.',
                    'context': None
                })
                
            # Store suggestions
            suggestion_id = f"sug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.suggestions[suggestion_id] = {
                'suggestions': suggestions,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {str(e)}")
            raise
            
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        try:
            words = text.split()
            sentences = [s for s in text.split('.') if s.strip()]
            syllables = sum(self._count_syllables(word) for word in words)
            
            if not words or not sentences:
                return 0.0
                
            # Flesch Reading Ease score
            words_per_sentence = len(words) / len(sentences)
            syllables_per_word = syllables / len(words)
            score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Failed to calculate readability: {str(e)}")
            return 0.0
            
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
            
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
            
        return count
        
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze text sentiment using transformer model"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            scores = outputs.logits.softmax(dim=1)
            sentiment_score = float(scores[0][1] - scores[0][0])  # Positive - Negative
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {str(e)}")
            return 0.0
            
    def get_suggestion_history(self) -> List[Dict[str, Any]]:
        """Get history of suggestions"""
        return [
            {
                'id': suggestion_id,
                **suggestion_data
            }
            for suggestion_id, suggestion_data in self.suggestions.items()
        ]
        
    def get_suggestion(self, suggestion_id: str) -> Optional[Dict[str, Any]]:
        """Get specific suggestion by ID"""
        return self.suggestions.get(suggestion_id)
        
    def clear_suggestion_history(self):
        """Clear suggestion history"""
        self.suggestions = {} 