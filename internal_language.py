#!/usr/bin/env python3
"""
Internal Language System

This module provides language processing capabilities for the Lumina GUI
system. It handles text processing, language pattern recognition, and 
semantic analysis functions.
"""

import logging
import random
import re
import string
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("internal_language")

class InternalLanguageSystem:
    """
    Internal Language System for Lumina GUI
    
    Provides core language processing capabilities including:
    - Text processing and transformation
    - Semantic analysis
    - Pattern recognition
    - Symbolic encoding/decoding
    """
    
    def __init__(self):
        """Initialize the internal language system"""
        logger.info("Initializing Internal Language System")
        
        # Initialize language processing components
        self.initialized = True
        self.symbols = {
            "circle": "unity, wholeness, infinity",
            "square": "stability, foundation, balance",
            "triangle": "change, transformation, direction",
            "spiral": "growth, evolution, expansion",
            "wave": "flow, rhythm, continuity",
            "grid": "structure, organization, framework",
            "star": "guidance, inspiration, aspiration",
            "hexagon": "harmony, cooperation, efficiency",
            "infinity": "endless potential, limitless possibilities",
            "network": "connection, relationship, complexity"
        }
        
        # Track internal state
        self.active_modules = ["text_processor", "symbolic_encoder", "pattern_detector"]
        self.processing_count = 0
        
        logger.info("Internal Language System initialized")
    
    def process_input(self, text, mode="standard"):
        """
        Process input text through the internal language system
        
        Args:
            text (str): Input text to process
            mode (str): Processing mode (standard, symbolic, technical, creative)
            
        Returns:
            str: Processed text
        """
        self.processing_count += 1
        
        logger.info(f"Processing input (#{self.processing_count}): '{text[:30]}...' in {mode} mode")
        
        if not text:
            return ""
        
        # Apply appropriate processing based on mode
        if mode == "symbolic":
            return self._symbolic_processing(text)
        elif mode == "technical":
            return self._technical_processing(text)
        elif mode == "creative":
            return self._creative_processing(text)
        else:
            return self._standard_processing(text)
    
    def _standard_processing(self, text):
        """Standard text processing with basic semantic analysis"""
        # Extract key entities
        entities = self._extract_entities(text)
        
        # Detect patterns in text
        patterns = self._detect_patterns(text)
        
        # Create processed response
        result = f"Processed: {text}\n"
        
        if entities:
            result += f"Entities: {', '.join(entities)}\n"
        
        if patterns:
            result += f"Patterns: {', '.join(patterns)}"
        
        return result.strip()
    
    def _symbolic_processing(self, text):
        """Process text with symbolic encoding"""
        # Find any matching symbols in text
        matched_symbols = []
        for symbol, meaning in self.symbols.items():
            if symbol.lower() in text.lower():
                matched_symbols.append(f"{symbol} ({meaning})")
        
        # Create base response
        result = f"Symbolic analysis of: {text}\n"
        
        if matched_symbols:
            result += f"Symbols detected: {', '.join(matched_symbols)}"
        else:
            # Suggest a symbol based on the content
            suggested_symbol = random.choice(list(self.symbols.keys()))
            result += f"Suggested symbol: {suggested_symbol} ({self.symbols[suggested_symbol]})"
        
        return result
    
    def _technical_processing(self, text):
        """Process text with technical analysis focus"""
        # Count word frequency
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Calculate basic text statistics
        char_count = len(text)
        word_count = len(words)
        sentence_count = len(re.split(r'[.!?]+', text)) - 1
        
        try:
            avg_word_length = sum(len(word) for word in words) / word_count
        except ZeroDivisionError:
            avg_word_length = 0
        
        result = f"Technical analysis:\n"
        result += f"Characters: {char_count}\n"
        result += f"Words: {word_count}\n"
        result += f"Sentences: {sentence_count}\n"
        result += f"Average word length: {avg_word_length:.2f}\n"
        
        if word_freq:
            top_words = word_freq.most_common(5)
            result += f"Top words: {', '.join(f'{word} ({count})' for word, count in top_words)}"
        
        return result
    
    def _creative_processing(self, text):
        """Process text with creative transformations"""
        # Generate a transformation of the input text
        transformations = [
            self._reverse_words,
            self._alliterate_text,
            self._create_acrostic,
            self._create_word_ladder
        ]
        
        # Apply a random transformation
        transformer = random.choice(transformations)
        transformed = transformer(text)
        
        result = f"Creative transformation:\n{transformed}"
        return result
    
    def _extract_entities(self, text):
        """Extract potential entities from text"""
        # Simple entity extraction using capitalized words
        entity_pattern = r'\b[A-Z][a-zA-Z]+\b'
        entities = re.findall(entity_pattern, text)
        
        # Filter common words that might be capitalized
        common_words = {'I', 'The', 'A', 'An', 'This', 'That', 'These', 'Those'}
        entities = [e for e in entities if e not in common_words]
        
        return entities
    
    def _detect_patterns(self, text):
        """Detect language patterns in text"""
        patterns = []
        
        # Check for question pattern
        if re.search(r'\?', text):
            patterns.append("question")
        
        # Check for definition pattern
        if re.search(r'\bis\b|\bare\b|\bmeans\b', text.lower()):
            patterns.append("definition")
        
        # Check for comparison pattern
        if re.search(r'\blike\b|\bthan\b|\bcompare\b|\bversus\b|\bvs\b', text.lower()):
            patterns.append("comparison")
        
        # Check for sequence pattern
        if re.search(r'\bfirst\b|\bsecond\b|\bnext\b|\bthen\b|\bfinally\b', text.lower()):
            patterns.append("sequence")
        
        # Check for causal pattern
        if re.search(r'\bbecause\b|\bcause\b|\beffect\b|\bresult\b', text.lower()):
            patterns.append("causation")
        
        return patterns
    
    # Creative text transformation methods
    def _reverse_words(self, text):
        """Reverse the words in the text"""
        words = text.split()
        reversed_words = [word[::-1] for word in words]
        return " ".join(reversed_words)
    
    def _alliterate_text(self, text):
        """Create an alliterative version of the text"""
        words = text.split()
        if not words:
            return text
            
        first_letter = words[0][0].lower() if words[0] else 'a'
        
        # Dictionary of alliterative replacements
        alliterative_words = {
            'a': ['amazing', 'awesome', 'abundant', 'accurate', 'adventurous'],
            'b': ['brilliant', 'bold', 'brave', 'bright', 'beautiful'],
            'c': ['creative', 'clear', 'clever', 'crafty', 'curious'],
            'd': ['dynamic', 'daring', 'delightful', 'detailed', 'diligent'],
            'e': ['efficient', 'elegant', 'energetic', 'enormous', 'excellent'],
            'f': ['fantastic', 'fast', 'focused', 'friendly', 'fascinating'],
            'g': ['great', 'genuine', 'graceful', 'grateful', 'glorious'],
            'h': ['helpful', 'happy', 'harmonious', 'humble', 'honest'],
            'i': ['innovative', 'intelligent', 'inspiring', 'insightful', 'impressive'],
            'j': ['joyful', 'jubilant', 'just', 'jolly', 'judicious'],
            'k': ['kind', 'keen', 'knowledgeable', 'kaleidoscopic', 'kinetic'],
            'l': ['logical', 'lively', 'loving', 'luminous', 'limitless'],
            'm': ['magnificent', 'magical', 'mindful', 'motivated', 'masterful'],
            'n': ['natural', 'noble', 'nurturing', 'noteworthy', 'nimble'],
            'o': ['optimistic', 'organized', 'original', 'observant', 'outstanding'],
            'p': ['powerful', 'peaceful', 'precise', 'practical', 'passionate'],
            'q': ['quick', 'quiet', 'qualified', 'quizzical', 'quintessential'],
            'r': ['resourceful', 'radiant', 'reliable', 'remarkable', 'resilient'],
            's': ['smart', 'strong', 'strategic', 'skillful', 'sincere'],
            't': ['thoughtful', 'talented', 'thorough', 'trustworthy', 'timely'],
            'u': ['unique', 'understanding', 'uplifting', 'unifying', 'unstoppable'],
            'v': ['valuable', 'versatile', 'vibrant', 'vigilant', 'virtuous'],
            'w': ['wise', 'wonderful', 'warm', 'willing', 'witty'],
            'x': ['extraordinary', 'exemplary', 'excellent', 'exciting', 'exuberant'],
            'y': ['young', 'yielding', 'yearning', 'youthful', 'yielding'],
            'z': ['zealous', 'zestful', 'zippy', 'zany', 'zen']
        }
        
        letter = first_letter if first_letter in alliterative_words else 'a'
        word_choices = alliterative_words[letter]
        
        # Create alliterative sentence
        result = " ".join(random.choice(word_choices) for _ in range(5))
        return result
    
    def _create_acrostic(self, text):
        """Create an acrostic from the text"""
        words = text.split()
        
        if not words:
            return text
            
        # Get first letters
        first_letters = [word[0].upper() for word in words if word]
        acrostic_word = ''.join(first_letters[:10])  # Limit to 10 letters
        
        # Generate phrases for each letter
        phrases = []
        for letter in acrostic_word:
            if letter in string.ascii_uppercase:
                words_starting_with = [
                    word for word in self._get_common_words() 
                    if word.startswith(letter.lower())
                ]
                
                if words_starting_with:
                    word = random.choice(words_starting_with)
                    phrases.append(f"{letter}... like {word}")
                else:
                    phrases.append(f"{letter}... for something amazing")
            
        return "\n".join(phrases)
    
    def _create_word_ladder(self, text):
        """Create a word ladder from the text"""
        words = [word for word in re.findall(r'\b\w+\b', text) if len(word) > 2]
        
        if not words:
            return text
            
        # Select a subset of words
        selected_words = words[:5] if len(words) > 5 else words
        
        # Create word ladder
        result = ""
        for i, word in enumerate(selected_words):
            result += " " * i + word + "\n"
            
        return result
    
    def _get_common_words(self):
        """Get a list of common English words"""
        common_words = [
            "time", "person", "year", "way", "day",
            "thing", "man", "world", "life", "hand",
            "part", "child", "eye", "woman", "place",
            "work", "week", "case", "point", "government",
            "company", "number", "group", "problem", "fact"
        ]
        return common_words
    
    def get_system_status(self):
        """Return current status of the internal language system"""
        return {
            "status": "active" if self.initialized else "inactive",
            "active_modules": self.active_modules,
            "processing_count": self.processing_count,
            "available_symbols": list(self.symbols.keys()),
            "memory_usage": f"{len(str(self))} bytes"
        }
    
    def analyze_text_semantics(self, text):
        """Perform semantic analysis on text"""
        # This is a simplified simulation of semantic analysis
        
        # Calculate text statistics
        word_count = len(re.findall(r'\b\w+\b', text))
        
        # Determine text complexity (simplified)
        avg_word_length = sum(len(word) for word in text.split()) / max(1, len(text.split()))
        complexity = "high" if avg_word_length > 6 else "medium" if avg_word_length > 4 else "low"
        
        # Determine sentiment (simplified)
        positive_words = ["good", "great", "excellent", "positive", "happy", "joy"]
        negative_words = ["bad", "poor", "negative", "sad", "problem", "difficult"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Determine primary topic (simplified)
        topics = {
            "technology": ["computer", "software", "technology", "digital", "code"],
            "science": ["science", "research", "experiment", "data", "theory"],
            "arts": ["art", "music", "creative", "design", "culture"],
            "business": ["business", "market", "economy", "finance", "company"],
            "health": ["health", "medical", "body", "wellness", "disease"]
        }
        
        topic_scores = {}
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            topic_scores[topic] = score
        
        primary_topic = max(topic_scores.items(), key=lambda x: x[1])[0] if any(topic_scores.values()) else "general"
        
        return {
            "word_count": word_count,
            "complexity": complexity,
            "sentiment": sentiment,
            "primary_topic": primary_topic,
            "topic_scores": topic_scores
        }

# For testing
if __name__ == "__main__":
    language_system = InternalLanguageSystem()
    
    # Test text processing
    test_text = "Neural networks can learn complex patterns from data using backpropagation."
    
    print("Standard processing:")
    print(language_system.process_input(test_text))
    print("\nSymbolic processing:")
    print(language_system.process_input(test_text, mode="symbolic"))
    print("\nTechnical processing:")
    print(language_system.process_input(test_text, mode="technical"))
    print("\nCreative processing:")
    print(language_system.process_input(test_text, mode="creative"))
    
    # Test system status
    print("\nSystem status:")
    print(language_system.get_system_status())
