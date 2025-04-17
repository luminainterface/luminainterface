#!/usr/bin/env python3
"""
English Language Trainer Module

Provides training capabilities for language modules by generating sample English text,
grammatical structures, and vocabulary enrichment.
"""

import random
import logging
import uuid
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("english_language_trainer")


class EnglishLanguageTrainer:
    """
    English Language Trainer Class
    
    Provides training capabilities for language systems through:
    - Vocabulary generation
    - Grammar and syntax training
    - Idiomatic expression training
    - Training data generation for language models
    """
    
    def __init__(self):
        """Initialize the English Language Trainer"""
        # Load vocabulary, grammar patterns, and idioms
        self.vocabulary = self._load_vocabulary()
        self.grammar_patterns = self._load_grammar_patterns()
        self.idioms = self._load_idioms()
        
        # Training statistics
        self.training_stats = {
            "sessions": 0,
            "examples_generated": 0,
            "start_time": datetime.now().isoformat()
        }
        
        logger.info("English Language Trainer initialized")
    
    def _load_vocabulary(self):
        """Load the vocabulary database"""
        # In a production system, this would load from files or a database
        # For this implementation, we'll use a simple in-memory dictionary
        
        return {
            "nouns": [
                "memory", "language", "system", "computer", "network", 
                "intelligence", "robot", "machine", "human", "person", 
                "concept", "knowledge", "information", "data", "algorithm",
                "learning", "training", "model", "pattern", "structure"
            ],
            "verbs": [
                "remember", "forget", "learn", "process", "understand", 
                "think", "analyze", "synthesize", "store", "retrieve", 
                "compute", "calculate", "predict", "generate", "train",
                "communicate", "connect", "recognize", "identify", "classify"
            ],
            "adjectives": [
                "intelligent", "neural", "cognitive", "linguistic", "semantic", 
                "syntactic", "computational", "algorithmic", "structured", "recursive", 
                "complex", "simple", "efficient", "robust", "scalable",
                "adaptive", "dynamic", "static", "interactive", "responsive"
            ],
            "adverbs": [
                "quickly", "efficiently", "intelligently", "recursively", "adaptively", 
                "interactively", "responsively", "computationally", "algorithmically", "dynamically"
            ],
            "prepositions": [
                "in", "on", "with", "by", "through", 
                "around", "between", "among", "across", "within"
            ],
            "determiners": [
                "the", "a", "an", "this", "that", 
                "these", "those", "some", "any", "all"
            ],
            "conjunctions": [
                "and", "or", "but", "so", "because", 
                "while", "if", "unless", "although", "since"
            ]
        }
    
    def _load_grammar_patterns(self):
        """Load common English grammar patterns"""
        # These patterns use placeholders that can be replaced with vocabulary
        return [
            {
                "name": "Simple Declarative",
                "pattern": "{subject} {verb} {object}.",
                "example": "The system processes information."
            },
            {
                "name": "Compound Sentence",
                "pattern": "{subject} {verb} {object}, and {subject} {verb} {object}.",
                "example": "The network learns patterns, and the model generates predictions."
            },
            {
                "name": "Complex Sentence",
                "pattern": "When {subject} {verb} {object}, {subject} {verb} {object}.",
                "example": "When the algorithm analyzes data, the system generates insights."
            },
            {
                "name": "Passive Voice",
                "pattern": "{object} {be} {verb_pp} by {subject}.",
                "example": "The data is processed by the system."
            },
            {
                "name": "Question",
                "pattern": "How does {subject} {verb} {object}?",
                "example": "How does the model recognize patterns?"
            },
            {
                "name": "Conditional",
                "pattern": "If {subject} {verb} {object}, then {subject} will {verb} {object}.",
                "example": "If the system learns efficiently, then it will perform better."
            },
            {
                "name": "Comparative",
                "pattern": "{subject} is more {adjective} than {subject}.",
                "example": "Neural networks are more adaptive than rule-based systems."
            }
        ]
    
    def _load_idioms(self):
        """Load a list of technical idioms"""
        # These are common expressions used in technical contexts
        return [
            {
                "idiom": "connecting the dots",
                "meaning": "Understanding the relationships between different pieces of information"
            },
            {
                "idiom": "thinking outside the box",
                "meaning": "Being creative and not limited by conventional approaches"
            },
            {
                "idiom": "at the cutting edge",
                "meaning": "Using the most advanced or innovative methods"
            },
            {
                "idiom": "information overload",
                "meaning": "Having too much information to process effectively"
            },
            {
                "idiom": "garbage in, garbage out",
                "meaning": "The quality of output is determined by the quality of input"
            },
            {
                "idiom": "reinventing the wheel",
                "meaning": "Unnecessarily duplicating a basic method that already exists"
            },
            {
                "idiom": "connecting the dots",
                "meaning": "Identifying the pattern or relationship between different ideas or events"
            },
            {
                "idiom": "getting up to speed",
                "meaning": "Learning enough to become proficient or productive"
            },
            {
                "idiom": "low-hanging fruit",
                "meaning": "Tasks or goals that are easily achievable"
            },
            {
                "idiom": "drill down",
                "meaning": "Examine something in detail, focusing on a specific aspect"
            }
        ]
    
    def generate_training_data(self, topic=None, count=10):
        """
        Generate training examples for language modules
        
        Args:
            topic: Optional topic to focus on (str)
            count: Number of examples to generate (int)
        
        Returns:
            List of generated sentences
        """
        generated_examples = []
        
        # Use topic-specific vocabulary if provided
        topic_vocabulary = self._get_topic_vocabulary(topic) if topic else None
        
        # Generate requested number of examples
        for _ in range(count):
            # Select a random grammar pattern
            pattern = random.choice(self.grammar_patterns)
            
            # Generate a sentence from the pattern
            sentence = self._generate_from_pattern(pattern, topic_vocabulary)
            generated_examples.append(sentence)
        
        # Update statistics
        self.training_stats["sessions"] += 1
        self.training_stats["examples_generated"] += len(generated_examples)
        
        logger.info(f"Generated {len(generated_examples)} training examples")
        if topic:
            logger.info(f"Training focused on topic: {topic}")
        
        return generated_examples
    
    def _get_topic_vocabulary(self, topic):
        """
        Get vocabulary specific to a topic
        
        Args:
            topic: The topic to focus on
        
        Returns:
            Dictionary of topic-specific vocabulary
        """
        # This is a simple implementation - in a real system, this could use
        # word embeddings, knowledge graphs, or other NLP techniques
        
        topic_words = {
            "learning": {
                "nouns": ["student", "teacher", "knowledge", "skill", "education", "brain", "practice"],
                "verbs": ["learn", "study", "practice", "teach", "understand", "comprehend", "acquire"],
                "adjectives": ["educational", "cognitive", "academic", "intelligent", "comprehensive"]
            },
            "memory": {
                "nouns": ["recall", "storage", "retrieval", "brain", "capacity", "recollection", "experience"],
                "verbs": ["remember", "store", "recall", "forget", "retain", "memorize", "recognize"],
                "adjectives": ["memorable", "forgettable", "long-term", "short-term", "episodic", "semantic"]
            },
            "language": {
                "nouns": ["word", "sentence", "grammar", "syntax", "vocabulary", "expression", "speech"],
                "verbs": ["speak", "communicate", "express", "convey", "articulate", "describe", "explain"],
                "adjectives": ["verbal", "linguistic", "grammatical", "expressive", "communicative"]
            },
            "consciousness": {
                "nouns": ["awareness", "perception", "mind", "cognition", "thought", "experience", "self"],
                "verbs": ["perceive", "think", "feel", "experience", "reflect", "contemplate", "realize"],
                "adjectives": ["aware", "conscious", "cognitive", "perceptual", "introspective", "reflexive"]
            },
            "technology": {
                "nouns": ["computer", "algorithm", "system", "device", "network", "hardware", "software"],
                "verbs": ["compute", "process", "analyze", "develop", "implement", "deploy", "upgrade"],
                "adjectives": ["digital", "electronic", "computational", "automated", "technical"]
            }
        }
        
        # If we have vocabulary for this topic, return it
        if topic.lower() in topic_words:
            # Make a copy of the full vocabulary
            topic_specific = self.vocabulary.copy()
            
            # Prioritize topic-specific words
            for word_type, words in topic_words[topic.lower()].items():
                if word_type in topic_specific:
                    # Combine topic words with general vocabulary, putting topic words first
                    topic_specific[word_type] = words + [w for w in topic_specific[word_type] if w not in words]
            
            return topic_specific
        
        # Fall back to general vocabulary
        return self.vocabulary
    
    def _generate_from_pattern(self, pattern, vocabulary=None):
        """
        Generate a sentence from a grammar pattern
        
        Args:
            pattern: Grammar pattern dictionary
            vocabulary: Optional specialized vocabulary
        
        Returns:
            Generated sentence (str)
        """
        # Use provided vocabulary or fall back to default
        vocab = vocabulary or self.vocabulary
        
        # Get the pattern template
        template = pattern["pattern"]
        
        # Common replacements
        replacements = {
            "{subject}": f"{random.choice(vocab['determiners'])} {random.choice(vocab['adjectives'])} {random.choice(vocab['nouns'])}",
            "{object}": f"{random.choice(vocab['determiners'])} {random.choice(vocab['adjectives'])} {random.choice(vocab['nouns'])}",
            "{verb}": random.choice(vocab['verbs']),
            "{verb_pp}": f"{random.choice(vocab['verbs'])}ed",  # Simple past participle
            "{be}": random.choice(["is", "was", "has been"]),
            "{adjective}": random.choice(vocab['adjectives'])
        }
        
        # Apply replacements
        for placeholder, replacement in replacements.items():
            template = template.replace(placeholder, replacement)
        
        return template
    
    def enrich_vocabulary(self, word_type, new_words):
        """
        Add new words to the vocabulary
        
        Args:
            word_type: Type of words to add (nouns, verbs, etc.)
            new_words: List of new words to add
        
        Returns:
            Number of words added
        """
        if word_type not in self.vocabulary:
            logger.warning(f"Unknown word type: {word_type}")
            return 0
        
        # Only add words that aren't already in the vocabulary
        existing_words = set(self.vocabulary[word_type])
        words_to_add = [word for word in new_words if word not in existing_words]
        
        # Add new words
        self.vocabulary[word_type].extend(words_to_add)
        
        logger.info(f"Added {len(words_to_add)} new {word_type} to vocabulary")
        return len(words_to_add)
    
    def evaluate_text(self, text):
        """
        Evaluate the quality and complexity of provided text
        
        Args:
            text: Text to evaluate (str)
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Simple text evaluation metrics
        words = text.lower().split()
        unique_words = set(words)
        
        # Count sentences (simple approximation)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Detect grammar patterns (simple implementation)
        detected_patterns = []
        for pattern in self.grammar_patterns:
            # This is a very simplistic check - a real implementation would use NLP
            if pattern["example"].lower() in text.lower():
                detected_patterns.append(pattern["name"])
        
        # Calculate lexical diversity (unique words / total words)
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        evaluation = {
            "word_count": len(words),
            "unique_words": len(unique_words),
            "sentence_count": len(sentences),
            "lexical_diversity": round(lexical_diversity, 2),
            "detected_patterns": detected_patterns,
            "evaluation_id": str(uuid.uuid4())
        }
        
        logger.info(f"Evaluated text ({len(words)} words, {len(sentences)} sentences)")
        return evaluation
    
    def teach_idiom(self):
        """
        Teach a random technical idiom
        
        Returns:
            Dictionary containing the idiom and its meaning
        """
        idiom = random.choice(self.idioms)
        
        logger.info(f"Teaching idiom: {idiom['idiom']}")
        return idiom
    
    def get_training_stats(self):
        """
        Get statistics about training sessions
        
        Returns:
            Dictionary of training statistics
        """
        current_stats = {
            "sessions": self.training_stats["sessions"],
            "examples_generated": self.training_stats["examples_generated"],
            "start_time": self.training_stats["start_time"],
            "vocabulary_size": sum(len(words) for words in self.vocabulary.values()),
            "grammar_patterns": len(self.grammar_patterns),
            "idioms": len(self.idioms)
        }
        
        return current_stats


# Testing
if __name__ == "__main__":
    # Create trainer
    trainer = EnglishLanguageTrainer()
    
    # Generate training examples
    examples = trainer.generate_training_data(topic="learning", count=5)
    print("\nGenerated Training Examples:")
    for i, example in enumerate(examples):
        print(f"{i+1}. {example}")
    
    # Generate technical text
    technical_text = " ".join(trainer.generate_training_data(topic="technology", count=3))
    print("\nGenerated Technical Text:")
    print(technical_text)
    
    # Evaluate the generated text
    evaluation = trainer.evaluate_text(technical_text)
    print("\nText Evaluation:")
    for key, value in evaluation.items():
        if key != "evaluation_id":
            print(f"{key}: {value}")
    
    # Get training statistics
    stats = trainer.get_training_stats()
    print("\nTraining Statistics:")
    for key, value in stats.items():
        if key != "start_time":
            print(f"{key}: {value}") 