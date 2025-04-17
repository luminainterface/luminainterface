#!/usr/bin/env python3
"""
English Language Trainer Module

This module provides training capabilities for language modules
by generating sample English text, grammatical structures, and
vocabulary enrichment.
"""

import logging
import random
import string
import re
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("english_trainer")

class EnglishLanguageTrainer:
    """
    English Language Trainer for neural language systems
    
    Provides training data and evaluation for English language processing:
    - Vocabulary generation and enrichment
    - Grammar and syntax training
    - Idiomatic expression training
    - Training data generation for language models
    """
    
    def __init__(self):
        """Initialize the English language trainer"""
        logger.info("Initializing English Language Trainer")
        
        # Initialize training components
        self.vocabulary = self._load_vocabulary()
        self.grammar_patterns = self._load_grammar_patterns()
        self.idioms = self._load_idioms()
        
        # Initialize training metrics
        self.training_sessions = 0
        self.generated_examples = 0
        self.last_training = None
        
        logger.info(f"English Language Trainer initialized with {len(self.vocabulary)} vocabulary items")
    
    def _load_vocabulary(self):
        """Load the core English vocabulary"""
        # In a production system, this would load from a file or database
        # Here we'll use a simplified set of common words
        return {
            "nouns": [
                "algorithm", "network", "data", "model", "system", "pattern", "structure", 
                "layer", "neuron", "processor", "machine", "computer", "intelligence", 
                "function", "node", "graph", "matrix", "vector", "tensor", "gradient",
                "problem", "solution", "code", "memory", "knowledge", "language", "input", "output"
            ],
            "verbs": [
                "process", "compute", "analyze", "learn", "train", "optimize", "recognize", 
                "classify", "predict", "generate", "transform", "extract", "implement", 
                "develop", "design", "iterate", "converge", "encode", "decode", "perform"
            ],
            "adjectives": [
                "neural", "intelligent", "computational", "advanced", "complex", "efficient", 
                "accurate", "powerful", "adaptive", "recursive", "parallel", "distributed", 
                "connected", "automated", "supervised", "unsupervised", "deep", "convolutional",
                "recurrent", "generative"
            ],
            "adverbs": [
                "efficiently", "accurately", "rapidly", "automatically", "intelligently", 
                "precisely", "effectively", "significantly", "computationally", "technically",
                "adaptively", "dynamically", "iteratively", "recursively", "mathematically"
            ],
            "prepositions": [
                "in", "on", "with", "through", "by", "from", "to", "between", "among", 
                "across", "during", "after", "before", "without", "within", "beyond"
            ],
            "determiners": [
                "the", "a", "an", "this", "that", "these", "those", "each", "every", 
                "some", "any", "all", "many", "much", "several", "few", "most"
            ],
            "conjunctions": [
                "and", "or", "but", "yet", "so", "while", "because", "although", 
                "since", "unless", "if", "when", "whenever", "where", "whereas"
            ]
        }
    
    def _load_grammar_patterns(self):
        """Load common English grammar patterns"""
        # Sample grammar patterns for sentence structure
        return [
            # Simple pattern - [Determiner] [Adjective]? [Noun] [Verb] [Adverb]?
            {"pattern": "DET ADJ? NOUN VERB ADV?", 
             "example": "The neural network processes efficiently."},
             
            # Object pattern - [Determiner] [Adjective]? [Noun] [Verb] [Determiner] [Adjective]? [Noun]
            {"pattern": "DET ADJ? NOUN VERB DET ADJ? NOUN", 
             "example": "The deep network classifies the complex patterns."},
             
            # Prepositional pattern - [Determiner] [Noun] [Verb] [Preposition] [Determiner] [Noun]
            {"pattern": "DET NOUN VERB PREP DET NOUN", 
             "example": "The algorithm learns from the data."},
             
            # Compound pattern - [Determiner] [Noun] [Verb] [Determiner] [Noun] [Conjunction] [Verb] [Determiner] [Noun]
            {"pattern": "DET NOUN VERB DET NOUN CONJ VERB DET NOUN", 
             "example": "The system processes the input and generates the output."},
             
            # Complex pattern - [Determiner] [Adjective] [Noun] [Verb] [Adverb] [Preposition] [Determiner] [Adjective] [Noun]
            {"pattern": "DET ADJ NOUN VERB ADV PREP DET ADJ NOUN", 
             "example": "The neural network trains rapidly on the large dataset."}
        ]
    
    def _load_idioms(self):
        """Load common English idioms and expressions"""
        # Technical idioms related to AI and computing
        return [
            {"idiom": "black box", "meaning": "A system where only the inputs and outputs are observable, but internal workings are hidden"},
            {"idiom": "garbage in, garbage out", "meaning": "Poor quality input data leads to poor quality output results"},
            {"idiom": "reinventing the wheel", "meaning": "Duplicating a basic solution that already exists"},
            {"idiom": "down the rabbit hole", "meaning": "Getting deeply involved in a complex or confusing problem"},
            {"idiom": "connect the dots", "meaning": "Understanding how different pieces of information relate to each other"},
            {"idiom": "needle in a haystack", "meaning": "Something very difficult to find among many other things"},
            {"idiom": "thinking outside the box", "meaning": "Thinking creatively, beyond conventional approaches"},
            {"idiom": "cutting edge", "meaning": "The most advanced stage of development in a field"},
            {"idiom": "hill climbing", "meaning": "Iteratively making incremental improvements to a solution"},
            {"idiom": "vanishing gradient", "meaning": "When gradients become too small during neural network training"}
        ]
    
    def generate_training_data(self, count=10, topic="neural_networks"):
        """
        Generate training data in English based on the specified topic
        
        Args:
            count (int): Number of training examples to generate
            topic (str): Topic to focus on (neural_networks, machine_learning, general_ai)
            
        Returns:
            list: Generated training examples
        """
        self.training_sessions += 1
        self.last_training = datetime.now()
        
        training_data = []
        
        for _ in range(count):
            # Generate an example based on grammar patterns
            pattern = random.choice(self.grammar_patterns)
            example = self._generate_from_pattern(pattern["pattern"], topic)
            
            # Record the example
            training_data.append({
                "text": example,
                "pattern": pattern["pattern"],
                "topic": topic,
                "timestamp": self.last_training.isoformat()
            })
            
            self.generated_examples += 1
        
        logger.info(f"Generated {count} training examples on topic: {topic}")
        return training_data
    
    def _generate_from_pattern(self, pattern, topic):
        """Generate a sentence from a grammatical pattern"""
        # Convert pattern string to list of token types
        token_types = pattern.split()
        
        # Replace each token type with a random word of that type
        sentence = []
        for token_type in token_types:
            # Handle optional elements (marked with ?)
            if token_type.endswith('?'):
                if random.random() < 0.5:  # 50% chance to include
                    token_type = token_type[:-1]  # Remove the ?
                else:
                    continue  # Skip this token
            
            # Generate appropriate word based on token type
            if token_type == "DET":
                sentence.append(random.choice(self.vocabulary["determiners"]))
            elif token_type == "NOUN":
                sentence.append(random.choice(self.vocabulary["nouns"]))
            elif token_type == "VERB":
                verb = random.choice(self.vocabulary["verbs"])
                # Simple present tense modification
                if random.random() < 0.3:  # 30% chance for third person singular
                    if verb.endswith('s'):
                        verb += 'es'
                    elif verb.endswith('y'):
                        verb = verb[:-1] + 'ies'
                    else:
                        verb += 's'
                sentence.append(verb)
            elif token_type == "ADJ":
                sentence.append(random.choice(self.vocabulary["adjectives"]))
            elif token_type == "ADV":
                sentence.append(random.choice(self.vocabulary["adverbs"]))
            elif token_type == "PREP":
                sentence.append(random.choice(self.vocabulary["prepositions"]))
            elif token_type == "CONJ":
                sentence.append(random.choice(self.vocabulary["conjunctions"]))
        
        # Join words into a sentence and capitalize first letter
        result = " ".join(sentence)
        result = result[0].upper() + result[1:] + "."
        
        return result
    
    def enrich_vocabulary(self, new_words, category):
        """
        Add new words to the vocabulary
        
        Args:
            new_words (list): List of new words to add
            category (str): Word category (nouns, verbs, adjectives, etc.)
            
        Returns:
            bool: Success status
        """
        if category not in self.vocabulary:
            logger.error(f"Invalid vocabulary category: {category}")
            return False
        
        # Filter words already in vocabulary
        existing_words = set(self.vocabulary[category])
        filtered_words = [word for word in new_words if word not in existing_words]
        
        # Add new words
        self.vocabulary[category].extend(filtered_words)
        
        logger.info(f"Added {len(filtered_words)} new words to {category} vocabulary")
        return True
    
    def evaluate_text(self, text):
        """
        Evaluate English text quality and provide metrics
        
        Args:
            text (str): Text to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        # Basic text analysis
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        unique_words = len(set(words))
        
        # Measure vocabulary usage
        category_usage = {}
        for category, word_list in self.vocabulary.items():
            category_words = set(word_list)
            matches = [word for word in words if word in category_words]
            category_usage[category] = len(matches)
        
        # Sentence complexity
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Grammar pattern matching (simplified)
        grammar_matches = []
        for pattern in self.grammar_patterns:
            if any(re.search(r'\b' + re.escape(word) + r'\b', text.lower()) for word in pattern["example"].lower().split()):
                grammar_matches.append(pattern["pattern"])
        
        return {
            "word_count": word_count,
            "unique_words": unique_words,
            "lexical_diversity": unique_words / max(1, word_count),
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "category_usage": category_usage,
            "grammar_patterns_detected": grammar_matches,
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    def teach_idiom(self):
        """
        Teach a random idiom with explanation
        
        Returns:
            dict: Idiom and its meaning
        """
        idiom = random.choice(self.idioms)
        logger.info(f"Teaching idiom: {idiom['idiom']}")
        return idiom
    
    def get_training_stats(self):
        """
        Get statistics about the training sessions
        
        Returns:
            dict: Training statistics
        """
        return {
            "training_sessions": self.training_sessions,
            "examples_generated": self.generated_examples,
            "vocabulary_size": {category: len(words) for category, words in self.vocabulary.items()},
            "total_vocabulary": sum(len(words) for words in self.vocabulary.values()),
            "last_training": self.last_training.isoformat() if self.last_training else None
        }
    
    def generate_technical_text(self, topic="neural_networks", paragraphs=1):
        """
        Generate technical English text on a specific topic
        
        Args:
            topic (str): Technical topic to focus on
            paragraphs (int): Number of paragraphs to generate
            
        Returns:
            str: Generated technical text
        """
        # Topics and their associated vocabulary
        topics = {
            "neural_networks": {
                "key_terms": ["neural network", "layers", "neurons", "activation", "weights", "backpropagation", 
                             "gradient descent", "deep learning", "training", "inference", "forward pass"],
                "structures": ["convolutional", "recurrent", "feed-forward", "transformer", "generative"]
            },
            "machine_learning": {
                "key_terms": ["algorithm", "model", "features", "classification", "regression", "clustering", 
                             "supervised", "unsupervised", "reinforcement", "cross-validation", "bias-variance"],
                "structures": ["decision tree", "random forest", "support vector machine", "k-means", "bayesian"]
            },
            "data_science": {
                "key_terms": ["dataset", "preprocessing", "analysis", "visualization", "statistics", "pipeline", 
                             "correlation", "distribution", "feature engineering", "dimensionality reduction"],
                "structures": ["exploratory analysis", "hypothesis testing", "regression analysis", "time series"]
            }
        }
        
        # Default to neural networks if topic not found
        topic_terms = topics.get(topic, topics["neural_networks"])
        
        result = []
        for _ in range(paragraphs):
            # Generate 3-6 sentences for this paragraph
            sentences = []
            sentence_count = random.randint(3, 6)
            
            for _ in range(sentence_count):
                # Pick a structure for this sentence
                if random.random() < 0.7:  # 70% chance to use a pattern
                    pattern = random.choice(self.grammar_patterns)
                    sentence = self._generate_from_pattern(pattern["pattern"], topic)
                else:
                    # Generate a technical definition sentence
                    term = random.choice(topic_terms["key_terms"])
                    structure = random.choice(topic_terms["structures"])
                    sentence = f"The {structure} {term} is {random.choice(['critical', 'essential', 'important', 'fundamental', 'key'])} for {random.choice(['effective', 'efficient', 'robust', 'accurate', 'optimal'])} {random.choice(['learning', 'processing', 'analysis', 'performance', 'results'])}."
                
                sentences.append(sentence)
            
            # Join sentences into a paragraph
            paragraph = " ".join(sentences)
            result.append(paragraph)
        
        self.generated_examples += paragraphs
        logger.info(f"Generated {paragraphs} technical paragraphs on {topic}")
        
        return "\n\n".join(result)

# For testing
if __name__ == "__main__":
    trainer = EnglishLanguageTrainer()
    
    # Generate and display training examples
    examples = trainer.generate_training_data(5)
    print("Generated Training Examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['text']}")
    
    # Generate technical text
    print("\nGenerated Technical Text:")
    technical_text = trainer.generate_technical_text(paragraphs=2)
    print(technical_text)
    
    # Evaluate the generated text
    print("\nText Evaluation:")
    evaluation = trainer.evaluate_text(technical_text)
    print(f"Word count: {evaluation['word_count']}")
    print(f"Unique words: {evaluation['unique_words']}")
    print(f"Lexical diversity: {evaluation['lexical_diversity']:.2f}")
    print(f"Sentence count: {evaluation['sentence_count']}")
    
    # Get training stats
    print("\nTraining Statistics:")
    stats = trainer.get_training_stats()
    print(f"Training sessions: {stats['training_sessions']}")
    print(f"Examples generated: {stats['examples_generated']}")
    print(f"Total vocabulary: {stats['total_vocabulary']} words") 