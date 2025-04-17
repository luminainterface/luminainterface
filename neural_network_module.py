#!/usr/bin/env python3
"""
Neural Network Module for Mistral Weighted Chat System

This module provides a neural network implementation for text processing
that can be integrated with the Mistral weighted chat system. It implements
a simple neural network for text analysis with weights that can be adjusted.
"""

import os
import sys
import logging
import time
import numpy as np
import json
from pathlib import Path
from collections import defaultdict, Counter
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/neural_network.log", mode='a')
    ]
)
logger = logging.getLogger("NeuralNetworkModule")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Running in limited mode.")
    TORCH_AVAILABLE = False


@dataclass
class NeuralProcessingState:
    """Class to store the state of neural processing"""
    text: str = ""
    embeddings: Optional[np.ndarray] = None
    activations: Optional[torch.Tensor] = None
    resonance: float = 0.0
    concepts: Dict[str, float] = field(default_factory=dict)
    word_frequencies: Counter = field(default_factory=Counter)
    token_count: int = 0


class SimpleTextEncoder:
    """Simple text encoder for when PyTorch is not available"""
    
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.word_vectors = {}
        
    def encode(self, text: str) -> np.ndarray:
        """Encode text into a simple embedding"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return np.zeros(self.embedding_dim)
            
        # Generate word vectors for unknown words
        for word in words:
            if word not in self.word_vectors:
                # Create a pseudo-random but deterministic vector for each word
                seed = sum(ord(c) for c in word)
                np.random.seed(seed)
                self.word_vectors[word] = np.random.randn(self.embedding_dim)
                
        # Average word vectors
        vectors = [self.word_vectors[word] for word in words if word in self.word_vectors]
        if not vectors:
            return np.zeros(self.embedding_dim)
            
        return np.mean(vectors, axis=0)


class TextEncoder(nn.Module):
    """Text encoder using a simple neural network"""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, word_indices):
        """Forward pass through the encoder"""
        embeddings = self.embedding(word_indices)
        # Average the embeddings for all words
        mean_embedding = torch.mean(embeddings, dim=0)
        encoded = self.encoder(mean_embedding)
        return encoded


class ConceptNetwork(nn.Module):
    """Neural network for concept identification in text"""
    
    def __init__(self, input_dim=256, concept_dim=64, num_concepts=32):
        super().__init__()
        self.concept_projection = nn.Linear(input_dim, concept_dim)
        self.concept_gates = nn.Linear(concept_dim, num_concepts)
        self.concept_values = nn.Linear(concept_dim, num_concepts)
        
    def forward(self, encoded_text):
        """Forward pass through the concept network"""
        projected = F.relu(self.concept_projection(encoded_text))
        gates = torch.sigmoid(self.concept_gates(projected))
        values = torch.tanh(self.concept_values(projected))
        
        # Gate the concept values
        activations = gates * values
        
        # Calculate overall resonance as the mean activation
        resonance = torch.mean(torch.abs(activations))
        
        return activations, resonance


class NeuralNetworkProcessor:
    """Neural Network Processor for text analysis"""
    
    def __init__(self, model_dir=None, device=None):
        self.model_dir = model_dir or os.path.join("data", "neural_models")
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Configuration
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.concept_dim = 64
        self.num_concepts = 32
        self.vocab_size = 10000
        self.temperature = 0.8
        
        # Set device
        if TORCH_AVAILABLE:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Initialize models
            self.initialize_models()
        else:
            logger.warning("Running without PyTorch - using simplified text processing")
            self.text_encoder = SimpleTextEncoder(embedding_dim=self.embedding_dim)
        
        # Vocabfularyi and tokenization
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.next_idx = 0
        
        # Weight for processing
        self.nn_weight = 0.5
        self.speaker_config = {"nn_weight": 0.5, "llm_weight": 0.5}
        
        # Load configuration if exists
        self.load_configuration()
    
    def initialize_models(self):
        """Initialize neural network models"""
        if not TORCH_AVAILABLE:
            return
            
        # Create text encoder
        self.encoder = TextEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Create concept network
        self.concept_network = ConceptNetwork(
            input_dim=self.hidden_dim,
            concept_dim=self.concept_dim,
            num_concepts=self.num_concepts
        )
        
        # Move models to device
        self.encoder.to(self.device)
        self.concept_network.to(self.device)
        
        # Load models if available
        encoder_path = os.path.join(self.model_dir, "encoder.pt")
        concept_path = os.path.join(self.model_dir, "concept_network.pt")
        
        if os.path.exists(encoder_path):
            try:
                self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
                logger.info(f"Loaded encoder from {encoder_path}")
            except Exception as e:
                logger.error(f"Error loading encoder: {e}")
                
        if os.path.exists(concept_path):
            try:
                self.concept_network.load_state_dict(torch.load(concept_path, map_location=self.device))
                logger.info(f"Loaded concept network from {concept_path}")
            except Exception as e:
                logger.error(f"Error loading concept network: {e}")
        
        # Set evaluation mode
        self.encoder.eval()
        self.concept_network.eval()
    
    def load_configuration(self):
        """Load configuration from file"""
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Update configuration
                self.embedding_dim = config.get("embedding_dim", self.embedding_dim)
                self.hidden_dim = config.get("hidden_dim", self.hidden_dim)
                self.concept_dim = config.get("concept_dim", self.concept_dim)
                self.num_concepts = config.get("num_concepts", self.num_concepts)
                self.vocab_size = config.get("vocab_size", self.vocab_size)
                self.temperature = config.get("temperature", self.temperature)
                self.nn_weight = config.get("nn_weight", self.nn_weight)
                
                # Load vocabulary if available
                if "vocabulary" in config:
                    self.word_to_idx = config["vocabulary"]
                    self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
                    self.next_idx = max(self.word_to_idx.values()) + 1 if self.word_to_idx else 0
                
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
    
    def save_configuration(self):
        """Save configuration to file"""
        config_path = os.path.join(self.model_dir, "config.json")
        try:
            config = {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "concept_dim": self.concept_dim,
                "num_concepts": self.num_concepts,
                "vocab_size": self.vocab_size,
                "temperature": self.temperature,
                "nn_weight": self.nn_weight,
                "vocabulary": self.word_to_idx
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token indices"""
        words = re.findall(r'\b\w+\b', text.lower())
        indices = []
        
        for word in words:
            if word not in self.word_to_idx:
                if self.next_idx < self.vocab_size:
                    self.word_to_idx[word] = self.next_idx
                    self.idx_to_word[self.next_idx] = word
                    self.next_idx += 1
                else:
                    # Skip word if vocabulary is full
                    continue
            
            indices.append(self.word_to_idx[word])
            
        return indices
    
    def set_nn_weight(self, weight: float):
        """Set the neural network weight"""
        self.nn_weight = max(0.0, min(1.0, weight))
        logger.info(f"Neural network weight set to {self.nn_weight}")
        
        # Update speaker config
        self.speaker_config["nn_weight"] = self.nn_weight
        
        # Save configuration
        self.save_configuration()
        
        return self.nn_weight
    
    def process_text(self, text: str) -> NeuralProcessingState:
        """Process text through the neural network"""
        # Create processing state
        state = NeuralProcessingState(text=text)
        
        # Count word frequencies
        words = re.findall(r'\b\w+\b', text.lower())
        state.word_frequencies = Counter(words)
        state.token_count = len(words)
        
        if TORCH_AVAILABLE:
            # Process with PyTorch models
            try:
                with torch.no_grad():
                    # Tokenize
                    token_indices = self.tokenize(text)
                    if not token_indices:
                        return state
                    
                    # Convert to tensor
                    indices_tensor = torch.tensor(token_indices, device=self.device)
                    
                    # Get text encoding
                    encoded = self.encoder(indices_tensor)
                    
                    # Process through concept network
                    activations, resonance = self.concept_network(encoded)
                    
                    # Store results
                    state.activations = activations
                    state.resonance = resonance.item()
                    
                    # Extract top concept activations
                    concept_values = {}
                    for i in range(self.num_concepts):
                        concept_name = f"C{i}"
                        concept_values[concept_name] = activations[i].item()
                    
                    # Sort concepts by absolute value of activation
                    state.concepts = {k: v for k, v in sorted(
                        concept_values.items(), 
                        key=lambda item: abs(item[1]),
                        reverse=True
                    )}
                    
                    logger.debug(f"Processed text with resonance: {state.resonance}")
            except Exception as e:
                logger.error(f"Error processing text through neural network: {e}")
        else:
            # Use simplified processing
            try:
                embeddings = self.text_encoder.encode(text)
                state.embeddings = embeddings
                
                # Calculate a simple resonance score
                state.resonance = float(np.mean(np.abs(embeddings)))
                
                # Generate mock concept activations
                for i in range(min(5, len(words))):
                    word = words[i] if i < len(words) else "unknown"
                    state.concepts[f"C{i}"] = np.sin(sum(ord(c) for c in word) / 100.0)
                
                logger.debug(f"Processed text with simplified resonance: {state.resonance}")
            except Exception as e:
                logger.error(f"Error in simplified text processing: {e}")
        
        return state
    
    def get_enhanced_response(self, user_input: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Process user input and generate enhanced response information"""
        start_time = time.time()
        
        # Process through neural network
        process_state = self.process_text(user_input)
        
        # Extract key information
        result = {
            "raw_input": user_input,
            "neural_score": process_state.resonance * self.nn_weight,
            "processing_time": time.time() - start_time,
            "concept_activations": {k: v for k, v in list(process_state.concepts.items())[:5]},
            "token_count": process_state.token_count,
            "word_frequencies": dict(process_state.word_frequencies.most_common(5))
        }
        
        # Add context if available
        if context:
            result["context_provided"] = True
            # Process context as well
            context_state = self.process_text(context)
            result["context_neural_score"] = context_state.resonance * self.nn_weight
        else:
            result["context_provided"] = False
        
        return result


# For testing
if __name__ == "__main__":
    # Create a neural network processor
    processor = NeuralNetworkProcessor()
    
    # Test text processing
    test_text = "This is a test of the neural network processing capabilities."
    
    print(f"Processing: '{test_text}'")
    state = processor.process_text(test_text)
    
    print(f"Resonance: {state.resonance:.4f}")
    print("Top concept activations:")
    for concept, value in list(state.concepts.items())[:5]:
        print(f"  {concept}: {value:.4f}")
    
    # Test enhanced response
    result = processor.get_enhanced_response(test_text)
    print("\nEnhanced response:")
    print(json.dumps(result, indent=2)) 