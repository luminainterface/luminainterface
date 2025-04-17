"""
Neural processor for text embedding and concept mapping with configurable models.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any, Union, NamedTuple
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import threading
from queue import Queue
from tenacity import retry, stop_after_attempt, wait_exponential
import warnings
from sentence_transformers import SentenceTransformer

# Optional imports with fallbacks
try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Using fallback model.")

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. Using basic tokenization.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NeuralState:
    """Represents the neural network's internal state."""
    activations: List[float]
    attention_weights: Optional[List[float]] = None
    concept_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralState':
        """Create state from dictionary data."""
        return cls(**data)

class ProcessingState(NamedTuple):
    """Holds the state of text processing including embeddings and activations."""
    text: str
    embedding: torch.Tensor
    activations: torch.Tensor
    resonance: float = 0.0

class ConceptMapper(nn.Module):
    """Maps text embeddings to concept space."""
    
    def __init__(self, input_dim: int, output_dim: int, num_concepts: int):
        super().__init__()
        self.dimension_adapter = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.concept_layer = nn.Linear(output_dim, num_concepts)
        self.activation = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dimension_adapter(x)
        return self.activation(self.concept_layer(x))

class NeuralProcessor:
    """
    Neural processor that combines sentence embeddings with concept mapping
    for enhanced text understanding and generation.
    """
    
    def __init__(
        self,
        model_dir: str,
        embedding_dim: int = 768,  # Updated to match mpnet's default
        output_dim: int = 512,    # Desired output dimension
        num_concepts: int = 200,
        vocab_size: int = 20000,
        temperature: float = 0.7,
        speaker_config: Optional[Dict[str, Any]] = None,
        embedding_model: str = 'paraphrase-mpnet-base-v2'
    ):
        """
        Initialize the neural processor.
        
        Args:
            model_dir: Directory to save/load models
            embedding_dim: Input dimension from the embedding model (default: 768 for mpnet)
            output_dim: Desired output dimension after adaptation (default: 512)
            num_concepts: Number of concept dimensions (default: 200)
            vocab_size: Size of vocabulary (default: 20000)
            temperature: Generation temperature (default: 0.7)
            speaker_config: Configuration for speaker behavior
            embedding_model: Name of the sentence-transformers model to use
        """
        self.model_dir = Path(model_dir)
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_concepts = num_concepts
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.speaker_config = speaker_config or {}
        
        # Initialize the sentence transformer
        try:
            self.encoder = SentenceTransformer(embedding_model)
            actual_dim = self.encoder.get_sentence_embedding_dimension()
            if actual_dim != embedding_dim:
                logger.warning(
                    f"Model {embedding_model} produces {actual_dim}-dim embeddings. "
                    f"Adjusting embedding_dim from {embedding_dim} to {actual_dim}"
                )
                self.embedding_dim = actual_dim
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
            
        # Initialize concept mapper with dimension adapter
        self.concept_mapper = ConceptMapper(self.embedding_dim, self.output_dim, num_concepts)
        
        # Load existing model if available
        self._load_model()
        
        logger.info(
            f"Initialized NeuralProcessor with {self.embedding_dim}->{self.output_dim} embedding dimensions "
            f"and {num_concepts} concepts using model {embedding_model}"
        )
        
        # Initialize storage
        self.concept_mapping = self._load_concept_mapping()
        self.state_history: List[NeuralState] = []
        self.state_queue = Queue()
        
        # Initialize resonance tracking
        self.resonance_history: List[Dict[str, float]] = []
        self.resonance_threshold = 2.0
        
        # Start background worker
        self._start_save_worker()
    
    def _load_model(self):
        """Load saved model state if available."""
        model_path = self.model_dir / 'final_model.pt'
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path)
                self.concept_mapper.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {str(e)}")
    
    def process_text(self, text: str) -> ProcessingState:
        """
        Process text through the neural pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            ProcessingState containing embeddings and activations
        """
        try:
            # Generate embedding
            with torch.no_grad():
                embedding = torch.tensor(
                    self.encoder.encode(text, convert_to_tensor=True).cpu().numpy()
                ).float()
            
            # Ensure correct shape
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
                
            # Generate concept activations
            self.concept_mapper.train()  # Ensure training mode for gradient tracking
            activations = self.concept_mapper(embedding)
            
            return ProcessingState(
                text=text,
                embedding=embedding,
                activations=activations,
                resonance=0.0
            )
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text response based on prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text response
        """
        try:
            # Process prompt
            state = self.process_text(prompt)
            
            # For now, return a simple response based on speaker config
            # This should be replaced with actual generation logic
            style_tokens = self.speaker_config.get('style_tokens', [])
            if style_tokens:
                import random
                return f"A response incorporating {random.choice(style_tokens)}"
            return "Processing complete"
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return "Error processing request"
    
    def update_resonance(self, text: str, score: float):
        """
        Update resonance score for generated text.
        
        Args:
            text: Generated text
            score: Resonance score (typically 0-5)
        """
        # This is a placeholder for resonance tracking
        # Could be expanded to maintain history, adjust weights, etc.
        logger.info(f"Updated resonance for text: score={score:.2f}")

    def _start_save_worker(self):
        """Start background worker for saving states."""
        def save_worker():
            while True:
                state = self.state_queue.get()
                if state is None:
                    break
                    
                try:
                    self._save_state(state)
                except Exception as e:
                    logger.error(f"Error saving state: {str(e)}")
                    
                self.state_queue.task_done()
                
        self.save_thread = threading.Thread(target=save_worker, daemon=True)
        self.save_thread.start()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _save_state(self, state: NeuralState):
        """Save neural state to persistent storage.
        
        Args:
            state: Neural state to save
        """
        try:
            # Ensure save directory exists
            save_dir = self.model_dir / 'states'
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save state to JSON file
            timestamp = datetime.fromisoformat(state.timestamp).strftime('%Y%m%d_%H%M%S')
            filename = save_dir / f"state_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            # Stop worker
            self.state_queue.put(None)
            if hasattr(self, 'save_thread'):
                self.save_thread.join(timeout=5)
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def _load_concept_mapping(self) -> List[str]:
        """Load concept mapping from file or generate default."""
        try:
            mapping_file = self.model_dir / 'concept_mapping.json'
            if mapping_file.exists():
                with open(mapping_file) as f:
                    concept_data = json.load(f)
                    # Flatten the structured concept mapping
                    concepts = []
                    for category, category_concepts in concept_data.items():
                        concepts.extend(category_concepts)
                    return concepts
        except Exception as e:
            logger.error(f"Error loading concept mapping: {str(e)}")
            
        # Generate default mapping
        return [f"concept_{i}" for i in range(self.num_concepts)] 