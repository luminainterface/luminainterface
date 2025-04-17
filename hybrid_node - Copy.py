import torch
import torch.nn as nn
import numpy as np
import json
import os
import random
import logging
from datetime import datetime
import traceback
import sqlite3
import threading
import time
from queue import Queue, Empty
from threading import Thread
from dataclasses import dataclass

# Import sentence_transformers only if available
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning(
        "sentence_transformers not available. Some functionality will be limited."
    )

# Import transformers only if available
# Temporarily disabled to avoid dependency issues
TRANSFORMERS_AVAILABLE = False
logging.warning("transformers disabled. Some functionality will be limited.")

# try:
#     from transformers import pipeline
#     TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False
#     logging.warning("transformers not available. Some functionality will be limited.")

from torch.optim import Adam
from wiki_vocabulary import WikiVocabulary
import quantum_infection
from english_language_trainer import EnglishLanguageTrainer
from physics_engine import PhysicsEngine, PhysicsResult
from physics_metaphysics_framework import PhysicsMetaphysicsFramework
from torch.nn import functional as F
import re
from typing import Dict, List, Tuple, Optional, Union, Any
import math
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PhysicsNarrativeClassifier(nn.Module):
    def __init__(self, input_dim=300):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim

        # Initialize layers
        self.linear1 = nn.Linear(input_dim, input_dim).to(self.device)
        self.linear2 = nn.Linear(input_dim, input_dim).to(self.device)
        self.attention = nn.MultiheadAttention(input_dim, num_heads=4).to(self.device)

        # Move entire model to device
        self.to(self.device)

    def forward(self, x):
        # Ensure input is on correct device
        x = x.to(self.device)

        # Process through layers
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        # Apply attention
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if needed
        x_permuted = x.permute(
            1, 0, 2
        )  # Reshape for attention (seq_len, batch, features)
        attn_output, attn_weights = self.attention(x_permuted, x_permuted, x_permuted)

        # Return to original shape
        output = attn_output.permute(1, 0, 2)  # Back to (batch, seq_len, features)
        if output.size(0) == 1:
            output = output.squeeze(0)  # Remove batch dimension if it was added

        return output, attn_weights


@dataclass
class NarrativeAnalysis:
    """Analysis result for physics/hyperphysics narratives."""

    text: str
    is_physics: bool
    is_hyperphysics: bool
    physics_truth_score: float
    hyperphysics_truth_score: float
    confidence: float
    explanation: str
    physics_principles: List[str]
    hyperphysics_concepts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "is_physics": self.is_physics,
            "is_hyperphysics": self.is_hyperphysics,
            "physics_truth_score": self.physics_truth_score,
            "hyperphysics_truth_score": self.hyperphysics_truth_score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "physics_principles": self.physics_principles,
            "hyperphysics_concepts": self.hyperphysics_concepts,
        }


class HybridNode(nn.Module):
    def __init__(self, embedding_dim: int = 300):
        """Initialize hybrid node.

        Args:
            embedding_dim: Dimension of input embeddings
        """
        super().__init__()

        # Load configuration
        config_instance = Config.get_instance()
        model_config = config_instance.get_model_config()

        # Model architecture
        self.embedding_dim = embedding_dim
        self.hidden_dim = model_config["hidden_dim"]
        self.num_layers = model_config["num_layers"]
        self.dropout = model_config["dropout"]

        # Neural network layers with layer normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,  # Use bidirectional LSTM for better feature extraction
        )

        # Output layer with batch normalization
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # *2 for bidirectional
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 2),  # Output 2 classes directly
            nn.LogSoftmax(dim=-1),  # Use LogSoftmax for numerical stability
        )

        # Initialize weights with Xavier/Glorot initialization
        self._init_weights()

        # State tracking
        self.last_confidence = None
        self.last_features = None

        self._initialized = False
        self._active = False
        # Don't call initialize here, let the main script do it
        # self.initialize()

    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (output, attention_weights)
        """
        try:
            # Input validation
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError("Input contains NaN or Inf values")

            # Ensure input is properly shaped
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension

            # Feature extraction with gradient scaling
            x = self.feature_extractor(x)
            x = torch.clamp(x, min=-1e7, max=1e7)  # Prevent extreme values

            # Reshape for LSTM if needed
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension

            # LSTM processing
            lstm_out, (hidden, _) = self.lstm(x)

            # Prevent exploding values
            lstm_out = torch.clamp(lstm_out, min=-1e7, max=1e7)
            hidden = torch.clamp(hidden, min=-1e7, max=1e7)

            # Concatenate forward and backward hidden states
            hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)

            # Output layer with stable softmax
            output = self.output_layer(hidden_concat)

            # Calculate attention weights with stable softmax
            attention_scores = torch.bmm(lstm_out, hidden_concat.unsqueeze(2))
            attention_scores = torch.clamp(attention_scores, min=-1e7, max=1e7)
            attention_weights = F.softmax(attention_scores.squeeze(2), dim=1)

            return output, attention_weights

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

    def get_state(self) -> Dict[str, Any]:
        """Get current node state.

        Returns:
            Dictionary containing current state
        """
        return {
            "last_confidence": self.last_confidence,
            "last_features": (
                self.last_features.detach() if self.last_features is not None else None
            ),
        }

    def update_from_feedback(
        self, feedback_score: float, learning_rate: float = 0.001
    ) -> None:
        """Update model based on feedback.

        Args:
            feedback_score: Feedback score from user
            learning_rate: Learning rate for update
        """
        if self.last_confidence is None:
            logger.warning("No previous interaction to update from")
            return

        try:
            # Calculate loss
            predicted = torch.tensor([self.last_confidence])
            target = torch.tensor([feedback_score])
            loss = nn.MSELoss()(predicted, target)

            # Backpropagate
            loss.backward()

            # Update weights
            with torch.no_grad():
                for param in self.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                        param.grad.zero_()

            logger.info(
                f"Updated model weights based on feedback: {feedback_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Error updating from feedback: {str(e)}")

    def save_state(self, path: str) -> None:
        """Save model state.

        Args:
            path: Path to save state to
        """
        try:
            state_dict = {
                "model_state": self.state_dict(),
                "last_confidence": self.last_confidence,
                "last_features": self.last_features,
            }
            torch.save(state_dict, path)
            logger.info(f"Saved model state to {path}")

        except Exception as e:
            logger.error(f"Error saving model state: {str(e)}")

    def load_state(self, path: str) -> None:
        """Load model state.

        Args:
            path: Path to load state from
        """
        try:
            state_dict = torch.load(path)
            self.load_state_dict(state_dict["model_state"])
            self.last_confidence = state_dict["last_confidence"]
            self.last_features = state_dict["last_features"]
            logger.info(f"Loaded model state from {path}")

        except Exception as e:
            logger.error(f"Error loading model state: {str(e)}")

    def _process_inputs(self, inputs, labels=None):
        """Process inputs and labels into proper tensor format."""
        if inputs is None:
            raise ValueError("Inputs cannot be None")

        # Handle different input types
        if isinstance(inputs, str):
            # For single string input
            inputs = [inputs]

        if isinstance(inputs, list):
            # Convert list of strings to tensors
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    inputs = self.tokenizer.encode(inputs, convert_to_tensor=True)
                except:
                    # Fallback to basic tokenization
                    inputs = torch.tensor(
                        [list(map(ord, text)) for text in inputs], dtype=torch.float32
                    )
            else:
                inputs = torch.tensor(
                    [list(map(ord, text)) for text in inputs], dtype=torch.float32
                )

        # Convert numpy arrays to tensors if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()

        # Ensure inputs are tensors
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Unsupported input type: {type(inputs)}")

        # Reshape if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # Add batch dimension

        # Ensure proper dimensions (batch_size x input_dim)
        if inputs.size(-1) != self.embedding_dim:
            inputs = self._handle_input_size_mismatch(inputs)

        # Process labels if provided
        if labels is not None:
            if isinstance(labels, (list, np.ndarray)):
                labels = torch.tensor(labels, dtype=torch.float32)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)

        return inputs, labels

    def _handle_input_size_mismatch(self, inputs):
        """Handle input size mismatches by padding or truncating."""
        batch_size = inputs.size(0)

        # If input is smaller than embedding_dim, pad with zeros
        if inputs.size(-1) < self.embedding_dim:
            padding = torch.zeros(batch_size, self.embedding_dim - inputs.size(-1))
            inputs = torch.cat([inputs, padding], dim=-1)

        # If input is larger than embedding_dim, truncate
        elif inputs.size(-1) > self.embedding_dim:
            inputs = inputs[:, : self.embedding_dim]

        return inputs

    def train_on_semantic_data(self, data):
        """Train the model on semantic data."""
        inputs, labels = self._process_inputs(data)
        return self._train_step(inputs, labels)

    def train_physics_model(self, data):
        """Train the model on physics data."""
        inputs, labels = self._process_inputs(data)
        return self._train_step(inputs, labels)

    def train_on_latin_data(self, data):
        """Train the model on Latin data."""
        inputs, labels = self._process_inputs(data)
        return self._train_step(inputs, labels)

    def train_on_arithmetic_data(self, data):
        """Train the model on arithmetic data."""
        inputs, labels = self._process_inputs(data)
        return self._train_step(inputs, labels)

    def _train_step(self, inputs, labels):
        """Perform a single training step.

        Args:
            inputs: Processed input tensor
            labels: Processed label tensor
        Returns:
            float: Loss value
        """
        self.train()
        if not hasattr(self, 'optimizer'):
            self.optimizer = Adam(self.parameters())
        if not hasattr(self, 'criterion'):
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer.zero_grad()

        outputs, _ = self(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def analyze_physics_narrative(self, text):
        """Analyze physics narrative in text."""
        inputs, _ = self._process_inputs({"inputs": text})

        self.model.eval()
        with torch.no_grad():
            outputs, attention_weights = self.model(inputs)
            predictions = torch.softmax(outputs, dim=-1)

        return {
            "text": text,
            "physics_score": predictions[0][1].item(),
            "hyperphysics_score": predictions[0][0].item(),
            "narrative_predictions": predictions.cpu().numpy(),
            "attention_weights": attention_weights.cpu().numpy(),
        }

    def initialize(self):
        """Initialize the hybrid node"""
        try:
            # Add initialization logic here
            self._initialized = True
            return True
        except Exception as e:
            self._initialized = False
            return False
            
    def activate(self):
        """Activate the hybrid node"""
        if self._initialized:
            self._active = True
            return True
        return False
        
    def deactivate(self):
        """Deactivate the hybrid node"""
        self._active = False
        
    def is_initialized(self):
        """Check if the node is initialized"""
        return self._initialized
        
    def is_active(self):
        """Check if the node is active"""
        return self._active and self._initialized
        
    def get_status(self):
        """Get the current status of the node"""
        if not self._initialized:
            return 'uninitialized'
        return 'active' if self._active else 'inactive'

    def load_pretrained_weights(self, model_path: str):
        """Load pre-trained weights from a .pt file into the node's model layers."""
        if not os.path.exists(model_path):
            logger.error(f"Model weights file not found at {model_path}")
            return False
            
        try:
            logger.info(f"Loading pre-trained weights for HybridNode from {model_path}...")
            # Determine the device to load onto
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load the state dict, mapping location to the determined device
            state_dict = torch.load(model_path, map_location=device)
            
            # Handle different save formats (e.g., checkpoint vs raw state_dict)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif isinstance(state_dict, dict) and "state_dict" in state_dict: # Another common pattern
                state_dict = state_dict["state_dict"]
            
            # Adjust keys if necessary (e.g., removing 'module.' prefix from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") # remove `module.` prefix
                new_state_dict[name] = v
            
            # Load the adjusted state dict into the current model (self is nn.Module)
            self.load_state_dict(new_state_dict, strict=False) # Use strict=False to ignore non-matching keys
            
            self.eval()  # Set model to evaluation mode after loading weights
            logger.info(f"Successfully loaded pre-trained weights into HybridNode from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pre-trained weights into HybridNode from {model_path}: {e}")
            logger.error(traceback.format_exc())
            return False
