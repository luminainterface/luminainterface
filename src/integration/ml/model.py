"""
Machine Learning Model Module

Handles self-writing and learning capabilities for the backend system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Any, Optional
import logging
import numpy as np
from pathlib import Path

from ..config import ML_CONFIG, DATA_DIR

logger = logging.getLogger(__name__)

class NeuralProcessor(nn.Module):
    """Neural processor for self-writing and learning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or ML_CONFIG
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained('gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
        # Additional layers
        self.hidden_size = self.config["model"]["hidden_size"]
        self.dropout = nn.Dropout(self.config["model"]["dropout"])
        
        # Task-specific heads
        self.code_generation_head = nn.Linear(self.hidden_size, self.tokenizer.vocab_size)
        self.pattern_recognition_head = nn.Linear(self.hidden_size, 256)
        self.state_prediction_head = nn.Linear(self.hidden_size, 128)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        # Get transformer outputs
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Task-specific predictions
        code_logits = self.code_generation_head(hidden_states)
        pattern_features = self.pattern_recognition_head(hidden_states[:, 0, :])
        state_predictions = self.state_prediction_head(hidden_states[:, 0, :])
        
        return {
            'code_logits': code_logits,
            'pattern_features': pattern_features,
            'state_predictions': state_predictions
        }
        
    def generate_code(self, prompt: str, max_length: int = 100) -> str:
        """Generate code based on prompt."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.transformer.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode and return
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_code
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return ""
            
    def recognize_patterns(self, states: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Recognize patterns in gate states."""
        try:
            # Convert states to tensors
            state_tensors = []
            for state in states:
                tensor = torch.tensor([
                    state['output'],
                    float(state['gate_state'] == 'OPEN'),
                    len(state['connections']),
                    len(state['inputs'])
                ], dtype=torch.float32)
                state_tensors.append(tensor)
                
            # Stack tensors and process
            state_batch = torch.stack(state_tensors).to(self.device)
            with torch.no_grad():
                pattern_features = self.pattern_recognition_head(state_batch)
                
            # Convert to pattern scores
            patterns = []
            for features in pattern_features:
                pattern = {
                    'temporal': float(features[0]),
                    'spatial': float(features[1]),
                    'causal': float(features[2]),
                    'structural': float(features[3])
                }
                patterns.append(pattern)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error recognizing patterns: {e}")
            return []
            
    def predict_states(self, current_states: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Predict future states based on current states."""
        try:
            # Prepare input tensors
            state_tensors = []
            for state in current_states:
                tensor = torch.tensor([
                    state['output'],
                    float(state['gate_state'] == 'OPEN'),
                    len(state['connections']),
                    len(state['inputs'])
                ], dtype=torch.float32)
                state_tensors.append(tensor)
                
            # Process states
            state_batch = torch.stack(state_tensors).to(self.device)
            with torch.no_grad():
                predictions = self.state_prediction_head(state_batch)
                
            # Convert to prediction dictionaries
            predicted_states = []
            for pred in predictions:
                state = {
                    'output_prob': float(pred[0]),
                    'state_change_prob': float(pred[1]),
                    'connection_change_prob': float(pred[2]),
                    'stability_score': float(pred[3])
                }
                predicted_states.append(state)
                
            return predicted_states
            
        except Exception as e:
            logger.error(f"Error predicting states: {e}")
            return []
            
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the model to disk."""
        try:
            save_path = path or str(DATA_DIR / "model_checkpoint.pt")
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': self.config
            }, save_path)
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            
    def load_model(self, path: Optional[str] = None) -> None:
        """Load the model from disk."""
        try:
            load_path = path or str(DATA_DIR / "model_checkpoint.pt")
            checkpoint = torch.load(load_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.config = checkpoint['config']
            logger.info(f"Model loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            
    @staticmethod
    def create_model(config: Optional[Dict[str, Any]] = None) -> 'NeuralProcessor':
        """Create a new instance of the model."""
        model = NeuralProcessor(config)
        return model 