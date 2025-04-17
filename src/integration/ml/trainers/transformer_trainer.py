#!/usr/bin/env python3
"""
Transformer Trainer

This module implements the trainer for transformer models, handling both
sequence prediction and generation tasks.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from ..core import BaseTrainer, BaseModel, MLConfig

class TransformerTrainer(BaseTrainer):
    """Trainer for transformer models"""
    
    def __init__(self, model: BaseModel, config: MLConfig):
        super().__init__(model, config)
        
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step"""
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(input_ids)
        
        # Calculate loss
        # Reshape outputs and labels for loss calculation
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)
        
        loss = self.loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=-1)
            correct = (predictions == labels).float().sum()
            total = labels.numel()
            accuracy = correct / total
            
            # Calculate perplexity
            perplexity = torch.exp(loss)
            
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'perplexity': perplexity.item()
        }
        
    def validate_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Single validation step"""
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids)
        
        # Calculate loss
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)
        
        loss = self.loss_fn(outputs, labels)
        
        # Calculate metrics
        predictions = torch.argmax(outputs, dim=-1)
        correct = (predictions == labels).float().sum()
        total = labels.numel()
        accuracy = correct / total
        
        perplexity = torch.exp(loss)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'perplexity': perplexity.item()
        }
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate sequence using the model"""
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                metrics = self.validate_step(batch)
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                total_perplexity += metrics['perplexity']
                num_batches += 1
                
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'perplexity': total_perplexity / num_batches
        }
        
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        return checkpoint['epoch']

# Register trainer
from ..trainers import TRAINER_REGISTRY
TRAINER_REGISTRY['transformer'] = TransformerTrainer 