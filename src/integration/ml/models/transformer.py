#!/usr/bin/env python3
"""
Transformer Model

This module implements a flexible transformer model that can be used for
various learning tasks including sequence prediction, classification, and generation.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..core import BaseModel, MLConfig

class TransformerConfig(MLConfig):
    """Transformer-specific configuration"""
    def __init__(
        self,
        vocab_size: int = 30000,
        max_seq_length: int = 512,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        intermediate_size: int = 1024,
        layer_norm_eps: float = 1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose and reshape tensor for attention computation"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Normalize the attention scores to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Project back to hidden size
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(attention_output + hidden_states)
        
        return attention_output

class TransformerLayer(nn.Module):
    """Transformer layer with self-attention and feed-forward network"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gelu = nn.GELU()
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        attention_output = self.attention(hidden_states, attention_mask)
        
        # Feed-forward network
        intermediate_output = self.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(layer_output + attention_output)
        
        return layer_output

class TransformerModel(BaseModel):
    """Transformer model for various learning tasks"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        
    def _setup_model(self) -> None:
        """Setup model architecture"""
        config = self.config
        
        # Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(config.num_layers)
        ])
        
        # Output head (can be customized based on task)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size)
        
    def _create_position_ids(self, input_shape: Tuple[int, ...]) -> torch.Tensor:
        """Create position IDs for input sequence"""
        seq_length = input_shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape[0], -1)
        return position_ids
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        input_shape = input_ids.size()
        
        # Create position IDs
        position_ids = self._create_position_ids(input_shape)
        
        # Get embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add embeddings and apply layer norm
        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        # Apply output head
        logits = self.output_head(hidden_states)
        
        return logits
        
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate sequence"""
        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)
        
        while cur_len < max_length:
            # Get model predictions
            with torch.no_grad():
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Append next tokens
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            cur_len = input_ids.size(1)
            
        return input_ids

# Register model
from ..models import MODEL_REGISTRY
MODEL_REGISTRY['transformer'] = TransformerModel 