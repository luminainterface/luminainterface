#!/usr/bin/env python3
"""
Sequence Dataset

This module implements a dataset for sequence data, supporting both
text and numerical sequences.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict

from ..core import BaseDataset, MLConfig

class SequenceDataset(BaseDataset):
    """Dataset for sequence data"""
    
    def __init__(
        self,
        sequences: List[List[Union[str, int]]],
        config: MLConfig,
        vocab: Optional[Dict[str, int]] = None,
        max_length: Optional[int] = None
    ):
        super().__init__(config)
        self.sequences = sequences
        self.max_length = max_length or config.max_seq_length
        
        # Build or load vocabulary
        self.vocab = vocab if vocab is not None else self._build_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        
        # Add special tokens to vocab if not present
        for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.reverse_vocab[len(self.reverse_vocab)] = token
                
        # Cache processed sequences
        self.processed_sequences = [
            self.preprocess(seq) for seq in sequences
        ]
        
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from sequences"""
        vocab = defaultdict(int)
        
        # Count token frequencies
        for sequence in self.sequences:
            for token in sequence:
                vocab[str(token)] += 1
                
        # Sort by frequency
        sorted_tokens = sorted(
            vocab.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create vocabulary mapping
        return {
            token: idx
            for idx, (token, _) in enumerate(sorted_tokens)
        }
        
    def encode(self, sequence: List[Union[str, int]]) -> List[int]:
        """Encode sequence to token IDs"""
        return [
            self.vocab.get(str(token), self.vocab[self.unk_token])
            for token in sequence
        ]
        
    def decode(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs to tokens"""
        return [
            self.reverse_vocab.get(token_id, self.unk_token)
            for token_id in token_ids
        ]
        
    def preprocess(self, sequence: List[Union[str, int]]) -> torch.Tensor:
        """Preprocess sequence"""
        # Encode sequence
        encoded = self.encode(sequence)
        
        # Add BOS and EOS tokens
        encoded = [self.vocab[self.bos_token]] + encoded + [self.vocab[self.eos_token]]
        
        # Pad or truncate
        if len(encoded) < self.max_length:
            encoded = encoded + [self.vocab[self.pad_token]] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
            
        return torch.tensor(encoded, dtype=torch.long)
        
    def create_attention_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        """Create attention mask for sequence"""
        # Create mask (1 for real tokens, 0 for padding)
        mask = (sequence != self.vocab[self.pad_token]).float()
        
        # Convert to attention mask
        attention_mask = mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0  # Set padding to large negative value
        
        return attention_mask
        
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item"""
        sequence = self.processed_sequences[idx]
        
        # Input is all tokens except last, target is all tokens except first
        input_ids = sequence[:-1]
        labels = sequence[1:]
        
        return input_ids, labels
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
        
    def save_vocab(self, path: str) -> None:
        """Save vocabulary to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.vocab, f)
            
    @classmethod
    def load_vocab(cls, path: str) -> Dict[str, int]:
        """Load vocabulary from file"""
        import json
        with open(path, 'r') as f:
            return json.load(f)
            
    def get_batch(
        self,
        indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get batch of sequences"""
        sequences = [self.processed_sequences[i] for i in indices]
        
        # Create input and target tensors
        input_ids = torch.stack([seq[:-1] for seq in sequences])
        labels = torch.stack([seq[1:] for seq in sequences])
        
        # Create attention masks
        attention_masks = torch.stack([
            self.create_attention_mask(seq[:-1])
            for seq in sequences
        ])
        
        return input_ids, attention_masks, labels

class SequenceDataLoader:
    """DataLoader for sequence data"""
    
    def __init__(
        self,
        dataset: SequenceDataset,
        batch_size: int,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        
    def __iter__(self):
        """Iterator"""
        # Create index array
        indices = np.arange(self.num_samples)
        
        # Shuffle if needed
        if self.shuffle:
            np.random.shuffle(indices)
            
        # Yield batches
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield self.dataset.get_batch(batch_indices)
            
    def __len__(self) -> int:
        """Number of batches"""
        return (self.num_samples + self.batch_size - 1) // self.batch_size 