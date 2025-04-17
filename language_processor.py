import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SymbolicTokenizer:
    """Tokenization module that handles different levels of text segmentation"""
    
    def __init__(self, tokenization_level: str = 'subword'):
        """
        Initialize tokenizer with specified level
        Args:
            tokenization_level: 'char', 'word', or 'subword'
        """
        self.tokenization_level = tokenization_level
        self.vocab = {}
        self.vocab_size = 0
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3
        }
        
        # Download NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
        
    def train(self, texts: List[str], max_vocab_size: int = 10000):
        """Train the tokenizer on a corpus of texts"""
        logger.info(f"Training {self.tokenization_level}-level tokenizer...")
        
        if self.tokenization_level == 'char':
            self._train_char_level(texts, max_vocab_size)
        elif self.tokenization_level == 'word':
            self._train_word_level(texts, max_vocab_size)
        else:  # subword
            self._train_subword_level(texts, max_vocab_size)
            
        logger.info(f"Tokenizer trained with vocabulary size: {self.vocab_size}")
        
    def _train_char_level(self, texts: List[str], max_vocab_size: int):
        """Train character-level tokenizer"""
        chars = set()
        for text in texts:
            chars.update(text.lower())
            
        # Sort characters by frequency
        char_counts = Counter(''.join(texts).lower())
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocabulary
        self.vocab = {**self.special_tokens}
        for char, _ in sorted_chars[:max_vocab_size - len(self.special_tokens)]:
            self.vocab[char] = len(self.vocab)
        self.vocab_size = len(self.vocab)
        
    def _train_word_level(self, texts: List[str], max_vocab_size: int):
        """Train word-level tokenizer"""
        words = []
        for text in texts:
            # Tokenize and clean text
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
            words.extend(tokens)
            
        # Sort words by frequency
        word_counts = Counter(words)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocabulary
        self.vocab = {**self.special_tokens}
        for word, _ in sorted_words[:max_vocab_size - len(self.special_tokens)]:
            self.vocab[word] = len(self.vocab)
        self.vocab_size = len(self.vocab)
        
    def _train_subword_level(self, texts: List[str], max_vocab_size: int):
        """Train subword-level tokenizer using Byte-Pair Encoding"""
        # Initialize with character vocabulary
        self._train_char_level(texts, max_vocab_size)
        
        # Implement BPE algorithm
        # This is a simplified version - in practice, you might want to use a library like sentencepiece
        pairs = Counter()
        for text in texts:
            words = word_tokenize(text.lower())
            for word in words:
                if word.isalpha() and word not in self.stop_words:
                    pairs.update(zip(word[:-1], word[1:]))
                    
        # Merge most frequent pairs
        while len(self.vocab) < max_vocab_size and pairs:
            most_frequent = pairs.most_common(1)[0][0]
            new_token = ''.join(most_frequent)
            self.vocab[new_token] = len(self.vocab)
            pairs.pop(most_frequent)
            
        self.vocab_size = len(self.vocab)
        
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        text = text.lower()
        
        if self.tokenization_level == 'char':
            return [self.vocab.get(c, self.special_tokens['<unk>']) for c in text]
        elif self.tokenization_level == 'word':
            tokens = word_tokenize(text)
            return [self.vocab.get(t, self.special_tokens['<unk>']) 
                   for t in tokens if t.isalpha() and t not in self.stop_words]
        else:  # subword
            # Simplified subword tokenization
            tokens = []
            current = ''
            for c in text:
                current += c
                if current in self.vocab:
                    tokens.append(self.vocab[current])
                    current = ''
            if current:
                tokens.append(self.vocab.get(current, self.special_tokens['<unk>']))
            return tokens

class ResonanceEmbedder(nn.Module):
    """Embedding module that captures both static and contextual aspects of language"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, context_window: int = 5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        
        # Static embeddings
        self.static_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Contextual embeddings
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim * context_window, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Resonance network
        self.resonance_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate embeddings that combine static and contextual information
        Args:
            token_ids: Tensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = token_ids.shape
        
        # Get static embeddings
        static_embeds = self.static_embeddings(token_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Generate contextual embeddings
        contextual_embeds = []
        for i in range(seq_len):
            # Get context window
            start = max(0, i - self.context_window // 2)
            end = min(seq_len, i + self.context_window // 2 + 1)
            context = static_embeds[:, start:end, :]
            
            # Pad if necessary
            if context.shape[1] < self.context_window:
                padding = torch.zeros(batch_size, self.context_window - context.shape[1], 
                                    self.embedding_dim, device=token_ids.device)
                context = torch.cat([padding, context], dim=1)
                
            # Flatten context
            context = context.reshape(batch_size, -1)
            
            # Generate contextual embedding
            contextual_embed = self.context_encoder(context)
            contextual_embeds.append(contextual_embed)
            
        contextual_embeds = torch.stack(contextual_embeds, dim=1)
        
        # Combine static and contextual embeddings through resonance
        combined = torch.cat([static_embeds, contextual_embeds], dim=-1)
        resonance_embeds = self.resonance_net(combined)
        
        return resonance_embeds

class LanguageProcessor:
    """Combined tokenization and embedding processor"""
    
    def __init__(self, tokenization_level: str = 'subword', 
                 embedding_dim: int = 300, context_window: int = 5):
        self.tokenizer = SymbolicTokenizer(tokenization_level)
        self.embedder = ResonanceEmbedder(
            vocab_size=0,  # Will be updated after tokenizer training
            embedding_dim=embedding_dim,
            context_window=context_window
        )
        
    def train(self, texts: List[str], max_vocab_size: int = 10000):
        """Train both tokenizer and embedder"""
        self.tokenizer.train(texts, max_vocab_size)
        self.embedder.vocab_size = self.tokenizer.vocab_size
        
    def process(self, text: str) -> torch.Tensor:
        """Process text through tokenization and embedding"""
        # Tokenize
        token_ids = self.tokenizer.tokenize(text)
        
        # Convert to tensor
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        
        # Generate embeddings
        embeddings = self.embedder(token_tensor)
        
        return embeddings

# Example usage
if __name__ == "__main__":
    # Sample texts for training
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Neural networks can learn complex patterns in language.",
        "Embeddings capture semantic relationships between words."
    ]
    
    # Initialize and train processor
    processor = LanguageProcessor(tokenization_level='subword')
    processor.train(texts)
    
    # Process a new text
    test_text = "Language processing is fascinating!"
    embeddings = processor.process(test_text)
    print(f"Generated embeddings shape: {embeddings.shape}") 