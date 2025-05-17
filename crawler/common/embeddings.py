"""Embedding functionality for the crawler service."""
import os
from typing import List, Optional
from langchain_community.embeddings import OllamaEmbeddings
from .config import EMBEDDING_MODEL, OLLAMA_URL, OLLAMA_MODEL
from .logging import get_logger

logger = get_logger(__name__)

# Global embedding model instance
_embedding_model = None

def get_embedding_model() -> OllamaEmbeddings:
    """Get or create the embedding model instance.
    
    Returns:
        OllamaEmbeddings model instance.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = OllamaEmbeddings(
                base_url=OLLAMA_URL,
                model=OLLAMA_MODEL
            )
            logger.info(f"Loaded Ollama embedding model: {OLLAMA_MODEL}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    return _embedding_model

async def encode_text(text: str) -> Optional[List[float]]:
    """Encode text into vector embedding.
    
    Args:
        text: Input text to encode.
        
    Returns:
        Vector embedding or None if encoding failed.
    """
    try:
        model = get_embedding_model()
        embedding = await model.aembed_query(text)
        return embedding
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        return None

async def encode_batch(texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
    """Encode a batch of texts into vector embeddings.
    
    Args:
        texts: List of input texts to encode.
        batch_size: Batch size for encoding.
        
    Returns:
        List of vector embeddings (None for failed encodings).
    """
    try:
        model = get_embedding_model()
        embeddings = await model.aembed_documents(texts)
        return embeddings
    except Exception as e:
        logger.error(f"Error encoding batch: {e}")
        return [None] * len(texts)

def compute_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Cosine similarity score.
    """
    try:
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0

def find_most_similar(query_vec: List[float], candidates: List[List[float]], top_k: int = 5) -> List[int]:
    """Find most similar vectors to query vector.
    
    Args:
        query_vec: Query vector.
        candidates: List of candidate vectors.
        top_k: Number of top results to return.
        
    Returns:
        Indices of most similar vectors.
    """
    try:
        import numpy as np
        query_vec = np.array(query_vec)
        candidates = np.array(candidates)
        
        # Compute similarities
        similarities = np.dot(candidates, query_vec) / (
            np.linalg.norm(candidates, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_indices.tolist()
    except Exception as e:
        logger.error(f"Error finding similar vectors: {e}")
        return [] 