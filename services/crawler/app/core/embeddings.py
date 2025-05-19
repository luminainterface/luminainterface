"""Custom embedding implementation to prevent invalid options."""
import os
from typing import List, Optional
import httpx
from langchain_core.embeddings import Embeddings

class CustomOllamaEmbeddings(Embeddings):
    """Custom Ollama embeddings implementation that only uses required parameters."""
    
    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        model: str = "nomic-embed-text"
    ):
        """Initialize with only required parameters."""
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(timeout=30.0)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            try:
                response = self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"])
            except Exception as e:
                raise RuntimeError(f"Failed to get embedding: {str(e)}")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            response = self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as e:
            raise RuntimeError(f"Failed to get embedding: {str(e)}")
    
    def __del__(self):
        """Clean up the HTTP client."""
        self.client.close() 