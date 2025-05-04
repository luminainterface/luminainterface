from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import os
import uuid
from lumina_core.utils.cache import EmbeddingCache

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://qdrant:6333")
        )
        self.encoder = SentenceTransformer("nomic-ai/nomic-embed-text-v1")
        self.collection_name = "lumina-chat"
        self._ensure_collection()
        self._conversation_count = 0
        
        # Initialize embedding cache
        redis_url = os.getenv("REDIS_URL")
        self.cache = EmbeddingCache(
            redis_url=redis_url,
            max_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))
        )

    def _ensure_collection(self):
        """Ensure the collection exists with proper configuration."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.encoder.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )

    async def upsert_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Upsert messages into Qdrant."""
        texts = [msg["content"] for msg in messages]
        
        # Get embeddings with caching
        embeddings = []
        for text in texts:
            embedding = await self.cache.get_embedding(
                text,
                lambda t: self.encoder.encode(t).tolist()
            )
            embeddings.append(embedding)
        
        # Generate conversation ID for new conversations
        conversation_id = str(uuid.uuid4())
        self._conversation_count += 1
        
        points = []
        for i, (msg, embedding) in enumerate(zip(messages, embeddings)):
            points.append(models.PointStruct(
                id=f"{conversation_id}-{i}",
                vector=embedding,
                payload={
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp"),
                    "conversation_id": conversation_id
                }
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    async def get_similar_messages(
        self, 
        query: str, 
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve similar messages from Qdrant."""
        # Get query embedding with caching
        query_vector = await self.cache.get_embedding(
            query,
            lambda t: self.encoder.encode(t).tolist()
        )
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {
                "content": hit.payload["content"],
                "role": hit.payload["role"],
                "score": hit.score,
                "conversation_id": hit.payload.get("conversation_id")
            }
            for hit in results
        ]

    async def get_metrics(self) -> Dict[str, int]:
        """Get collection metrics."""
        collection_info = self.client.get_collection(self.collection_name)
        return {
            "vectors": collection_info.points_count,
            "conversations": self._conversation_count
        } 