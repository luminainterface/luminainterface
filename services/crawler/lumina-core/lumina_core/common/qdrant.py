"""Qdrant client wrapper for Lumina Core."""

from typing import Optional, List, Dict, Any
import logging
from qdrant_client import QdrantClient as BaseQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from .config import get_settings

logger = logging.getLogger(__name__)

class QdrantClient:
    """Wrapper for Qdrant client with proper error handling and retries."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        timeout: Optional[int] = None,
        prefer_grpc: bool = False
    ):
        """Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL. If None, uses settings.QDRANT_URL
            timeout: Request timeout in seconds. If None, uses settings.QDRANT_TIMEOUT
            prefer_grpc: Whether to prefer gRPC over HTTP
        """
        settings = get_settings()
        self.url = url or settings.QDRANT_URL
        self.timeout = timeout or settings.QDRANT_TIMEOUT
        self.prefer_grpc = prefer_grpc
        self._client = None
        
    async def connect(self) -> None:
        """Verify connection to Qdrant server."""
        try:
            # Initialize client if not already done
            if self._client is None:
                self._client = BaseQdrantClient(
                    url=self.url,
                    timeout=self.timeout,
                    prefer_grpc=self.prefer_grpc
                )
            
            # Verify connection by getting collections
            self._client.get_collections()
            logger.info(f"Successfully connected to Qdrant at {self.url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.url}: {e}")
            raise
            
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        on_disk_payload: bool = True
    ) -> bool:
        """Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of vectors to store
            distance: Distance metric (Cosine, Euclidean, Dot)
            on_disk_payload: Whether to store payload on disk
            
        Returns:
            True if collection was created, False if it already exists
        """
        try:
            # Ensure client is initialized
            if self._client is None:
                await self.connect()
                
            # Check if collection exists
            collections = self._client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                logger.info(f"Collection {collection_name} already exists")
                return False
                
            # Create collection
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance[distance.upper()]
                ),
                on_disk_payload=on_disk_payload
            )
            logger.info(f"Created collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
            
    async def upsert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """Upsert points into collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points to upsert
            batch_size: Size of batches for upserting
        """
        try:
            # Ensure client is initialized
            if self._client is None:
                await self.connect()
                
            # Process in batches
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self._client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=p["id"],
                            vector=p["vector"],
                            payload=p.get("payload", {})
                        )
                        for p in batch
                    ]
                )
            logger.info(f"Upserted {len(points)} points to {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to upsert points to {collection_name}: {e}")
            raise
            
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            with_payload: Whether to return payloads
            with_vectors: Whether to return vectors
            
        Returns:
            List of search results with scores
        """
        try:
            # Ensure client is initialized
            if self._client is None:
                await self.connect()
                
            results = self._client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload if with_payload else None,
                    "vector": r.vector if with_vectors else None
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Search failed in {collection_name}: {e}")
            raise
            
    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
        """
        try:
            # Ensure client is initialized
            if self._client is None:
                await self.connect()
                
            self._client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise 