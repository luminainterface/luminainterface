"""Qdrant client module for vector storage operations."""
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from .config import QDRANT_URL, QDRANT_COLLECTION
from .logging import get_logger

logger = get_logger(__name__)

class QdrantClient:
    """Qdrant client for vector storage operations."""
    
    def __init__(self, url: str = QDRANT_URL):
        """Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL.
        """
        self.url = url
        self._client = None
        
    async def connect(self) -> None:
        """Create Qdrant client connection."""
        if not self._client:
            self._client = QdrantClient(self.url)
            logger.info("Connected to Qdrant")
            
    async def close(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            await self._client.close()
            logger.info("Closed Qdrant connection")
            
    async def init_collection(self, collection_name: str = QDRANT_COLLECTION, vector_size: int = 384) -> bool:
        """Initialize vector collection.
        
        Args:
            collection_name: Name of the collection.
            vector_size: Size of the vectors.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Check if collection exists
            collections = await self._client.get_collections()
            exists = any(c.name == collection_name for c in collections.collections)
            
            if not exists:
                # Create collection
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection {collection_name}")
            else:
                logger.info(f"Collection {collection_name} already exists")
                
            return True
        except Exception as e:
            logger.error(f"Error initializing collection {collection_name}: {e}")
            return False
            
    async def upsert(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """Upsert points to collection.
        
        Args:
            collection_name: Name of the collection.
            points: List of points to upsert.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Convert points to Qdrant format
            qdrant_points = []
            for point in points:
                qdrant_points.append(models.PointStruct(
                    id=point['id'],
                    vector=point['vector'],
                    payload=point['payload']
                ))
                
            # Upsert points
            await self._client.upsert(
                collection_name=collection_name,
                points=qdrant_points
            )
            return True
        except Exception as e:
            logger.error(f"Error upserting to collection {collection_name}: {e}")
            return False
            
    async def search(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            collection_name: Name of the collection.
            query_vector: Query vector.
            limit: Maximum number of results.
            
        Returns:
            List of search results.
        """
        try:
            results = await self._client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return [
                {
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
            return []
            
    async def delete(self, collection_name: str, points: List[str]) -> bool:
        """Delete points from collection.
        
        Args:
            collection_name: Name of the collection.
            points: List of point IDs to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            await self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=points
                )
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting from collection {collection_name}: {e}")
            return False

# Create global Qdrant client instance
qdrant_client = QdrantClient()

async def init_qdrant() -> None:
    """Initialize Qdrant connection."""
    await qdrant_client.connect()

async def close_qdrant() -> None:
    """Close Qdrant connection."""
    await qdrant_client.close() 