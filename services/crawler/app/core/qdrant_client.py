"""Qdrant client for the crawler service."""
import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from qdrant_client import QdrantClient as BaseQdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class QdrantClient:
    """Qdrant client wrapper with connection management."""
    def __init__(self, url: Optional[str] = None):
        self.url = url or os.getenv("QDRANT_URL", "http://qdrant:6333")
        self._client: Optional[BaseQdrantClient] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        # Initialize the client synchronously
        try:
            self._client = BaseQdrantClient(url=self.url)
            self._initialized = True
            logger.info("Connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def ensure_initialized(self):
        """Ensure the client is initialized."""
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:  # Double check after acquiring lock
                    try:
                        self._client = BaseQdrantClient(url=self.url)
                        self._initialized = True
                        logger.info("Connected to Qdrant")
                    except Exception as e:
                        logger.error(f"Failed to connect to Qdrant: {e}")
                        raise

    async def connect(self):
        """Connect to Qdrant (for compatibility with other clients)."""
        await self.ensure_initialized()

    async def close(self):
        """Close Qdrant connection."""
        if self._client:
            self._client.close()
            self._initialized = False
            logger.info("Closed Qdrant connection")

    async def retrieve(self, collection_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve points by their IDs."""
        await self.ensure_initialized()
        try:
            points = self._client.retrieve(
                collection_name=collection_name,
                ids=ids
            )
            return [point.dict() for point in points]
        except Exception as e:
            logger.error(f"Error retrieving points from {collection_name}: {e}")
            raise

    async def upsert(self, collection_name: str, points: List[Dict[str, Any]]):
        """Upsert points to a collection."""
        await self.ensure_initialized()
        try:
            self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point["payload"]
                    )
                    for point in points
                ]
            )
            logger.debug(f"Upserted {len(points)} points to {collection_name}")
        except Exception as e:
            logger.error(f"Error upserting to {collection_name}: {e}")
            raise

    async def get_collection(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information."""
        await self.ensure_initialized()
        try:
            collection = self._client.get_collection(collection_name=collection_name)
            return collection.dict()
        except Exception as e:
            logger.error(f"Error getting collection {collection_name}: {e}")
            raise

    async def get_collections(self):
        """Get all collections from Qdrant."""
        await self.ensure_initialized()
        try:
            return self._client.get_collections()
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            raise

    async def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine"):
        """Create a new collection in Qdrant."""
        await self.ensure_initialized()
        try:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE if distance.lower() == "cosine" else models.Distance.EUCLID
                )
            )
            logger.info(f"Created collection {collection_name}")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Collection {collection_name} already exists")
                return False
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    async def delete_collection(self, collection_name: str):
        """Delete a collection."""
        await self.ensure_initialized()
        try:
            self._client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            raise 