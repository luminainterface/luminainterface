"""Qdrant client for the crawler service."""
import os
import logging
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

    async def connect(self):
        """Connect to Qdrant."""
        if not self._initialized:
            try:
                self._client = BaseQdrantClient(url=self.url)
                self._initialized = True
                logger.info("Connected to Qdrant")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise

    async def close(self):
        """Close Qdrant connection."""
        if self._client:
            self._client.close()
            self._initialized = False
            logger.info("Closed Qdrant connection")

    async def init_collection(self, collection_name: str, vector_size: int = 384):
        """Initialize a collection if it doesn't exist."""
        if not self._initialized:
            await self.connect()
        try:
            collections = self._client.get_collections().collections
            if not any(c.name == collection_name for c in collections):
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection {collection_name}")
        except Exception as e:
            logger.error(f"Error initializing collection {collection_name}: {e}")
            raise

    async def upsert(self, collection_name: str, points: List[Dict[str, Any]]):
        """Upsert points to a collection."""
        if not self._initialized:
            await self.connect()
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

    async def get_collections(self):
        """Get all collections from Qdrant."""
        if not self._initialized:
            await self.connect()
        try:
            return self._client.get_collections()
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            raise

    async def create_collection(self, collection_name: str, vectors_config: Dict[str, Any]):
        """Create a new collection in Qdrant."""
        if not self._initialized:
            await self.connect()
        try:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vectors_config["size"],
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

# Create a singleton instance
qdrant_client = QdrantClient() 