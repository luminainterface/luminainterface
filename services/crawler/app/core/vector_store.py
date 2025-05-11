from typing import List, Dict, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, url: str, collection_name: str = "wiki_concepts"):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        
    def init_collection(self, vector_size: int = 384):
        """Initialize or recreate the collection with specified parameters"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists:
                logger.info(f"Collection {self.collection_name} already exists")
                return
                
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    def upsert_vectors(self, vectors: List[np.ndarray], metadata: List[Dict], ids: Optional[List[str]] = None):
        """Upsert vectors and their metadata to the collection, with flow-aware logging."""
        try:
            points = []
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                point_id = ids[i] if ids else str(i)
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=meta
                    )
                )
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info({
                "event": "data_flow",
                "source": "crawler",
                "destination": "qdrant",
                "protocol": "REST VEC",
                "action": "upsert",
                "status": "success",
                "details": {"collection": self.collection_name, "count": len(points), "ids": [p.id for p in points]}
            })
        except Exception as e:
            logger.error({
                "event": "data_flow",
                "source": "crawler",
                "destination": "qdrant",
                "protocol": "REST VEC",
                "action": "upsert",
                "status": "error",
                "details": {"collection": self.collection_name, "error": str(e)}
            })
            raise
            
    def search_similar(self, query_vector: np.ndarray, limit: int = 5) -> List[Dict]:
        """Search for similar vectors in the collection"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit
            )
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            return [] 