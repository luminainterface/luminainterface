from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
import asyncio

class VectorPruner:
    def __init__(
        self,
        qdrant_url: str = None,
        collection_name: str = "lumina-chat",
        max_age_days: int = 30,
        min_similarity: float = 0.3
    ):
        """Initialize the vector pruner.
        
        Args:
            qdrant_url: Qdrant server URL
            collection_name: Name of the collection to prune
            max_age_days: Maximum age of vectors in days
            min_similarity: Minimum similarity score to keep
        """
        self.client = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL", "http://qdrant:6333")
        )
        self.collection_name = collection_name
        self.max_age_days = max_age_days
        self.min_similarity = min_similarity

    async def prune_vectors(self) -> Dict[str, int]:
        """Prune old and low-similarity vectors.
        
        Returns:
            Dict with counts of pruned vectors and remaining vectors
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            total_vectors = collection_info.points_count
            
            # Calculate cutoff date
            cutoff_date = datetime.utcnow() - timedelta(days=self.max_age_days)
            
            # Get all points with their timestamps
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your needs
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Filter points to delete
            points_to_delete = []
            for point in points:
                timestamp = datetime.fromisoformat(point.payload.get("timestamp", "2000-01-01"))
                if timestamp < cutoff_date:
                    # Check similarity if available
                    if "similarity" in point.payload:
                        if point.payload["similarity"] < self.min_similarity:
                            points_to_delete.append(point.id)
                    else:
                        points_to_delete.append(point.id)
            
            # Delete points in batches
            if points_to_delete:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=points_to_delete
                    )
                )
            
            # Get new count
            new_count = self.client.get_collection(self.collection_name).points_count
            
            return {
                "pruned": len(points_to_delete),
                "remaining": new_count,
                "total_before": total_vectors
            }
            
        except Exception as e:
            logger.error(f"Error pruning vectors: {e}")
            raise

async def run_pruning_job():
    """Run the pruning job and log results."""
    try:
        pruner = VectorPruner()
        results = await pruner.prune_vectors()
        
        logger.info(
            "Pruning complete",
            extra={
                "pruned": results["pruned"],
                "remaining": results["remaining"],
                "total_before": results["total_before"]
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Pruning job failed: {e}")
        raise

if __name__ == "__main__":
    # Run the pruning job
    asyncio.run(run_pruning_job()) 