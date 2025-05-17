import redis
import redis.asyncio as aioredis
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from datetime import datetime
import numpy as np
from prometheus_client import Counter, Histogram

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://:02211998@redis:6379")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# Initialize Redis clients
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)  # Sync client for compatibility
redis_async_client = aioredis.from_url(REDIS_URL, decode_responses=True)  # Async client for stream operations

# Initialize Qdrant client
qdrant_client = QdrantClient(os.getenv("QDRANT_URL", "http://qdrant:6333"))

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Metrics
CONCEPT_DEDUPLICATIONS = Counter(
    'concept_dictionary_deduplications_total',
    'Number of concept deduplications',
    ['type']  # 'exact' or 'similar'
)

CONCEPT_MERGES = Counter(
    'concept_dictionary_merges_total',
    'Number of concept merges'
)

USAGE_UPDATES = Counter(
    'concept_dictionary_usage_updates_total',
    'Number of concept usage updates'
)

DEDUPLICATION_TIME = Histogram(
    'concept_dictionary_deduplication_seconds',
    'Time spent on deduplication operations'
)

# Constants
SIMILARITY_THRESHOLD = float(os.getenv("CONCEPT_SIMILARITY_THRESHOLD", "0.85"))
MIN_USAGE_COUNT = int(os.getenv("MIN_USAGE_COUNT", "3"))
BLOCKED_LICENSES = set(os.getenv("BLOCKED_LICENSES", "proprietary,restricted").split(","))

class ConceptMetadata:
    """Metadata for a concept with usage tracking and license info."""
    def __init__(
        self,
        term: str,
        definition: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        usage_count: int = 1,
        license_type: Optional[str] = None,
        last_updated: Optional[str] = None,
        similar_concepts: Optional[List[str]] = None
    ):
        self.term = term
        self.definition = definition
        self.embedding = embedding
        self.metadata = metadata or {}
        self.usage_count = usage_count
        self.license_type = license_type
        self.last_updated = last_updated or datetime.utcnow().isoformat()
        self.similar_concepts = similar_concepts or []

    def to_dict(self) -> Dict:
        return {
            "term": self.term,
            "definition": self.definition,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "usage_count": self.usage_count,
            "license_type": self.license_type,
            "last_updated": self.last_updated,
            "similar_concepts": self.similar_concepts
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptMetadata':
        return cls(**data)

class ConceptDB:
    """Enhanced database operations for concepts with deduplication and usage tracking."""
    def __init__(self):
        self.redis = redis_client  # Sync client for compatibility
        self.redis_async = redis_async_client  # Async client for stream operations
        self.qdrant = qdrant_client
        self.model = model

    def _get_redis_key(self, term: str) -> str:
        """Get Redis key for a concept."""
        return f"concept:{term}"

    def _get_usage_key(self, term: str) -> str:
        """Get Redis key for concept usage tracking."""
        return f"concept:usage:{term}"

    def _get_similarity_key(self, term: str) -> str:
        """Get Redis key for similar concepts tracking."""
        return f"concept:similar:{term}"

    async def find_similar_concepts(
        self,
        embedding: List[float],
        threshold: float = SIMILARITY_THRESHOLD,
        limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar concepts using vector similarity."""
        try:
            results = self.qdrant.search(
                collection_name="concepts",
                query_vector=embedding,
                limit=limit,
                score_threshold=threshold
            )
            return [(point.id, point.score) for point in results]
        except Exception as e:
            logger.error(f"Error finding similar concepts: {e}")
            return []

    async def check_duplicate(
        self,
        term: str,
        definition: str,
        embedding: Optional[List[float]] = None
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check if a concept is a duplicate of an existing one."""
        with DEDUPLICATION_TIME.time():
            try:
                # First check exact match
                existing = self.redis.get(self._get_redis_key(term))
                if existing:
                    CONCEPT_DEDUPLICATIONS.labels(type="exact").inc()
                    return True, term, 1.0

                # If no exact match and we have an embedding, check similarity
                if embedding:
                    similar = await self.find_similar_concepts(embedding)
                    if similar:
                        best_match, score = similar[0]
                        if score >= SIMILARITY_THRESHOLD:
                            CONCEPT_DEDUPLICATIONS.labels(type="similar").inc()
                            return True, best_match, score

                return False, None, None
            except Exception as e:
                logger.error(f"Error checking duplicate: {e}")
                return False, None, None

    async def merge_concepts(
        self,
        source_term: str,
        target_term: str,
        merge_metadata: bool = True
    ) -> bool:
        """Merge two concepts, combining their metadata and usage counts."""
        try:
            source_data = self.redis.get(self._get_redis_key(source_term))
            target_data = self.redis.get(self._get_redis_key(target_term))
            
            if not source_data or not target_data:
                return False

            source = ConceptMetadata.from_dict(json.loads(source_data))
            target = ConceptMetadata.from_dict(json.loads(target_data))

            # Merge metadata
            if merge_metadata:
                target.metadata.update(source.metadata)
                target.usage_count += source.usage_count
                if source.license_type and not target.license_type:
                    target.license_type = source.license_type
                target.similar_concepts.extend(source.similar_concepts)
                target.similar_concepts = list(set(target.similar_concepts))

            # Update target in Redis
            self.redis.set(
                self._get_redis_key(target_term),
                json.dumps(target.to_dict())
            )

            # Update target in Qdrant if embeddings exist
            if target.embedding:
                self.qdrant.upsert(
                    collection_name="concepts",
                    points=[{
                        "id": target_term,
                        "vector": target.embedding,
                        "payload": target.to_dict()
                    }]
                )

            # Delete source
            self.redis.delete(self._get_redis_key(source_term))
            self.qdrant.delete(
                collection_name="concepts",
                points_selector={"ids": [source_term]}
            )

            CONCEPT_MERGES.inc()
            return True
        except Exception as e:
            logger.error(f"Error merging concepts: {e}")
            return False

    async def update_usage(
        self,
        term: str,
        increment: int = 1,
        update_metadata: Optional[Dict] = None
    ) -> bool:
        """Update usage count and metadata for a concept."""
        try:
            key = self._get_redis_key(term)
            data = self.redis.get(key)
            
            if not data:
                return False

            concept = ConceptMetadata.from_dict(json.loads(data))
            concept.usage_count += increment
            concept.last_updated = datetime.utcnow().isoformat()
            
            if update_metadata:
                concept.metadata.update(update_metadata)

            # Update in Redis
            self.redis.set(key, json.dumps(concept.to_dict()))
            
            # Update in Qdrant if embedding exists
            if concept.embedding:
                self.qdrant.upsert(
                    collection_name="concepts",
                    points=[{
                        "id": term,
                        "vector": concept.embedding,
                        "payload": concept.to_dict()
                    }]
                )

            USAGE_UPDATES.inc()
            return True
        except Exception as e:
            logger.error(f"Error updating usage: {e}")
            return False

    async def add_concept(
        self,
        term: str,
        definition: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
        license_type: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Add a new concept with deduplication and license checking."""
        try:
            # Check license
            if license_type and license_type.lower() in BLOCKED_LICENSES:
                logger.warning(f"Blocked concept with license {license_type}: {term}")
                return False, "license_blocked"

            # Check for duplicates
            is_duplicate, existing_term, similarity = await self.check_duplicate(
                term, definition, embedding
            )

            if is_duplicate:
                if similarity and similarity < 1.0:  # Similar but not exact
                    # Merge similar concepts
                    await self.merge_concepts(term, existing_term)
                # Update usage count for existing concept
                await self.update_usage(existing_term)
                return True, existing_term

            # Create new concept
            concept = ConceptMetadata(
                term=term,
                definition=definition,
                embedding=embedding,
                metadata=metadata,
                license_type=license_type
            )

            # Store in Redis
            self.redis.set(
                self._get_redis_key(term),
                json.dumps(concept.to_dict())
            )

            # Store in Qdrant if embedding exists
            if embedding:
                self.qdrant.upsert(
                    collection_name="concepts",
                    points=[{
                        "id": term,
                        "vector": embedding,
                        "payload": concept.to_dict()
                    }]
                )

            return True, term
        except Exception as e:
            logger.error(f"Error adding concept: {e}")
            return False, None

    def find(self, term: str) -> Optional[ConceptMetadata]:
        """Find a concept by term."""
        try:
            data = self.redis.get(self._get_redis_key(term))
            if data:
                return ConceptMetadata.from_dict(json.loads(data))
            return None
        except Exception as e:
            logger.error(f"Error finding concept: {e}")
            return None

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all concepts."""
        try:
            stats = {
                "total_concepts": 0,
                "total_usage": 0,
                "avg_usage": 0,
                "usage_distribution": {},
                "license_distribution": {}
            }

            for key in self.redis.scan_iter("concept:*"):
                if not key.startswith("concept:usage:") and not key.startswith("concept:similar:"):
                    data = self.redis.get(key)
                    if data:
                        concept = ConceptMetadata.from_dict(json.loads(data))
                        stats["total_concepts"] += 1
                        stats["total_usage"] += concept.usage_count
                        
                        # Update usage distribution
                        usage_range = (concept.usage_count // 10) * 10
                        stats["usage_distribution"][usage_range] = stats["usage_distribution"].get(usage_range, 0) + 1
                        
                        # Update license distribution
                        if concept.license_type:
                            stats["license_distribution"][concept.license_type] = stats["license_distribution"].get(concept.license_type, 0) + 1

            if stats["total_concepts"] > 0:
                stats["avg_usage"] = stats["total_usage"] / stats["total_concepts"]

            return stats
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {} 