import redis
import redis.asyncio as aioredis
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from datetime import datetime, timedelta
import numpy as np
from prometheus_client import Counter, Histogram
import logging
import uuid
from uuid import uuid4
import asyncio
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("concept-dictionary-db")

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://:02211998@redis:6379")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "concepts")  # Add collection name constant

# Initialize Redis clients
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)  # Sync client for compatibility
redis_async_client = aioredis.from_url(REDIS_URL, decode_responses=True)  # Async client for stream operations

# Initialize Qdrant client
qdrant_client = QdrantClient(QDRANT_URL)

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

class ConceptMetadata(BaseModel):
    term: str
    definition: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}
    usage_count: int = 0
    last_updated: str = ""
    license_type: Optional[str] = None
    similar_concepts: List[str] = []
    version: int = 1  # Track concept version
    sync_state: Dict[str, Any] = {  # Enhanced sync state tracking
        "redis_synced": True,
        "qdrant_synced": True,
        "last_sync_attempt": None,
        "last_successful_sync": None,
        "sync_error": None,
        "sync_retry_count": 0,
        "sync_history": [],  # List of recent sync attempts
        "last_redis_version": 1,  # Version last synced to Redis
        "last_qdrant_version": 1,  # Version last synced to Qdrant
        "sync_priority": 0,  # Higher priority concepts sync first
        "sync_lock": None,  # Lock ID if sync is in progress
        "sync_lock_expiry": None,  # When the lock expires
        "sync_required": False,  # Flag if sync is needed
        "sync_required_reason": None  # Why sync is needed
    }

    def to_dict(self) -> Dict:
        return {
            "term": self.term,
            "definition": self.definition,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "usage_count": self.usage_count,
            "license_type": self.license_type,
            "last_updated": self.last_updated,
            "similar_concepts": self.similar_concepts,
            "version": self.version,
            "sync_state": self.sync_state
        }

    def mark_sync_attempt(self, success: bool, error: Optional[str] = None, target: str = "both") -> None:
        """Record a sync attempt in the sync history."""
        timestamp = datetime.utcnow().isoformat()
        attempt = {
            "timestamp": timestamp,
            "success": success,
            "error": error,
            "target": target,
            "version": self.version
        }
        
        # Keep only last 10 sync attempts
        self.sync_state["sync_history"] = (self.sync_state.get("sync_history", []) + [attempt])[-10:]
        
        if success:
            self.sync_state["last_successful_sync"] = timestamp
            self.sync_state["sync_error"] = None
            self.sync_state["sync_retry_count"] = 0
            if target in ["redis", "both"]:
                self.sync_state["redis_synced"] = True
                self.sync_state["last_redis_version"] = self.version
            if target in ["qdrant", "both"]:
                self.sync_state["qdrant_synced"] = True
                self.sync_state["last_qdrant_version"] = self.version
        else:
            self.sync_state["sync_error"] = error
            self.sync_state["sync_retry_count"] = self.sync_state.get("sync_retry_count", 0) + 1
            self.sync_state["sync_required"] = True
            self.sync_state["sync_required_reason"] = error

    def needs_sync(self, target: str = "both") -> bool:
        """Check if concept needs syncing to specified target(s)."""
        if target == "both":
            return (not self.sync_state["redis_synced"] or 
                   not self.sync_state["qdrant_synced"] or
                   self.sync_state["sync_required"])
        elif target == "redis":
            return (not self.sync_state["redis_synced"] or
                   self.sync_state["last_redis_version"] < self.version)
        elif target == "qdrant":
            return (not self.sync_state["qdrant_synced"] or
                   self.sync_state["last_qdrant_version"] < self.version)
        return False

    def acquire_sync_lock(self, lock_id: str, expiry_seconds: int = 30) -> bool:
        """Try to acquire a sync lock for this concept."""
        if self.sync_state["sync_lock"] and self.sync_state["sync_lock_expiry"]:
            # Check if existing lock is expired
            if datetime.fromisoformat(self.sync_state["sync_lock_expiry"]) > datetime.utcnow():
                return False
        
        self.sync_state["sync_lock"] = lock_id
        self.sync_state["sync_lock_expiry"] = (datetime.utcnow() + timedelta(seconds=expiry_seconds)).isoformat()
        return True

    def release_sync_lock(self, lock_id: str) -> bool:
        """Release the sync lock if it matches the given lock_id."""
        if self.sync_state["sync_lock"] == lock_id:
            self.sync_state["sync_lock"] = None
            self.sync_state["sync_lock_expiry"] = None
            return True
        return False

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptMetadata':
        return cls(**data)

async def ensure_qdrant_collection():
    """Ensure Qdrant collection exists with proper configuration."""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if QDRANT_COLLECTION not in collection_names:
            logger.info(f"Creating Qdrant collection: {QDRANT_COLLECTION}")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={
                    "size": 384,  # Size for all-MiniLM-L6-v2 model
                    "distance": "Cosine"
                }
            )
            logger.info(f"Created Qdrant collection: {QDRANT_COLLECTION}")
        return True
    except Exception as e:
        logger.error(f"Error ensuring Qdrant collection exists: {e}")
        return False

# Initialize Qdrant collection
asyncio.create_task(ensure_qdrant_collection())

class ConceptDB:
    """Enhanced database operations for concepts with deduplication and usage tracking."""
    def __init__(self):
        self.redis = redis_client
        self.redis_async = redis_async_client
        self.qdrant = qdrant_client
        self.model = model
        self._term_to_uuid = {}
        self.collection_name = QDRANT_COLLECTION
        self._sync_lock_prefix = "concept:sync_lock:"
        self._sync_retry_delay = 5  # seconds
        self._max_sync_retries = 3

    def _get_uuid_for_term(self, term: str) -> str:
        """Get or create a UUID for a term."""
        if term not in self._term_to_uuid:
            self._term_to_uuid[term] = str(uuid4())
        return self._term_to_uuid[term]

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
                collection_name=self.collection_name,  # Use stored collection name
                query_vector=embedding,
                limit=limit,
                score_threshold=threshold,
                with_payload=True
            )
            return [(point.payload.get("term", str(point.id)), point.score) for point in results]
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
                target_uuid = self._get_uuid_for_term(target_term)
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=[{
                        "id": target_uuid,
                        "vector": target.embedding,
                        "payload": {
                            **target.to_dict(),
                            "term": target_term,
                            "uuid": target_uuid
                        }
                    }]
                )

            # Delete source from Qdrant
            source_uuid = self._get_uuid_for_term(source_term)
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector={"ids": [source_uuid]}
            )

            # Delete source from Redis
            self.redis.delete(self._get_redis_key(source_term))

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
                    collection_name=self.collection_name,
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
        """Add a new concept with atomic write and sync tracking."""
        try:
            # Generate UUID for the concept
            concept_uuid = str(uuid4())
            
            # Create concept metadata
            concept = ConceptMetadata(
                term=term,
                definition=definition,
                embedding=embedding,
                metadata=metadata or {},
                license_type=license_type,
                last_updated=datetime.utcnow().isoformat(),
                version=1,
                sync_state={
                    "redis_synced": False,
                    "qdrant_synced": False,
                    "last_sync_attempt": datetime.utcnow().isoformat(),
                    "sync_error": None
                }
            )

            # Start Redis transaction
            async with redis_async_client.pipeline(transaction=True) as pipe:
                # Check if concept exists
                exists = await pipe.get(self._get_redis_key(term))
                await pipe.execute()
                
                if exists:
                    existing = ConceptMetadata.from_dict(json.loads(exists))
                    concept.version = existing.version + 1

                # Update Redis first (SST)
                try:
                    await redis_async_client.set(
                        self._get_redis_key(term),
                        json.dumps(concept.to_dict())
                    )
                    concept.sync_state["redis_synced"] = True
                except Exception as e:
                    logger.error(f"Failed to write concept '{term}' to Redis: {e}")
                    concept.sync_state["sync_error"] = f"Redis write failed: {str(e)}"
                    raise

                # Update Qdrant if embedding exists
                if embedding:
                    try:
                        qdrant_id = self._get_uuid_for_term(term)
                        await self.qdrant.upsert(
                            collection_name=self.collection_name,
                            points=[{
                                "id": qdrant_id,
                                "vector": embedding,
                                "payload": {
                                    **concept.to_dict(),
                                    "term": term,
                                    "uuid": qdrant_id,
                                    "version": concept.version
                                }
                            }]
                        )
                        concept.sync_state["qdrant_synced"] = True
                    except Exception as e:
                        logger.error(f"Failed to write concept '{term}' to Qdrant: {e}")
                        concept.sync_state["sync_error"] = f"Qdrant write failed: {str(e)}"
                        # Don't raise here - we want to keep the Redis write
                        # The concept will be marked for resync

                # Update Redis with final sync state
                await redis_async_client.set(
                    self._get_redis_key(term),
                    json.dumps(concept.to_dict())
                )

                return True, term
        except Exception as e:
            logger.error(f"Error adding concept: {e}")
            return False, None

    async def _acquire_redis_lock(self, term: str, lock_id: str, expiry_seconds: int = 30) -> bool:
        """Try to acquire a Redis-based lock for syncing a concept."""
        lock_key = f"{self._sync_lock_prefix}{term}"
        return await self.redis_async.set(
            lock_key,
            lock_id,
            ex=expiry_seconds,
            nx=True  # Only set if not exists
        )

    async def _release_redis_lock(self, term: str, lock_id: str) -> bool:
        """Release a Redis-based lock if it matches the given lock_id."""
        lock_key = f"{self._sync_lock_prefix}{term}"
        # Use Lua script for atomic check-and-delete
        script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        end
        return 0
        """
        result = await self.redis_async.eval(script, 1, lock_key, lock_id)
        return bool(result)

    async def atomic_write(self, concept: ConceptMetadata, update_type: str = "manual") -> bool:
        """Atomically write a concept to both Redis and Qdrant."""
        lock_id = str(uuid4())
        try:
            # Try to acquire Redis lock
            if not await self._acquire_redis_lock(concept.term, lock_id):
                logger.warning(f"Could not acquire sync lock for concept {concept.term}")
                return False

            # Increment version
            concept.version += 1
            
            # Write to Redis first (SST)
            try:
                await self.redis_async.set(
                    self._get_redis_key(concept.term),
                    json.dumps(concept.to_dict())
                )
                concept.mark_sync_attempt(True, target="redis")
            except Exception as e:
                logger.error(f"Failed to write concept {concept.term} to Redis: {e}")
                concept.mark_sync_attempt(False, str(e), target="redis")
                return False

            # Write to Qdrant if embedding exists
            if concept.embedding:
                try:
                    qdrant_id = self._get_uuid_for_term(concept.term)
                    await self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=[{
                            "id": qdrant_id,
                            "vector": concept.embedding,
                            "payload": {
                                **concept.to_dict(),
                                "term": concept.term,
                                "uuid": qdrant_id,
                                "version": concept.version
                            }
                        }]
                    )
                    concept.mark_sync_attempt(True, target="qdrant")
                except Exception as e:
                    logger.error(f"Failed to write concept {concept.term} to Qdrant: {e}")
                    concept.mark_sync_attempt(False, str(e), target="qdrant")
                    # Don't fail the whole operation - we'll heal on read
                    # But update Redis to reflect the sync state
                    await self.redis_async.set(
                        self._get_redis_key(concept.term),
                        json.dumps(concept.to_dict())
                    )

            return True
        finally:
            # Always release the lock
            await self._release_redis_lock(concept.term, lock_id)

    async def heal_concept(self, term: str, force: bool = False) -> bool:
        """Heal a concept by ensuring it's properly synced between Redis and Qdrant."""
        lock_id = str(uuid4())
        try:
            # Get concept from Redis (SST)
            data = await self.redis_async.get(self._get_redis_key(term))
            if not data:
                logger.warning(f"Concept {term} not found in Redis during healing")
                return False

            concept = ConceptMetadata.from_dict(json.loads(data))
            
            # Skip if no healing needed and not forced
            if not force and not concept.needs_sync():
                return True

            # Try to acquire lock
            if not await self._acquire_redis_lock(term, lock_id):
                logger.warning(f"Could not acquire sync lock for healing concept {term}")
                return False

            healed = False
            retry_count = 0

            while retry_count < self._max_sync_retries and not healed:
                try:
                    # Check Qdrant state
                    qdrant_id = self._get_uuid_for_term(term)
                    point = await self.qdrant.retrieve(
                        collection_name=self.collection_name,
                        ids=[qdrant_id],
                        with_payload=True
                    )

                    needs_qdrant_update = (
                        not point or
                        point.payload.get("version", 0) < concept.version or
                        not concept.sync_state["qdrant_synced"]
                    )

                    if needs_qdrant_update and concept.embedding:
                        # Update Qdrant
                        await self.qdrant.upsert(
                            collection_name=self.collection_name,
                            points=[{
                                "id": qdrant_id,
                                "vector": concept.embedding,
                                "payload": {
                                    **concept.to_dict(),
                                    "term": term,
                                    "uuid": qdrant_id,
                                    "version": concept.version
                                }
                            }]
                        )
                        concept.mark_sync_attempt(True, target="qdrant")
                        healed = True
                    elif not needs_qdrant_update:
                        # Qdrant is up to date
                        concept.mark_sync_attempt(True, target="qdrant")
                        healed = True

                    # Update Redis with new sync state
                    await self.redis_async.set(
                        self._get_redis_key(term),
                        json.dumps(concept.to_dict())
                    )

                except Exception as e:
                    logger.error(f"Error healing concept {term} (attempt {retry_count + 1}): {e}")
                    concept.mark_sync_attempt(False, str(e))
                    retry_count += 1
                    if retry_count < self._max_sync_retries:
                        await asyncio.sleep(self._sync_retry_delay)

            return healed

        finally:
            await self._release_redis_lock(term, lock_id)

    async def find(self, term: str, heal_on_read: bool = True) -> Optional[ConceptMetadata]:
        """Find a concept by term, with optional healing on read."""
        try:
            # Get from Redis (SST)
            data = await self.redis_async.get(self._get_redis_key(term))
            if not data:
                return None

            concept = ConceptMetadata.from_dict(json.loads(data))
            
            # If healing is enabled and concept needs sync
            if heal_on_read and concept.needs_sync():
                await self.heal_concept(term)
                # Re-fetch to get updated sync state
                data = await self.redis_async.get(self._get_redis_key(term))
                if data:
                    concept = ConceptMetadata.from_dict(json.loads(data))

            return concept
        except Exception as e:
            logger.error(f"Error finding concept {term}: {e}")
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

    def is_concept_key(self, key):
        """Check if a Redis key is a string (concept)."""
        try:
            key_type = self.redis.type(key)
            return key_type == 'string'  # Redis returns 'string' when decode_responses=True
        except Exception as e:
            logging.getLogger("concept-dictionary").error(f"Error checking Redis key type for {key}: {e}")
            return False

    async def start_healing_cron(self, interval_seconds: int = 300):
        """Start a periodic healing job."""
        while True:
            try:
                # Get all concept keys
                async for key in self.redis_async.scan_iter("concept:*"):
                    if await self.is_concept_key(key):
                        term = key.replace("concept:", "")
                        try:
                            # Get concept data
                            data = await self.redis_async.get(key)
                            if data:
                                concept = ConceptMetadata.from_dict(json.loads(data))
                                if concept.needs_sync():
                                    # Prioritize concepts with more retries
                                    concept.sync_state["sync_priority"] = concept.sync_state.get("sync_retry_count", 0)
                                    # Update Redis with priority
                                    await self.redis_async.set(key, json.dumps(concept.to_dict()))
                        except Exception as e:
                            logger.error(f"Error processing concept {term} in healing cron: {e}")
                            continue

                # Sort concepts by priority and heal them
                async for key in self.redis_async.scan_iter("concept:*"):
                    if await self.is_concept_key(key):
                        term = key.replace("concept:", "")
                        try:
                            data = await self.redis_async.get(key)
                            if data:
                                concept = ConceptMetadata.from_dict(json.loads(data))
                                if concept.sync_state.get("sync_priority", 0) > 0:
                                    await self.heal_concept(term)
                        except Exception as e:
                            logger.error(f"Error healing concept {term} in cron: {e}")
                            continue

            except Exception as e:
                logger.error(f"Error in healing cron: {e}")

            await asyncio.sleep(interval_seconds) 