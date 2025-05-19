from fastapi import FastAPI, HTTPException, Request, APIRouter, Depends, Header, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import json
import os
import logging
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
from datetime import datetime, timedelta
import httpx
import threading
import time
import asyncio
from .ingest_consumer import start_consumers, _service_paused, SYNC_THRESHOLD, reconcile, RECONCILING, SYNC_DIFF, REPAIR, DRIFT
from .db import redis_client, redis_async_client, qdrant_client, model, ConceptDB, SIMILARITY_THRESHOLD, MIN_USAGE_COUNT, BLOCKED_LICENSES, QDRANT_COLLECTION, ConceptMetadata
from .models import Concept
from .auto_digest import AutoDigest
from fastapi.responses import JSONResponse
import redis
import uuid
from redis.asyncio import Redis
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from .metrics import (
    CONCEPT_OPERATIONS,
    CONCEPT_SYNC_OPERATIONS,
    CONCEPT_DIGEST_OPERATIONS,
    CONCEPT_OPERATION_LATENCY,
    CONCEPT_SYNC_LATENCY,
    CONCEPT_DIGEST_LATENCY,
    CONCEPT_QUALITY_SCORE,
    CONCEPT_SYNC_ERRORS,
    CONCEPT_DIGEST_ERRORS,
    CONCEPT_QUEUE_SIZE,
    CONCEPT_DIGEST_QUEUE_SIZE,
    CONCEPT_TRAINING_QUEUE_SIZE,
    CONCEPT_TRAINING_STATUS,
    CONCEPT_TRAINING_LATENCY,
    CONCEPT_TRAINING_ERRORS
)

# Environment variables
API_KEY = os.getenv("CONCEPT_DICT_API_KEY", "changeme")
ENV = os.getenv("ENV", "dev")
DEBUG = os.getenv("DEBUG", "1")

# Initialize FastAPI app
app = FastAPI(title="Concept Dictionary", root_path="")
Instrumentator().instrument(app).expose(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("concept-dictionary")

# Initialize metrics
CONCEPT_UPDATES = Counter(
    'concept_dictionary_updates_total',
    'Number of concept updates'
)

CONCEPT_RETRIEVALS = Counter(
    'concept_dictionary_retrievals_total',
    'Number of concept retrievals'
)

RETRIEVAL_LATENCY = Histogram(
    'concept_dictionary_retrieval_latency_seconds',
    'Time spent retrieving concepts'
)

# Initialize database
db = ConceptDB()

# Start healing cron in background
asyncio.create_task(db.start_healing_cron())

# Initialize auto-digestion
auto_digest = None

class MetaRequest(BaseModel):
    cids: List[str]

class ConceptUpdate(BaseModel):
    """Update request for a concept."""
    definition: Optional[str] = None
    metadata: Optional[Dict] = None
    license_type: Optional[str] = None

class ConceptMerge(BaseModel):
    """Request to merge two concepts."""
    source_term: str
    target_term: str
    merge_metadata: bool = True

class ConceptStats(BaseModel):
    """Response model for concept statistics."""
    total_concepts: int
    total_usage: int
    avg_usage: float
    usage_distribution: Dict[int, int]
    license_distribution: Dict[str, int]
    last_updated: str

class ConceptTrainingStatus(BaseModel):
    """Model for tracking concept training status"""
    term: str
    status: str  # "pending", "training", "trained", "failed"
    training_started: Optional[str] = None
    training_completed: Optional[str] = None
    error: Optional[str] = None
    training_version: int = 1
    last_attempt: Optional[str] = None
    retry_count: int = 0

class FetchConceptsRequest(BaseModel):
    """Request model for fetching untrained concepts"""
    batch_size: int = 10
    max_retries: int = 3
    min_quality_score: float = 0.0

class FetchConceptsResponse(BaseModel):
    """Response model for fetched concepts"""
    concepts: List[Dict[str, Any]]
    batch_id: str
    total_available: int

class TestConceptInput(BaseModel):
    """Model for test concept injection."""
    term: str = Field(..., description="The concept term")
    definition: str = Field(..., description="The concept definition")
    examples: List[str] = Field(default_factory=list, description="Example usages")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    force_quality_score: Optional[float] = Field(None, description="Force a specific quality score for testing")

async def is_concept_key(key: str) -> bool:
    """Helper to check if a Redis key is a string (concept)."""
    try:
        key_type = await redis_async_client.type(key)
        return key_type == 'string'  # Redis returns 'string' when decode_responses=True
    except redis.RedisError as e:
        logger.error(f"Error checking Redis key type for {key}: {e}")
        return False

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check Redis connection
        await redis_async_client.ping()
        
        # Check Qdrant connection
        qdrant_client.get_collections()
        
        # Get concept counts (async)
        redis_keys = await scan_redis_keys("concept:*")
        redis_count = 0
        for key in redis_keys:
            if await is_concept_key(key):
                redis_count += 1
        qdrant_count = await get_qdrant_point_count()
        
        # Calculate sync status
        difference = abs(redis_count - qdrant_count)
        sync_status = "healthy" if difference <= SYNC_THRESHOLD else "degraded"
        
        return JSONResponse(
            status_code=200 if sync_status == "healthy" else 503,
            content={
                "status": sync_status,
                "dependencies": {
                    "redis": "connected",
                    "qdrant": "connected"
                },
                "sync_status": {
                    "redis_concepts": redis_count,
                    "qdrant_concepts": qdrant_count,
                    "difference": redis_count - qdrant_count,
                    "status": sync_status,
                    "threshold": SYNC_THRESHOLD,
                    "service_paused": _service_paused
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/concepts")
async def get_concepts() -> List[Dict]:
    """Get all concepts, skipping non-string keys (e.g., streams)"""
    with RETRIEVAL_LATENCY.time():
        try:
            concepts = []
            redis_keys = await scan_redis_keys("concept:*")
            for key in redis_keys:
                try:
                    # Skip non-string keys (like streams)
                    if not await is_concept_key(key):
                        continue
                    concept_data = await redis_async_client.get(key)
                    if concept_data:
                        concepts.append(json.loads(concept_data))
                except redis.ResponseError as e:
                    # Log and skip keys that cause Redis errors
                    logger.warning(f"Skipping key {key} due to Redis error: {e}")
                    continue
            CONCEPT_RETRIEVALS.inc()
            return concepts
        except Exception as e:
            logger.error(f"Error retrieving concepts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/concepts/{term}")
async def get_concept(term: str, heal: bool = True):
    """Get a concept with optional healing on read."""
    try:
        concept = await db.find(term, heal_on_read=heal)
        if not concept:
            raise HTTPException(status_code=404, detail=f"Concept {term} not found")
        return concept.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting concept {term}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/sync_pdf_embeddings")
async def sync_pdf_embeddings():
    """Sync PDF embeddings from the pdf_vectors_768 collection to the concepts collection."""
    try:
        # Get all vectors from the PDF collection
        pdf_vectors = await qdrant_client.scroll(
            collection_name="pdf_vectors_768",
            limit=1000,
            with_payload=True,
            with_vectors=True
        )
        imported = 0
        scroll_offset = None
        # Get initial counts for logging
        initial_redis_count = len([key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)])
        initial_qdrant_count = await get_qdrant_point_count()
        logger.info(f"Starting PDF embeddings sync. Initial counts - Redis: {initial_redis_count}, Qdrant: {initial_qdrant_count}")
        # Get PDF embeddings from the 768d collection
        while True:
            result, next_page = qdrant_client.scroll(
                collection_name="pdf_vectors_768",
                limit=256,  # reasonable batch size
                with_payload=True,
                with_vectors=True,
                offset=scroll_offset
            )
            if not result:
                break
            for point in result:
                try:
                    payload = point.payload or {}
                    text = payload.get('text', '')
                    metadata = payload.get('metadata', {})
                    term = metadata.get('title', f"pdf_concept_{point.id}")
                    concept_embedding = model.encode(text)
                    concept = {
                        "term": term,
                        "definition": text[:500],
                        "embedding": concept_embedding.tolist(),
                        "metadata": {
                            **metadata,
                            "source": "pdf_embedding",
                            "original_id": point.id,
                            "original_embedding_dim": 768
                        },
                        "last_updated": datetime.utcnow().isoformat()
                    }
                    redis_client.set(f"concept:{term}", json.dumps(concept))
                    qdrant_client.upsert(
                        collection_name="concepts",
                        points=[{
                            "id": term,
                            "vector": concept_embedding.tolist(),
                            "payload": concept
                        }]
                    )
                    imported += 1
                except Exception as e:
                    logger.error(f"Error processing PDF embedding {point.id}: {e}")
                    continue
            if not next_page:
                break
            scroll_offset = next_page
        final_redis_count = len([key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)])
        final_qdrant_count = await get_qdrant_point_count()
        logger.info(
            f"PDF embeddings sync completed:\n"
            f"  - Imported: {imported} concepts\n"
            f"  - Final counts - Redis: {final_redis_count}, Qdrant: {final_qdrant_count}\n"
            f"  - Net change: {final_redis_count - initial_redis_count}"
        )
        return {
            "status": "success",
            "imported": imported,
            "counts": {
                "initial_redis": initial_redis_count,
                "initial_qdrant": initial_qdrant_count,
                "final_redis": final_redis_count,
                "final_qdrant": final_qdrant_count
            }
        }
    except Exception as e:
        logger.error(f"Error syncing PDF embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_embedding(text: str) -> List[float]:
    """Generate an embedding for a text using the sentence transformer model."""
    try:
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

@app.post("/concepts", response_model=Concept)
async def add_concept(concept: Concept):
    """Add or update a concept with atomic write."""
    try:
        # Convert to ConceptMetadata
        metadata = ConceptMetadata(
            term=concept.term,
            definition=concept.definition,
            embedding=concept.embedding,
            metadata=concept.metadata or {},
            license_type=concept.license_type
        )

        # Use atomic write
        success = await db.atomic_write(metadata)
        if not success:
            raise HTTPException(
                status_code=503,
                detail="Failed to write concept - service temporarily unavailable"
            )

        # Return the concept
        return concept
    except Exception as e:
        logger.error(f"Exception in add_concept: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.put("/concepts/{term}")
async def update_concept(term: str, concept: Concept):
    """Update a concept with atomic write."""
    try:
        # Get existing concept
        existing = await db.find(term, heal_on_read=True)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Concept {term} not found")

        # Update fields
        metadata = ConceptMetadata(
            term=term,  # Keep original term
            definition=concept.definition,
            embedding=concept.embedding or existing.embedding,
            metadata=concept.metadata or existing.metadata,
            license_type=concept.license_type or existing.license_type,
            usage_count=existing.usage_count,
            similar_concepts=existing.similar_concepts,
            version=existing.version  # Will be incremented in atomic_write
        )

        # Use atomic write
        success = await db.atomic_write(metadata)
        if not success:
            raise HTTPException(
                status_code=503,
                detail="Failed to update concept - service temporarily unavailable"
            )

        return {"status": "success", "message": f"Concept {term} updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating concept {term}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/concepts/{term}")
async def delete_concept(term: str):
    """Delete a concept"""
    try:
        # Remove from Redis
        if redis_client.delete(f"concept:{term}") == 0:
            raise HTTPException(status_code=404, detail="Concept not found")
        
        # Remove from Qdrant
        try:
            qdrant_client.delete(
                collection_name="concepts",
                points_selector={"ids": [term]}
            )
        except Exception as e:
            logger.warning(f"Error removing concept from Qdrant: {e}")
        
        return {"status": "success", "message": f"Concept {term} deleted"}
    except Exception as e:
        logger.error(f"Error deleting concept {term}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/reembed")
async def reembed_concepts():
    """Re-embed all concepts and update Qdrant"""
    try:
        concepts = await get_concepts()
        reembedded = 0
        for concept in concepts:
            try:
                # Generate new embedding
                text = f"{concept['term']} {concept.get('definition', '')}"
                embedding = generate_embedding(text)
                if embedding:
                    concept['embedding'] = embedding
                    # Update in Redis
                    redis_client.set(
                        f"concept:{concept['term']}",
                        json.dumps(concept)
                    )
                    # Update in Qdrant
                    qdrant_client.upsert(
                        collection_name="concepts",
                        points=[{
                            "id": concept['term'],
                            "vector": embedding,
                            "payload": {
                                "term": concept['term'],
                                "definition": concept.get('definition', ''),
                                "metadata": concept.get('metadata', {})
                            }
                        }]
                    )
                    reembedded += 1
            except Exception as e:
                logger.error(f"Error re-embedding concept {concept.get('term', 'unknown')}: {e}")
        
        return {"status": "success", "message": f"Re-embedded {reembedded} concepts"}
    except Exception as e:
        logger.error(f"Error during re-embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_qdrant_point_count() -> int:
    """Get the total number of points in the Qdrant concepts collection using the scroll API."""
    try:
        total = 0
        scroll_offset = None
        while True:
            points, next_page = qdrant_client.scroll(
                collection_name="concepts",
                limit=256,
                with_payload=False,
                with_vectors=False,
                offset=scroll_offset
            )
            total += len(points)
            if not next_page:
                break
            scroll_offset = next_page
        return total
    except Exception as e:
        logger.error(f"Error getting Qdrant point count via scroll: {e}")
        return -1

def log_concept_count():
    while True:
        try:
            redis_count = len([key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)])
            # Get Qdrant count using asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            qdrant_count = loop.run_until_complete(get_qdrant_point_count())
            loop.close()
            logger.info(
                f"Concept Dictionary Status:\n"
                f"  - Redis concepts: {redis_count}\n"
                f"  - Qdrant concepts: {qdrant_count}\n"
                f"  - Difference: {redis_count - qdrant_count if qdrant_count >= 0 else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"Error counting concepts: {e}")
        time.sleep(30)

@app.get("/concepts/status")
async def get_concept_status():
    """Get the current status of concepts in both Redis and Qdrant"""
    try:
        redis_count = len([key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)])
        qdrant_count = await get_qdrant_point_count()
        
        # Update sync difference metric
        SYNC_DIFF.set(redis_count - qdrant_count)
        
        # Check if reconciliation is needed
        needs_reconciliation = abs(redis_count - qdrant_count) > SYNC_THRESHOLD
        if needs_reconciliation and not RECONCILING._value.get():  # Use _value.get() to check current value
            logger.warning(f"Sync difference detected: Redis={redis_count}, Qdrant={qdrant_count}, diff={redis_count - qdrant_count}")
            # Trigger reconciliation in background
            asyncio.create_task(reconcile())
        
        return {
            "redis_count": redis_count,
            "qdrant_count": qdrant_count,
            "difference": redis_count - qdrant_count,
            "needs_reconciliation": needs_reconciliation,
            "is_reconciling": bool(RECONCILING._value.get())  # Use _value.get() to check current value
        }
    except Exception as e:
        logger.error(f"Error getting concept status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def scan_redis_keys(pattern: str) -> List[str]:
    """Async helper to scan Redis keys matching a pattern."""
    keys = []
    cursor = 0
    while True:
        cursor, batch = await redis_async_client.scan(cursor, match=pattern, count=100)
        keys.extend(batch)
        if cursor == 0:
            break
    return keys

@app.post("/concepts/import_from_redis")
async def import_from_redis():
    """
    Import all concepts from Redis into Qdrant. This will ensure concepts in Redis are also in Qdrant.
    """
    try:
        imported = 0
        redis_keys = await scan_redis_keys("concept:*")
        redis_keys = [key for key in redis_keys if await redis_async_client.type(key) == 'string']
        
        # Get initial counts for logging
        initial_redis_count = len(redis_keys)
        initial_qdrant_count = await get_qdrant_point_count()
        logger.info(f"Starting import from Redis. Initial counts - Redis: {initial_redis_count}, Qdrant: {initial_qdrant_count}")
        
        # Initialize UUID mapping
        term_to_uuid = {}
        
        for key in redis_keys:
            try:
                concept_data = await redis_async_client.get(key)
                if not concept_data:
                    continue
                concept = json.loads(concept_data)
                term = key.replace("concept:", "")
                
                # Generate or get UUID for term
                if term not in term_to_uuid:
                    term_to_uuid[term] = str(uuid.uuid4())
                qdrant_id = term_to_uuid[term]
                
                if not concept.get("embedding"):
                    text = f"{term} {concept.get('definition', '')}"
                    embedding = model.encode(text)
                    concept["embedding"] = embedding.tolist()
                    await redis_async_client.set(key, json.dumps(concept))
                
                # Update concept with UUID
                concept["uuid"] = qdrant_id
                
                qdrant_client.upsert(
                    collection_name="concepts",
                    points=[{
                        "id": qdrant_id,
                        "vector": concept["embedding"],
                        "payload": {
                            **concept,
                            "term": term,  # Keep term in payload
                            "uuid": qdrant_id  # Store UUID in payload
                        }
                    }]
                )
                imported += 1
            except Exception as e:
                logger.error(f"Error importing concept {key} from Redis: {e}")
                continue
                
        final_redis_count = len(redis_keys)
        final_qdrant_count = await get_qdrant_point_count()
        logger.info(
            f"Import from Redis completed:\n"
            f"  - Imported: {imported} concepts\n"
            f"  - Final counts - Redis: {final_redis_count}, Qdrant: {final_qdrant_count}\n"
            f"  - Net change: {final_qdrant_count - initial_qdrant_count}"
        )
        return {
            "status": "success",
            "imported": imported,
            "counts": {
                "initial_redis": initial_redis_count,
                "initial_qdrant": initial_qdrant_count,
                "final_redis": final_redis_count,
                "final_qdrant": final_qdrant_count
            }
        }
    except Exception as e:
        logger.error(f"Error importing from Redis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/import_from_qdrant")
async def import_from_qdrant():
    """
    Import all concepts from Qdrant into Redis, forcing updates for all concepts.
    This will overwrite any existing concepts in Redis with data from Qdrant.
    """
    try:
        imported = 0
        failed = 0
        retry_count = 0
        max_retries = 3
        scroll_offset = None
        
        # Get initial counts for logging (async)
        redis_keys = await scan_redis_keys("concept:*")
        initial_redis_count = 0
        for key in redis_keys:
            if await is_concept_key(key):
                initial_redis_count += 1
        initial_qdrant_count = await get_qdrant_point_count()
        
        logger.info(f"Starting import from Qdrant. Initial counts - Redis: {initial_redis_count}, Qdrant: {initial_qdrant_count}")
        
        # Clear all concept keys from Redis first
        logger.info("Clearing all concept keys from Redis...")
        deleted = 0
        for key in redis_keys:
            if await is_concept_key(key):
                await redis_async_client.delete(key)
                deleted += 1
        logger.info(f"Deleted {deleted} concept keys from Redis")
        
        # Verify Redis is empty before proceeding
        redis_keys_after_clear = await scan_redis_keys("concept:*")
        remaining = 0
        for key in redis_keys_after_clear:
            if await is_concept_key(key):
                remaining += 1
        logger.info(f"After clearing, {remaining} concept keys remain in Redis")
        if remaining > 0:
            raise HTTPException(status_code=500, detail=f"Failed to clear Redis: {remaining} concept keys still present")
        
        logger.info("Redis cleared successfully, proceeding with import...")
        
        while True:
            try:
                # Use scroll API with pagination
                result, next_page = qdrant_client.scroll(
                    collection_name="concepts",
                    limit=1000,  # Increased batch size
                    with_payload=True,
                    with_vectors=True,
                    offset=scroll_offset
                )
                
                if not result:
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.warning(f"No results returned, retrying (attempt {retry_count}/{max_retries})...")
                        await asyncio.sleep(5)  # Wait before retry
                        continue
                    break
                
                retry_count = 0
                
                # Process batch of concepts
                batch = []
                for point in result:
                    try:
                        payload = point.payload or {}
                        term = payload.get("term") or str(point.id)
                        if not term:
                            logger.warning(f"Skipping point with empty term: {point.id}")
                            continue
                        concept = {
                            "term": term,
                            "definition": payload.get("definition", ""),
                            "embedding": point.vector,
                            "metadata": payload.get("metadata", {}),
                            "last_updated": payload.get("last_updated", datetime.utcnow().isoformat()),
                            "uuid": str(point.id)
                        }
                        for k, v in payload.items():
                            if k not in concept and k not in ["term", "uuid"]:
                                concept[k] = v
                        batch.append((f"concept:{term}", json.dumps(concept)))
                    except Exception as e:
                        logger.error(f"Error processing concept {term}: {e}")
                        failed += 1
                        continue
                if batch:
                    pipe = redis_async_client.pipeline()
                    for key, value in batch:
                        pipe.set(key, value)
                    results = await pipe.execute()
                    successful = sum(1 for r in results if r)
                    imported += successful
                    failed += len(batch) - successful
                    if imported % 1000 == 0:
                        logger.info(f"Imported {imported} concepts so far...")
                if not next_page:
                    break
                scroll_offset = next_page
            except Exception as e:
                if retry_count < max_retries:
                    retry_count += 1
                    logger.error(f"Error during Qdrant scroll, retrying (attempt {retry_count}/{max_retries}): {e}")
                    await asyncio.sleep(5)
                    continue
                else:
                    logger.error(f"Error during Qdrant scroll after retries: {e}")
                    raise
        # Get final counts for logging (async)
        redis_keys_final = await scan_redis_keys("concept:*")
        final_redis_count = 0
        for key in redis_keys_final:
            if await is_concept_key(key):
                final_redis_count += 1
        final_qdrant_count = await get_qdrant_point_count()
        logger.info(
            f"Import from Qdrant completed:\n"
            f"  - Imported: {imported} concepts\n"
            f"  - Failed: {failed} concepts\n"
            f"  - Final counts - Redis: {final_redis_count}, Qdrant: {final_qdrant_count}\n"
            f"  - Net change: {final_redis_count - initial_redis_count}\n"
            f"  - Sync ratio: {((imported / (imported + failed)) * 100 if (imported + failed) > 0 else 0):.2f}%"
        )
        if failed > 0:
            logger.warning(f"Import completed with {failed} failures")
        if final_redis_count != final_qdrant_count:
            logger.error(f"Sync verification failed: Redis count ({final_redis_count}) != Qdrant count ({final_qdrant_count})")
            raise HTTPException(
                status_code=500,
                detail=f"Sync verification failed: Redis count ({final_redis_count}) != Qdrant count ({final_qdrant_count})"
            )
        return {
            "status": "success",
            "imported": imported,
            "failed": failed,
            "counts": {
                "initial_redis": initial_redis_count,
                "initial_qdrant": initial_qdrant_count,
                "final_redis": final_redis_count,
                "final_qdrant": final_qdrant_count
            },
            "sync_ratio": ((imported / (imported + failed)) * 100 if (imported + failed) > 0 else 0)
        }
    except Exception as e:
        logger.error(f"Error importing from Qdrant: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/concepts/test_qdrant")
async def test_qdrant():
    """Test Qdrant connection and (optionally) scroll a few points from the 'concepts' collection."""
    try:
        # Test connection by listing collections
        collections = qdrant_client.get_collections()
        logger.info(f"Qdrant connection test: collections={collections}")

        # Optionally, scroll a few points from the 'concepts' collection
        (points, _) = qdrant_client.scroll(collection_name="concepts", limit=5, with_payload=True, with_vectors=False)
        logger.info(f"Qdrant scroll test: {len(points)} points returned (sample)")

        return {"status": "ok", "collections": [c.name for c in collections], "sample_points": len(points)}
    except Exception as e:
        logger.error(f"Qdrant test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Qdrant test failed: {e}")

@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    # Allow health checks and docs access
    if request.url.path.startswith(("/health", "/docs", "/openapi.json", "/redoc")):
        return await call_next(request)
    # Skip API key check in dev mode
    if ENV == "dev" or DEBUG == "1":
        logger.info(f"Bypassing API key check in {ENV} mode for {request.url.path}")
        return await call_next(request)
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)

async def periodic_sync():
    """Background task that periodically syncs concepts between Redis and Qdrant."""
    consecutive_failures = 0
    max_consecutive_failures = 3
    sync_interval = 300  # 5 minutes
    alert_threshold = 0.95  # Alert if sync ratio is below 95%
    
    while True:
        try:
            # Get initial counts
            initial_redis_count = len([key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)])
            initial_qdrant_count = await get_qdrant_point_count()
            
            logger.info(f"Starting periodic sync. Initial counts - Redis: {initial_redis_count}, Qdrant: {initial_qdrant_count}")
            
            # Sync from Redis to Qdrant
            logger.info("Starting periodic import from Redis...")
            redis_resp = await import_from_redis()
            logger.info(f"Periodic import from Redis completed: {redis_resp}")
            
            # Sync from Qdrant to Redis
            logger.info("Starting periodic import from Qdrant...")
            qdrant_resp = await import_from_qdrant()
            logger.info(f"Periodic import from Qdrant completed: {qdrant_resp}")
            
            # Get final counts
            final_redis_count = len([key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)])
            final_qdrant_count = await get_qdrant_point_count()
            
            # Calculate sync metrics
            redis_sync_ratio = redis_resp.get("sync_ratio", 0) if isinstance(redis_resp, dict) else 0
            qdrant_sync_ratio = qdrant_resp.get("sync_ratio", 0) if isinstance(qdrant_resp, dict) else 0
            if redis_sync_ratio == 0 and qdrant_sync_ratio == 0:
                overall_sync_ratio = 0
            else:
                overall_sync_ratio = (redis_sync_ratio + qdrant_sync_ratio) / 2
            
            # Check if sync was successful
            if overall_sync_ratio < alert_threshold:
                logger.error(
                    f"Sync quality below threshold:\n"
                    f"  - Redis sync ratio: {redis_sync_ratio:.2%}\n"
                    f"  - Qdrant sync ratio: {qdrant_sync_ratio:.2%}\n"
                    f"  - Overall sync ratio: {overall_sync_ratio:.2%}\n"
                    f"  - Initial counts - Redis: {initial_redis_count}, Qdrant: {initial_qdrant_count}\n"
                    f"  - Final counts - Redis: {final_redis_count}, Qdrant: {final_qdrant_count}"
                )
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                logger.info(
                    f"Sync completed successfully:\n"
                    f"  - Redis sync ratio: {redis_sync_ratio:.2%}\n"
                    f"  - Qdrant sync ratio: {qdrant_sync_ratio:.2%}\n"
                    f"  - Overall sync ratio: {overall_sync_ratio:.2%}\n"
                    f"  - Initial counts - Redis: {initial_redis_count}, Qdrant: {initial_qdrant_count}\n"
                    f"  - Final counts - Redis: {final_redis_count}, Qdrant: {final_qdrant_count}"
                )
            
            # If we've had too many consecutive failures, reduce sync interval
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    f"Too many consecutive sync failures ({consecutive_failures}). "
                    f"Reducing sync interval to 60 seconds until next successful sync."
                )
                sync_interval = 60
            else:
                sync_interval = 300  # Reset to 5 minutes
            
        except Exception as e:
            logger.error(f"Periodic sync failed: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    f"Too many consecutive sync failures ({consecutive_failures}). "
                    f"Reducing sync interval to 60 seconds until next successful sync."
                )
                sync_interval = 60
        
        await asyncio.sleep(sync_interval)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        # Start the concept count logging thread
        threading.Thread(target=log_concept_count, daemon=True).start()
        
        # Start the consumers
        asyncio.create_task(start_consumers())
        
        # Initialize and start auto-digestion
        global auto_digest
        auto_digest = AutoDigest(db)
        asyncio.create_task(auto_digest.start())
        
        logger.info("Started concept dictionary services")

        # Ensure collection exists
        try:
            if QDRANT_COLLECTION not in [c.name for c in qdrant_client.get_collections().collections]:
                qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config={"size": 384, "distance": "Cosine"},  # Updated to match model dimension
                    optimizers_config={"indexing_threshold": 10000}
                )
                logger.info(f"Created Qdrant collection '{QDRANT_COLLECTION}'")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection exists: {e}")

        # Start initial sync in the background
        async def initial_sync():
            try:
                # Initial sync from Redis to Qdrant
                logger.info("Starting import from Redis on startup...")
                try:
                    redis_resp = await import_from_redis()
                    logger.info(f"Startup import from Redis completed: {redis_resp}")
                except Exception as e:
                    logger.error(f"Startup import from Redis failed: {e}")

                # Initial sync from Qdrant to Redis
                logger.info("Starting import from Qdrant on startup...")
                try:
                    qdrant_resp = await import_from_qdrant()
                    logger.info(f"Startup import from Qdrant completed: {qdrant_resp}")
                except Exception as e:
                    logger.error(f"Startup import from Qdrant failed: {e}")
            except Exception as e:
                logger.error(f"Error in initial sync: {e}")

        # Start the initial sync in the background without waiting
        asyncio.create_task(initial_sync())

        # Start the periodic sync
        asyncio.create_task(periodic_sync())

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.post("/meta")
async def get_metadata(request: MetaRequest) -> Dict[str, Dict[str, Any]]:
    """Get metadata for a batch of concept IDs"""
    try:
        meta = {}
        for cid in request.cids:
            concept = db.find(cid)
            if concept:
                meta[cid] = {
                    "title": concept.title,
                    "description": concept.description,
                    "source": concept.source,
                    "file": concept.file,
                    "created_at": concept.created_at,
                    "updated_at": concept.updated_at
                }
        return meta
    except Exception as e:
        logger.error(f"Error fetching metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/merge")
async def merge_concepts(merge: ConceptMerge):
    """Merge two concepts, combining their metadata and usage counts."""
    try:
        success = await db.merge_concepts(
            merge.source_term,
            merge.target_term,
            merge.merge_metadata
        )
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to merge concepts - one or both not found"
            )
        return {
            "status": "success",
            "message": f"Merged {merge.source_term} into {merge.target_term}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/{term}/usage")
async def increment_usage(
    term: str,
    increment: int = 1,
    update_metadata: Optional[Dict] = None
):
    """Increment usage count for a concept."""
    try:
        success = await db.update_usage(term, increment, update_metadata)
        if not success:
            raise HTTPException(status_code=404, detail="Concept not found")
        return {
            "status": "success",
            "message": f"Updated usage count for {term}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating usage for {term}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/concepts/stats", response_model=ConceptStats)
async def get_concept_stats():
    """Get usage statistics for all concepts."""
    try:
        stats = await db.get_usage_stats()
        stats["last_updated"] = datetime.utcnow().isoformat()
        return stats
    except Exception as e:
        logger.error(f"Error getting concept stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/concepts/similar/{term}")
async def get_similar_concepts(
    term: str,
    threshold: float = SIMILARITY_THRESHOLD,
    limit: int = 5
):
    """Get similar concepts for a given term."""
    try:
        concept = db.find(term)
        if not concept or not concept.embedding:
            raise HTTPException(
                status_code=404,
                detail="Concept not found or has no embedding"
            )

        similar = await db.find_similar_concepts(
            concept.embedding,
            threshold,
            limit
        )

        return {
            "term": term,
            "similar_concepts": [
                {
                    "term": similar_term,
                    "similarity": float(score)
                }
                for similar_term, score in similar
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar concepts for {term}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/deduplicate")
async def deduplicate_concepts(
    threshold: float = SIMILARITY_THRESHOLD,
    min_usage: int = MIN_USAGE_COUNT
):
    """Run deduplication on all concepts."""
    try:
        stats = {
            "processed": 0,
            "merged": 0,
            "skipped": 0,
            "errors": 0
        }
        concepts = []
        for key in redis_client.scan_iter("concept:*"):
            if not is_concept_key(key):
                continue
            if not key.startswith("concept:usage:") and not key.startswith("concept:similar:"):
                data = redis_client.get(key)
                if data:
                    concept = ConceptMetadata.from_dict(json.loads(data))
                    if concept.usage_count >= min_usage:
                        concepts.append(concept)
        concepts.sort(key=lambda x: x.usage_count, reverse=True)
        for i, concept in enumerate(concepts):
            stats["processed"] += 1
            try:
                if not concept.embedding:
                    stats["skipped"] += 1
                    continue
                similar = await db.find_similar_concepts(
                    concept.embedding,
                    threshold,
                    limit=5
                )
                for similar_term, score in similar:
                    if similar_term == concept.term:
                        continue
                    similar_concept = db.find(similar_term)
                    if not similar_concept:
                        continue
                    if similar_concept.usage_count < concept.usage_count:
                        success = await db.merge_concepts(
                            similar_term,
                            concept.term
                        )
                        if success:
                            stats["merged"] += 1
            except Exception as e:
                logger.error(f"Error processing concept {concept.term}: {e}")
                stats["errors"] += 1
        return {
            "status": "success",
            "stats": stats,
            "threshold": threshold,
            "min_usage": min_usage
        }
    except Exception as e:
        logger.error(f"Error during deduplication: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/concepts/digest/status")
async def get_digest_status():
    """Get the status of the auto-digestion process."""
    try:
        if not auto_digest:
            return {
                "status": "not_initialized",
                "message": "Auto-digestion not initialized"
            }
        
        return {
            "status": "running" if auto_digest._running else "stopped",
            "last_digest": datetime.fromtimestamp(auto_digest._last_digest).isoformat() if auto_digest._last_digest else None,
            "last_quality_check": datetime.fromtimestamp(auto_digest._last_quality_check).isoformat() if auto_digest._last_quality_check else None,
            "last_merge": datetime.fromtimestamp(auto_digest._last_merge).isoformat() if auto_digest._last_merge else None,
            "settings": {
                "digest_interval": auto_digest.digest_interval,
                "quality_check_interval": auto_digest.quality_check_interval,
                "merge_interval": auto_digest.merge_interval,
                "min_concept_length": auto_digest.min_concept_length,
                "max_concept_length": auto_digest.max_concept_length,
                "min_quality_score": auto_digest.min_quality_score
            }
        }
    except Exception as e:
        logger.error(f"Error getting digest status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/digest/trigger")
async def trigger_digest():
    """Manually trigger a digestion cycle."""
    try:
        if not auto_digest:
            raise HTTPException(status_code=503, detail="Auto-digestion not initialized")
        
        # Run all processes
        await auto_digest._run_digest()
        await auto_digest._run_quality_checks()
        await auto_digest._run_auto_merge()
        
        return {
            "status": "success",
            "message": "Digestion cycle completed",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering digest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/reset_pools")
async def reset_pools(request: Request):
    """Reset both Redis and Qdrant pools to ensure they are in sync."""
    try:
        # Skip authentication in dev mode
        if ENV != "dev" and DEBUG != "1":
            api_key = request.headers.get("X-API-Key")
            if api_key != API_KEY:
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

        # First, get all concepts from Redis that we want to preserve
        logger.info("Starting pool reset process...")
        preserved_concepts = {}
        redis_keys = await scan_redis_keys("concept:*")
        redis_keys = [key for key in redis_keys if await redis_async_client.type(key) == 'string']
        
        # Store all valid concepts temporarily
        for key in redis_keys:
            try:
                concept_data = await redis_async_client.get(key)
                if concept_data:
                    concept = json.loads(concept_data)
                    if concept.get("embedding"):  # Only preserve concepts with embeddings
                        preserved_concepts[key] = concept
            except Exception as e:
                logger.error(f"Error reading concept {key} from Redis: {e}")
                continue
        
        logger.info(f"Found {len(preserved_concepts)} valid concepts to preserve")
        
        # Clear Qdrant collection
        try:
            logger.info("Clearing Qdrant collection...")
            qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={"size": 384, "distance": "Cosine"},
                optimizers_config={"indexing_threshold": 10000}
            )
            logger.info("Qdrant collection cleared and recreated")
        except Exception as e:
            logger.error(f"Error clearing Qdrant collection: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to clear Qdrant: {str(e)}")
        
        # Clear Redis concepts
        try:
            logger.info("Clearing Redis concepts...")
            for key in redis_keys:
                await redis_async_client.delete(key)
            logger.info("Redis concepts cleared")
        except Exception as e:
            logger.error(f"Error clearing Redis concepts: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to clear Redis: {str(e)}")
        
        # Restore preserved concepts
        restored_count = 0
        for key, concept in preserved_concepts.items():
            try:
                # Restore to Redis
                await redis_async_client.set(key, json.dumps(concept))
                
                # Restore to Qdrant
                qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=[{
                        "id": concept["term"],
                        "vector": concept["embedding"],
                        "payload": concept
                    }]
                )
                restored_count += 1
            except Exception as e:
                logger.error(f"Error restoring concept {key}: {e}")
                continue
        
        # Verify final counts
        final_redis_count = len([key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)])
        final_qdrant_count = await get_qdrant_point_count()
        
        logger.info(
            f"Pool reset completed:\n"
            f"  - Preserved concepts: {len(preserved_concepts)}\n"
            f"  - Successfully restored: {restored_count}\n"
            f"  - Final counts - Redis: {final_redis_count}, Qdrant: {final_qdrant_count}"
        )
        
        return {
            "status": "success",
            "message": "Pools reset successfully",
            "stats": {
                "preserved_concepts": len(preserved_concepts),
                "restored_concepts": restored_count,
                "final_redis_count": final_redis_count,
                "final_qdrant_count": final_qdrant_count
            }
        }
    except Exception as e:
        logger.error(f"Error during pool reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resync")
async def trigger_resync(api_key: str = Header(..., alias="X-API-Key")) -> Dict:
    """Trigger a reconciliation between Redis and Qdrant concepts."""
    if api_key != os.getenv("CONCEPT_DICT_API_KEY", "changeme"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if RECONCILING._value.get() == 1:
        return {
            "status": "already_running",
            "message": "A reconciliation is already in progress",
            "current_sync_diff": SYNC_DIFF._value.get()
        }
    
    # Start reconciliation in background
    asyncio.create_task(reconcile())
    
    return {
        "status": "started",
        "message": "Reconciliation started",
        "current_sync_diff": SYNC_DIFF._value.get()
    }

@app.get("/sync-status")
async def get_sync_status(api_key: str = Header(..., alias="X-API-Key")) -> Dict:
    """Get current sync status between Redis and Qdrant."""
    if api_key != os.getenv("CONCEPT_DICT_API_KEY", "changeme"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {
        "sync_difference": SYNC_DIFF._value.get(),
        "is_reconciling": RECONCILING._value.get() == 1,
        "total_repairs": REPAIR._value.get(),
        "total_drift": DRIFT._value.get()
    }

@app.post("/concepts/{term}/heal")
async def heal_concept(term: str, force: bool = False):
    """Manually trigger healing for a concept."""
    try:
        success = await db.heal_concept(term, force=force)
        if not success:
            raise HTTPException(
                status_code=404 if not force else 500,
                detail=f"Failed to heal concept {term}"
            )
        return {"status": "success", "message": f"Concept {term} healed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error healing concept {term}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/concepts/sync/status")
async def get_sync_status():
    """Get sync status for all concepts."""
    try:
        stats = {
            "total": 0,
            "synced": 0,
            "needs_sync": 0,
            "sync_errors": 0,
            "by_priority": {0: 0, 1: 0, 2: 0, 3: 0}  # Count by retry count
        }
        
        async for key in db.redis_async.scan_iter("concept:*"):
            if await db.is_concept_key(key):
                try:
                    data = await db.redis_async.get(key)
                    if data:
                        concept = ConceptMetadata.from_dict(json.loads(data))
                        stats["total"] += 1
                        
                        if concept.sync_state["redis_synced"] and concept.sync_state["qdrant_synced"]:
                            stats["synced"] += 1
                        else:
                            stats["needs_sync"] += 1
                            
                        if concept.sync_state.get("sync_error"):
                            stats["sync_errors"] += 1
                            
                        retry_count = concept.sync_state.get("sync_retry_count", 0)
                        stats["by_priority"][min(retry_count, 3)] += 1
                except Exception as e:
                    logger.error(f"Error processing concept {key} in sync status: {e}")
                    continue
                    
        return stats
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/fetch_queue", response_model=FetchConceptsResponse)
async def fetch_untrained_concepts(request: FetchConceptsRequest) -> FetchConceptsResponse:
    """Fetch a batch of untrained concepts for training."""
    try:
        batch_id = str(uuid.uuid4())
        concepts = []
        total_available = 0
        
        # Scan Redis for concepts
        async for key in redis_async_client.scan_iter("concept:*"):
            if not await db.is_concept_key(key):
                continue
                
            data = await redis_async_client.get(key)
            if not data:
                continue
                
            concept = ConceptMetadata.from_dict(json.loads(data))
            
            # Skip concepts that are already being trained or have failed too many times
            training_status = concept.metadata.get("training_status", {})
            if training_status.get("status") == "training":
                continue
            if training_status.get("retry_count", 0) >= request.max_retries:
                continue
            if training_status.get("status") == "trained":
                continue
                
            # Check quality score if specified
            quality_score = concept.metadata.get("quality_score", 0.0)
            if quality_score < request.min_quality_score:
                continue
                
            total_available += 1
            
            if len(concepts) < request.batch_size:
                # Mark concept as pending training
                concept.metadata["training_status"] = {
                    "status": "pending",
                    "batch_id": batch_id,
                    "last_attempt": datetime.utcnow().isoformat(),
                    "retry_count": training_status.get("retry_count", 0)
                }
                
                # Update Redis with new training status
                await redis_async_client.set(key, json.dumps(concept.to_dict()))
                
                # Add to response
                concepts.append(concept.to_dict())
        
        return FetchConceptsResponse(
            concepts=concepts,
            batch_id=batch_id,
            total_available=total_available
        )
        
    except Exception as e:
        logger.error(f"Error fetching untrained concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/training_status/{term}")
async def update_training_status(term: str, status: ConceptTrainingStatus):
    """Update the training status of a concept."""
    try:
        key = db._get_redis_key(term)
        data = await redis_async_client.get(key)
        if not data:
            raise HTTPException(status_code=404, detail=f"Concept {term} not found")
            
        concept = ConceptMetadata.from_dict(json.loads(data))
        
        # Update training status
        concept.metadata["training_status"] = status.dict()
        
        # If training completed successfully, update version
        if status.status == "trained":
            concept.version += 1
            concept.sync_state["sync_required"] = True
            concept.sync_state["sync_required_reason"] = "training_completed"
        
        # Save updated concept
        await redis_async_client.set(key, json.dumps(concept.to_dict()))
        
        # Trigger healing if needed
        if status.status == "trained":
            asyncio.create_task(db.heal_concept(term))
            
        return {"status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating training status for concept {term}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/concepts/training_stats")
async def get_training_stats():
    """Get statistics about concept training status."""
    try:
        stats = {
            "total": 0,
            "pending": 0,
            "training": 0,
            "trained": 0,
            "failed": 0,
            "by_retry_count": {0: 0, 1: 0, 2: 0, 3: 0}
        }
        
        async for key in redis_async_client.scan_iter("concept:*"):
            if not await db.is_concept_key(key):
                continue
                
            data = await redis_async_client.get(key)
            if not data:
                continue
                
            concept = ConceptMetadata.from_dict(json.loads(data))
            training_status = concept.metadata.get("training_status", {})
            
            stats["total"] += 1
            status = training_status.get("status", "untrained")
            stats[status] = stats.get(status, 0) + 1
            
            retry_count = training_status.get("retry_count", 0)
            stats["by_retry_count"][min(retry_count, 3)] += 1
            
        return stats
        
    except Exception as e:
        logger.error(f"Error getting training stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_db() -> ConceptDB:
    """Dependency injection for database access."""
    return db

@app.post("/test/inject_concept", 
         summary="Inject a test concept",
         description="Endpoint for testing and manual concept injection. This endpoint bypasses normal validation.")
async def inject_test_concept(
    concept: TestConceptInput,
    background_tasks: BackgroundTasks,
    db: ConceptDB = Depends(get_db)
) -> Dict[str, Any]:
    """Inject a test concept into the system."""
    try:
        # Create concept metadata
        metadata = ConceptMetadata(
            term=concept.term,
            definition=concept.definition,
            examples=concept.examples,
            metadata=concept.metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            quality_score=concept.force_quality_score if concept.force_quality_score is not None else 1.0,
            sync_state={
                "redis": {"synced": False, "error": None, "retry_count": 0, "sync_lock": None},
                "qdrant": {"synced": False, "error": None, "retry_count": 0, "sync_lock": None}
            }
        )

        # Store in Redis
        await db.store_concept(metadata)
        
        # Queue for auto-digestion
        background_tasks.add_task(db.queue_concept_for_digest, concept.term)
        
        return {
            "status": "success",
            "message": f"Test concept '{concept.term}' injected successfully",
            "concept": {
                "term": concept.term,
                "definition": concept.definition,
                "examples": concept.examples,
                "quality_score": metadata.quality_score,
                "created_at": metadata.created_at.isoformat(),
                "sync_state": metadata.sync_state
            }
        }
    except Exception as e:
        logger.error(f"Error injecting test concept: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to inject test concept: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 