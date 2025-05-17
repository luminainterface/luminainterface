from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import os
import logging
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
from datetime import datetime
import httpx
import threading
import time
import asyncio
from .ingest_consumer import start_consumers
from .db import redis_client, qdrant_client, model, ConceptDB, SIMILARITY_THRESHOLD, MIN_USAGE_COUNT, BLOCKED_LICENSES
from .models import Concept
from fastapi.responses import JSONResponse
import redis

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

# Helper to check if a Redis key is a string (concept)
def is_concept_key(key):
    try:
        key_type = redis_client.type(key)
        return key_type == 'string'  # Redis returns 'string' when decode_responses=True
    except redis.RedisError as e:
        logger.error(f"Error checking Redis key type for {key}: {e}")
        return False

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_client.ping()
        
        # Check Qdrant connection
        qdrant_client.get_collections()
        
        return JSONResponse(
            status_code=200,
            content={
            "status": "healthy",
            "dependencies": {
                "redis": "connected",
                "qdrant": "connected"
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
            for key in redis_client.scan_iter("concept:*"):
                try:
                    # Skip non-string keys (like streams)
                    if redis_client.type(key) != b'string':
                        continue
                    concept_data = redis_client.get(key)
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
async def get_concept(term: str) -> Dict:
    """Get a specific concept"""
    with RETRIEVAL_LATENCY.time():
        try:
            concept_data = redis_client.get(f"concept:{term}")
            if not concept_data:
                raise HTTPException(status_code=404, detail="Concept not found")
            CONCEPT_RETRIEVALS.inc()
            return json.loads(concept_data)
        except Exception as e:
            logger.error(f"Error retrieving concept {term}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/sync_pdf_embeddings")
async def sync_pdf_embeddings():
    """Sync PDF embeddings from the pdf_embeddings_768d collection to the concepts collection."""
    try:
        imported = 0
        scroll_offset = None
        # Get initial counts for logging
        initial_redis_count = len([key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)])
        initial_qdrant_count = await get_qdrant_point_count()
        logger.info(f"Starting PDF embeddings sync. Initial counts - Redis: {initial_redis_count}, Qdrant: {initial_qdrant_count}")
        # Get PDF embeddings from the 768d collection
        while True:
            result, next_page = qdrant_client.scroll(
                collection_name="pdf_embeddings_768d",
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

@app.put("/concepts/{term}")
async def update_concept(term: str, update: ConceptUpdate):
    """Update a concept's metadata and/or definition."""
    try:
        concept = db.find(term)
        if not concept:
            raise HTTPException(status_code=404, detail="Concept not found")

        # Update fields if provided
        if update.definition is not None:
            concept.definition = update.definition
        if update.metadata is not None:
            concept.metadata.update(update.metadata)
        if update.license_type is not None:
            # Check if new license is blocked
            if update.license_type.lower() in BLOCKED_LICENSES:
                raise HTTPException(
                    status_code=400,
                    detail=f"License type {update.license_type} is blocked"
                )
            concept.license_type = update.license_type

        # Update in Redis
        redis_client.set(
            f"concept:{term}",
            json.dumps(concept.to_dict())
        )

        # Update in Qdrant if embedding exists
        if concept.embedding:
            qdrant_client.upsert(
                collection_name="concepts",
                points=[{
                    "id": term,
                    "vector": concept.embedding,
                    "payload": concept.to_dict()
                }]
            )

        CONCEPT_UPDATES.inc()
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
        return {
            "status": "success",
            "counts": {
                "redis_concepts": redis_count,
                "qdrant_concepts": qdrant_count,
                "difference": redis_count - qdrant_count if qdrant_count >= 0 else None
            },
            "last_sync": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting concept status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/concepts/import_from_qdrant")
async def import_from_qdrant():
    """
    Import all concepts from Qdrant into Redis. This implementation is generic and will adapt to new fields in Qdrant payloads automatically.
    """
    try:
        imported = 0
        failed = 0
        scroll_offset = None
        
        # Get initial counts for logging
        initial_redis_count = len([key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)])
        initial_qdrant_count = await get_qdrant_point_count()
        
        logger.info(f"Starting import from Qdrant. Initial counts - Redis: {initial_redis_count}, Qdrant: {initial_qdrant_count}")
        
        while True:
            # Use scroll API with pagination
            result, next_page = qdrant_client.scroll(
                collection_name="concepts",
                limit=256,  # reasonable batch size
                with_payload=True,
                with_vectors=True,
                offset=scroll_offset
            )
            
            if not result:
                break
                
            for point in result:
                try:
                    # Use all available fields, fallback to id if term is missing
                    payload = point.payload or {}
                    term = payload.get("term") or str(point.id)
                    concept = {
                        "term": term,
                        "definition": payload.get("definition", ""),
                        "embedding": point.vector,
                        "metadata": payload.get("metadata", {}),
                        "last_updated": payload.get("last_updated", datetime.utcnow().isoformat())
                    }
                    # Add any extra fields from payload for future-proofing
                    for k, v in payload.items():
                        if k not in concept:
                            concept[k] = v
                    
                    # Try to write to Redis and verify
                    key = f"concept:{term}"
                    redis_client.set(key, json.dumps(concept))
                    # Verify the write
                    stored_data = redis_client.get(key)
                    if not stored_data:
                        logger.error(f"Failed to verify Redis write for concept {term}")
                        failed += 1
                        continue
                    imported += 1
                    logger.debug(f"Successfully imported concept {term} to Redis")
                except redis.RedisError as e:
                    logger.error(f"Redis error importing concept {term}: {e}")
                    failed += 1
                    continue
                except Exception as e:
                    logger.error(f"Error processing concept {term}: {e}")
                    failed += 1
                    continue
                
            if not next_page:
                break
            scroll_offset = next_page
            
        # Get final counts for logging
        final_redis_count = len([key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)])
        final_qdrant_count = await get_qdrant_point_count()
        
        logger.info(
            f"Import from Qdrant completed:\n"
            f"  - Imported: {imported} concepts\n"
            f"  - Failed: {failed} concepts\n"
            f"  - Final counts - Redis: {final_redis_count}, Qdrant: {final_qdrant_count}\n"
            f"  - Net change: {final_redis_count - initial_redis_count}"
        )
        
        if failed > 0:
            logger.warning(f"Import completed with {failed} failures")
        
        return {
            "status": "success",
            "imported": imported,
            "failed": failed,
            "counts": {
                "initial_redis": initial_redis_count,
                "initial_qdrant": initial_qdrant_count,
                "final_redis": final_redis_count,
                "final_qdrant": final_qdrant_count
            }
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

# Start the logging thread on startup
threading.Thread(target=log_concept_count, daemon=True).start()

@app.post("/concepts/import_from_redis")
async def import_from_redis():
    """
    Import all concepts from Redis into Qdrant. This will ensure concepts in Redis are also in Qdrant.
    """
    try:
        imported = 0
        redis_keys = [key for key in redis_client.scan_iter("concept:*") if is_concept_key(key)]
        # Get initial counts for logging
        initial_redis_count = len(redis_keys)
        initial_qdrant_count = await get_qdrant_point_count()
        logger.info(f"Starting import from Redis. Initial counts - Redis: {initial_redis_count}, Qdrant: {initial_qdrant_count}")
        for key in redis_keys:
            try:
                concept_data = redis_client.get(key)
                if not concept_data:
                    continue
                concept = json.loads(concept_data)
                term = key.replace("concept:", "")
                if not concept.get("embedding"):
                    text = f"{term} {concept.get('definition', '')}"
                    embedding = model.encode(text)
                    concept["embedding"] = embedding.tolist()
                    redis_client.set(key, json.dumps(concept))
                qdrant_client.upsert(
                    collection_name="concepts",
                    points=[{
                        "id": term,
                        "vector": concept["embedding"],
                        "payload": concept
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

async def periodic_sync():
    """Background task that periodically syncs concepts between Redis and Qdrant."""
    while True:
        try:
            # Sync from Redis to Qdrant
            logger.info("Starting periodic import from Redis...")
            redis_resp = await import_from_redis()
            logger.info(f"Periodic import from Redis completed: {redis_resp}")
            
            # Sync from Qdrant to Redis
            logger.info("Starting periodic import from Qdrant...")
            qdrant_resp = await import_from_qdrant()
            logger.info(f"Periodic import from Qdrant completed: {qdrant_resp}")
            
        except Exception as e:
            logger.error(f"Periodic sync failed: {e}")
        await asyncio.sleep(300)  # Sync every 5 minutes

@app.on_event("startup")
async def startup_event():
    logger.info("[DEBUG] FastAPI startup_event called.")
    try:
        # Start the consumer in the background
        asyncio.create_task(start_consumers())
        logger.info("Started concept dictionary consumers in background")
    except Exception as e:
        logger.error(f"Failed to start consumers: {e}", exc_info=True)
        raise

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

    # Start the periodic sync
    asyncio.create_task(periodic_sync())

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 