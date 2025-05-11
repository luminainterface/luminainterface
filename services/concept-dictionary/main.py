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
from ingest_consumer import start_consumers
from db import redis_client, qdrant_client, model, ConceptDB

# Environment variables
API_KEY = os.getenv("CONCEPT_DICT_API_KEY", "changeme")

# Initialize FastAPI app
app = FastAPI(title="Concept Dictionary")
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

class Concept(BaseModel):
    term: str
    definition: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict] = None
    last_updated: Optional[str] = None

class MetaRequest(BaseModel):
    cids: List[str]

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_client.ping()
        
        # Check Qdrant connection
        qdrant_client.get_collections()
        
        return {
            "status": "healthy",
            "dependencies": {
                "redis": "connected",
                "qdrant": "connected"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/concepts")
async def get_concepts() -> List[Dict]:
    """Get all concepts"""
    with RETRIEVAL_LATENCY.time():
        try:
            concepts = []
            for key in redis_client.scan_iter("concept:*"):
                concept_data = redis_client.get(key)
                if concept_data:
                    concepts.append(json.loads(concept_data))
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

def generate_embedding(text: str) -> List[float]:
    """Generate an embedding for a text using the sentence transformer model."""
    try:
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

@app.put("/concepts/{term}")
async def update_concept(term: str, concept: Concept):
    """Update or create a concept"""
    try:
        # Update timestamp
        concept.last_updated = datetime.utcnow().isoformat()
        
        # Generate embedding if not provided
        if not concept.embedding:
            text = f"{concept.term} {concept.definition}"
            concept.embedding = generate_embedding(text)
            if not concept.embedding:
                logger.warning(f"Failed to generate embedding for concept {term}")
        
        # Store in Redis
        redis_client.set(
            f"concept:{term}",
            json.dumps(concept.dict())
        )
        
        # If embedding exists, store in Qdrant
        if concept.embedding:
            qdrant_client.upsert(
                collection_name="concepts",
                points=[{
                    "id": term,
                    "vector": concept.embedding,
                    "payload": {
                        "term": term,
                        "definition": concept.definition,
                        "metadata": concept.metadata or {}
                    }
                }]
            )
        
        CONCEPT_UPDATES.inc()
        return {"status": "success", "message": f"Concept {term} updated"}
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
            redis_count = len([key for key in redis_client.scan_iter("concept:*")])
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
        redis_count = len([key for key in redis_client.scan_iter("concept:*")])
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
        scroll_offset = None
        
        # Get initial counts for logging
        initial_redis_count = len([key for key in redis_client.scan_iter("concept:*")])
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
                # Use all available fields, fallback to id if term is missing
                payload = point.payload or {}
                term = payload.get("term") or str(point.id)
                concept = {
                    "term": term,
                    "definition": payload.get("definition", ""),
                    "embedding": point.vector,
                    "metadata": payload.get("metadata", {}),
                    "last_updated": payload.get("last_updated", None)
                }
                # Add any extra fields from payload for future-proofing
                for k, v in payload.items():
                    if k not in concept:
                        concept[k] = v
                redis_client.set(f"concept:{term}", json.dumps(concept))
                imported += 1
                
            if not next_page:
                break
            scroll_offset = next_page
            
        # Get final counts for logging
        final_redis_count = len([key for key in redis_client.scan_iter("concept:*")])
        final_qdrant_count = await get_qdrant_point_count()
        
        logger.info(
            f"Import from Qdrant completed:\n"
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
    if request.url.path.startswith("/health"):  # Allow health checks
        return await call_next(request)
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)

# Start the logging thread on startup
threading.Thread(target=log_concept_count, daemon=True).start()

async def periodic_sync():
    """Background task that periodically calls import_from_qdrant() so that the concept dictionary is automatically updated (pulled) from Qdrant every N seconds (for example, every 300 seconds)."""
    while True:
        try:
            logger.info("Starting periodic import from Qdrant...")
            resp = await import_from_qdrant()
            logger.info(f"Periodic import from Qdrant completed: {resp}")
        except Exception as e:
            logger.error(f"Periodic import from Qdrant failed: {e}")
        await asyncio.sleep(300)  # (adjust the sleep interval as needed)

@app.on_event("startup")
async def startup_event():
    """On startup, ensure that the Qdrant collection 'concepts' exists (create it if missing), then pull (import) concepts from Qdrant, and start a periodic sync (so that the concept dictionary is always updated by Qdrant)."""
    logger.info("Ensuring Qdrant collection 'concepts' exists...")
    try:
        collections = qdrant_client.get_collections()
        if not any(c.name == "concepts" for c in collections.collections):
            logger.info("Collection 'concepts' not found; creating it...")
            qdrant_client.create_collection(
                collection_name="concepts",
                vectors_config={"size": 384, "distance": "Cosine"}
            )
            logger.info("Collection 'concepts' created.")
        else:
            logger.info("Collection 'concepts' already exists.")
    except Exception as e:
        logger.error(f"Error ensuring Qdrant collection 'concepts' exists: {e}")
        raise

    logger.info("Starting import from Qdrant on startup...")
    try:
        resp = await import_from_qdrant()
        logger.info(f"Startup import from Qdrant completed: {resp}")
    except Exception as e:
        logger.error(f"Startup import from Qdrant failed: {e}")

    # Start the periodic sync
    asyncio.create_task(periodic_sync())
    
    # Start the ingest consumer
    asyncio.create_task(start_consumers())

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 