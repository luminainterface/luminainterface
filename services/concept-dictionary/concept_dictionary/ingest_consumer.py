import asyncio
import time
import logging
import json
import os
import redis.asyncio as aioredis
from uuid import uuid4
from typing import Dict, Optional, Any, List
from lumina_core.common.bus import BusClient, StreamMessage
from .db import redis_async_client, qdrant_client, model, ConceptDB, ConceptMetadata
from pydantic import BaseModel
from prometheus_client import Counter, Gauge
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("concept-dictionary")

# Initialize database and bus
db = ConceptDB()
bus = BusClient(redis_url=os.getenv("REDIS_URL", "redis://:02211998@redis:6379"))

# Track Redis connection state
_redis_connected = False

# Track service pause state
_service_paused = False
SYNC_THRESHOLD = 10  # Threshold below which we consider sync healthy
SYNC_CHECK_INTERVAL = 30  # Check sync status every 30 seconds

# Metrics
DRIFT = Counter('dict_qdrant_drift_total', 'Total number of concept-vector mismatches detected')
REPAIR = Counter('dict_qdrant_repair_total', 'Total number of repairs performed')
SYNC_DIFF = Gauge('dict_qdrant_sync_difference', 'Current difference between Redis and Qdrant concept counts')
RECONCILING = Gauge('dict_qdrant_reconciling', 'Whether a reconciliation is currently running (0/1)')

# Constants
BATCH_SIZE = 1000  # Number of vectors to process in each batch

async def ensure_redis_connection():
    """Ensure Redis connection is established."""
    global _redis_connected
    try:
        await redis_async_client.ping()
        _redis_connected = True
        return True
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        _redis_connected = False
        return False

# Initialize Redis connection
asyncio.create_task(ensure_redis_connection())

class CrawlIn(BaseModel):
    """Message from crawler."""
    url: str
    title: Optional[str]
    vec_id: str
    ts: float
    license_type: Optional[str] = None
    metadata: Optional[Dict] = None

class PdfIn(BaseModel):
    """Message from PDF processor."""
    file_path: str
    vec_id: str
    ts: float
    license_type: Optional[str] = None
    metadata: Optional[Dict] = None

class ConceptOut(BaseModel):
    """Message for new/updated concepts."""
    cid: str
    term: str
    definition: str
    embedding_id: str
    meta: Dict
    ts: float
    usage_count: int
    license_type: Optional[str] = None
    similar_concepts: Optional[List[str]] = None

async def process_concept(
    vec_id: str,
    meta: Dict,
    term: str,
    definition: str,
    embedding: Optional[List[float]] = None,
    license_type: Optional[str] = None
) -> Optional[ConceptOut]:
    """Process a concept with deduplication and license checking. If embedding is provided but its dimension is not 768, it is removed so that the concept is re-embedded (via the concept-dictionary's model) later."""
    try:
        # Remove embedding if its dimension is not 768 (so that it is re-embedded later)
        if embedding is not None and len(embedding) != 768:
            logger.warning(f"Removing embedding for concept {term} (dimension {len(embedding)} != 768). It will be re-embedded.")
            embedding = None

        # Add concept with deduplication (embedding is now None if it was removed)
        success, result = await db.add_concept(
            term=term,
            definition=definition,
            embedding=embedding,
            metadata=meta,
            license_type=license_type
        )

        if not success:
            if result == "license_blocked":
                logger.warning(f"Blocked concept with license {license_type}: {term}")
            return None

        # Get the concept (either new or existing after deduplication)
        concept = db.find(result)
        if not concept:
            logger.error(f"Failed to retrieve concept after adding: {term}")
            return None

        # Ensure concept is in Qdrant (if embedding is available)
        try:
            # Check if concept exists in Qdrant
            point = await qdrant_client.retrieve(
                collection_name="concepts",
                ids=[term],
                with_payload=False
            )
            if not point and concept.embedding:
                # Add to Qdrant if not present and embedding is available (now guaranteed to be 768)
                await qdrant_client.upsert(
                    collection_name="concepts",
                    points=[{
                        "id": term,
                        "vector": concept.embedding,
                        "payload": {
                            "term": term,
                            "definition": definition,
                            "metadata": meta
                        }
                    }]
                )
                logger.info(f"Added concept {term} to Qdrant (embedding dimension: {len(concept.embedding)})")
            elif not concept.embedding:
                logger.warning(f"No embedding (or removed embedding) for concept {term}, skipping Qdrant sync.")
        except Exception as e:
            logger.error(f"Error syncing concept {term} to Qdrant: {e}")
            # Continue processing even if Qdrant sync fails

        # Create output message
        return ConceptOut(
            cid=concept.term,  # Use term as ID for consistency
            term=concept.term,
            definition=concept.definition,
            embedding_id=vec_id,
            meta=concept.metadata,
            ts=time.time(),
            usage_count=concept.usage_count,
            license_type=concept.license_type,
            similar_concepts=concept.similar_concepts
        )

    except Exception as e:
        logger.error(f"Error processing concept {term}: {e}", exc_info=True)
        return None

async def handle_crawl_message(msg: CrawlIn) -> None:
    """Handle incoming messages from crawler."""
    try:
        # Extract concept info from metadata
        meta = {
            "url": msg.url,
            "title": msg.title,
            "source": "crawl",
            "timestamp": msg.ts,
            **(msg.metadata or {})
        }

        # Use title as term and URL as definition if available
        term = msg.title or msg.url.split("/")[-1]
        definition = msg.url

        # Get embedding from Qdrant
        try:
            point = qdrant_client.retrieve(
                collection_name="embeddings",
                ids=[msg.vec_id],
                with_payload=False
            )
            embedding = point[0].vector if point else None
        except Exception as e:
            logger.warning(f"Failed to get embedding for {msg.vec_id}: {e}")
            embedding = None

        # Process concept
        concept_out = await process_concept(
            vec_id=msg.vec_id,
            meta=meta,
            term=term,
            definition=definition,
            embedding=embedding,
            license_type=msg.license_type
        )

        if concept_out:
            await bus.publish("concept.new", concept_out.dict())
            logger.info(f"Published concept {concept_out.term} to concept.new stream")

    except Exception as e:
        logger.error(f"Error handling crawl message: {e}")

async def handle_pdf_message(msg: PdfIn) -> None:
    """Handle incoming messages from PDF processor."""
    try:
        # Extract concept info from metadata
        meta = {
            "file": msg.file_path,
            "source": "pdf",
            "timestamp": msg.ts,
            **(msg.metadata or {})
        }

        # Use filename as term and path as definition
        term = os.path.basename(msg.file_path)
        definition = msg.file_path

        # Get embedding from Qdrant
        try:
            point = qdrant_client.retrieve(
                collection_name="embeddings",
                ids=[msg.vec_id],
                with_payload=False
            )
            embedding = point[0].vector if point else None
        except Exception as e:
            logger.warning(f"Failed to get embedding for {msg.vec_id}: {e}")
            embedding = None

        # Process concept
        concept_out = await process_concept(
            vec_id=msg.vec_id,
            meta=meta,
            term=term,
            definition=definition,
            embedding=embedding,
            license_type=msg.license_type
        )

        if concept_out:
            await bus.publish("concept.new", concept_out.dict())
            logger.info(f"Published concept {concept_out.term} to concept.new stream")

    except Exception as e:
        logger.error(f"Error handling PDF message: {e}")

async def reconcile() -> None:
    """Reconcile concepts between Redis and Qdrant with version tracking."""
    try:
        RECONCILING.set(1)
        logger.info("Starting reconciliation between Redis and Qdrant")
        
        # Set crawler pause flag
        await redis_async_client.set("crawler.PAUSE", "1")
        
        # Get all concept keys from Redis
        concept_keys = []
        async for key in redis_async_client.scan_iter("concept:*"):
            if await is_concept_key(key):
                concept_keys.append(key)
        
        logger.info(f"Found {len(concept_keys)} concepts in Redis")
        
        # Collect concepts and their data
        redis_concepts = {}
        for key in concept_keys:
            try:
                data = await redis_async_client.get(key)
                if data:
                    concept = json.loads(data)
                    if concept.get("embedding"):  # Only process concepts with embeddings
                        redis_concepts[key] = concept
            except Exception as e:
                logger.error(f"Error reading concept {key}: {e}")
        
        # Get all points from Qdrant with version info
        qdrant_points = {}
        offset = None
        while True:
            try:
                result, next_page = await qdrant_client.scroll(
                    collection_name="concepts",
                    limit=1000,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset
                )
                
                if not result:
                    break
                
                for point in result:
                    payload = point.payload or {}
                    term = payload.get("term") or str(point.id)
                    if term:
                        qdrant_points[term] = {
                            "point": point,
                            "version": payload.get("version", 0),
                            "sync_state": payload.get("sync_state", {})
                        }
                
                if not next_page:
                    break
                offset = next_page
                
            except Exception as e:
                logger.error(f"Error scrolling Qdrant points: {e}")
                break
        
        logger.info(f"Found {len(qdrant_points)} concepts in Qdrant")
        
        # Track reconciliation metrics
        stats = {
            "missing_vectors": 0,
            "orphan_vectors": 0,
            "version_mismatches": 0,
            "sync_errors": 0,
            "healed": 0
        }
        
        # Process Redis concepts
        for key, concept in redis_concepts.items():
            term = key.replace("concept:", "")
            qdrant_data = qdrant_points.get(term)
            
            try:
                if not qdrant_data:
                    # Concept missing in Qdrant
                    stats["missing_vectors"] += 1
                    await qdrant_client.upsert(
                        collection_name="concepts",
                        points=[{
                            "id": term,
                            "vector": concept["embedding"],
                            "payload": {
                                **concept,
                                "term": term,
                                "version": concept.get("version", 1),
                                "sync_state": {
                                    "redis_synced": True,
                                    "qdrant_synced": True,
                                    "last_sync_attempt": datetime.utcnow().isoformat(),
                                    "sync_error": None
                                }
                            }
                        }]
                    )
                    stats["healed"] += 1
                elif qdrant_data["version"] < concept.get("version", 1):
                    # Version mismatch - Redis has newer version
                    stats["version_mismatches"] += 1
                    await qdrant_client.upsert(
                        collection_name="concepts",
                        points=[{
                            "id": term,
                            "vector": concept["embedding"],
                            "payload": {
                                **concept,
                                "term": term,
                                "version": concept.get("version", 1),
                                "sync_state": {
                                    "redis_synced": True,
                                    "qdrant_synced": True,
                                    "last_sync_attempt": datetime.utcnow().isoformat(),
                                    "sync_error": None
                                }
                            }
                        }]
                    )
                    stats["healed"] += 1
                elif not qdrant_data["sync_state"].get("qdrant_synced", True):
                    # Sync state indicates Qdrant needs update
                    stats["sync_errors"] += 1
                    await qdrant_client.upsert(
                        collection_name="concepts",
                        points=[{
                            "id": term,
                            "vector": concept["embedding"],
                            "payload": {
                                **concept,
                                "term": term,
                                "version": concept.get("version", 1),
                                "sync_state": {
                                    "redis_synced": True,
                                    "qdrant_synced": True,
                                    "last_sync_attempt": datetime.utcnow().isoformat(),
                                    "sync_error": None
                                }
                            }
                        }]
                    )
                    stats["healed"] += 1
            except Exception as e:
                logger.error(f"Error reconciling concept {term}: {e}")
                continue
        
        # Find and remove orphan vectors
        redis_terms = {key.replace("concept:", "") for key in redis_concepts.keys()}
        orphan_vectors = []
        for term, data in qdrant_points.items():
            if term not in redis_terms:
                orphan_vectors.append(data["point"].id)
                stats["orphan_vectors"] += 1
        
        if orphan_vectors:
            try:
                await qdrant_client.delete(
                    collection_name="concepts",
                    points_selector={"ids": orphan_vectors}
                )
                logger.info(f"Removed {len(orphan_vectors)} orphan vectors from Qdrant")
            except Exception as e:
                logger.error(f"Error removing orphan vectors: {e}")
        
        # Log reconciliation results
        logger.info(
            f"Reconciliation completed:\n"
            f"  - Missing vectors added: {stats['missing_vectors']}\n"
            f"  - Version mismatches fixed: {stats['version_mismatches']}\n"
            f"  - Sync errors resolved: {stats['sync_errors']}\n"
            f"  - Orphan vectors removed: {stats['orphan_vectors']}\n"
            f"  - Total concepts healed: {stats['healed']}"
        )
        
        # Update metrics
        REPAIR.labels(type="missing").inc(stats["missing_vectors"])
        REPAIR.labels(type="version").inc(stats["version_mismatches"])
        REPAIR.labels(type="sync").inc(stats["sync_errors"])
        REPAIR.labels(type="orphan").inc(stats["orphan_vectors"])
        
    except Exception as e:
        logger.error(f"Error during reconciliation: {e}")
    finally:
        RECONCILING.set(0)
        # Clear crawler pause flag
        await redis_async_client.delete("crawler.PAUSE")

async def check_sync_status() -> None:
    """Check sync status between Redis and Qdrant."""
    try:
        redis_count = len(await redis_async_client.keys("concept:*"))
        qdrant_count = len(qdrant_client.scroll(
            collection_name="concepts",
            limit=1,  # We only need the count
            with_payload=False
        )[0])
        
        sync_diff = abs(redis_count - qdrant_count)
        SYNC_DIFF.set(sync_diff)
        
        if sync_diff > SYNC_THRESHOLD:
            logger.warning(f"Sync difference detected: {sync_diff} concepts")
            # Trigger reconciliation if not already running
            if RECONCILING._value.get() == 0:  # Access the internal value
                asyncio.create_task(reconcile())
        
    except Exception as e:
        logger.error(f"Error checking sync status: {e}")

async def periodic_sync_check() -> None:
    """Periodically check sync status."""
    while True:
        await check_sync_status()
        await asyncio.sleep(SYNC_CHECK_INTERVAL)

async def handle_message(msg: StreamMessage) -> None:
    """Handle incoming stream messages."""
    global _service_paused
    
    # Check if service is paused
    if _service_paused:
        logger.warning(f"Skipping message from {msg.stream} - service is paused due to sync issues")
        return
        
    logger.info(f"[DEBUG] Received message in handle_message from stream {msg.stream}: {msg}")
    try:
        # Check sync status before processing
        await check_sync_status()
        if _service_paused:
            logger.warning(f"Skipping message from {msg.stream} - service is paused due to sync issues")
            return
            
        data = msg.data
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message data as JSON: {e}")
                return

        # Handle based on stream type
        if msg.stream == "ingest.crawl":
            if not all(k in data for k in ["url", "title", "vec_id", "ts"]):
                logger.error(f"Invalid crawl message format: {data}")
                return
            await handle_crawl_message(CrawlIn(**data))
        elif msg.stream == "ingest.pdf":
            if not all(k in data for k in ["file_path", "vec_id", "ts"]):
                logger.error(f"Invalid PDF message format: {data}")
                return
            await handle_pdf_message(PdfIn(**data))
        else:
            logger.warning(f"Unknown stream type: {msg.stream}")
    except Exception as e:
        logger.error(f"Error handling message from {msg.stream}: {e}", exc_info=True)

async def start_consumers():
    """Start consuming from streams."""
    global _redis_connected
    logger.info("[DEBUG] Entered start_consumers() - launching consumer loop...")
    
    # Start the periodic sync check
    asyncio.create_task(periodic_sync_check())
    
    while True:
        try:
            if not _redis_connected:
                logger.info("[DEBUG] Redis not connected, calling ensure_redis_connection()...")
                if not await ensure_redis_connection():
                    logger.error("Failed to establish Redis connection, retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                logger.info("[DEBUG] ensure_redis_connection() complete.")
            
            # Start consumer tasks
            logger.info("[DEBUG] Starting consumer tasks...")
            tasks = [
                bus.consume("ingest.pdf", "concept-dict", "concept-dict-1", handle_message),
                bus.consume("ingest.crawl", "concept-dict", "concept-dict-1", handle_message)
            ]
            
            try:
                logger.info("[DEBUG] Consumer tasks created, awaiting asyncio.gather...")
                await asyncio.gather(*tasks, return_exceptions=True)
            except (aioredis.ConnectionError, aioredis.TimeoutError) as e:
                logger.error(f"Redis connection lost during consumption: {e}")
                _redis_connected = False
                await asyncio.sleep(5)  # Wait before reconnecting
                continue
            except Exception as e:
                logger.error(f"Error in consumer tasks: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on errors
                
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}", exc_info=True)
            await asyncio.sleep(1)  # Prevent tight loop on errors

async def consume_concept_new():
    while True:
        try:
            # Poll for new messages on stream "concept:new" (group "concept-dict")
            if not (msg := await redis_async_client.xreadgroup("concept-dict", "concept-dict-consumer", {"concept:new": ">"}, count=1, block=1000)):
                logger.error("No new message received on stream concept:new (group concept-dict).")
            else:
                logger.info(f"Received new concept message on stream concept:new: {msg}")
        except Exception as e:
            logger.error(f"Error consuming concept:new stream: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(start_consumers())
    except KeyboardInterrupt:
        logger.info("Shutting down consumers...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise 