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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("concept-dictionary")

# Initialize database and bus
db = ConceptDB()
bus = BusClient(redis_url=os.getenv("REDIS_URL", "redis://:02211998@redis:6379"))

# Track Redis connection state
_redis_connected = False

async def ensure_redis_connection():
    """Ensure Redis connection is established and maintained."""
    global _redis_connected
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            if not _redis_connected:
                logger.info(f"[DEBUG] Attempting Redis connection (attempt {attempt + 1}/{max_retries})...")
                await bus.connect()
                # Verify connection with ping
                await redis_async_client.ping()
                _redis_connected = True
                logger.info("Connected to Redis (async)")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis (attempt {attempt + 1}/{max_retries}): {e}")
            _redis_connected = False
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Max retries reached for Redis connection")
                raise
    return False

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
    """Process a concept with deduplication and license checking."""
    try:
        # Add concept with deduplication
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

        # Ensure concept is in Qdrant
        try:
            # Check if concept exists in Qdrant
            point = await qdrant_client.retrieve(
                collection_name="concepts",
                ids=[term],
                with_payload=False
            )
            
            if not point:
                # Add to Qdrant if not present
                if embedding:
                    await qdrant_client.upsert(
                        collection_name="concepts",
                        points=[{
                            "id": term,
                            "vector": embedding,
                            "payload": {
                                "term": term,
                                "definition": definition,
                                "metadata": meta
                            }
                        }]
                    )
                    logger.info(f"Added concept {term} to Qdrant")
                else:
                    logger.warning(f"No embedding available for concept {term}, skipping Qdrant sync")
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

async def handle_message(msg: StreamMessage) -> None:
    """Handle incoming stream messages."""
    logger.info(f"[DEBUG] Received message in handle_message from stream {msg.stream}: {msg}")
    try:
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

if __name__ == "__main__":
    try:
        asyncio.run(start_consumers())
    except KeyboardInterrupt:
        logger.info("Shutting down consumers...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise 