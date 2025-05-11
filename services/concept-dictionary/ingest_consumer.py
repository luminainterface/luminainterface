import asyncio
import time
import logging
import json
import os
from uuid import uuid4
from typing import Dict, Optional
from lumina_core.common.bus import BusClient
from db import redis_client, qdrant_client, model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("concept-dictionary")

# Initialize bus client
bus = BusClient(redis_url=os.getenv("REDIS_URL", "redis://redis:6379"))

class CrawlIn(StreamMessage):
    url: str
    title: Optional[str]
    vec_id: str
    ts: float

class PdfIn(StreamMessage):
    id: str
    file_path: str
    vec_id: str
    ts: float

class ConceptOut(StreamMessage):
    cid: str
    embedding_id: str
    meta: Dict
    ts: float
    usage_count: int

async def upsert_concept(vec_id: str, meta: Dict, fp: str) -> str:
    """Upsert a concept by fp and return its ID. If fp exists, increment usage_count."""
    try:
        fp_key = f"concept:fp:{fp}"
        existing = redis_client.get(fp_key)
        if existing:
            concept_data = json.loads(existing)
            concept_data["usage_count"] = concept_data.get("usage_count", 1) + 1
            concept_data["last_updated"] = time.time()
            redis_client.set(fp_key, json.dumps(concept_data))
            logger.info(f"Incremented usage_count for concept fp={fp}")
            return concept_data["cid"]
        else:
            # Generate a consistent concept ID
            cid = str(uuid4())
            concept_data = {
                "cid": cid,
                "embedding_id": vec_id,
                "meta": meta,
                "fp": fp,
                "usage_count": 1,
                "last_updated": time.time()
            }
            redis_client.set(fp_key, json.dumps(concept_data))
            # Store in Qdrant if we have an embedding
            if "embedding" in meta:
                qdrant_client.upsert(
                    collection_name="concepts",
                    points=[{
                        "id": cid,
                        "vector": meta["embedding"],
                        "payload": {
                            "cid": cid,
                            "embedding_id": vec_id,
                            "meta": meta,
                            "fp": fp,
                            "usage_count": 1
                        }
                    }]
                )
            logger.info(f"Upserted new concept {cid} with fp={fp}")
            return cid
    except Exception as e:
        logger.error(f"Error upserting concept: {e}")
        raise

async def handle_message(msg: StreamMessage) -> None:
    """Handle incoming messages from ingest streams."""
    try:
        data = msg.data
        fp = data.get("fp")
        if not fp:
            logger.warning(f"No fp in message: {data}")
            return
        if "url" in data:  # from crawler
            m = CrawlIn(**data)
            meta = {
                "url": m.url,
                "title": m.title,
                "source": "crawl",
                "timestamp": m.ts
            }
        else:  # from pdf trainer
            m = PdfIn(**data)
            meta = {
                "file": m.file_path,
                "source": "pdf",
                "timestamp": m.ts,
                **getattr(m, "metadata", {})
            }
        # Upsert concept by fp
        cid = await upsert_concept(m.vec_id, meta, fp)
        # Publish to concept.new stream
        out_msg = ConceptOut(
            cid=cid,
            embedding_id=m.vec_id,
            meta=meta,
            ts=time.time(),
            usage_count=1  # For new concepts; for existing, could fetch actual count if needed
        )
        await bus.publish("concept.new", out_msg.dict())
        logger.info(f"Published new/updated concept {cid} to concept.new stream")
    except Exception as e:
        logger.error(f"Error handling message: {e}")

async def start_consumers():
    """Start consuming from both ingest streams."""
    try:
        # Connect to Redis
        await bus.connect()
        
        # Start consumers
        logger.info("Starting ingest stream consumers...")
        await asyncio.gather(
            bus.consume(
                stream="ingest.pdf",
                group="dict",
                consumer="pdf-consumer",
                handler=handle_message,
                block_ms=1000,
                count=10
            ),
            bus.consume(
                stream="ingest.crawl",
                group="dict",
                consumer="crawl-consumer",
                handler=handle_message,
                block_ms=1000,
                count=10
            )
        )
    except Exception as e:
        logger.error(f"Error in consumer loop: {e}")
        raise
    finally:
        await bus.close()

async def main():
    """Main entry point."""
    while True:
        try:
            await start_consumers()
        except Exception as e:
            logger.error(f"Consumer loop failed: {e}")
            await asyncio.sleep(5)  # Wait before retry

if __name__ == "__main__":
    asyncio.run(main()) 