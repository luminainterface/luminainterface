from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
import os
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import redis.asyncio as redis
from typing import Dict

from sentence_transformers import SentenceTransformer
from lumina_core.common.bus import BusClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("batch-embedder")

# Initialize FastAPI app
app = FastAPI(title="Batch Embedder")
Instrumentator().instrument(app).expose(app)

# Initialize metrics
BATCHES_RECEIVED = Counter(
    'batch_embedder_batches_total',
    'Number of batches received'
)

BATCH_SIZE = Histogram(
    'batch_embedder_batch_size',
    'Size of received batches'
)

PROCESSING_TIME = Histogram(
    'batch_embedder_processing_seconds',
    'Time spent processing batches'
)

# Initialize Redis client
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))

# Initialize bus client and model
MODEL = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
bus = BusClient(redis_url=os.getenv("REDIS_URL", "redis://redis:6379"))
TRAIN_STREAM = "trainer.vectors"

class TrainBatch(BaseModel):
    batch_id: str
    embed_ids: list[str]
    vectors: list[list[float]]

class HealthResponse(BaseModel):
    status: str
    dependencies: Dict[str, str]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint that verifies Redis and bus connections"""
    try:
        # Check Redis connection
        await redis_client.ping()
        # Check bus connection
        await bus.ping()
        return HealthResponse(
            status="healthy",
            dependencies={
                "redis": "connected",
                "bus": "connected"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.post("/train_batch")
async def train(batch: TrainBatch):
    """Process a batch of vectors and forward to trainer"""
    try:
        with PROCESSING_TIME.time():
            # Validate batch
            if not batch.embed_ids or not batch.vectors:
                raise HTTPException(status_code=400, detail="Empty batch")
            if len(batch.embed_ids) != len(batch.vectors):
                raise HTTPException(status_code=400, detail="Mismatched batch sizes")
                
            # Forward to trainer stream
            await bus.publish(TRAIN_STREAM, batch.dict())
            
            # Update metrics
            BATCHES_RECEIVED.inc()
            BATCH_SIZE.observe(len(batch.embed_ids))
            
            logger.info(f"Forwarded batch {batch.batch_id} with {len(batch.embed_ids)} vectors")
            return {"status": "queued", "batch_id": batch.batch_id}
            
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_raw_html(msg):
    data = msg.data
    url = data.get("url")
    html = data.get("html")
    fp = data.get("fp")
    ts = data.get("ts")
    if not html or not url or not fp:
        logger.warning(f"Malformed ingest.raw_html message: {data}")
        return
    # Embed html (simple: use first 1000 chars)
    text = html[:1000]
    vector = MODEL.encode(text).tolist()
    batch = {
        "batch_id": f"html-{fp}-{ts}",
        "embed_ids": [fp],
        "vectors": [vector]
    }
    await bus.publish(TRAIN_STREAM, batch)
    logger.info(f"Embedded and forwarded HTML for {url} (fp={fp})")

async def handle_raw_pdf(msg):
    data = msg.data
    file_id = data.get("file_id")
    fp = data.get("fp")
    ts = data.get("ts")
    if not file_id or not fp:
        logger.warning(f"Malformed ingest.raw_pdf message: {data}")
        return
    # Simulate embedding (real: load and embed PDF)
    vector = [0.0] * MODEL.get_sentence_embedding_dimension()
    batch = {
        "batch_id": f"pdf-{fp}-{ts}",
        "embed_ids": [fp],
        "vectors": [vector]
    }
    await bus.publish(TRAIN_STREAM, batch)
    logger.info(f"Forwarded PDF for {file_id} (fp={fp})")

@app.on_event("startup")
async def startup_event():
    await bus.connect()
    # Start consumers for both streams
    import asyncio
    asyncio.create_task(bus.consume(
        stream="ingest.raw_html",
        group="embed",
        consumer="batch-embedder",
        handler=handle_raw_html,
        block_ms=1000,
        count=1
    ))
    asyncio.create_task(bus.consume(
        stream="ingest.raw_pdf",
        group="embed",
        consumer="batch-embedder",
        handler=handle_raw_pdf,
        block_ms=1000,
        count=1
    ))

@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown"""
    await bus.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8709) 