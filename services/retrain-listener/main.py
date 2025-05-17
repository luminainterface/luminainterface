import asyncio
import os
import tempfile
import time
import uuid
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import aiohttp
from datetime import datetime
from lumina_core.common.bus import BusClient
from lumina_core.common.retry import with_retry

from sentence_transformers import SentenceTransformer
from lumina_core.common.stream_message import StreamMessage
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrain-listener")

# Initialize FastAPI app
app = FastAPI(title="Retrain Listener Service")
Instrumentator().instrument(app).expose(app)

# Initialize metrics
RETRAIN_TRIGGERED = Counter("retrain_triggered_total", "Number of retraining triggers", ["reason"])
RETRAIN_BATCH_SIZE = Histogram("retrain_batch_size", "Size of retraining batches",
    buckets=[1, 2, 5, 10, 20, 50, 100])
RETRAIN_CONFIDENCE = Histogram("retrain_confidence", "Confidence scores of retrained concepts",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
RETRAIN_SECONDS = Histogram("retrain_seconds", "Time spent processing retraining requests",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

# Initialize bus client and model
bus = BusClient(redis_url=os.getenv("REDIS_URL", "redis://redis:6379"))
BATCH_STREAM = os.getenv("SOURCE_STREAM", "output.generated")
EMBEDDER = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
TRAIN_ENDPOINT = os.getenv("TRAIN_URL", "http://concept-trainer-growable:8710/train_batch")

class TurnMessage(StreamMessage):
    turn_id: str
    concepts_used: List[str]
    text: str
    confidence: float

class TrainBatch(BaseModel):
    batch_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    embed_ids: List[str]
    vectors: List[List[float]]

class RetrainListener:
    def __init__(self, redis_url: str, train_url: str):
        self.bus = BusClient(redis_url=redis_url)
        self.train_url = train_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.min_confidence = 0.4
        self.batch_size = 10
        self.batch_timeout = 60  # seconds
        self.current_batch: List[Dict[str, Any]] = []
        self.last_batch_time = datetime.now()
        
    async def connect(self):
        """Connect to Redis and create consumer group"""
        await self.bus.connect()
        try:
            await self.bus.create_group("output.generated", "retrain")
        except Exception as e:
            logger.info(f"Group may exist: {e}")
            
        # Create aiohttp session
        self.session = aiohttp.ClientSession()
        
    async def close(self):
        """Close connections"""
        if self.session:
            await self.session.close()
            
    def should_trigger_retrain(self, msg: Dict[str, Any]) -> bool:
        """Determine if a message should trigger retraining"""
        # Check confidence
        confidence = msg.get("confidence", 0.0)
        if confidence < self.min_confidence:
            RETRAIN_TRIGGERED.labels(reason="low_confidence").inc()
            return True
            
        # Check for unknown concepts
        concepts_used = msg.get("concepts_used", [])
        if not concepts_used:
            RETRAIN_TRIGGERED.labels(reason="no_concepts").inc()
            return True
            
        return False
        
    async def process_batch(self):
        """Process the current batch of training data"""
        if not self.current_batch:
            return
            
        try:
            # Prepare batch data
            batch_data = {
                "concepts": [
                    {
                        "text": msg.get("text", ""),
                        "concepts": msg.get("concepts_used", []),
                        "confidence": msg.get("confidence", 0.0),
                        "timestamp": msg.get("timestamp")
                    }
                    for msg in self.current_batch
                ]
            }
            
            # Send to trainer
            async with self.session.post(
                self.train_url,
                json=batch_data
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail="Failed to trigger retraining"
                    )
                    
                result = await response.json()
                logger.info(f"Retraining triggered: {result}")
                
                # Record metrics
                RETRAIN_BATCH_SIZE.observe(len(self.current_batch))
                for msg in self.current_batch:
                    RETRAIN_CONFIDENCE.observe(msg.get("confidence", 0.0))
                    
        except Exception as e:
            logger.error(f"Error processing retraining batch: {e}")
            raise
        finally:
            # Clear batch
            self.current_batch = []
            self.last_batch_time = datetime.now()
            
    @with_retry("output.generated", max_attempts=3, dead_letter_stream="retrain.dlq")
    async def process_output(self, msg: Dict[str, Any]):
        """Process generated output with retry logic and DLQ support"""
        start_time = datetime.now()
        try:
            # Check if we should trigger retraining
            if not self.should_trigger_retrain(msg):
                return
                
            # Add to current batch
            self.current_batch.append(msg)
            
            # Check if we should process the batch
            now = datetime.now()
            if (len(self.current_batch) >= self.batch_size or
                (self.current_batch and
                 (now - self.last_batch_time).total_seconds() >= self.batch_timeout)):
                await self.process_batch()
                
            # Record metrics
            RETRAIN_SECONDS.observe(
                (datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Error processing output: {e}")
            raise
            
    async def start(self):
        """Start consuming from output.generated stream"""
        while True:
            try:
                # Process any pending batch
                if self.current_batch:
                    now = datetime.now()
                    if (now - self.last_batch_time).total_seconds() >= self.batch_timeout:
                        await self.process_batch()
                        
                # Consume messages
                await self.bus.consume(
                    stream="output.generated",
                    group="retrain",
                    consumer="worker",
                    handler=self.process_output,
                    block_ms=1000,
                    count=10
                )
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)

@app.on_event("startup")
async def startup():
    """Initialize listener on startup"""
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    train_url = os.getenv("TRAIN_URL", "http://batch-embedder:8709/train_batch")
    
    listener = RetrainListener(redis_url, train_url)
    await listener.connect()
    app.state.listener = listener
    # Start consumer loop
    asyncio.create_task(listener.start())

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    if hasattr(app.state, "listener"):
        await app.state.listener.close()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    from prometheus_client import generate_latest
    return Response(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 