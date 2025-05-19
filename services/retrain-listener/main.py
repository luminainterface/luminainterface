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
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
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

# Environment variables
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8000")
CRAWLER_URL = os.getenv("CRAWLER_URL", "http://crawler:7000")
DATA_ANALYZER_URL = os.getenv("DATA_ANALYZER_URL", "http://data-analyzer:8500")
API_KEY = os.getenv("RETRAIN_API_KEY", "changeme")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "admin_key_change_me")

# Initialize metrics
RETRAIN_TRIGGERED = Counter("retrain_triggered_total", "Number of retraining triggers", ["reason"])
RETRAIN_BATCH_SIZE = Histogram("retrain_batch_size", "Size of retraining batches",
    buckets=[1, 2, 5, 10, 20, 50, 100])
RETRAIN_CONFIDENCE = Histogram("retrain_confidence", "Confidence scores of retrained concepts",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
RETRAIN_SECONDS = Histogram("retrain_seconds", "Time spent processing retraining requests",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
SYNC_TRIGGERED = Counter("sync_triggered_total", "Number of sync triggers", ["reason"])
SYNC_DURATION = Histogram("sync_duration_seconds", "Time spent in sync operations",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0])

# API key security
api_key_header = APIKeyHeader(name="X-API-Key")
admin_key_header = APIKeyHeader(name="X-Admin-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

async def verify_admin_key(admin_key: str = Depends(admin_key_header)):
    if admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return admin_key

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
        self.last_sync_time: Optional[datetime] = None
        self.sync_cooldown = 300  # 5 minutes between syncs
        
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

    async def should_trigger_sync(self) -> bool:
        """Determine if a sync should be triggered"""
        if not self.last_sync_time:
            return True
            
        time_since_sync = (datetime.now() - self.last_sync_time).total_seconds()
        return time_since_sync >= self.sync_cooldown

    async def trigger_sync(self, reason: str) -> Optional[str]:
        """Trigger a sync operation between services"""
        if not await self.should_trigger_sync():
            logger.info("Sync cooldown active, skipping sync")
            return None

        try:
            SYNC_TRIGGERED.labels(reason=reason).inc()
            start_time = time.time()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{DATA_ANALYZER_URL}/sync/trigger",
                    json={
                        "force": False,
                        "services": ["concept-dict", "crawler", "ollama"],
                        "dry_run": False
                    },
                    headers={"X-Admin-Key": ADMIN_API_KEY}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sync_id = result.get("sync_id")
                    if sync_id:
                        self.last_sync_time = datetime.now()
                        SYNC_DURATION.observe(time.time() - start_time)
                        logger.info(f"Sync triggered successfully: {sync_id}")
                        return sync_id
                        
            logger.error("Failed to trigger sync")
            return None
            
        except Exception as e:
            logger.error(f"Error triggering sync: {e}")
            return None
        
    async def process_batch(self):
        """Process the current batch of training data"""
        if not self.current_batch:
            return
            
        try:
            start_time = time.time()
            
            # Check if we should trigger a sync
            if await self.should_trigger_sync():
                sync_id = await self.trigger_sync("batch_processing")
                if sync_id:
                    # Wait for sync to complete
                    await self.wait_for_sync(sync_id)
            
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
                RETRAIN_SECONDS.observe(time.time() - start_time)
                    
        except Exception as e:
            logger.error(f"Error processing retraining batch: {e}")
            raise
        finally:
            # Clear batch
            self.current_batch = []
            self.last_batch_time = datetime.now()

    async def wait_for_sync(self, sync_id: str, timeout: int = 300) -> bool:
        """Wait for a sync operation to complete"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{DATA_ANALYZER_URL}/sync/status/{sync_id}",
                        headers={"X-API-Key": API_KEY}
                    )
                    
                    if response.status_code == 200:
                        status = response.json()
                        if status["status"] in ["completed", "failed"]:
                            return status["status"] == "completed"
                            
            except Exception as e:
                logger.error(f"Error checking sync status: {e}")
                
            await asyncio.sleep(5)  # Check every 5 seconds
            
        logger.error(f"Sync {sync_id} timed out after {timeout} seconds")
        return False

    async def start(self):
        """Start consuming messages"""
        while True:
            try:
                messages = await self.bus.consume("output.generated", "retrain", count=10)
                
                for msg in messages:
                    if self.should_trigger_retrain(msg):
                        self.current_batch.append(msg)
                        
                        # Process batch if full or timeout reached
                        if len(self.current_batch) >= self.batch_size or \
                           (datetime.now() - self.last_batch_time).total_seconds() >= self.batch_timeout:
                            await self.process_batch()
                            
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

@app.post("/sync/trigger")
async def trigger_sync(
    reason: str = "manual",
    admin_key: str = Depends(verify_admin_key)
) -> Dict[str, Any]:
    """Manually trigger a sync operation"""
    if not hasattr(app.state, "listener"):
        raise HTTPException(status_code=503, detail="Service not ready")
        
    sync_id = await app.state.listener.trigger_sync(reason)
    if not sync_id:
        raise HTTPException(status_code=500, detail="Failed to trigger sync")
        
    return {
        "status": "triggered",
        "sync_id": sync_id,
        "message": "Sync operation triggered"
    }

@app.get("/metrics/sync")
async def get_sync_metrics(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Get sync-related metrics"""
    return {
        "last_sync_time": app.state.listener.last_sync_time.isoformat() if app.state.listener.last_sync_time else None,
        "sync_cooldown": app.state.listener.sync_cooldown,
        "sync_triggered_total": {
            "low_confidence": SYNC_TRIGGERED.labels(reason="low_confidence")._value.get(),
            "batch_processing": SYNC_TRIGGERED.labels(reason="batch_processing")._value.get(),
            "manual": SYNC_TRIGGERED.labels(reason="manual")._value.get()
        },
        "sync_duration_seconds": {
            "count": SYNC_DURATION._count.get(),
            "sum": SYNC_DURATION._sum.get()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8680, log_level="info") 