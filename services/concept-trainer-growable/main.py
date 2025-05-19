import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, start_http_server, CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.cors import CORSMiddleware
import redis.asyncio as aioredis
from qdrant_client import QdrantClient
import numpy as np
from contextlib import asynccontextmanager
import uvicorn
import httpx
from time import sleep

from model import GrowableConceptNet
from routes import router as model_router
from concept_pipeline import GrowableConceptPipeline
from services.crawler.app.core.concept_client import ConceptClient
from services.crawler.app.core.embeddings import CustomOllamaEmbeddings
from services.crawler.app.core.sync_manager import SyncManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "768"))  # Default to BERT-like size
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
TRAINING_INTERVAL = int(os.getenv("TRAINING_INTERVAL", "60"))  # seconds
GROWTH_THRESHOLD = float(os.getenv("GROWTH_THRESHOLD", "0.1"))
API_PORT = int(os.getenv("API_PORT", "8710"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8711"))
DICT_URL = os.getenv("DICT_URL", "http://concept-dict:8828")
AUTO_INGEST_INTERVAL = int(os.getenv("AUTO_INGEST_INTERVAL", "120"))  # seconds
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Prometheus metrics
TRAINING_REQUESTS = Counter(
    'concept_trainer_training_requests_total',
    'Total number of training requests',
    ['concept_id']
)
TRAINING_DURATION = Histogram(
    'concept_trainer_training_duration_seconds',
    'Time spent training',
    ['concept_id']
)
GROWTH_EVENTS = Counter(
    'concept_trainer_growth_events_total',
    'Total number of model growth events',
    ['concept_id', 'layer_idx']
)
DRIFT_LEVEL = Histogram(
    'concept_trainer_drift_level',
    'Concept drift level',
    ['concept_id']
)

# Data models
class VectorBatch(BaseModel):
    """A batch of vectors for training"""
    concept_id: str
    vectors: List[List[float]]
    labels: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None

class TrainingMetrics(BaseModel):
    """Training metrics for a concept"""
    concept_id: str
    loss: float
    accuracy: float
    drift: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class GrowthRequest(BaseModel):
    """Request to grow the model"""
    concept_id: str
    layer_idx: int
    new_size: int
    reason: str

# Global state
redis_client = None
qdrant_client = None
model = None
concept_client = None
embeddings = None
sync_manager = None
pipeline = None
training_queue = asyncio.Queue()
is_training = False
training_lock = asyncio.Lock()

# Track last seen update per concept
last_concept_updates = {}

def get_concept_vectors(concept: dict) -> Optional[List[List[float]]]:
    # Try to extract vectors from the concept dict (adapt as needed)
    embedding = concept.get("embedding")
    if embedding and isinstance(embedding, list) and all(isinstance(x, (float, int)) for x in embedding):
        return [embedding]
    return None

async def auto_ingest_from_concept_dictionary():
    global last_concept_updates
    while True:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:  # Add timeout
                try:
                    resp = await client.get(f"{DICT_URL}/concepts")
                    if resp.status_code == 200:
                        concepts = resp.json()
                        new_count = 0
                        for concept in concepts:
                            try:
                                cid = concept.get("term")
                                last_updated = concept.get("last_updated") or concept.get("updated_at")
                                if not cid or not last_updated:
                                    logger.debug(f"Skipping concept without term or last_updated: {concept}")
                                    continue
                                
                                # Only process if new or updated
                                if cid not in last_concept_updates or last_concept_updates[cid] != last_updated:
                                    vectors = get_concept_vectors(concept)
                                    if vectors:
                                        batch = VectorBatch(
                                            concept_id=cid,
                                            vectors=vectors,
                                            labels=None,
                                            metadata=concept.get("metadata")
                                        )
                                        await training_queue.put(batch)
                                        last_concept_updates[cid] = last_updated
                                        new_count += 1
                                        logger.info(f"Queued concept {cid} for training")
                                    else:
                                        logger.debug(f"No vectors found for concept {cid}")
                            except Exception as e:
                                logger.error(f"Error processing concept {concept.get('term', 'unknown')}: {str(e)}")
                                continue
                        
                        if new_count:
                            logger.info(f"Auto-ingested {new_count} new/updated concepts from Concept Dictionary")
                            await trigger_training()
                        else:
                            logger.debug("No new concepts to ingest")
                    else:
                        logger.warning(f"Failed to fetch concepts from Concept Dictionary: {resp.status_code} - {resp.text}")
                except httpx.ConnectError as e:
                    logger.warning(f"Cannot connect to Concept Dictionary at {DICT_URL}: {str(e)}")
                except httpx.TimeoutException as e:
                    logger.warning(f"Timeout connecting to Concept Dictionary at {DICT_URL}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error during concept dictionary fetch: {str(e)}")
        except Exception as e:
            logger.error(f"Auto-ingest error: {str(e)}", exc_info=True)
        
        # Sleep before next attempt, with exponential backoff if there were errors
        sleep_time = AUTO_INGEST_INTERVAL
        if len(last_concept_updates) == 0:  # If we haven't successfully ingested anything yet
            sleep_time = min(sleep_time * 2, 300)  # Double the interval up to 5 minutes
        logger.debug(f"Sleeping for {sleep_time} seconds before next auto-ingest attempt")
        await asyncio.sleep(sleep_time)

async def trigger_training():
    global is_training
    if not is_training and not training_lock.locked() and not training_queue.empty():
        async with training_lock:
            is_training = True
            try:
                while not training_queue.empty():
                    batch = await training_queue.get()
                    # Actual training logic
                    vectors = torch.tensor(batch.vectors, dtype=torch.float32)
                    labels = torch.zeros(len(batch.vectors), dtype=torch.long) if batch.labels is None else torch.tensor(batch.labels, dtype=torch.long)
                    outputs = model(vectors)
                    criterion = torch.nn.NLLLoss()
                    loss = criterion(outputs, labels)
                    model.optimizer.zero_grad()
                    loss.backward()
                    model.optimizer.step()
                    # Optionally record metrics
                    accuracy = (torch.argmax(outputs, dim=1) == labels).float().mean().item()
                    model.record_training(batch.concept_id, loss.item(), accuracy, drift=0.0)
                    logger.info(f"Auto-trained on concept: {batch.concept_id}, loss={loss.item():.4f}, acc={accuracy:.4f}")
                # Save model after training
                torch.save(model.state_dict(), "model_final.pth")
                logger.info("Model state saved after auto-training.")
            finally:
                is_training = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global redis_client, qdrant_client, model, concept_client, embeddings, sync_manager, pipeline
    
    # Initialize Redis client
    redis_client = await aioredis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True
    )
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY
    )
    
    # Initialize concept client
    concept_client = ConceptClient(base_url=DICT_URL)
    
    # Initialize embeddings
    embeddings = CustomOllamaEmbeddings()
    
    # Initialize model
    model = GrowableConceptNet(
        input_size=MODEL_INPUT_SIZE,
        hidden_sizes=[MODEL_INPUT_SIZE, MODEL_INPUT_SIZE // 2, MODEL_INPUT_SIZE // 4],
        output_size=2,  # Binary classification for now
        learning_rate=1e-4
    )
    
    # Load saved model if exists
    try:
        model.load_state_dict(torch.load("model_final.pth"))
        logger.info("Loaded saved model state")
    except:
        logger.info("No saved model found, starting fresh")
    
    # Initialize sync manager
    sync_manager = SyncManager(
        concept_client=concept_client,
        qdrant_client=qdrant_client,
        sync_interval=3600.0  # 1 hour
    )
    
    # Initialize and start pipeline
    pipeline = GrowableConceptPipeline(
        concept_client=concept_client,
        embeddings=embeddings,
        model=model,
        sync_manager=sync_manager,
        poll_interval=1.0,
        growth_threshold=GROWTH_THRESHOLD
    )
    
    # Start services
    await sync_manager.start()
    await pipeline.start()
    
    # Start metrics server
    start_http_server(METRICS_PORT)
    
    yield
    
    # Cleanup
    if pipeline:
        await pipeline.stop()
    if sync_manager:
        await sync_manager.stop()
    if redis_client:
        await redis_client.close()
    if concept_client:
        await concept_client.__aexit__(None, None, None)

# Initialize FastAPI app
app = FastAPI(
    title="Concept Trainer Growable Service",
    description="A service for training and growing concept models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(model_router, prefix="/api/v1")

# Setup Prometheus instrumentation if enabled
if not os.getenv("DISABLE_METRICS", "").lower() in ("1", "true", "yes"):
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "pipeline_running": pipeline._running if pipeline else False,
        "sync_manager_running": sync_manager._running if sync_manager else False,
        "last_sync": sync_manager.last_sync_time.isoformat() if sync_manager and sync_manager.last_sync_time else None
    }

@app.get("/pipeline/status")
async def pipeline_status():
    """Get pipeline status and metrics"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    return {
        "running": pipeline._running,
        "model_stats": model.get_network_stats() if model else None,
        "last_sync": sync_manager.last_sync_time.isoformat() if sync_manager and sync_manager.last_sync_time else None,
        "time_since_sync": sync_manager.time_since_last_sync.total_seconds() if sync_manager and sync_manager.time_since_last_sync else None
    }

@app.post("/pipeline/retry-failed")
async def retry_failed():
    """Retry failed concepts"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    await pipeline.retry_failed_concepts()
    return {"status": "retry_triggered"}

@app.post("/pipeline/force-sync")
async def force_sync():
    """Force a sync between Redis and Qdrant"""
    if not sync_manager:
        raise HTTPException(status_code=503, detail="Sync manager not initialized")
        
    await sync_manager.force_sync()
    return {"status": "sync_triggered"}

@app.post("/model/train")
async def train_model(batch: VectorBatch, background_tasks: BackgroundTasks):
    """Add a batch of vectors to the training queue"""
    try:
        # Validate input
        if not batch.vectors or not batch.concept_id:
            raise HTTPException(status_code=400, detail="Invalid batch data")
            
        # Add to training queue
        await training_queue.put(batch)
        CONCEPT_QUEUE_SIZE.inc()
        
        return {"status": "queued", "concept_id": batch.concept_id}
    except Exception as e:
        TRAINING_REQUESTS.labels(concept_id=batch.concept_id).inc()
        logger.error(f"Error queueing training batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def model_status():
    """Get current model status"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {
        "queue_size": training_queue.qsize(),
        "is_training": is_training,
        "model_stats": model.get_network_stats(),
        "last_training": getattr(model, 'last_training', None)
    }

@app.post("/model/grow")
async def grow_model(request: GrowthRequest):
    """Grow the model's capacity"""
    try:
        if not model:
            raise HTTPException(status_code=503, detail="Model not initialized")
            
        # Grow the specified layer
        model.grow_layer(request.layer_idx, request.new_size)
        GROWTH_EVENTS.labels(concept_id=request.concept_id, layer_idx=request.layer_idx).inc()
        
        return {
            "status": "grown",
            "concept_id": request.concept_id,
            "layer_idx": request.layer_idx,
            "new_size": request.new_size
        }
    except Exception as e:
        logger.error(f"Error growing model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_training_queue():
    """Process the training queue periodically"""
    global is_training
    
    while True:
        try:
            # Get next batch from Redis
            # TODO: Implement proper queue processing
            pass
        except Exception as e:
            logger.error(f"Error processing training queue: {str(e)}")
        finally:
            is_training = False
            await asyncio.sleep(TRAINING_INTERVAL)

async def monitor_concept_queue():
    """Monitor concept queue for growth opportunities"""
    while True:
        try:
            # Get all tracked concepts from model metrics
            stats = model.get_network_stats()
            tracked_concepts = list(model.concept_metrics.keys())
            
            # Check each concept for growth
            for concept_id in tracked_concepts:
                should_grow, layer_idx = model.should_grow(concept_id)
                if should_grow:
                    logger.info(f"Growth opportunity detected for concept {concept_id} at layer {layer_idx}")
                    # TODO: Implement automatic growth
                    pass
        except Exception as e:
            logger.error(f"Error monitoring concept queue: {str(e)}")
        await asyncio.sleep(TRAINING_INTERVAL)

@app.on_event("startup")
async def startup_event():
    logger.info("Concept trainer growable startup event triggered. (The service will connect to qdrant and redis as soon as they are available.)") 