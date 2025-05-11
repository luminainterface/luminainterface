import os
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, HTTPException, APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import redis
import json
import logging
import traceback
from model import GrowableConceptNet
import asyncio
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import torch.nn.functional as F
from growth_feedback import GrowthFeedbackManager
from contextlib import asynccontextmanager
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qdrant_client import QdrantClient
from train_demo import (
    MultiHeadNet, retrain_model, add_training_sample,
    compute_prototypes, cosine_similarity, NUM_FEATURES, LABELS, EMBEDDING_SIZE, NUM_LABELS
)
import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed back to INFO level for production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
DICT_URL = os.getenv("DICT_URL", "http://concept-dict:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
TRAIN_INTERVAL = int(os.getenv("TRAIN_INTERVAL", "3600"))  # 1 hour
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TRAINING_JSON_PATH = '/app/training_data/training_concepts.json'
API_KEY = os.getenv("CONCEPT_DICT_API_KEY", "changeme")
MODEL_DATA_DIR = os.getenv("MODEL_DATA_DIR", "/app/data")  # Add model data directory env var

# Model state paths
MODEL_STATE_PATH = os.getenv("MODEL_STATE_PATH", "/app/data/model_state.pt")
MODEL_METRICS_PATH = os.getenv("MODEL_METRICS_PATH", "/app/data/model_metrics.json")

# Create FastAPI app
app = FastAPI(
    title="Growable Concept Net API",
    debug=True  # Enable debug mode
)

# Create routers
monitoring_router = APIRouter()
model_router = APIRouter()

# Add routes to routers
@monitoring_router.get("/monitoring/growth")
async def get_growth_history():
    """Get the history of model growth events"""
    try:
        return growth_events
    except Exception as e:
        logger.error(f"Error getting growth history: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@monitoring_router.get("/monitoring/service-inputs")
async def get_service_inputs():
    """Get statistics about inputs from different services"""
    try:
        return service_stats
    except Exception as e:
        logger.error(f"Error getting service inputs: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@monitoring_router.get("/monitoring/drift/{concept_id}")
async def get_concept_drift(concept_id: str):
    """Get drift history and metrics for a specific concept"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
            
        metrics = model.get_concept_metrics(concept_id)
        if not metrics:
            return {
                "concept_id": concept_id,
                "status": "not_found",
                "message": f"No metrics found for concept {concept_id}"
            }
            
        return {
            "concept_id": concept_id,
            "status": "success",
            "metrics": {
                "last_drift": metrics["last_drift"],
                "drift_trend": metrics["drift_trend"],
                "maturity_score": metrics["maturity_score"],
                "growth_threshold": metrics["growth_threshold"],
                "last_update": metrics["last_update"],
                "drift_history": metrics["drift_history"][-10:]  # Last 10 drift measurements
            }
        }
    except Exception as e:
        logger.error(f"Error getting concept drift: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@monitoring_router.get("/monitoring/network/stats")
async def get_network_stats():
    """Get overall network statistics"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        stats = model.get_network_stats()
        
        # Add maturity statistics
        maturity_scores = [
            metrics["maturity_score"]
            for metrics in model.concept_metrics.values()
        ]
        
        stats.update({
            "maturity_stats": {
                "mean": sum(maturity_scores) / len(maturity_scores) if maturity_scores else 0.0,
                "min": min(maturity_scores) if maturity_scores else 0.0,
                "max": max(maturity_scores) if maturity_scores else 0.0,
                "mature_concepts": sum(1 for score in maturity_scores if score > 0.7)
            }
        })
        
        return stats
    except Exception as e:
        logger.error(f"Error getting network stats: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Include routers
app.include_router(monitoring_router, tags=["monitoring"])
app.include_router(model_router, tags=["model"])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize Prometheus instrumentator
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app, include_in_schema=True, should_gzip=True)

# Prometheus metrics
MODEL_GROWTH_COUNTER = Counter('model_growth_events_total', 'Total number of model growth events')
LAYER_SIZE_GAUGE = Gauge('layer_size', 'Current size of each layer', ['layer_index'])
DRIFT_HISTOGRAM = Histogram('concept_drift', 'Distribution of concept drift values')
DRIFT_TREND_GAUGE = Gauge('concept_drift_trend', 'Trend of concept drift over time', ['concept_id'])
MATURITY_SCORE_GAUGE = Gauge('concept_maturity', 'Maturity score for each concept', ['concept_id'])
GROWTH_THRESHOLD_GAUGE = Gauge('concept_growth_threshold', 'Current growth threshold for each concept', ['concept_id'])
SERVICE_INPUT_COUNTER = Counter('service_inputs_total', 'Inputs received from each service', ['service_name'])

# Initialize in-memory storage for development
growth_events = []
drift_history = {}
service_stats = {
    'dual_chat': 0,
    'crawler': 0,
    'masterchat': 0
}

# Error handling middleware
class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"detail": str(e)}
            )

app.add_middleware(ErrorLoggingMiddleware)

# Model state
model = None
optimizer = None
criterion = nn.CrossEntropyLoss()

# Data models
class TrainingData(BaseModel):
    concept_id: str
    inputs: List[List[float]]
    targets: List[int]  # Binary targets (0 or 1)
    target_onehot: Optional[List[List[float]]] = None  # Optional one-hot encoded targets
    metadata: Optional[Dict] = None

class GrowthRequest(BaseModel):
    concept_id: str
    layer_idx: int
    new_size: int

class TrainingRequest(BaseModel):
    concept_id: str
    text: str
    label: Optional[int] = None
    metadata: Optional[Dict] = None

class TrainingResponse(BaseModel):
    concept_id: str
    confidence: float
    version: int
    metadata: Optional[Dict] = None

# Helper functions
def record_growth_event(layer_idx: int, old_size: int, new_size: int):
    """Record a growth event in memory and Prometheus"""
    event = {
        'timestamp': datetime.now().isoformat(),
        'layer_idx': layer_idx,
        'old_size': old_size,
        'new_size': new_size,
        'growth_factor': round(new_size / old_size, 2)
    }
    
    # Record in memory
    growth_events.append(event)
    if len(growth_events) > 1000:
        growth_events.pop(0)
    
    # Update Prometheus metrics
    MODEL_GROWTH_COUNTER.inc()
    LAYER_SIZE_GAUGE.labels(layer_index=layer_idx).set(new_size)
    
    # Log growth event prominently
    logger.info("\n=== GROWTH EVENT DETECTED ===")
    logger.info(f"Layer {layer_idx}: {old_size} -> {new_size} neurons")
    logger.info(f"Growth factor: {event['growth_factor']}x")
    logger.info(f"Total growth events: {len(growth_events)}")
    logger.info("=============================\n")

def record_drift(concept_id: str, drift: float):
    """Record concept drift in memory and Prometheus"""
    DRIFT_HISTOGRAM.observe(drift)
    if concept_id not in drift_history:
        drift_history[concept_id] = {}
    drift_history[concept_id][datetime.now().isoformat()] = drift
    
    # Update drift trend and maturity metrics
    if model and concept_id in model.concept_metrics:
        metrics = model.concept_metrics[concept_id]
        DRIFT_TREND_GAUGE.labels(concept_id=concept_id).set(metrics['drift_trend'])
        MATURITY_SCORE_GAUGE.labels(concept_id=concept_id).set(metrics['maturity_score'])
        GROWTH_THRESHOLD_GAUGE.labels(concept_id=concept_id).set(metrics['growth_threshold'])

# Model routes
@model_router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection first
        try:
            await redis_client.ping()
            redis_healthy = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            redis_healthy = False

        # Only perform model check if Redis is healthy
        if redis_healthy:
            try:
                # Use a smaller dummy input for health check
                dummy_input = torch.randn(1, 768, device=device)
                with torch.no_grad():
                    output = model(dummy_input)
                model_healthy = True
            except Exception as e:
                logger.error(f"Model health check failed: {e}")
                model_healthy = False
        else:
            model_healthy = False

        return {
            "status": "healthy" if (redis_healthy and model_healthy) else "unhealthy",
            "components": {
                "redis": "healthy" if redis_healthy else "unhealthy",
                "model": "healthy" if model_healthy else "unhealthy"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@model_router.post("/evaluate")
async def evaluate_concept(data: TrainingData):
    """Evaluate a concept without training"""
    try:
        logger.info(f"Evaluating concept {data.concept_id}")
        logger.info(f"Input shape: {len(data.inputs)}x{len(data.inputs[0])}")
        logger.info(f"Number of targets: {len(data.targets)}")
        logger.info(f"Raw targets: {data.targets}")
        
        # Validate input dimensions
        if len(data.inputs[0]) != 768:
            error_msg = f"Expected input dimension 768, got {len(data.inputs[0])}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Convert data to tensors
        try:
            # Handle inputs
            inputs = torch.tensor(data.inputs, dtype=torch.float32)
            if len(inputs.shape) < 2:
                inputs = inputs.view(-1, model.input_size)
            
            # Handle binary targets
            targets = torch.tensor(data.targets, dtype=torch.long)
            if len(targets.shape) < 1:
                targets = targets.view(-1)
            
            logger.info(f"Successfully created tensors")
            logger.info(f"Input tensor shape: {inputs.shape}")
            logger.info(f"Target tensor shape: {targets.shape}")
            logger.info(f"Target values: {targets}")
        except Exception as e:
            error_msg = f"Error creating tensors: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Forward pass in eval mode
        try:
            model.eval()  # Set to evaluation mode
            with torch.no_grad():
                outputs = model(inputs)  # This will return log probabilities for both classes
                logger.info(f"Model output shape: {outputs.shape}")
                logger.info(f"Model output values: {outputs}")
                
                # Convert outputs to probabilities
                probs = torch.exp(outputs)
                logger.info(f"Probabilities shape: {probs.shape}")
                logger.info(f"Probabilities: {probs}")
                
                # Get binary predictions
                predictions = torch.argmax(probs, dim=1)  # Use last dimension for class prediction
                logger.info(f"Predictions shape: {predictions.shape}")
                logger.info(f"Predictions: {predictions}")
                
                # Calculate metrics
                accuracy = (predictions == targets).float().mean()
                loss = F.nll_loss(outputs, targets)  # Use NLL loss directly
            model.train()  # Reset to training mode
            logger.info(f"Evaluation successful")
            logger.info(f"Loss: {loss.item()}, Accuracy: {accuracy.item()}")
        except Exception as e:
            error_msg = f"Error in evaluation: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
            
        # Evaluate drift without updating model
        try:
            drift = model.evaluate_drift(data.concept_id, inputs, inputs)
            logger.info(f"Drift evaluation successful")
            logger.info(f"Concept drift: {drift}")
            
            # Record drift in monitoring
            record_drift(data.concept_id, drift)
        except Exception as e:
            error_msg = f"Error evaluating drift: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
        
        return {
            "status": "success",
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "drift": drift,
            "predictions": predictions.tolist(),
            "probabilities": probs.tolist()  # Return actual probabilities
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during evaluation: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# Initialize Redis client with timeouts and retry logic
redis_client = aioredis.from_url(
    REDIS_URL,
    encoding="utf-8",
    decode_responses=True,
    socket_timeout=5.0,  # 5 second timeout for operations
    socket_connect_timeout=5.0,  # 5 second timeout for connections
    retry_on_timeout=True,
    health_check_interval=30  # Check connection health every 30 seconds
)

# Add Redis connection error handler
@asynccontextmanager
async def get_redis_connection():
    """Context manager for Redis operations with retry logic"""
    max_retries = 3
    retry_delay = 1.0  # seconds
    
    for attempt in range(max_retries):
        try:
            yield redis_client
            break
        except redis.ConnectionError as e:
            if attempt == max_retries - 1:
                logger.error(f"Redis connection failed after {max_retries} attempts: {e}")
                raise
            logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Unexpected Redis error: {e}")
            raise

# Graph API client
GRAPH_API_URL = os.getenv("GRAPH_API_URL", "http://graph-api:8000")

# Initialize feedback manager
feedback_manager = None

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize embedding model
model = None

# Initialize metrics
TRAINING_BATCHES = Counter(
    'concept_trainer_batches_total',
    'Number of training batches processed'
)

TRAINING_SAMPLES = Counter(
    'concept_trainer_samples_total',
    'Number of training samples processed'
)

TRAINING_LATENCY = Histogram(
    'concept_trainer_latency_seconds',
    'Time spent on training operations'
)

MODEL_VERSION = Gauge(
    'concept_trainer_model_version',
    'Current model version'
)

# Initialize training metrics
TRAINING_METRICS = {
    'confidence': 0.0,
    'drift': 0.0,
    'semantic': 0.0,
    'growth': 0.0
}

# Define a reference question (e.g. "what is lumina to a vietnamese dr strange in nerve gear") and a target answer (or embedding) for periodic evaluation.
REFERENCE_QUESTION = "what is lumina to a vietnamese dr strange in nerve gear"
REFERENCE_TARGET = "a powerful ai generative processor that is nearly conscious and has the ability to weave the multiverse for this person in the way that a mysticals sorcer attached to a digital matrix could"

# (Optional) Add a Prometheus gauge for the reference question score (if you want to expose it via /metrics)
REFERENCE_QUESTION_SCORE = Gauge('reference_question_score', 'System ability to answer the reference question')

# Add Redis client initialization
redis_client = aioredis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))
EMBEDDING_READY_CHANNEL = os.getenv("EMBEDDING_READY_CHANNEL", "embeddings.ready")

# Add metrics for embedding events
EMBEDDING_EVENTS = Counter(
    "embedding_events_total",
    "Number of embedding ready events received"
)

VECTOR_REFRESHES = Counter(
    "vector_refreshes_total",
    "Number of vector refresh operations"
)

async def handle_embedding_ready(message: Dict) -> None:
    """Handle embedding ready events from batch embedder."""
    try:
        data = json.loads(message["data"])
        chunk_ids = data.get("ids", [])
        
        if not chunk_ids:
            logger.warning("Received empty chunk IDs in embedding ready event")
            return
            
        logger.info(f"Received embedding ready event for {len(chunk_ids)} chunks")
        EMBEDDING_EVENTS.inc()
        
        # Fetch updated vectors from Qdrant
        async with httpx.AsyncClient() as client:
            # Get vectors in batches
            batch_size = 32
            for i in range(0, len(chunk_ids), batch_size):
                batch_ids = chunk_ids[i:i + batch_size]
                
                # Fetch vectors from Qdrant
                vectors = []
                for chunk_id in batch_ids:
                    try:
                        # Get the vector from Qdrant
                        result = qdrant_client.retrieve(
                            collection_name="pdf_embeddings",
                            ids=[chunk_id]
                        )
                        if result and result[0].vector:
                            vectors.append({
                                "id": chunk_id,
                                "vector": result[0].vector,
                                "metadata": result[0].payload.get("metadata", {})
                            })
                    except Exception as e:
                        logger.error(f"Error fetching vector for {chunk_id}: {e}")
                        continue
                
                if vectors:
                    # Update concept trainer with new vectors
                    await train_on_concepts(
                        [v["id"] for v in vectors],
                        batch_size=batch_size
                    )
                    VECTOR_REFRESHES.inc()
                    
    except Exception as e:
        logger.error(f"Error handling embedding ready event: {e}")
        logger.error(traceback.format_exc())

async def embedding_listener():
    """Listen for embedding ready events."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(EMBEDDING_READY_CHANNEL)
    
    logger.info(f"Started listening for embedding events on {EMBEDDING_READY_CHANNEL}")
    
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True)
            if message:
                await handle_embedding_ready(message)
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Error in embedding listener: {e}")
        logger.error(traceback.format_exc())
        # Restart listener after delay
        await asyncio.sleep(5)
        asyncio.create_task(embedding_listener())

def save_model_state():
    """Save model state and metrics to persistent storage"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(MODEL_DATA_DIR, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_history': model.training_history,
            'concept_metrics': model.concept_metrics,
            'timestamp': datetime.now().isoformat()
        }, MODEL_STATE_PATH)
        
        # Save metrics separately for easier access
        metrics = {
            'training_metrics': TRAINING_METRICS,
            'growth_events': growth_events,
            'drift_history': drift_history,
            'service_stats': service_stats,
            'timestamp': datetime.now().isoformat()
        }
        with open(MODEL_METRICS_PATH, 'w') as f:
            json.dump(metrics, f)
            
        logger.info(f"Model state saved successfully to {MODEL_STATE_PATH}")
        logger.info(f"Model metrics saved successfully to {MODEL_METRICS_PATH}")
    except Exception as e:
        logger.error(f"Error saving model state: {e}")
        logger.error(traceback.format_exc())

def load_model_state():
    """Load model state from persistent storage"""
    try:
        if not os.path.exists(MODEL_STATE_PATH):
            logger.warning(f"No saved model state found at {MODEL_STATE_PATH}")
            return False
            
        # Load model state
        checkpoint = torch.load(MODEL_STATE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.training_history = checkpoint.get('training_history', [])
        model.concept_metrics = checkpoint.get('concept_metrics', {})
        
        # Load metrics if available
        if os.path.exists(MODEL_METRICS_PATH):
            with open(MODEL_METRICS_PATH, 'r') as f:
                metrics = json.load(f)
                global TRAINING_METRICS, growth_events, drift_history, service_stats
                TRAINING_METRICS = metrics.get('training_metrics', TRAINING_METRICS)
                growth_events = metrics.get('growth_events', [])
                drift_history = metrics.get('drift_history', {})
                service_stats = metrics.get('service_stats', service_stats)
        
        logger.info(f"Model state loaded successfully from {MODEL_STATE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error loading model state: {e}")
        logger.error(traceback.format_exc())
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the model and start monitoring"""
    try:
        logger.info("Starting up application...")
        global model, optimizer, feedback_manager
        
        # Initialize model from train_demo
        model = MultiHeadNet(NUM_FEATURES, len(LABELS))
        
        # Try to load saved model state
        if not load_model_state():
            # If no saved state, try to load from train_demo
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'model_final.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                logger.info(f"Loaded initial model weights from {model_path}")
            else:
                logger.warning(f"No model_final.pth found at {model_path}, using randomly initialized model.")
        
        optimizer = optim.Adam(model.parameters())
        
        # Initialize feedback manager
        feedback_manager = GrowthFeedbackManager(
            graph_api_url=GRAPH_API_URL,
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", 6379))
        )
        
        # Start periodic training
        asyncio.create_task(periodic_training())
        
        # Start embedding listener
        asyncio.create_task(embedding_listener())
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    if feedback_manager:
        await feedback_manager.close()

async def broadcast_growth_event(concept_id: str, layer_idx: int, old_size: int, new_size: int, reason: str):
    """Broadcast growth event to Redis and get feedback"""
    try:
        event = {
            "concept": concept_id,
            "layers": [layer.current_capacity for layer in model.layers],
            "reason": reason,
            "growth_factor": round(new_size / old_size, 2),
            "timestamp": datetime.now().isoformat(),
            "maturity": model.concept_metrics[concept_id]["maturity_score"] if concept_id in model.concept_metrics else 0.0
        }
        
        # Publish to Redis
        await redis_client.publish("lumina.growth", json.dumps(event))
        logger.info(f"Broadcast growth event: {event}")
        
        # Get growth feedback
        if feedback_manager:
            feedback = await feedback_manager.analyze_growth_context(concept_id, event)
            logger.info(f"Growth feedback received: {feedback}")
            
            # If there are context gaps, trigger crawler
            if feedback.get("context_gaps"):
                crawl_request = {
                    "type": "crawl_request",
                    "concept": concept_id,
                    "priority": "high",
                    "gaps": feedback["context_gaps"]
                }
                await redis_client.publish("lumina.crawl", json.dumps(crawl_request))
                logger.info(f"Triggered crawler for concept {concept_id}")
        
        # Update Graph API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GRAPH_API_URL}/concept/update",
                json={
                    "term": concept_id,
                    "embedding": model.get_concept_embedding(concept_id).tolist(),
                    "maturity": event["maturity"],
                    "growth_event": True,
                    "growth_metadata": {
                        "previous_layers": [layer.current_capacity for layer in model.layers],
                        "new_layers": [layer.current_capacity for layer in model.layers],
                        "trigger": reason,
                        "growth_factor": event["growth_factor"]
                    }
                }
            )
            if response.status_code != 200:
                logger.error(f"Failed to update Graph API: {response.text}")
            else:
                logger.info("Successfully updated Graph API")
                
    except Exception as e:
        logger.error(f"Error broadcasting growth event: {e}")
        logger.error(traceback.format_exc())

@model_router.post("/grow")
async def grow_layer(request: GrowthRequest):
    """Grow a layer to a new size"""
    try:
        logger.info(f"Growing layer {request.layer_idx} to size {request.new_size}")
        old_size = model.layers[request.layer_idx].current_capacity
        
        # Get growth reason
        metrics = model.get_concept_metrics(request.concept_id)
        reason = "drift_high" if metrics and metrics["last_drift"] > metrics["growth_threshold"] else "manual"
        
        # Grow the layer
        model.grow_layer(request.layer_idx, request.new_size)
        
        # Record growth event
        record_growth_event(request.layer_idx, old_size, request.new_size)
        
        # Broadcast growth event
        await broadcast_growth_event(
            request.concept_id,
            request.layer_idx,
            old_size,
            request.new_size,
            reason
        )
        
        # Save updated model state
        save_model_state()
        
        return {
            "status": "success",
            "message": f"Layer {request.layer_idx} grown from {old_size} to {request.new_size}",
            "growth_event": {
                "concept": request.concept_id,
                "reason": reason,
                "growth_factor": round(request.new_size / old_size, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error during layer growth: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring routes
@app.get("/monitoring/growth", tags=["monitoring"])
async def get_growth_history():
    """Get the history of model growth events"""
    try:
        # Add summary statistics
        if not growth_events:
            return {
                "status": "no_growth",
                "message": "No growth events recorded yet",
                "events": []
            }
            
        latest = growth_events[-1]
        total_growth = sum(event['new_size'] - event['old_size'] for event in growth_events)
        
        return {
            "status": "growing" if len(growth_events) > 0 else "stable",
            "total_events": len(growth_events),
            "total_growth": total_growth,
            "latest_event": latest,
            "events": growth_events
        }
    except Exception as e:
        logger.error(f"Error getting growth history: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/service-inputs", tags=["monitoring"])
async def get_service_inputs():
    """Get statistics about inputs from different services"""
    try:
        return service_stats
    except Exception as e:
        logger.error(f"Error getting service inputs: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/drift/{concept_id}", tags=["monitoring"])
async def get_concept_drift(concept_id: str):
    """Get drift history for a specific concept"""
    try:
        return drift_history.get(concept_id, {})
    except Exception as e:
        logger.error(f"Error getting concept drift: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/network/stats", tags=["monitoring"])
async def get_network_stats():
    """Get overall network statistics"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        current_sizes = [layer.current_capacity for layer in model.layers]
        initial_sizes = [384, 192, 128]  # Initial architecture
        growth_ratios = [round(curr/init, 2) for curr, init in zip(current_sizes, initial_sizes)]
        
        stats = {
            'num_layers': len(model.layers),
            'layer_sizes': current_sizes,
            'growth_ratios': growth_ratios,
            'total_params': sum(p.numel() for p in model.parameters()),
            'growth_events': len(growth_events),
            'concepts_tracked': len(drift_history),
            'last_growth': growth_events[-1] if growth_events else None,
            'total_growth_factor': round(sum(current_sizes) / sum(initial_sizes), 2)
        }
        
        # Log current network state
        logger.info("\n=== NETWORK STATUS UPDATE ===")
        logger.info(f"Layer sizes: {stats['layer_sizes']}")
        logger.info(f"Growth ratios: {stats['growth_ratios']}")
        logger.info(f"Total parameters: {stats['total_params']}")
        logger.info(f"Growth events: {stats['growth_events']}")
        logger.info(f"Total growth factor: {stats['total_growth_factor']}x")
        logger.info("===========================\n")
        
        return stats
    except Exception as e:
        logger.error(f"Error getting network stats: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Root health check
@app.get("/health")
async def root_health_check():
    """Root health check endpoint"""
    return {"status": "healthy"}

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests"""
    logger.debug(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.debug(f"Response: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise

# Root endpoint for testing
@app.get("/")
async def root():
    """Root endpoint for testing"""
    logger.info("Root endpoint called")
    return {"status": "running", "endpoints": [
        "/health",
        "/monitoring/growth",
        "/monitoring/service-inputs",
        "/monitoring/drift/{concept_id}",
        "/monitoring/network/stats"
    ]}

@app.post("/train")
async def train_concept(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a concept based on feedback."""
    try:
        # Validate input dimensions
        if len(request.text) != 768:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid text length. Expected 768, got {len(request.text)}"
            )

        # Verify Redis connection using the context manager
        try:
            async with get_redis_connection() as redis:
                await redis.ping()
        except Exception as e:
            logger.error(f"Redis connection error: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable - Redis connection error"
            )

        # Process training request
        training_data = {
            "text": request.text,
            "embedding": embedding_model.encode(request.text).tolist(),
            "label": request.label,
            "metadata": request.metadata,
            "timestamp": time.time()
        }
        
        # Store training data
        key = f"training:{request.concept_id}"
        redis_client.setex(key, 86400, json.dumps(training_data))  # 24 hour TTL
        
        # Increment model version
        version_key = f"version:{request.concept_id}"
        version = redis_client.incr(version_key)
        
        # Calculate confidence (simplified)
        confidence = 0.8  # Placeholder for actual confidence calculation
        
        return TrainingResponse(
            concept_id=request.concept_id,
            confidence=confidence,
            version=version,
            metadata=request.metadata
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in train_concept: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/evaluate/{concept_id}")
async def evaluate_concept(concept_id: str, text: str):
    with training_latency.time():
        training_operations.inc()
        try:
            # Get training data
            key = f"training:{concept_id}"
            training_data = redis_client.get(key)
            if not training_data:
                raise HTTPException(status_code=404, detail="Concept not found")
            
            training_data = json.loads(training_data)
            
            # Generate query embedding
            query_embedding = embedding_model.encode(text)
            
            # Calculate similarity
            similarity = np.dot(query_embedding, training_data["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(training_data["embedding"])
            )
            
            return {
                "concept_id": concept_id,
                "similarity": float(similarity),
                "confidence": float(similarity),
                "version": int(redis_client.get(f"version:{concept_id}") or 0)
            }
        except Exception as e:
            logger.error(f"Error evaluating concept: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

async def train_on_concepts(concept_ids: List[str], batch_size: int):
    """Train on a batch of concepts"""
    try:
        with TRAINING_LATENCY.time():
            # Fetch concepts with authentication
            headers = {}
            if API_KEY:
                headers["X-API-Key"] = API_KEY
                
            async with httpx.AsyncClient() as client:
                concepts = []
                for cid in concept_ids:
                    resp = await client.get(f"{DICT_URL}/concepts/{cid}", headers=headers)
                    if resp.status_code == 200:
                        concepts.append(resp.json())
                    elif resp.status_code == 401:
                        logger.error(f"Authentication failed when fetching concept {cid}. Please check CONCEPT_DICT_API_KEY.")
                        continue
            
            if not concepts:
                logger.warning("No concepts found for training")
                return
            
            # Prepare training data
            texts = [
                f"{c['term']} {c.get('definition', '')}"
                for c in concepts
            ]
            
            # Generate embeddings in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                
                # Update concepts with new embeddings
                for j, embedding in enumerate(embeddings):
                    concept = concepts[i + j]
                    async with httpx.AsyncClient() as client:
                        await client.put(
                            f"{DICT_URL}/concepts/{concept['term']}",
                            json={
                                "term": concept["term"],
                                "definition": concept.get("definition", ""),
                                "embedding": embedding.tolist(),
                                "metadata": concept.get("metadata", {}),
                                "last_updated": datetime.utcnow().isoformat()
                            },
                            headers=headers
                        )
                
                TRAINING_BATCHES.inc()
                TRAINING_SAMPLES.inc(len(batch))
            
            # Increment model version
            MODEL_VERSION.inc()
            
            logger.info(f"Trained on {len(concepts)} concepts")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

@app.post("/train")
async def train(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train on specified concepts"""
    try:
        background_tasks.add_task(
            train_on_concepts,
            request.concept_ids,
            request.batch_size or BATCH_SIZE
        )
        return {
            "status": "success",
            "message": f"Training started on {len(request.concept_ids)} concepts"
        }
    except Exception as e:
        logger.error(f"Error scheduling training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_on_concepts(concepts):
    """Train the model on a list of concepts."""
    try:
        valid_concepts = []
        for concept in concepts:
            try:
                # Extract and validate embedding
                vector = concept.get('embedding', [])
                if not vector or len(vector) != EMBEDDING_SIZE:
                    logger.warning(f"Invalid embedding shape for concept {concept['term']}: expected {EMBEDDING_SIZE}, got {len(vector)}")
                    continue
                    
                # Extract and validate label
                try:
                    label = int(concept.get('metadata', {}).get('label', -1))
                    if not 0 <= label < NUM_LABELS:
                        logger.warning(f"Invalid label for concept {concept['term']}: {label}")
                        continue
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid label type for concept {concept['term']}: {e}")
                    continue
                    
                # Extract and validate truth/growth with better validation
                try:
                    metadata = concept.get('metadata', {})
                    truth = float(metadata.get('truth', 0.5))
                    growth = float(metadata.get('growth', 0.5))
                    
                    # Enhanced validation for truth and growth values
                    if not (0.0 <= truth <= 1.0):
                        logger.warning(f"Invalid truth value for concept {concept['term']}: {truth}")
                        truth = max(0.0, min(1.0, truth))  # Clamp to valid range
                        
                    if not (0.0 <= growth <= 1.0):
                        logger.warning(f"Invalid growth value for concept {concept['term']}: {growth}")
                        growth = max(0.0, min(1.0, growth))  # Clamp to valid range
                        
                    # Calculate growth pressure based on drift and maturity
                    if model and concept['term'] in model.concept_metrics:
                        metrics = model.concept_metrics[concept['term']]
                        drift = metrics.get('last_drift', 0.0)
                        maturity = metrics.get('maturity_score', 0.0)
                        growth_pressure = (
                            0.4 * drift +  # Drift component
                            0.3 * (1 - maturity) +  # Maturity component
                            0.3 * growth  # Growth component
                        )
                        # Update growth value based on pressure
                        growth = max(growth, growth_pressure)
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid truth/growth type for concept {concept['term']}: {e}")
                    continue
                    
                valid_concepts.append((vector, label, truth, growth, concept['term']))
                
            except Exception as e:
                logger.error(f"Error processing concept {concept.get('term', 'unknown')}: {e}")
                logger.error(traceback.format_exc())
                continue
                
        if not valid_concepts:
            logger.error("No valid concepts found for training")
            return None
            
        # Process valid concepts with enhanced metrics tracking
        for vector, label, truth, growth, term in valid_concepts:
            try:
                # Add training sample with growth tracking
                add_training_sample(vector, label, truth, growth, term)
                
                # Update concept metrics if model exists
                if model and term in model.concept_metrics:
                    metrics = model.concept_metrics[term]
                    metrics['growth'] = growth
                    metrics['last_update'] = datetime.now().isoformat()
                    
                    # Update growth threshold based on recent performance
                    if len(metrics.get('drift_history', [])) >= 5:
                        recent_drifts = [entry['drift'] for entry in metrics['drift_history'][-5:]]
                        drift_std = np.std(recent_drifts)
                        metrics['growth_threshold'] = max(0.001, min(0.1, drift_std * 2))
                        
                logger.debug(f"Added training sample for concept: {term} with growth={growth:.4f}")
                
            except Exception as e:
                logger.error(f"Error adding training sample for concept {term}: {e}")
                logger.error(traceback.format_exc())
                continue
                
        # Retrain model with enhanced monitoring
        try:
            model = retrain_model()
            if model is None:
                logger.error("Model retraining failed")
                return None
                
            # Evaluate model performance with growth metrics
            metrics = await evaluate_model_performance()
            if metrics:
                logger.info(f"Training completed with metrics: {metrics}")
                # Update global training metrics
                global TRAINING_METRICS
                TRAINING_METRICS.update(metrics)
                
                # Save model state after successful training
                save_model_state()
                
            return model
            
        except Exception as e:
            logger.error("Error during model retraining")
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error("Error in train_model_on_concepts")
        logger.error(traceback.format_exc())
        return None

async def evaluate_model_performance():
    """Evaluate model performance and update metrics"""
    try:
        # Get current model state
        if model is None:
            logger.error("No model available for evaluation")
            return None
            
        # Get training data from Redis
        redis_client = await get_redis_client()
        training_data = await redis_client.get("training_data")
        
        if not training_data:
            logger.warning("No training data found in Redis")
            return None
            
        try:
            data = json.loads(training_data)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding training data: {e}")
            return None
            
        if not data:
            logger.warning("Empty training data")
            return None
            
        # Calculate metrics with enhanced growth tracking
        valid_concepts = []
        growth_metrics = {
            'total_growth': 0.0,
            'growth_count': 0,
            'drift_scores': [],
            'maturity_scores': [],
            'growth_pressures': []
        }
        
        for concept in data:
            try:
                # Extract and validate vector
                vector = concept.get('vector', [])
                if not vector or len(vector) != EMBEDDING_SIZE:
                    continue
                    
                # Extract and validate label
                try:
                    label = int(concept.get('label', -1))
                    if not 0 <= label < NUM_LABELS:
                        continue
                except (ValueError, TypeError):
                    continue
                    
                # Extract and validate truth and growth
                try:
                    truth = float(concept.get('truth', 0.5))
                    growth = float(concept.get('growth', 0.5))
                    if not (0.0 <= truth <= 1.0 and 0.0 <= growth <= 1.0):
                        continue
                except (ValueError, TypeError):
                    continue
                    
                # Get concept metrics if available
                concept_id = concept.get('term', 'unknown')
                if model and concept_id in model.concept_metrics:
                    metrics = model.concept_metrics[concept_id]
                    
                    # Track drift and maturity
                    drift = metrics.get('last_drift', 0.0)
                    maturity = metrics.get('maturity_score', 0.0)
                    growth_threshold = metrics.get('growth_threshold', 0.001)
                    
                    # Calculate growth pressure
                    growth_pressure = (
                        0.4 * drift +  # Drift component
                        0.3 * (1 - maturity) +  # Maturity component
                        0.3 * growth  # Growth component
                    )
                    
                    # Update growth metrics
                    growth_metrics['drift_scores'].append(drift)
                    growth_metrics['maturity_scores'].append(maturity)
                    growth_metrics['growth_pressures'].append(growth_pressure)
                    
                    # Track total growth
                    if growth_pressure > growth_threshold:
                        growth_metrics['total_growth'] += growth_pressure
                        growth_metrics['growth_count'] += 1
                        
                    # Update concept metrics
                    metrics.update({
                        'growth_pressure': growth_pressure,
                        'last_evaluation': datetime.now().isoformat()
                    })
                    
                valid_concepts.append((vector, label, truth, growth, concept_id))
                
            except Exception as e:
                logger.error(f"Error processing concept {concept.get('term', 'unknown')}: {e}")
                continue
                
        if not valid_concepts:
            logger.error("No valid concepts found for evaluation")
            return None
            
        # Calculate prototypes for semantic similarity
        vectors = np.array([v[0] for v in valid_concepts])
        labels = np.array([v[1] for v in valid_concepts])
        prototypes = compute_prototypes(vectors, labels)
        
        # Calculate semantic scores and confidence
        semantic_scores = []
        confidence_scores = []
        
        model.eval()
        with torch.no_grad():
            for vector, label, _, _, term in valid_concepts:
                # Get model predictions
                output = model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0))
                pred_label = output.argmax().item()
                confidence = torch.softmax(output, dim=1).max().item()
                
                # Calculate semantic similarity
                sem_score = cosine_similarity(vector, prototypes[pred_label])
                semantic_scores.append(sem_score)
                confidence_scores.append(confidence)
                
                logger.debug(f"Evaluation for {term}: confidence={confidence:.4f}, semantic={sem_score:.4f}")
                
        # Calculate aggregate metrics
        avg_confidence = np.mean(confidence_scores)
        avg_semantic = np.mean(semantic_scores)
        avg_drift = np.mean(growth_metrics['drift_scores']) if growth_metrics['drift_scores'] else 0.0
        avg_maturity = np.mean(growth_metrics['maturity_scores']) if growth_metrics['maturity_scores'] else 0.0
        avg_growth_pressure = np.mean(growth_metrics['growth_pressures']) if growth_metrics['growth_pressures'] else 0.0
        
        # Calculate growth rate
        growth_rate = growth_metrics['total_growth'] / max(1, growth_metrics['growth_count'])
        
        # Update metrics
        metrics = {
            'confidence': avg_confidence,
            'drift': avg_drift,
            'semantic': avg_semantic,
            'growth': growth_rate,
            'maturity': avg_maturity,
            'growth_pressure': avg_growth_pressure,
            'growth_count': growth_metrics['growth_count'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Update Prometheus metrics
        TRAINING_METRICS.update(metrics)
        MODEL_VERSION.inc()
        
        logger.info(f"Model evaluation completed: {json.dumps(metrics, indent=2)}")
        return metrics
        
    except Exception as e:
        logger.error("Error in evaluate_model_performance")
        logger.error(traceback.format_exc())
        return None

async def ingest_training_json():
    """Read and process training concepts from the JSON file, then clear it."""
    try:
        if not os.path.exists(TRAINING_JSON_PATH):
            logger.warning(f"Training JSON file not found at {TRAINING_JSON_PATH}")
            return []

        concepts = []
        try:
            with open(TRAINING_JSON_PATH, 'r') as f:
                for line in f:
                    try:
                        if line.strip():  # Skip empty lines
                            concept = json.loads(line)
                            concepts.append(concept)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse line: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error reading training file: {e}")
            return []

        # Clear the file after successful reading
        try:
            with open(TRAINING_JSON_PATH, 'w') as f:
                f.write('')  # Clear the file
            logger.info(f"Successfully ingested {len(concepts)} concepts and cleared training file")
        except Exception as e:
            logger.error(f"Failed to clear training file: {e}")
            # Continue anyway since we have the concepts in memory

        return concepts
    except Exception as e:
        logger.error(f"Error in ingest_training_json: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

async def fetch_concepts_from_dict():
    """Fetch concepts from concept-dictionary service"""
    try:
        headers = {}
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        else:
            logger.warning("No CONCEPT_DICT_API_KEY provided - authentication may fail")
            
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{DICT_URL}/concepts", headers=headers)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                logger.error("Authentication failed when fetching concepts. Please check CONCEPT_DICT_API_KEY.")
                return []
            else:
                logger.error(f"Failed to fetch concepts: {resp.status_code}")
                return []
    except Exception as e:
        logger.error(f"Error fetching concepts: {e}")
        return []

async def periodic_training():
    """Periodic training task that integrates with train_demo and ingests from JSON, or falls back to concept-dictionary."""
    while True:
        try:
            logger.info("[periodic_training] Starting training cycle...")
            
            # Try to load model state first
            if not load_model_state():
                logger.warning("[periodic_training] No saved model state found, initializing new model")
                global model
                model = MultiHeadNet(NUM_FEATURES, len(LABELS))
            
            # Try to get concepts from JSON first
            concepts = await ingest_training_json()
            
            if not concepts:
                logger.info("[periodic_training] No concepts from JSON, trying concept dictionary...")
                # Fall back to concept dictionary
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{DICT_URL}/concepts")
                    if response.status_code == 200:
                        concepts = response.json()
                    else:
                        logger.error(f"[periodic_training] Failed to get concepts from dictionary: {response.status_code}")
                        concepts = []
            
            if concepts:
                logger.info(f"[periodic_training] Training on {len(concepts)} concepts...")
                
                # Track growth metrics before training
                pre_training_metrics = await evaluate_model_performance()
                if pre_training_metrics:
                    logger.info(f"[periodic_training] Pre-training metrics: {json.dumps(pre_training_metrics, indent=2)}")
                
                # Train model
                model = await train_model_on_concepts(concepts)
                if model is None:
                    logger.error("[periodic_training] Model training failed")
                    await asyncio.sleep(60)
                    continue
                
                # Evaluate model performance
                post_training_metrics = await evaluate_model_performance()
                if post_training_metrics:
                    logger.info(f"[periodic_training] Post-training metrics: {json.dumps(post_training_metrics, indent=2)}")
                    
                    # Calculate growth changes
                    if pre_training_metrics:
                        growth_change = post_training_metrics['growth'] - pre_training_metrics['growth']
                        maturity_change = post_training_metrics['maturity'] - pre_training_metrics['maturity']
                        drift_change = post_training_metrics['drift'] - pre_training_metrics['drift']
                        
                        logger.info(f"[periodic_training] Growth changes:")
                        logger.info(f"Growth: {growth_change:+.4f}")
                        logger.info(f"Maturity: {maturity_change:+.4f}")
                        logger.info(f"Drift: {drift_change:+.4f}")
                        
                        # Check if growth is needed
                        if growth_change > 0.1 or drift_change > 0.1:  # Significant changes
                            logger.info("[periodic_training] Significant changes detected, checking for growth needs...")
                            
                            # Check each concept for growth needs
                            for concept in concepts:
                                concept_id = concept.get('term', 'unknown')
                                if model and concept_id in model.concept_metrics:
                                    should_grow, layer_idx = model.should_grow(concept_id)
                                    if should_grow:
                                        logger.info(f"[periodic_training] Growth needed for concept {concept_id} at layer {layer_idx}")
                                        
                                        # Calculate new size based on growth pressure
                                        metrics = model.concept_metrics[concept_id]
                                        growth_pressure = metrics.get('growth_pressure', 0.0)
                                        current_size = model.layers[layer_idx].current_capacity
                                        new_size = int(current_size * (1.0 + growth_pressure))
                                        
                                        # Grow the layer
                                        try:
                                            model.grow_layer(layer_idx, new_size)
                                            logger.info(f"[periodic_training] Grew layer {layer_idx} from {current_size} to {new_size}")
                                            
                                            # Record growth event
                                            model.record_growth(
                                                concept_id,
                                                layer_idx=layer_idx,
                                                old_size=current_size,
                                                new_size=new_size,
                                                reason="periodic_training_growth"
                                            )
                                        except Exception as e:
                                            logger.error(f"[periodic_training] Error growing layer: {e}")
                
                # Save model state after training
                save_model_state()
                
                logger.info(f"[periodic_training] Training cycle completed at {datetime.now()}")
            else:
                logger.info("[periodic_training] No concepts to train on.")
                
            logger.info("[periodic_training] Sleeping for 60 seconds before next cycle...")
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"[periodic_training] Error in training cycle: {e}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(60)  # Sleep before retrying

# (Optional) Add a new endpoint to expose the reference question score (if you want a REST endpoint)
@app.get("/monitoring/reference", tags=["monitoring"], response_class=PlainTextResponse)
async def get_reference_score():
    score = REFERENCE_QUESTION_SCORE._value.get()
    return f"Reference question score (dummy): {score:.2f} ({score * 100:.1f}%)"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8905, log_level="info") 