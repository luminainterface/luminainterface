from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import torch
import numpy as np
from redis import Redis
from qdrant_client import QdrantClient

from model import GrowableConceptNet

logger = logging.getLogger(__name__)

# Data models
class VectorBatch(BaseModel):
    """A batch of vectors for training"""
    concept_id: str
    vectors: List[List[float]]
    labels: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class TrainingMetrics(BaseModel):
    """Training metrics for a concept"""
    concept_id: str
    loss: float
    accuracy: float
    drift: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class GrowthRequest(BaseModel):
    """Request to grow a concept model"""
    concept_id: str
    target_size: Optional[int] = None
    growth_factor: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_status: Dict[str, Any]
    redis_status: bool
    qdrant_status: bool
    last_training: Optional[datetime] = None

class OutputRequest(BaseModel):
    """Request for model output/inference"""
    concept_id: str
    vectors: List[List[float]]
    metadata: Optional[Dict[str, Any]] = None

class OutputResponse(BaseModel):
    """Response with model predictions"""
    concept_id: str
    predictions: List[List[float]]
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

# Router setup
router = APIRouter()

# Dependencies
def get_model() -> GrowableConceptNet:
    """Get the model instance"""
    from main import model
    return model

def get_redis() -> Redis:
    """Get Redis client"""
    from main import redis_client
    return redis_client

def get_qdrant() -> QdrantClient:
    """Get Qdrant client"""
    from main import qdrant_client
    return qdrant_client

# Health endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check(
    model: GrowableConceptNet = Depends(get_model),
    redis: Redis = Depends(get_redis),
    qdrant: QdrantClient = Depends(get_qdrant)
) -> HealthResponse:
    """Basic health check endpoint"""
    try:
        # Check model status
        model_stats = model.get_network_stats()
        
        # Check Redis connection
        try:
            await redis.ping()
            redis_status = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            redis_status = False
        
        # Check Qdrant connection
        try:
            collections = qdrant.get_collections()
            qdrant_status = True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            qdrant_status = False
        
        # Determine overall status
        status = "healthy" if all([redis_status, qdrant_status]) else "degraded"
        
        return HealthResponse(
            status=status,
            version="1.0.0",
            model_status=model_stats,
            redis_status=redis_status,
            qdrant_status=qdrant_status,
            last_training=datetime.fromisoformat(model_stats['last_training']) if model_stats['last_training'] else None
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.debug("Health check failed (detail: {})", e, exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))

@router.get("/model/health", response_model=Dict[str, Any])
async def model_health(
    model: GrowableConceptNet = Depends(get_model)
) -> Dict[str, Any]:
    """Detailed model health check"""
    try:
        stats = model.get_network_stats()
        return {
            "status": "healthy",
            "model_stats": stats,
            "concepts": {
                concept_id: model.get_concept_metrics(concept_id)
                for concept_id in model.concept_metrics.keys()
            }
        }
    except Exception as e:
        logger.error(f"Model health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Training endpoints
@router.post("/model/train")
async def train_concept(
    batch: VectorBatch,
    background_tasks: BackgroundTasks,
    model: GrowableConceptNet = Depends(get_model),
    redis: Redis = Depends(get_redis)
) -> Dict[str, Any]:
    """Train on a batch of vectors"""
    try:
        # Convert inputs to tensors
        vectors = torch.tensor(batch.vectors, dtype=torch.float32)
        if batch.labels is not None:
            labels = torch.tensor(batch.labels, dtype=torch.long)
        else:
            # Generate pseudo-labels if none provided
            labels = torch.zeros(len(batch.vectors), dtype=torch.long)
        
        # Store batch in Redis for drift calculation
        batch_key = f"concept:{batch.concept_id}:batch:{batch.timestamp.isoformat()}"
        redis.set(batch_key, vectors.numpy().tobytes())
        
        # Schedule training in background
        background_tasks.add_task(
            _train_batch,
            model=model,
            concept_id=batch.concept_id,
            vectors=vectors,
            labels=labels,
            metadata=batch.metadata
        )
        
        return {
            "status": "training_started",
            "concept_id": batch.concept_id,
            "batch_size": len(batch.vectors),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/grow")
async def grow_model(
    request: GrowthRequest,
    model: GrowableConceptNet = Depends(get_model)
) -> Dict[str, Any]:
    """Grow the model for a concept"""
    try:
        # Check if model should grow
        should_grow, layer_idx = model.should_grow(request.concept_id)
        
        if not should_grow:
            return {
                "status": "no_growth_needed",
                "concept_id": request.concept_id,
                "reason": "Model is stable"
            }
        
        # Calculate new size
        if request.target_size is not None:
            new_size = request.target_size
        elif request.growth_factor is not None:
            current_size = model.layers[layer_idx].current_capacity
            new_size = int(current_size * request.growth_factor)
        else:
            # Default growth: 50% increase
            current_size = model.layers[layer_idx].current_capacity
            new_size = int(current_size * 1.5)
        
        # Grow the layer
        model.grow_layer(layer_idx, new_size)
        
        return {
            "status": "grown",
            "concept_id": request.concept_id,
            "layer_idx": layer_idx,
            "old_size": model.layers[layer_idx].current_capacity,
            "new_size": new_size,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Growth failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/status/{concept_id}")
async def get_concept_status(
    concept_id: str,
    model: GrowableConceptNet = Depends(get_model)
) -> Dict[str, Any]:
    """Get status and metrics for a concept"""
    try:
        metrics = model.get_concept_metrics(concept_id)
        if not metrics:
            raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")
        
        return {
            "concept_id": concept_id,
            "metrics": metrics,
            "model_stats": model.get_network_stats()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get concept status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/output", response_model=OutputResponse)
async def model_output(
    request: OutputRequest,
    model: GrowableConceptNet = Depends(get_model)
) -> OutputResponse:
    """Get model predictions for a batch of vectors"""
    try:
        vectors = torch.tensor(request.vectors, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(vectors)
            predictions = outputs.exp().tolist()  # Convert log-probs to probabilities
        return OutputResponse(
            concept_id=request.concept_id,
            predictions=predictions,
            metadata=request.metadata
        )
    except Exception as e:
        logger.error(f"Model output failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def _train_batch(
    model: GrowableConceptNet,
    concept_id: str,
    vectors: torch.Tensor,
    labels: torch.Tensor,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Train on a batch of data"""
    try:
        # Forward pass
        outputs = model(vectors)
        
        # Calculate loss
        criterion = torch.nn.NLLLoss()
        loss = criterion(outputs, labels)
        
        # Backward pass
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean().item()
        
        # Calculate drift (simplified)
        drift = 0.0  # TODO: Implement proper drift calculation
        
        # Record training metrics
        model.record_training(
            concept_id=concept_id,
            loss=loss.item(),
            accuracy=accuracy,
            drift=drift
        )
        
        logger.info(f"Training completed for concept {concept_id}: "
                   f"loss={loss.item():.4f}, accuracy={accuracy:.4f}, drift={drift:.4f}")
    except Exception as e:
        logger.error(f"Background training failed: {str(e)}")
        raise 