from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
import uvicorn
from datetime import datetime
import redis.asyncio as redis
import json
import os
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger
from qdrant_client import QdrantClient
from batch_processor import BatchProcessor, TrainerVectorBatch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../concept-trainer-growable')))
from model import GrowableConceptNet

app = FastAPI(title="Concept Trainer Service")

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://:02211998@redis:6379")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = "lumina_embeddings"

# Initialize Redis client
redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

# Initialize Qdrant client
qdrant_client = QdrantClient(QDRANT_URL)

# Training configuration
class TrainingConfig:
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 768):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

# Set seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and batch processor
config = TrainingConfig()
model = GrowableConceptNet(
    input_size=config.input_dim,
    hidden_sizes=[config.hidden_dim, config.hidden_dim],
    output_size=config.output_dim,
    activation='relu'
)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
batch_processor = BatchProcessor(model, device, REDIS_URL, QDRANT_URL)

class TrainingRequest(BaseModel):
    concept_id: str
    nn_response: List[float]
    mistral_response: List[float]
    confidence_delta: float
    feedback_score: float

@app.post("/train")
async def train(request: TrainingRequest):
    """Train the model with new feedback data"""
    try:
        # Convert inputs to tensors
        nn_tensor = torch.tensor(request.nn_response, dtype=torch.float32, device=device)
        mistral_tensor = torch.tensor(request.mistral_response, dtype=torch.float32, device=device)
        
        # Forward pass
        model.train()
        output = model(nn_tensor)
        
        # Calculate loss
        criterion = nn.MSELoss()
        loss = criterion(output, mistral_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store updated embedding
        await redis_client.set(
            f"concept:embedding:{request.concept_id}",
            json.dumps(output.detach().cpu().numpy().tolist())
        )
        
        return {
            "status": "success",
            "loss": loss.item(),
            "concept_id": request.concept_id
        }
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/batch")
async def train_batch(batch: TrainerVectorBatch):
    """Process a batch of vectors for training"""
    try:
        # Process the batch using the batch processor
        result = await batch_processor.process_batch({
            "vectors": batch.vectors,
            "batch_id": batch.batch_id,
            "metadata": batch.metadata
        })
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return result
        
    except Exception as e:
        logger.error(f"Error in batch training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        await redis_client.ping()
        
        # Test Qdrant connection
        qdrant_client.get_collections()
        
        return {
            "status": "healthy",
            "service": "concept-trainer",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "model": {
                "type": "growable-concept-net",
                "device": str(device),
                "input_dim": config.input_dim,
                "hidden_dim": config.hidden_dim,
                "output_dim": config.output_dim
            },
            "batch_processing": {
                "chunk_size": batch_processor.CHUNK_SIZE,
                "max_concurrent_chunks": batch_processor.MAX_CONCURRENT_CHUNKS
            },
            "connections": {
                "redis": "connected",
                "qdrant": "connected"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8813) 