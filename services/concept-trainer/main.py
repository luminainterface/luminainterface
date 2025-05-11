from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import Dict, Optional, List
import uvicorn
from datetime import datetime
import redis.asyncio as redis
import json
import os
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger
from qdrant_client import QdrantClient

app = FastAPI(title="Concept Trainer Service")

# Initialize Redis client
redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    encoding="utf-8",
    decode_responses=True
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = "lumina_embeddings"

# Utility to fetch system vectors from Qdrant
def fetch_system_vectors(top_k=20, query_text=None, embedding_model=None):
    client = QdrantClient(QDRANT_URL)
    filters = {"must": [{"key": "type", "match": {"value": "system"}}]}
    if query_text and embedding_model:
        query_vec = embedding_model.encode([query_text])[0]
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=query_vec,
            limit=top_k,
            query_filter=filters
        )
    else:
        hits = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=filters,
            limit=top_k
        )[0]
    return [h.payload for h in hits]

class TrainingConfig:
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

class ConceptEmbeddingModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc3 = nn.Linear(config.hidden_dim, config.output_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Set seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
config = TrainingConfig()
model = ConceptEmbeddingModel(config)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        await redis_client.ping()
        
        # Test Qdrant connection
        client = QdrantClient(QDRANT_URL)
        client.get_collections()
        
        return {
            "status": "healthy",
            "service": "concept-trainer",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "model": {
                "type": "concept-embedding",
                "device": str(device)
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