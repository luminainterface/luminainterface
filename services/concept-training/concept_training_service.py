#!/usr/bin/env python3
"""
Concept Training Worker Service
Trains and adapts concept understanding
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import List, Dict, Any
import time
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Concept Training Worker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    concept: str
    examples: List[str]
    method: str = "incremental"
    
# In-memory training storage
training_sessions = {}
concept_models = {}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "concept_training", 
        "version": "1.0.0",
        "active_sessions": len(training_sessions),
        "trained_concepts": len(concept_models)
    }

@app.post("/train")
async def train_concept(request: TrainRequest):
    """Train a concept with examples"""
    try:
        session_id = str(uuid.uuid4())
        
        # Simulate training process
        training_sessions[session_id] = {
            "concept": request.concept,
            "examples": request.examples,
            "method": request.method,
            "status": "training",
            "started_at": time.time(),
            "progress": 0
        }
        
        # Simple concept analysis
        example_count = len(request.examples)
        avg_length = sum(len(ex) for ex in request.examples) / example_count if example_count > 0 else 0
        
        # Store trained concept
        concept_models[request.concept] = {
            "examples": request.examples,
            "example_count": example_count,
            "avg_example_length": avg_length,
            "trained_at": time.time(),
            "method": request.method
        }
        
        training_sessions[session_id]["status"] = "completed"
        training_sessions[session_id]["progress"] = 100
        
        return {
            "session_id": session_id,
            "concept": request.concept,
            "status": "completed",
            "training_time": time.time() - training_sessions[session_id]["started_at"],
            "examples_processed": example_count,
            "method": request.method
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/{session_id}")
async def get_training_status(session_id: str):
    """Get training session status"""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return training_sessions[session_id]

@app.get("/concepts")
async def list_concepts():
    """List all trained concepts"""
    return {"concepts": list(concept_models.keys()), "count": len(concept_models)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8851)
