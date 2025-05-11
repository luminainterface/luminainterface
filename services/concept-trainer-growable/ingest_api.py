from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import numpy as np
import sys
import os
import logging
import traceback

sys.path.append(os.path.dirname(__file__))
from train_demo import add_training_sample, EMBEDDING_SIZE, NUM_LABELS

# Set up logging
logger = logging.getLogger(__name__)

app = FastAPI()

class VectorIngestRequest(BaseModel):
    vector: List[float]  # Should be length 384 (base embedding only)
    label: int
    truth: float
    growth: float
    question: str

    @validator('vector')
    def validate_vector(cls, v):
        if len(v) != EMBEDDING_SIZE:
            raise ValueError(f"Vector must have length {EMBEDDING_SIZE} (base embedding only), got {len(v)}")
        return v

    @validator('label')
    def validate_label(cls, v):
        if not 0 <= v < NUM_LABELS:
            raise ValueError(f"Label must be between 0 and {NUM_LABELS-1}, got {v}")
        return v

    @validator('truth', 'growth')
    def validate_float_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value must be between 0.0 and 1.0, got {v}")
        return v

@app.post("/ingest")
async def ingest_vector(request: VectorIngestRequest):
    """Ingest a new training vector. The vector should be the base embedding only (length 384)."""
    try:
        add_training_sample(
            vector=request.vector,
            label=request.label,
            truth=request.truth,
            growth=request.growth,
            question=request.question
        )
        return {"status": "success", "message": f"Successfully ingested vector for concept: {request.question}"}
    except Exception as e:
        logger.error(f"Error ingesting vector: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8681) 