from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import redis.asyncio as redis
import os
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram

app = FastAPI(title="Feedback Handler Service")

# Initialize Redis client
redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    encoding="utf-8",
    decode_responses=True
)

# Prometheus metrics
FEEDBACK_TOTAL = Counter(
    'feedback_total',
    'Total number of feedback events',
    ['source_type', 'quality_level']
)

FEEDBACK_QUALITY = Gauge(
    'feedback_quality',
    'Quality score of feedback',
    ['concept_id', 'source_type']
)

FEEDBACK_CONFIDENCE = Gauge(
    'feedback_confidence',
    'Confidence score of feedback',
    ['concept_id', 'source_type']
)

FEEDBACK_PROCESSING_TIME = Histogram(
    'feedback_processing_seconds',
    'Time spent processing feedback',
    ['source_type']
)

class FeedbackRequest(BaseModel):
    concept_id: str
    feedback_score: float = Field(..., ge=0.0, le=1.0)
    feedback_confidence: float = Field(..., ge=0.0, le=1.0)
    response_quality: float = Field(..., ge=0.0, le=1.0)
    source_type: str = Field(..., pattern="^(user|llm|automated|demo)$")
    user_id: str
    context: dict

@app.post("/feedback")
async def handle_feedback(request: FeedbackRequest):
    """Handle user feedback for concepts"""
    try:
        # Store feedback in Redis
        feedback_key = f"feedback:{request.concept_id}:{datetime.utcnow().isoformat()}"
        await redis_client.hset(feedback_key, mapping={
            "concept_id": request.concept_id,
            "feedback_score": str(request.feedback_score),
            "feedback_confidence": str(request.feedback_confidence),
            "response_quality": str(request.response_quality),
            "source_type": request.source_type,
            "user_id": request.user_id,
            "context": str(request.context),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update Prometheus metrics
        quality_level = "high" if request.feedback_confidence > 0.8 else "medium" if request.feedback_confidence > 0.5 else "low"
        FEEDBACK_TOTAL.labels(source_type=request.source_type, quality_level=quality_level).inc()
        FEEDBACK_QUALITY.labels(concept_id=request.concept_id, source_type=request.source_type).set(request.response_quality)
        FEEDBACK_CONFIDENCE.labels(concept_id=request.concept_id, source_type=request.source_type).set(request.feedback_confidence)
        
        # Publish feedback event
        await redis_client.publish(
            "feedback_events",
            feedback_key
        )
        
        return {
            "status": "success",
            "feedback_id": feedback_key
        }
        
    except Exception as e:
        logger.error(f"Error handling feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "feedback-handler",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 