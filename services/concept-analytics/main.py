from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import redis.asyncio as redis
import os
from datetime import datetime, timedelta
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
import numpy as np
from typing import List, Dict, Optional
import json
from fastapi.responses import Response

app = FastAPI(title="Concept Analytics Service")

# Initialize Redis client
redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    encoding="utf-8",
    decode_responses=True
)

# Initialize Prometheus metrics
registry = CollectorRegistry()

# Metrics for concept drift
concept_drift_velocity = Gauge(
    'concept_drift_velocity',
    'Rate of concept drift over time',
    ['concept_id'],
    registry=registry
)

concept_confidence_delta = Gauge(
    'concept_confidence_delta',
    'Change in concept confidence over time',
    ['concept_id'],
    registry=registry
)

concept_feedback_quality_avg = Gauge(
    'concept_feedback_quality_avg',
    'Average feedback quality by source type',
    ['concept_id', 'source_type'],
    registry=registry
)

concept_retrain_frequency = Counter(
    'concept_retrain_frequency',
    'Number of times a concept has been retrained',
    ['concept_id'],
    registry=registry
)

class ConceptAnalytics(BaseModel):
    concept_id: str
    drift_velocity: float
    confidence_delta: float
    feedback_quality: Dict[str, float]
    retrain_count: int
    last_updated: datetime

async def get_concept_history(concept_id: str, time_window: timedelta = timedelta(days=7)) -> List[Dict]:
    """Retrieve concept history from Redis."""
    history_key = f"concept_history:{concept_id}"
    history = redis_client.lrange(history_key, 0, -1)
    return [json.loads(entry) for entry in history]

def calculate_drift_velocity(history: List[Dict]) -> float:
    """Calculate the rate of concept drift over time."""
    if len(history) < 2:
        return 0.0
    
    drift_values = [entry.get('drift_score', 0.0) for entry in history]
    timestamps = [datetime.fromisoformat(entry.get('timestamp', '')) for entry in history]
    
    if len(drift_values) < 2:
        return 0.0
    
    # Calculate velocity using linear regression
    x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
    y = np.array(drift_values)
    slope, _ = np.polyfit(x, y, 1)
    
    return float(slope)

def calculate_confidence_delta(history: List[Dict]) -> float:
    """Calculate the change in concept confidence over time."""
    if len(history) < 2:
        return 0.0
    
    confidences = [entry.get('confidence', 0.0) for entry in history]
    return float(confidences[-1] - confidences[0])

def calculate_feedback_quality(history: List[Dict]) -> Dict[str, float]:
    """Calculate average feedback quality by source type."""
    feedback_by_source = {}
    
    for entry in history:
        feedback = entry.get('feedback', {})
        for source, quality in feedback.items():
            if source not in feedback_by_source:
                feedback_by_source[source] = []
            feedback_by_source[source].append(quality)
    
    return {
        source: float(np.mean(qualities))
        for source, qualities in feedback_by_source.items()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.get("/concepts/{concept_id}/analytics")
async def get_concept_analytics(concept_id: str, time_window: Optional[int] = 7):
    """Get analytics for a specific concept."""
    try:
        history = await get_concept_history(concept_id, timedelta(days=time_window))
        
        if not history:
            raise HTTPException(status_code=404, detail="Concept history not found")
        
        drift_velocity = calculate_drift_velocity(history)
        confidence_delta = calculate_confidence_delta(history)
        feedback_quality = calculate_feedback_quality(history)
        retrain_count = len([entry for entry in history if entry.get('retrained', False)])
        
        # Update Prometheus metrics
        concept_drift_velocity.labels(concept_id=concept_id).set(drift_velocity)
        concept_confidence_delta.labels(concept_id=concept_id).set(confidence_delta)
        
        for source, quality in feedback_quality.items():
            concept_feedback_quality_avg.labels(
                concept_id=concept_id,
                source_type=source
            ).set(quality)
        
        concept_retrain_frequency.labels(concept_id=concept_id).inc(retrain_count)
        
        return ConceptAnalytics(
            concept_id=concept_id,
            drift_velocity=drift_velocity,
            confidence_delta=confidence_delta,
            feedback_quality=feedback_quality,
            retrain_count=retrain_count,
            last_updated=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/top_drift")
async def get_top_drift_concepts(limit: int = 10):
    """Get concepts with highest drift velocity."""
    try:
        concepts = redis_client.keys("concept_history:*")
        drift_scores = []
        
        for concept_id in concepts:
            concept_id = concept_id.split(":")[1]
            history = await get_concept_history(concept_id)
            drift_velocity = calculate_drift_velocity(history)
            drift_scores.append((concept_id, drift_velocity))
        
        # Sort by drift velocity in descending order
        drift_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "concepts": [
                {
                    "concept_id": concept_id,
                    "drift_velocity": velocity
                }
                for concept_id, velocity in drift_scores[:limit]
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/top_growth")
async def get_top_growth_concepts(limit: int = 10):
    """Get concepts with highest confidence growth."""
    try:
        concepts = redis_client.keys("concept_history:*")
        growth_scores = []
        
        for concept_id in concepts:
            concept_id = concept_id.split(":")[1]
            history = await get_concept_history(concept_id)
            confidence_delta = calculate_confidence_delta(history)
            growth_scores.append((concept_id, confidence_delta))
        
        # Sort by confidence delta in descending order
        growth_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "concepts": [
                {
                    "concept_id": concept_id,
                    "confidence_growth": growth
                }
                for concept_id, growth in growth_scores[:limit]
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8905) 