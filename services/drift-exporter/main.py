from fastapi import FastAPI
import redis.asyncio as redis
import os
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger

app = FastAPI(title="Drift Exporter Service")

# Initialize Redis client
redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379"),
    encoding="utf-8",
    decode_responses=True
)

# Prometheus metrics
DRIFT_SCORE = Gauge(
    'concept_drift_score',
    'Drift score for each concept',
    ['concept_id']
)

CONCEPT_UPDATES = Counter(
    'concept_updates_total',
    'Total number of concept updates',
    ['concept_id']
)

UPDATE_LATENCY = Histogram(
    'concept_update_latency_seconds',
    'Time taken to update concept',
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "drift-exporter",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get current drift metrics"""
    try:
        # Get all concept keys from Redis
        concept_keys = await redis_client.keys("concept:drift:*")
        metrics = {}
        
        for key in concept_keys:
            concept_id = key.split(":")[-1]
            drift_score = float(await redis_client.get(key) or 0.0)
            DRIFT_SCORE.labels(concept_id=concept_id).set(drift_score)
            metrics[concept_id] = drift_score
            
        return {
            "status": "success",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 