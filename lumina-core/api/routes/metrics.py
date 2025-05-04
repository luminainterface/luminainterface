from fastapi import APIRouter
from typing import Dict
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import redis.asyncio as aioredis
import json

router = APIRouter()

@router.get("/metrics/summary")
async def get_metrics_summary():
    # Connect to Redis for real-time metrics
    redis_client = aioredis.from_url("redis://redis:6379")
    
    # Get basic metrics
    metrics = {
        "chat": {
            "total_requests": await redis_client.get("metrics:chat:total") or 0,
            "active_sessions": await redis_client.scard("metrics:sessions:active") or 0,
            "avg_response_time": await redis_client.get("metrics:chat:avg_response_time") or 0
        },
        "memory": {
            "total_vectors": await redis_client.get("metrics:memory:total_vectors") or 0,
            "recall_rate": await redis_client.get("metrics:memory:recall_rate") or 0
        },
        "llm": {
            "total_tokens": await redis_client.get("metrics:llm:total_tokens") or 0,
            "avg_generation_time": await redis_client.get("metrics:llm:avg_generation_time") or 0
        }
    }
    
    return metrics

@router.get("/metrics/raw")
async def get_metrics_raw():
    """Get raw Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST) 