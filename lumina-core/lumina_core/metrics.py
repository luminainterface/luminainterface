from prometheus_client import Histogram, Counter, Gauge
from prometheus_client.openmetrics.exposition import generate_latest
from fastapi import Response
from typing import Dict, Any
import asyncio
from loguru import logger
import time

# Request latency histogram
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    ['endpoint', 'method'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)

# Pruning metrics
PRUNE_DURATION = Histogram(
    'prune_duration_seconds',
    'Time spent pruning vectors',
    buckets=(1, 5, 10, 30, 60, 120)
)

PRUNED_VECTORS = Counter(
    'pruned_vectors_total',
    'Total number of vectors pruned',
    ['reason']  # reason = age | similarity
)

# Cache metrics
CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Total number of cache operations',
    ['operation', 'layer', 'result']  # operation: 'get'/'set', layer: 'redis'/'lru', result: 'hit'/'miss'
)

CACHE_SIZE = Gauge(
    'cache_size',
    'Number of items in cache',
    ['layer']  # layer = redis | lru
)

# Rate limit metrics
RATE_LIMIT_HITS = Counter(
    'rate_limit_hits_total',
    'Total requests hitting the rate limiter',
    ['endpoint', 'api_key', 'result']  # result = hit | block
)

RATE_LIMIT_TOKENS = Gauge(
    'rate_limit_tokens',
    'Current rate limit tokens available',
    ['endpoint', 'api_key']
)

async def update_cache_metrics(cache):
    """Background task to update cache size metrics."""
    while True:
        try:
            # Update LRU cache size
            CACHE_SIZE.labels(layer="lru").set(len(cache._lru_cache.cache_info()))
            
            # Update Redis cache size if available
            if cache._redis:
                try:
                    redis_size = await cache._redis.dbsize()
                    CACHE_SIZE.labels(layer="redis").set(redis_size)
                except Exception as e:
                    logger.error(f"Error getting Redis size: {e}")
                    CACHE_SIZE.labels(layer="redis").set(-1)  # Indicate error state
            
            await asyncio.sleep(60)  # Update every minute
            
        except Exception as e:
            logger.error(f"Error updating cache metrics: {e}")
            await asyncio.sleep(60)  # Still wait before retry

def get_metrics() -> Dict[str, Any]:
    """Get current metrics in Prometheus format."""
    return generate_latest().decode('utf-8')

def create_metrics_response() -> Response:
    """Create a FastAPI Response with Prometheus metrics."""
    return Response(
        content=get_metrics(),
        media_type="text/plain; version=0.0.4"
    ) 