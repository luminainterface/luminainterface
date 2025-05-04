import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import redis.asyncio as redis
from loguru import logger

# Redis client
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

@asynccontextmanager
async def redis_rate_limit(key: str, max_tokens: int, window_seconds: int):
    """
    Rate limit using Redis token bucket.
    
    Args:
        key: Redis key for this rate limit
        max_tokens: Maximum tokens per window
        window_seconds: Window size in seconds
        
    Raises:
        Exception: If rate limit exceeded
    """
    # Get current tokens
    tokens = await redis_client.get(key)
    if tokens is None:
        # Initialize bucket
        await redis_client.set(key, max_tokens)
        await redis_client.expire(key, window_seconds)
        tokens = max_tokens
    else:
        tokens = int(tokens)
    
    if tokens <= 0:
        logger.warning(f"Rate limit exceeded for {key}")
        raise Exception(f"Rate limit exceeded for {key}")
    
    try:
        # Consume token
        await redis_client.decr(key)
        yield
    finally:
        # Refill bucket
        await redis_client.incr(key) 