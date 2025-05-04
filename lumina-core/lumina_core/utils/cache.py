import hashlib
import json
from functools import lru_cache
from typing import List, Optional, Union
import redis.asyncio as redis
from loguru import logger
from lumina_core.metrics import CACHE_OPERATIONS

class EmbeddingCache:
    def __init__(self, redis_url: Optional[str] = None, max_size: int = 1000):
        """Initialize embedding cache with optional Redis backend."""
        self._lru_cache = lru_cache(maxsize=max_size)(self._compute_embedding)
        self._redis = None
        if redis_url:
            self._redis = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            logger.info(f"Using Redis cache at {redis_url}")
        else:
            logger.info("Using in-memory LRU cache")

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding hash for cache key."""
        return hashlib.sha256(text.encode()).hexdigest()

    async def get_embedding(
        self, 
        text: str, 
        compute_fn: callable
    ) -> List[float]:
        """Get embedding from cache or compute if missing."""
        cache_key = self._compute_embedding(text)
        
        # Try Redis first if available
        if self._redis:
            try:
                cached = await self._redis.get(cache_key)
                if cached:
                    logger.debug(f"Cache hit (Redis): {text[:50]}...")
                    CACHE_OPERATIONS.labels(
                        operation="get",
                        layer="redis",
                        result="hit"
                    ).inc()
                    return json.loads(cached)
                CACHE_OPERATIONS.labels(
                    operation="get",
                    layer="redis",
                    result="miss"
                ).inc()
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
                CACHE_OPERATIONS.labels(
                    operation="get",
                    layer="redis",
                    result="error"
                ).inc()
        
        # Try LRU cache
        try:
            cached = self._lru_cache(text)
            if cached:
                logger.debug(f"Cache hit (LRU): {text[:50]}...")
                CACHE_OPERATIONS.labels(
                    operation="get",
                    layer="lru",
                    result="hit"
                ).inc()
                return cached
            CACHE_OPERATIONS.labels(
                operation="get",
                layer="lru",
                result="miss"
            ).inc()
        except Exception as e:
            logger.warning(f"LRU cache error: {e}")
        
        # Compute and cache
        embedding = compute_fn(text)
        
        # Store in Redis if available
        if self._redis:
            try:
                await self._redis.set(
                    cache_key,
                    json.dumps(embedding),
                    ex=86400  # 24 hour expiry
                )
                CACHE_OPERATIONS.labels(
                    operation="set",
                    layer="redis",
                    result="success"
                ).inc()
            except Exception as e:
                logger.warning(f"Redis cache store error: {e}")
                CACHE_OPERATIONS.labels(
                    operation="set",
                    layer="redis",
                    result="error"
                ).inc()
        
        return embedding

    async def clear(self):
        """Clear all caches."""
        self._lru_cache.cache_clear()
        if self._redis:
            await self._redis.flushdb() 