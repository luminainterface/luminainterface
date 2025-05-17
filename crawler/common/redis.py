"""Redis client module for the crawler service."""
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import redis.asyncio as redis
from .config import REDIS_URL, CACHE_TTL
from .logging import get_logger

logger = get_logger(__name__)

class RedisClient:
    """Redis client with connection pooling and stream operations."""
    
    def __init__(self, url: str = REDIS_URL):
        """Initialize Redis client.
        
        Args:
            url: Redis connection URL.
        """
        self.url = url
        self._client = None
        
    async def connect(self) -> None:
        """Create Redis connection."""
        if not self._client:
            self._client = redis.from_url(self.url, decode_responses=True)
            logger.info("Connected to Redis")
            
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            logger.info("Closed Redis connection")
            
    async def ping(self) -> bool:
        """Check Redis connection."""
        try:
            await self._client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
            
    async def get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached value or None if not found.
        """
        try:
            value = await self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting cache for {key}: {e}")
            return None
            
    async def set_cache(self, key: str, value: Dict[str, Any], ttl: int = CACHE_TTL) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time to live in seconds.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            await self._client.set(key, json.dumps(value), ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Error setting cache for {key}: {e}")
            return False
            
    async def xadd(self, stream: str, data: Dict[str, Any]) -> Optional[str]:
        """Add entry to Redis stream.
        
        Args:
            stream: Stream name.
            data: Stream entry data.
            
        Returns:
            Stream entry ID or None if failed.
        """
        try:
            # Convert data to string values
            stream_data = {k: str(v) for k, v in data.items()}
            entry_id = await self._client.xadd(stream, stream_data)
            return entry_id
        except Exception as e:
            logger.error(f"Error adding to stream {stream}: {e}")
            return None
            
    async def xread(self, streams: List[str], count: int = 10) -> List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]:
        """Read entries from Redis streams.
        
        Args:
            streams: List of stream names.
            count: Maximum number of entries to read per stream.
            
        Returns:
            List of (stream, entries) tuples.
        """
        try:
            # Convert stream names to (name, last_id) tuples
            stream_ids = {stream: "0" for stream in streams}
            entries = await self._client.xread(stream_ids, count=count)
            return entries
        except Exception as e:
            logger.error(f"Error reading streams {streams}: {e}")
            return []
            
    async def add_to_crawl_queue(self, concept: str, weight: float, source: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add item to crawl queue.
        
        Args:
            concept: Concept to crawl.
            weight: Priority weight.
            source: Source type (git, pdf, url, graph).
            metadata: Optional metadata.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            data = {
                "concept": concept,
                "weight": str(weight),
                "source": source,
                "timestamp": str(time.time())
            }
            if metadata:
                data["metadata"] = json.dumps(metadata)
                
            await self.xadd("crawl_queue", data)
            return True
        except Exception as e:
            logger.error(f"Error adding to crawl queue: {e}")
            return False
            
    async def publish_crawl_result(self, concept: str, result: Dict[str, Any]) -> bool:
        """Publish crawl result.
        
        Args:
            concept: Crawled concept.
            result: Crawl result data.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            data = {
                "concept": concept,
                "result": json.dumps(result),
                "timestamp": str(time.time())
            }
            await self.xadd("crawl_results", data)
            return True
        except Exception as e:
            logger.error(f"Error publishing crawl result: {e}")
            return False

# Create global Redis client instance
redis_client = RedisClient()

async def init_redis() -> None:
    """Initialize Redis connection."""
    await redis_client.connect()

async def close_redis() -> None:
    """Close Redis connection.""" 