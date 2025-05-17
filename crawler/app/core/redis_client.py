"""Redis client for the crawler service."""
import os
import logging
from redis.asyncio import Redis
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class RedisClient:
    """Redis client wrapper with connection management."""
    def __init__(self, url: Optional[str] = None):
        self.url = url or os.getenv("REDIS_URL", "redis://:02211998@redis:6379")
        self._client: Optional[Redis] = None
        self._initialized = False

    async def connect(self):
        """Connect to Redis."""
        if not self._initialized:
            try:
                self._client = Redis.from_url(
                    self.url,
                    encoding="utf-8",
                    decode_responses=True
                )
                self._initialized = True
                logger.info("Connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._initialized = False
            logger.info("Closed Redis connection")

    async def ping(self) -> bool:
        """Ping Redis server."""
        if not self._initialized:
            await self.connect()
        return await self._client.ping()

    async def xadd(self, stream: str, data: Dict[str, Any]) -> str:
        """Add data to a Redis stream."""
        if not self._initialized:
            await self.connect()
        return await self._client.xadd(stream, data)

    async def xread(self, streams: List[str], count: int = 10) -> List[Tuple[str, List[Tuple[str, Dict[str, Any]]]]]:
        """Read from Redis streams."""
        if not self._initialized:
            await self.connect()
        return await self._client.xread({stream: "0" for stream in streams}, count=count)

    async def xack(self, stream: str, group: str, message_id: str) -> int:
        """Acknowledge a message in a Redis stream."""
        if not self._initialized:
            await self.connect()
        return await self._client.xack(stream, group, message_id)

    async def xlen(self, stream: str) -> int:
        """Get the length of a Redis stream."""
        if not self._initialized:
            await self.connect()
        return await self._client.xlen(stream)

# Create a singleton instance
redis_client = RedisClient() 