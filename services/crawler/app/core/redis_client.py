from typing import Any, Optional, List, Dict, Tuple
import json
import redis.asyncio as redis
import logging
from datetime import timedelta, datetime
import os
from .vector_store import VectorStore
import time
from lumina_core.common.bus import BusClient
import aioredis
import asyncio

logger = logging.getLogger(__name__)

class RedisClient:
    def __init__(self, redis_url: str, cache_ttl: int = 86400):
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self._redis = None
        self._lock = asyncio.Lock()
        self.bus = BusClient(redis_url=redis_url)
        self.system_crawl_channel = "lumina.system.crawl"
        self.crawl_queue = "crawler.queue"
        self.dead_letter_queue = "crawler.deadletter"
        self.vector_store = "crawler.vectors"
        self.inquiry_tracking = "crawler.inquiries"
        # Qdrant integration
        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.qdrant = VectorStore(qdrant_url)
        self.client = None
        
    async def __aenter__(self):
        await self.bus.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.redis.close()
        await self.bus.close()
        
    async def connect(self):
        """Connect to Redis"""
        if not self._redis:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            
    async def get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache"""
        if not self._redis:
            await self.connect()
        try:
            data = await self._redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting cache for {key}: {e}")
            return None
            
    async def set_cache(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        if not self._redis:
            await self.connect()
        try:
            await self._redis.set(
                key,
                json.dumps(value),
                ex=ttl or self.cache_ttl
            )
            return True
        except Exception as e:
            logger.error(f"Error setting cache for {key}: {e}")
            return False
            
    async def get_stream_info(self, stream: str) -> Dict[str, Any]:
        """Get information about a stream"""
        if not self._redis:
            await self.connect()
        try:
            info = await self._redis.xinfo_stream(stream)
            return {
                "length": info["length"],
                "radix_tree_keys": info["radix-tree-keys"],
                "radix_tree_nodes": info["radix-tree-nodes"],
                "groups": info["groups"],
                "last_generated_id": info["last-generated-id"],
                "first_entry": info["first-entry"],
                "last_entry": info["last-entry"]
            }
        except Exception as e:
            logger.error(f"Error getting stream info for {stream}: {e}")
            return {}
            
    async def get_group_info(self, stream: str, group: str) -> Dict[str, Any]:
        """Get information about a consumer group"""
        if not self._redis:
            await self.connect()
        try:
            info = await self._redis.xinfo_groups(stream)
            for group_info in info:
                if group_info["name"] == group:
                    return {
                        "name": group_info["name"],
                        "consumers": group_info["consumers"],
                        "pending": group_info["pending"],
                        "last_delivered_id": group_info["last-delivered-id"]
                    }
            return {}
        except Exception as e:
            logger.error(f"Error getting group info for {stream}:{group}: {e}")
            return {}
            
    async def get_consumer_info(self, stream: str, group: str, consumer: str) -> Dict[str, Any]:
        """Get information about a consumer"""
        if not self._redis:
            await self.connect()
        try:
            info = await self._redis.xinfo_consumers(stream, group)
            for consumer_info in info:
                if consumer_info["name"] == consumer:
                    return {
                        "name": consumer_info["name"],
                        "pending": consumer_info["pending"],
                        "idle": consumer_info["idle"]
                    }
            return {}
        except Exception as e:
            logger.error(f"Error getting consumer info for {stream}:{group}:{consumer}: {e}")
            return {}
            
    async def get_pending_messages(self, stream: str, group: str, consumer: str) -> List[Dict[str, Any]]:
        """Get pending messages for a consumer"""
        if not self._redis:
            await self.connect()
        try:
            pending = await self._redis.xpending(stream, group, "-", "+", 100, consumer)
            return [
                {
                    "message_id": msg[0],
                    "consumer": msg[1],
                    "idle_time": msg[2],
                    "delivery_count": msg[3]
                }
                for msg in pending
            ]
        except Exception as e:
            logger.error(f"Error getting pending messages for {stream}:{group}:{consumer}: {e}")
            return []
            
    async def get_dlq_stats(self, dlq_stream: str) -> Dict[str, Any]:
        """Get statistics about the dead-letter queue"""
        if not self._redis:
            await self.connect()
        try:
            info = await self.get_stream_info(dlq_stream)
            if not info:
                return {
                    "total_messages": 0,
                    "oldest_message": None,
                    "newest_message": None,
                    "reasons": {}
                }
                
            # Get all messages to analyze reasons
            messages = await self._redis.xrange(dlq_stream, "-", "+", count=1000)
            reasons = {}
            for _, msg in messages:
                reason = msg.get("reason", "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
                
            return {
                "total_messages": info["length"],
                "oldest_message": info["first_entry"][0] if info["first_entry"] else None,
                "newest_message": info["last_entry"][0] if info["last_entry"] else None,
                "reasons": reasons
            }
        except Exception as e:
            logger.error(f"Error getting DLQ stats for {dlq_stream}: {e}")
            return {
                "total_messages": 0,
                "oldest_message": None,
                "newest_message": None,
                "reasons": {}
            }
            
    async def reprocess_dlq_message(self, dlq_stream: str, message_id: str, target_stream: str) -> bool:
        """Reprocess a message from the dead-letter queue"""
        if not self._redis:
            await self.connect()
        try:
            # Get the message from DLQ
            messages = await self._redis.xrange(dlq_stream, message_id, message_id)
            if not messages:
                logger.error(f"Message {message_id} not found in DLQ {dlq_stream}")
                return False
                
            # Extract original message
            _, msg = messages[0]
            original_msg = json.loads(msg.get("original_message", "{}"))
            if not original_msg:
                logger.error(f"No original message found in DLQ message {message_id}")
                return False
                
            # Publish to target stream
            await self._redis.xadd(
                target_stream,
                {
                    **original_msg,
                    "reprocessed_from": message_id,
                    "reprocessed_at": datetime.utcnow().isoformat()
                }
            )
            
            # Remove from DLQ
            await self._redis.xdel(dlq_stream, message_id)
            return True
            
        except Exception as e:
            logger.error(f"Error reprocessing DLQ message {message_id}: {e}")
            return False
            
    async def cleanup_old_dlq_messages(self, dlq_stream: str, max_age_seconds: int = 604800) -> int:
        """Clean up old messages from the dead-letter queue"""
        if not self._redis:
            await self.connect()
        try:
            # Get all messages
            messages = await self._redis.xrange(dlq_stream, "-", "+")
            now = datetime.utcnow().timestamp()
            deleted = 0
            
            for msg_id, msg in messages:
                try:
                    timestamp = datetime.fromisoformat(msg.get("timestamp", "")).timestamp()
                    if now - timestamp > max_age_seconds:
                        await self._redis.xdel(dlq_stream, msg_id)
                        deleted += 1
                except (ValueError, TypeError):
                    # Skip messages with invalid timestamps
                    continue
                    
            return deleted
            
        except Exception as e:
            logger.error(f"Error cleaning up old DLQ messages: {e}")
            return 0

    async def enqueue(self, queue_name: str, item: Any) -> bool:
        """Add an item to a queue"""
        try:
            serialized = json.dumps(item)
            await self._redis.rpush(queue_name, serialized)
            return True
        except Exception as e:
            logger.error(f"Error enqueueing item to {queue_name}: {str(e)}")
            return False
            
    async def dequeue(self, queue_name: str, timeout: int = 0) -> Optional[Any]:
        """Get and remove an item from a queue with optional timeout"""
        try:
            if timeout > 0:
                result = await self._redis.blpop(queue_name, timeout=timeout)
                if result:
                    return json.loads(result[1])
            else:
                result = await self._redis.lpop(queue_name)
                if result:
                    return json.loads(result)
            return None
        except Exception as e:
            logger.error(f"Error dequeuing item from {queue_name}: {str(e)}")
            return None
            
    async def get_queue_length(self, queue_name: str) -> int:
        """Get the current length of a Redis stream"""
        try:
            if not self._redis:
                await self.connect()
            return await self._redis.xlen(queue_name)
        except Exception as e:
            logger.error(f"Error getting queue length for {queue_name}: {e}")
            return 0
            
    async def clear_queue(self, queue_name: str) -> bool:
        """Clear all items from a queue"""
        try:
            await self._redis.delete(queue_name)
            return True
        except Exception as e:
            logger.error(f"Error clearing queue {queue_name}: {str(e)}")
            return False 

    async def get_system_crawl_requests(self) -> List[Dict]:
        """Get pending system crawl requests from Redis Stream."""
        try:
            # Use BusClient to consume from crawl_request stream
            requests = []
            async def handle_request(msg: StreamMessage):
                try:
                    request = msg.data
                    requests.append(request)
                except Exception as e:
                    logger.error(f"Error processing crawl request: {e}")
            
            # Consume from crawl_request stream
            await self.bus.consume(
                stream="crawl_request",
                group="crawler",
                consumer="system",
                handler=handle_request,
                block_ms=1000,
                count=10
            )
            return requests
        except Exception as e:
            logger.error(f"Error getting system crawl requests: {e}")
            return []

    async def track_inquiry(self, inquiry_id: str, concept: str, weight: float, source: str = "system") -> bool:
        """Track an inquiry and its associated concept with weight"""
        try:
            inquiry_data = {
                "concept": concept,
                "weight": weight,
                "source": source,
                "timestamp": time.time(),
                "status": "pending"
            }
            await self._redis.hset(self.inquiry_tracking, inquiry_id, json.dumps(inquiry_data))
            # Add to crawl queue with weight-based priority
            await self.add_to_crawl_queue(concept, weight, source, inquiry_id)
            return True
        except Exception as e:
            logger.error(f"Error tracking inquiry: {e}")
            return False

    async def add_to_crawl_queue(self, concept: str, weight: float, source: str = "system", inquiry_id: Optional[str] = None) -> bool:
        """Add a concept to the crawl queue with weight-based priority"""
        try:
            # Calculate priority based on weight and source
            base_priority = weight
            if source == "system":
                base_priority *= 1.5  # Boost system inquiries
            
            request = {
                "concept": concept,
                "weight": weight,
                "priority": base_priority,
                "source": source,
                "inquiry_id": inquiry_id,
                "timestamp": time.time()
            }
            
            # Publish to crawl_request stream
            await self.bus.publish(
                stream="crawl_request",
                data=request,
                maxlen=1000  # Keep last 1000 requests
            )
            return True
        except Exception as e:
            logger.error(f"Error adding to crawl queue: {e}")
            return False

    async def store_vector(self, concept: str, vector: List[float], metadata: Dict) -> bool:
        """Store a vector with associated metadata in Redis and Qdrant"""
        try:
            vector_data = {
                "concept": concept,
                "vector": vector,
                "metadata": {
                    **metadata,
                    "timestamp": time.time(),
                    "format": "float32",
                    "dimensions": len(vector)
                }
            }
            # Store in Redis
            await self._redis.hset(
                self.vector_store,
                concept,
                json.dumps(vector_data)
            )
            # Store in Qdrant (sync call)
            self.qdrant.upsert_vectors(
                vectors=[vector],
                metadata=[metadata],
                ids=[concept]
            )
            return True
        except Exception as e:
            logger.error(f"Error storing vector: {e}")
            return False

    async def get_vector(self, concept: str) -> Optional[Dict]:
        """Retrieve a stored vector and its metadata"""
        try:
            data = await self._redis.hget(self.vector_store, concept)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving vector: {e}")
            return None

    async def get_next_crawl_request(self, source: Optional[str] = None) -> Optional[Dict]:
        """Get the next crawl request from the stream, optionally filtering by source."""
        try:
            # Use BusClient to consume from crawl_request stream
            request = None
            
            async def handle_request(msg: StreamMessage):
                nonlocal request
                try:
                    data = msg.data
                    if not source or data.get("source") == source:
                        request = data
                        if request.get("inquiry_id"):
                            await self.update_inquiry_status(request["inquiry_id"], "processing")
                except Exception as e:
                    logger.error(f"Error processing crawl request: {e}")
            
            # Consume one message from crawl_request stream
            await self.bus.consume(
                stream="crawl_request",
                group="crawler",
                consumer="worker",
                handler=handle_request,
                block_ms=1000,
                count=1
            )
            return request
        except Exception as e:
            logger.error(f"Error getting next crawl request: {e}")
            return None

    async def update_inquiry_status(self, inquiry_id: str, status: str) -> bool:
        """Update the status of an inquiry"""
        try:
            data = await self._redis.hget(self.inquiry_tracking, inquiry_id)
            if data:
                inquiry_data = json.loads(data)
                inquiry_data["status"] = status
                await self._redis.hset(self.inquiry_tracking, inquiry_id, json.dumps(inquiry_data))
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating inquiry status: {e}")
            return False

    async def add_to_dead_letter_queue(self, concept: str, error: str) -> None:
        """Add a failed crawl request to the dead letter queue"""
        try:
            await self._redis.hset(
                self.dead_letter_queue,
                concept,
                json.dumps({
                    "error": error,
                    "timestamp": time.time()
                })
            )
        except Exception as e:
            logger.error(f"Error adding to dead letter queue: {e}")

    async def get_queue_stats(self) -> Dict:
        """Get statistics about the crawl queues and inquiries"""
        try:
            return {
                "queue_size": await self._redis.zcard(self.crawl_queue),
                "dead_letter_size": await self._redis.hlen(self.dead_letter_queue),
                "vector_count": await self._redis.hlen(self.vector_store),
                "inquiry_count": await self._redis.hlen(self.inquiry_tracking),
                "priority_distribution": {
                    "high": await self._redis.zcount(self.crawl_queue, 0.7, 1.0),
                    "medium": await self._redis.zcount(self.crawl_queue, 0.3, 0.7),
                    "low": await self._redis.zcount(self.crawl_queue, 0, 0.3)
                },
                "source_distribution": {
                    "system": await self._redis.zcount(self.crawl_queue, 0, 1.0, "system"),
                    "graph": await self._redis.zcount(self.crawl_queue, 0, 1.0, "graph")
                }
            }
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {}

    async def publish_crawl_result(self, concept: str, result: Dict) -> bool:
        """Publish crawl result to ingest.crawl stream."""
        try:
            # Prepare ingest.crawl message
            message = {
                "url": result.get("url", ""),
                "title": result.get("title", concept),
                "vec_id": result.get("page_id", ""),
                "ts": datetime.utcnow().isoformat()
            }
            
            # Publish to ingest.crawl stream
            await self.bus.publish(
                stream="ingest.crawl",
                data=message,
                maxlen=1000  # Keep last 1000 results
            )
            return True
        except Exception as e:
            logger.error(f"Error publishing crawl result: {e}")
            return False

    async def get_consumer_lag(self, queue_name: str, consumer_group: str) -> int:
        """Get the consumer lag for a Redis stream consumer group"""
        try:
            if not self._redis:
                await self.connect()
                
            # Get consumer group info
            info = await self._redis.xinfo_groups(queue_name)
            
            # Find our consumer group
            for group in info:
                if group[b'name'].decode() == consumer_group:
                    # Return pending messages count
                    return group[b'pending']
                    
            return 0
        except Exception as e:
            logger.error(f"Error getting consumer lag for {queue_name}:{consumer_group}: {e}")
            return 0
            
    async def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        try:
            if not self._redis:
                await self.connect()
                
            # Get all keys matching page:*
            keys = await self._redis.keys("page:*")
            
            # Count hits and misses
            hits = 0
            misses = 0
            
            for key in keys:
                key_str = key.decode()
                if await self._redis.exists(key_str):
                    hits += 1
                else:
                    misses += 1
                    
            return {
                "hits": hits,
                "misses": misses,
                "total": len(keys)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"hits": 0, "misses": 0, "total": 0} 