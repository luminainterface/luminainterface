from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from lumina_core.common.bus import BusClient
from lumina_core.common.config import get_settings
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import logging
from pydantic import BaseModel
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

app = FastAPI(title="Dead-Letter UI")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables with defaults
DLQ_PREFIX = os.getenv("DLQ_PREFIX", "dlq.")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# Initialize Redis client using settings
bus = BusClient(redis_url=settings.REDIS_URL)

# Metrics
DLQ_MESSAGES_TOTAL = Counter(
    'dlq_messages_total',
    'Total number of messages in DLQ',
    ['stream', 'error_type']
)

DLQ_RETRY_TOTAL = Counter(
    'dlq_retry_total',
    'Total number of retry attempts',
    ['stream', 'status']  # status: success, failure
)

DLQ_PROCESSING_TIME = Histogram(
    'dlq_processing_seconds',
    'Time spent processing DLQ messages',
    ['operation']  # operation: retry, delete, inspect
)

class DLQMessage(BaseModel):
    """Model for DLQ message display."""
    id: str
    stream: str
    original_stream: str
    error: str
    attempt: int
    timestamp: datetime
    payload: Dict[str, Any]
    skip: bool = False

class RetryRequest(BaseModel):
    """Model for retry request."""
    message_ids: List[str]
    stream: str

@app.on_event("startup")
async def startup():
    """Initialize connections on startup."""
    await bus.connect()
    logger.info("Connected to Redis")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    await bus.close()
    logger.info("Disconnected from Redis")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Redis connection
        await bus._redis.ping()
        
        return {
            "status": "healthy",
            "service": "dead-letter-ui",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "redis": "connected"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return JSONResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/streams")
async def list_streams():
    """List all dead-letter streams."""
    try:
        streams = []
        async for key in bus._redis.scan_iter(f"{DLQ_PREFIX}*"):
            info = await bus.get_stream_info(key)
            streams.append({
                "name": key,
                "length": info["length"],
                "last_message": info["last_entry"],
                "first_message": info["first_entry"]
            })
        return streams
    except Exception as e:
        logger.error(f"Failed to list streams: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/messages/{stream}")
async def get_messages(
    stream: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    error_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[DLQMessage]:
    """Get messages from a dead-letter stream with filtering."""
    try:
        # Validate stream
        if not stream.startswith(DLQ_PREFIX):
            stream = f"{DLQ_PREFIX}{stream}"
            
        # Get stream info
        info = await bus.get_stream_info(stream)
        if not info["length"]:
            return []
            
        # Get messages
        messages = []
        async for msg_id, data in bus._redis.xrange(stream, count=limit, offset=offset):
            try:
                # Parse message
                msg_data = {k: json.loads(v) for k, v in data.items()}
                timestamp = datetime.fromisoformat(msg_data["timestamp"])
                
                # Apply filters
                if error_type and error_type not in msg_data["error"]:
                    continue
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                    
                messages.append(DLQMessage(
                    id=msg_id,
                    stream=stream,
                    original_stream=msg_data["original_stream"],
                    error=msg_data["error"],
                    attempt=msg_data["attempt"],
                    timestamp=timestamp,
                    payload=msg_data["original_msg"],
                    skip=msg_data.get("skip", False)
                ))
                
            except Exception as e:
                logger.error(f"Failed to parse message {msg_id}: {e}")
                continue
                
        return messages
        
    except Exception as e:
        logger.error(f"Failed to get messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retry")
async def retry_messages(request: RetryRequest):
    """Retry failed messages."""
    start_time = datetime.utcnow()
    success_count = 0
    failure_count = 0
    
    try:
        for msg_id in request.message_ids:
            try:
                # Get message from DLQ
                msg_data = await bus._redis.xrange(request.stream, msg_id, msg_id, count=1)
                if not msg_data:
                    logger.warning(f"Message {msg_id} not found in {request.stream}")
                    failure_count += 1
                    continue
                    
                # Parse message
                data = {k: json.loads(v) for k, v in msg_data[0][1].items()}
                original_stream = data["original_stream"]
                original_msg = data["original_msg"]
                
                # Publish back to original stream
                await bus.publish(original_stream, original_msg)
                
                # Delete from DLQ
                await bus._redis.xdel(request.stream, msg_id)
                
                success_count += 1
                DLQ_RETRY_TOTAL.labels(stream=request.stream, status="success").inc()
                
            except Exception as e:
                logger.error(f"Failed to retry message {msg_id}: {e}")
                failure_count += 1
                DLQ_RETRY_TOTAL.labels(stream=request.stream, status="failure").inc()
                
        # Record processing time
        DLQ_PROCESSING_TIME.labels(operation="retry").observe(
            (datetime.utcnow() - start_time).total_seconds()
        )
        
        return {
            "status": "completed",
            "success_count": success_count,
            "failure_count": failure_count,
            "total_time": (datetime.utcnow() - start_time).total_seconds()
        }
        
    except Exception as e:
        logger.error(f"Failed to process retry request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/messages/{stream}")
async def delete_messages(
    stream: str,
    message_ids: List[str],
    before_time: Optional[datetime] = None
):
    """Delete messages from a dead-letter stream."""
    start_time = datetime.utcnow()
    deleted_count = 0
    
    try:
        if not stream.startswith(DLQ_PREFIX):
            stream = f"{DLQ_PREFIX}{stream}"
            
        # Get messages to delete
        to_delete = []
        if before_time:
            # Get all messages before the specified time
            async for msg_id, data in bus._redis.xrange(stream):
                try:
                    msg_data = {k: json.loads(v) for k, v in data.items()}
                    timestamp = datetime.fromisoformat(msg_data["timestamp"])
                    if timestamp < before_time:
                        to_delete.append(msg_id)
                except Exception as e:
                    logger.error(f"Failed to parse message {msg_id}: {e}")
                    continue
        else:
            # Delete specific messages
            to_delete = message_ids
            
        # Delete messages
        if to_delete:
            deleted = await bus._redis.xdel(stream, *to_delete)
            deleted_count = deleted
            
        # Record processing time
        DLQ_PROCESSING_TIME.labels(operation="delete").observe(
            (datetime.utcnow() - start_time).total_seconds()
        )
        
        return {
            "status": "completed",
            "deleted_count": deleted_count,
            "total_time": (datetime.utcnow() - start_time).total_seconds()
        }
        
    except Exception as e:
        logger.error(f"Failed to delete messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get dead-letter queue statistics."""
    try:
        stats = {
            "total_streams": 0,
            "total_messages": 0,
            "error_types": {},
            "streams": []
        }
        
        # Scan for DLQ streams
        async for key in bus._redis.scan_iter(f"{DLQ_PREFIX}*"):
            info = await bus.get_stream_info(key)
            stats["total_streams"] += 1
            stats["total_messages"] += info["length"]
            
            # Get error distribution
            error_types = {}
            async for msg_id, data in bus._redis.xrange(key, count=1000):
                try:
                    msg_data = {k: json.loads(v) for k, v in data.items()}
                    error = msg_data["error"]
                    error_types[error] = error_types.get(error, 0) + 1
                except Exception:
                    continue
                    
            stats["error_types"].update(error_types)
            stats["streams"].append({
                "name": key,
                "length": info["length"],
                "error_types": error_types,
                "last_message": info["last_entry"],
                "first_message": info["first_entry"]
            })
            
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 