import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import httpx
from .utils import setup_logging, setup_scheduler, verify_api_key
from .models import ChatRequest
from . import memory

app = FastAPI()

# Configure CORS
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add staging/prod UI URLs here when ready
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

async def update_cache_metrics(cache):
    """Background task to update cache metrics."""
    while True:
        try:
            # Update metrics every 60 seconds
            await asyncio.sleep(60)
            # TODO: Implement actual metrics collection
            pass
        except Exception as e:
            print(f"Error updating cache metrics: {e}")

# Rate limiter setup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    setup_logging()
    setup_scheduler()
    
    # Initialize rate limiter
    redis_client = redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        encoding="utf-8",
        decode_responses=True
    )
    await FastAPILimiter.init(redis_client)
    
    # Start cache metrics background task
    asyncio.create_task(update_cache_metrics(memory.cache))

# Rate limit decorators
CHAT_RATE_LIMIT = "10/minute"  # 10 requests per minute per key
ADMIN_RATE_LIMIT = "5/minute"  # 5 requests per minute per key

# Add OPTIONS route handlers for all endpoints
@app.options("/v1/chat/completions")
async def options_chat():
    return {}

@app.options("/metrics/summary")
async def options_metrics():
    return {}

@app.options("/health")
async def options_health():
    return {}

@app.options("/admin/cache/clear")
async def options_admin_cache_clear():
    return {}

@app.options("/admin/cache/stats")
async def options_admin_cache_stats():
    return {}

@app.post("/v1/chat/completions")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(RateLimiter(times=10, seconds=60))
):
    """Handle chat completion requests."""
    try:
        logger.debug(f"Received chat request: {chat_request}")
        # Get the last user message
        user_message = next((msg.content for msg in reversed(chat_request.messages) 
                           if msg.role == "user"), None)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        logger.debug(f"Calling LLM engine with message: {user_message}")
        # Call the LLM engine
        async with httpx.AsyncClient() as client:
            llm_url = f"{os.getenv('LLM_ENGINE_URL', 'http://llm-engine:11434')}/api/generate"
            logger.debug(f"LLM engine URL: {llm_url}")
            response = await client.post(
                llm_url,
                json={
                    "model": chat_request.model,
                    "prompt": user_message,
                    "stream": False
                }
            )
            logger.debug(f"LLM engine response status: {response.status_code}")
            response.raise_for_status()
            llm_response = response.json()
            logger.debug(f"LLM engine response: {llm_response}")

        # Format the response
        return {
            "id": f"chatcmpl-{int(asyncio.get_event_loop().time())}",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": chat_request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": llm_response.get("response", "No response from model")
                },
                "finish_reason": "stop"
            }]
        }
    except httpx.HTTPError as e:
        logger.error(f"HTTP error calling LLM engine: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling LLM engine: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/prune")
async def trigger_pruning(
    _: bool = Depends(verify_api_key),
    __: bool = Depends(RateLimiter(times=5, seconds=60))
):
    """Trigger cache pruning."""
    try:
        # TODO: Implement cache pruning
        return {"status": "success", "message": "Cache pruning triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/summary")
async def get_metrics(
    _: bool = Depends(verify_api_key),
    __: bool = Depends(RateLimiter(times=30, seconds=60))
):
    """Return metrics summary."""
    try:
        # Get cache metrics
        cache_metrics = {
            "cache_size": len(memory.cache._cache) if hasattr(memory, 'cache') else 0,
            "cache_hits": getattr(memory.cache, 'hits', 0) if hasattr(memory, 'cache') else 0,
            "cache_misses": getattr(memory.cache, 'misses', 0) if hasattr(memory, 'cache') else 0,
            "cache_hit_rate": (
                getattr(memory.cache, 'hits', 0) / 
                (getattr(memory.cache, 'hits', 0) + getattr(memory.cache, 'misses', 1))
            ) if hasattr(memory, 'cache') else 0,
            "latency_p95_ms": 0,  # TODO: Implement latency tracking
            "requests_per_minute": 0,  # TODO: Implement request rate tracking
            "error_rate": 0,  # TODO: Implement error rate tracking
            "active_connections": 0,  # TODO: Implement connection tracking
        }
        
        return cache_metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check health of all services."""
    try:
        # Check Redis
        redis_client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
        redis_status = "ok"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = "down"

    # Check Vector DB
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.getenv('VECTOR_DB_URL', 'http://vector-db:6333')}/healthz"
            )
            response.raise_for_status()
            vector_db_status = "ok"
    except Exception as e:
        logger.error(f"Vector DB health check failed: {e}")
        vector_db_status = "down"

    # Check LLM Engine
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.getenv('LLM_ENGINE_URL', 'http://llm-engine:11434')}/api/tags"
            )
            response.raise_for_status()
            llm_status = "ok"
    except Exception as e:
        logger.error(f"LLM Engine health check failed: {e}")
        llm_status = "down"

    # Check Scheduler
    try:
        # TODO: Implement actual scheduler health check
        scheduler_status = "ok"
    except Exception as e:
        logger.error(f"Scheduler health check failed: {e}")
        scheduler_status = "down"

    return {
        "redis": redis_status,
        "qdrant": vector_db_status,
        "ollama": llm_status,
        "scheduler": scheduler_status
    }

# ... existing code ... 