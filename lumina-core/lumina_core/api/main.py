from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import asyncio
import time
import uuid
from datetime import datetime
from fastapi.security import APIKeyHeader
import os

from lumina_core.llm.ollama_bridge import OllamaBridge
from lumina_core.memory.qdrant_store import QdrantStore
from lumina_core.utils.logging import setup_logging, log_request, log_error
from lumina_core.api.openai_compat import router as openai_router
from lumina_core.memory.pruning import run_pruning_job
from lumina_core.scheduler import setup_scheduler, shutdown_scheduler, scheduler
from lumina_core.metrics import (
    REQUEST_LATENCY, PRUNE_DURATION, PRUNED_VECTORS,
    RATE_LIMIT_HITS, RATE_LIMIT_TOKENS,
    create_metrics_response, update_cache_metrics
)

# Setup logging
setup_logging(json_logs=False)

app = FastAPI()
ollama = OllamaBridge()
memory = QdrantStore()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Include OpenAI compatibility router
app.include_router(openai_router)

# API key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key from header."""
    expected_key = os.getenv("LUMINA_API_KEY")
    if not expected_key:
        return True  # Skip auth if no key configured
    
    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return True

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    confidence: float
    cite_ids: List[str]

# Rate limit decorators
CHAT_RATE_LIMIT = "10/minute"  # 10 requests per minute per key
ADMIN_RATE_LIMIT = "5/minute"  # 5 requests per minute per key

@app.post("/chat")
@app.middleware("http")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(RateLimiter(times=10, seconds=60, key_func=lambda: request.headers.get("X-API-Key")))
):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Get similar messages for context
        similar_messages = await memory.get_similar_messages(chat_request.message)
        context = [msg["content"] for msg in similar_messages]
        
        # Store user message
        user_message = {
            "role": "user",
            "content": chat_request.message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Stream response from Ollama
        async def generate():
            full_response = ""
            async for chunk in ollama.generate_stream(chat_request.message, context):
                if "response" in chunk:
                    full_response += chunk["response"]
                    yield f"data: {json.dumps({'chunk': chunk['response']})}\n\n"
            
            # Store assistant message
            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store both messages in memory
            await memory.upsert_messages([user_message, assistant_message])
            
            # Send final response with metrics
            yield f"data: {json.dumps({'done': True, 'tokens_used': ollama.tokens_used})}\n\n"
        
        response = StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
        # Log request completion
        duration = time.time() - start_time
        log_request(request_id, request.method, request.url.path, 200, duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        log_error(e, {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "duration_ms": round(duration * 1000, 2)
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/prune")
async def trigger_pruning(
    request: Request,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(RateLimiter(times=5, seconds=60, key_func=lambda: request.headers.get("X-API-Key")))
):
    """Trigger vector pruning job."""
    try:
        with PRUNE_DURATION.time():
            results = await run_pruning_job()
            
            # Record pruned vectors by reason
            if results["pruned"] > 0:
                PRUNED_VECTORS.labels(reason="age").inc(results["pruned"])
            
            return {
                "status": "success",
                "results": results
            }
    except Exception as e:
        log_error(e, {"endpoint": "/admin/prune"})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/summary")
async def get_metrics(
    request: Request,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(RateLimiter(times=30, seconds=60, key_func=lambda: request.headers.get("X-API-Key")))
):
    try:
        qdrant_metrics = await memory.get_metrics()
        metrics = {
            "conversations": qdrant_metrics["conversations"],
            "tokens_used": ollama.tokens_used,
            "qdrant_vectors": qdrant_metrics["vectors"]
        }
        
        # Add scheduler health metric
        if hasattr(scheduler, 'last_run'):
            metrics["prune_last_run_timestamp"] = scheduler.last_run
        else:
            metrics["prune_last_run_timestamp"] = 0
            
        return metrics
    except Exception as e:
        log_error(e, {"endpoint": "/metrics/summary"})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Deep health check of all services."""
    health_status = {
        "status": "ok",
        "services": {
            "ollama": "ok",
            "qdrant": "ok",
            "embeddings": "ok"
        }
    }
    
    try:
        # Check Ollama
        async for _ in ollama.generate_stream("test", []):
            break
    except Exception as e:
        health_status["services"]["ollama"] = "error"
        health_status["status"] = "degraded"
        log_error(e, {"service": "ollama"})
    
    try:
        # Check Qdrant
        await memory.get_metrics()
    except Exception as e:
        health_status["services"]["qdrant"] = "error"
        health_status["status"] = "degraded"
        log_error(e, {"service": "qdrant"})
    
    try:
        # Check embeddings
        memory.encoder.encode("test")
    except Exception as e:
        health_status["services"]["embeddings"] = "error"
        health_status["status"] = "degraded"
        log_error(e, {"service": "embeddings"})
    
    return health_status

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return create_metrics_response()

@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    """Middleware to track request latency and rate limits."""
    start_time = time.time()
    
    # Track rate limit hits
    api_key = request.headers.get("X-API-Key", "anonymous")
    endpoint = request.url.path
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record successful request
        REQUEST_LATENCY.labels(
            endpoint=endpoint,
            method=request.method
        ).observe(duration)
        
        RATE_LIMIT_HITS.labels(
            endpoint=endpoint,
            api_key=api_key,
            result="hit"
        ).inc()
        
        # Update available tokens if rate limited
        if response.status_code == 429:
            RATE_LIMIT_HITS.labels(
                endpoint=endpoint,
                api_key=api_key,
                result="block"
            ).inc()
            RATE_LIMIT_TOKENS.labels(
                endpoint=endpoint,
                api_key=api_key
            ).set(0)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        log_error(e, {
            "method": request.method,
            "path": request.url.path,
            "duration_ms": round(duration * 1000, 2)
        })
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    shutdown_scheduler()  # Stop the scheduler
    await ollama.close() 