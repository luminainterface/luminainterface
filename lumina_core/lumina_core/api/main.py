from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

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

@app.post("/chat")
@app.middleware("http")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(RateLimiter(times=10, seconds=60, key_func=lambda: request.headers.get("X-API-Key")))
):
    // ... existing code ...

@app.post("/admin/prune")
async def trigger_pruning(
    _: bool = Depends(verify_api_key),
    __: bool = Depends(RateLimiter(times=5, seconds=60, key_func=lambda: request.headers.get("X-API-Key")))
):
    // ... existing code ...

@app.get("/metrics/summary")
async def get_metrics(
    _: bool = Depends(verify_api_key),
    __: bool = Depends(RateLimiter(times=30, seconds=60, key_func=lambda: request.headers.get("X-API-Key")))
):
    // ... existing code ...

// ... existing code ... 