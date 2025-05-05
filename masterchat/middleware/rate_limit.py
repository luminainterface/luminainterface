from fastapi import Request, status
from fastapi.responses import JSONResponse
import time
import redis
from typing import Optional

class RateLimiter:
    def __init__(self, redis_url: str, window: int = 60, max_requests: int = 30):
        self.redis = redis.from_url(redis_url)
        self.window = window  # Time window in seconds
        self.max_requests = max_requests

    async def is_rate_limited(self, key: str) -> bool:
        """Check if a request should be rate limited."""
        now = int(time.time())
        window_key = f"ratelimit:{key}:{now // self.window}"
        
        # Increment counter and get current count
        count = self.redis.incr(window_key)
        if count == 1:  # First request in window
            self.redis.expire(window_key, self.window)
        
        return count > self.max_requests

def rate_limit(app, redis_url: str):
    limiter = RateLimiter(redis_url)
    
    @app.middleware("http")
    async def _rate_limit_middleware(request: Request, call_next):
        # Only apply to /tasks endpoints
        if not request.url.path.startswith("/tasks"):
            return await call_next(request)
            
        # Get client IP
        client_ip = request.client.host
        if await limiter.is_rate_limited(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": limiter.window
                }
            )
            
        return await call_next(request) 