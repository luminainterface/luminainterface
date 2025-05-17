from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import httpx
import redis.asyncio as redis
import logging
from prometheus_client import Gauge, Histogram
from .config import settings
import time

# Configure logging
logger = logging.getLogger(__name__)

# Metrics
service_health = Gauge(
    f"{settings.METRICS_PREFIX}_service_health",
    "Service health status (1 = healthy, 0 = unhealthy)",
    ["service", "check"]
)

health_check_duration = Histogram(
    f"{settings.METRICS_PREFIX}_health_check_duration_seconds",
    "Duration of health checks in seconds",
    ["service", "check"]
)

class HealthResponse(BaseModel):
    status: str
    checks: Dict[str, str]
    error: Optional[str] = None
    version: Optional[str] = None
    duration_s: Optional[float] = None

router = APIRouter()

async def check_redis() -> bool:
    start_time = time.time()
    try:
        redis_client = redis.from_url(str(settings.REDIS_URL))
        await redis_client.ping()
        service_health.labels(service="redis", check="ping").set(1)
        health_check_duration.labels(service="redis", check="ping").observe(time.time() - start_time)
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        service_health.labels(service="redis", check="ping").set(0)
        health_check_duration.labels(service="redis", check="ping").observe(time.time() - start_time)
        return False

async def check_qdrant() -> bool:
    start_time = time.time()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.QDRANT_URL}/readyz")
            response.raise_for_status()
            service_health.labels(service="qdrant", check="http").set(1)
            health_check_duration.labels(service="qdrant", check="http").observe(time.time() - start_time)
            return True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {str(e)}")
        service_health.labels(service="qdrant", check="http").set(0)
        health_check_duration.labels(service="qdrant", check="http").observe(time.time() - start_time)
        return False

async def check_prometheus() -> bool:
    start_time = time.time()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.PROMETHEUS_URL}/-/ready")
            response.raise_for_status()
            service_health.labels(service="prometheus", check="ready").set(1)
            health_check_duration.labels(service="prometheus", check="ready").observe(time.time() - start_time)
            return True
    except Exception as e:
        logger.error(f"Prometheus health check failed: {str(e)}")
        service_health.labels(service="prometheus", check="ready").set(0)
        health_check_duration.labels(service="prometheus", check="ready").observe(time.time() - start_time)
        return False

async def check_ollama() -> bool:
    start_time = time.time()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.OLLAMA_URL}/api/tags")
            response.raise_for_status()
            service_health.labels(service="ollama", check="api").set(1)
            health_check_duration.labels(service="ollama", check="api").observe(time.time() - start_time)
            return True
    except Exception as e:
        logger.error(f"Ollama health check failed: {str(e)}")
        service_health.labels(service="ollama", check="api").set(0)
        health_check_duration.labels(service="ollama", check="api").observe(time.time() - start_time)
        return False

async def check_event_mux() -> bool:
    start_time = time.time()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.EVENT_MUX_URL}/health")
            response.raise_for_status()
            service_health.labels(service="event_mux", check="http").set(1)
            health_check_duration.labels(service="event_mux", check="http").observe(time.time() - start_time)
            return True
    except Exception as e:
        logger.error(f"Event-mux health check failed: {str(e)}")
        service_health.labels(service="event_mux", check="http").set(0)
        health_check_duration.labels(service="event_mux", check="http").observe(time.time() - start_time)
        return False

@router.get("/health", response_model=HealthResponse)
async def health_check():
    start_time = time.time()
    checks = {}
    all_healthy = True

    # Check Redis
    redis_healthy = await check_redis()
    checks["redis"] = "ok" if redis_healthy else "error"
    all_healthy = all_healthy and redis_healthy

    # Check Qdrant
    qdrant_healthy = await check_qdrant()
    checks["qdrant"] = "ok" if qdrant_healthy else "error"
    all_healthy = all_healthy and qdrant_healthy

    # Check Prometheus if metrics enabled
    if settings.ENABLE_METRICS:
        prom_healthy = await check_prometheus()
        checks["prometheus"] = "ok" if prom_healthy else "error"
        all_healthy = all_healthy and prom_healthy

    # Check Ollama
    ollama_healthy = await check_ollama()
    checks["ollama"] = "ok" if ollama_healthy else "error"
    all_healthy = all_healthy and ollama_healthy

    # Check Event-mux
    event_mux_healthy = await check_event_mux()
    checks["event_mux"] = "ok" if event_mux_healthy else "error"
    all_healthy = all_healthy and event_mux_healthy

    # Overall service health metric
    service_health.labels(service="overall", check="aggregate").set(1 if all_healthy else 0)

    response = HealthResponse(
        status="ok" if all_healthy else "error",
        checks=checks,
        version=settings.VERSION if hasattr(settings, "VERSION") else None,
        duration_s=time.time() - start_time
    )

    if not all_healthy:
        response.error = "One or more dependency checks failed"
        raise HTTPException(status_code=503, detail=response.dict())

    return response 