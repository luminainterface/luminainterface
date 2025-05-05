from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import os
from redis import Redis
from qdrant_client import QdrantClient
from prometheus_client import generate_latest, CollectorRegistry, Counter, Histogram

app = FastAPI(title="Graph API")

# Initialize metrics
reg = CollectorRegistry()
request_total = Counter("graphapi_requests_total", "REST requests", ["path"], registry=reg)
latency = Histogram("graphapi_request_seconds", "Latency", ["path"], registry=reg)

# Initialize clients
redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))

@app.get("/metrics")
def metrics():
    return generate_latest(reg), 200, {"Content-Type": "text/plain"}

@app.middleware("http")
async def metrics_middleware(request, call_next):
    path = request.url.path.split("?")[0]
    with latency.labels(path=path).time():
        response = await call_next(request)
    request_total.labels(path=path).inc()
    return response

@app.get("/health")
async def health_check() -> Dict[str, str]:
    try:
        redis_client.ping()
        qdrant_client.get_collections()
        return {
            "redis": "ok",
            "qdrant": "ok",
            "ollama": "ok",
            "scheduler": "ok"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hierarchy")
async def get_hierarchy() -> Dict[str, Any]:
    """Get the current knowledge graph hierarchy"""
    try:
        # TODO: Implement actual hierarchy logic
        return {
            "nodes": [],
            "edges": [],
            "clusters": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 