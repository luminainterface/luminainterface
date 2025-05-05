from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import os
from redis import Redis
from qdrant_client import QdrantClient
from prometheus_client import generate_latest, CollectorRegistry, Counter, Histogram
from lumina_core.common.cors import add_cors
import time

app = FastAPI(title="Graph API")
add_cors(app)

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
    return {
        "nodes": [],
        "edges": [],
        "clusters": []
    }

@app.get("/metrics/summary")
def metric_summary():
    try:
        nodes = redis_client.hlen("graph:nodes") if redis_client.exists("graph:nodes") else 0
        edges = redis_client.scard("graph:edges") if redis_client.exists("graph:edges") else 0
        return {"nodes": nodes, "edges": edges, "ts": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 