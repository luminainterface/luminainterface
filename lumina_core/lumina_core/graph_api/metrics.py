from fastapi import APIRouter
from prometheus_client import generate_latest, CollectorRegistry, Counter, Histogram

router = APIRouter()
reg = CollectorRegistry()

request_total = Counter("graphapi_requests_total", "REST requests", ["path"], registry=reg)
latency = Histogram("graphapi_request_seconds", "Latency", ["path"], registry=reg)

@router.get("/metrics")
def metrics():
    return generate_latest(reg), 200, {"Content-Type": "text/plain"}

# middleware to count + time
def install_metrics(app):
    @app.middleware("http")
    async def _metrics(request, call_next):
        path = request.url.path.split("?")[0]
        with latency.labels(path=path).time():
            response = await call_next(request)
        request_total.labels(path=path).inc()
        return response 