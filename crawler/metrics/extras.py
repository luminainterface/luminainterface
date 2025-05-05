from prometheus_client import Gauge, Histogram, Counter, REGISTRY
import time

# Queue depth tracking
queue_depth = Gauge(
    "crawler_queue_depth",
    "Number of URLs waiting to be fetched",
    ["priority"],  # Track by priority level
    registry=REGISTRY
)

# Response size tracking with appropriate buckets
bytes_pulled = Histogram(
    "crawler_response_bytes",
    "Size of fetched pages in bytes",
    buckets=[1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6],  # 1 KB → 1 MB
    registry=REGISTRY
)

# HTTP status code tracking
http_status = Counter(
    "crawler_http_status_total",
    "HTTP response codes by status",
    ["code", "domain"],  # Track by status code and domain
    registry=REGISTRY
)

# Fetch timing with domain tracking
fetch_duration = Histogram(
    "crawler_fetch_duration_seconds",
    "Time spent fetching pages",
    ["domain"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],  # 100ms → 10s
    registry=REGISTRY
)

def track_fetch(domain: str, size: int, status: int, duration: float):
    """Helper to record all metrics for a single fetch."""
    queue_depth.labels(priority="normal").dec()  # Decrement queue
    bytes_pulled.observe(size)
    http_status.labels(code=str(status), domain=domain).inc()
    fetch_duration.labels(domain=domain).observe(duration)

def track_queue_add(priority: str = "normal"):
    """Helper to increment queue counter."""
    queue_depth.labels(priority=priority).inc() 