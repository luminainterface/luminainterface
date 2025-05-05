from fastapi import FastAPI, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, generate_latest
import time
import os

app = FastAPI()
reg = CollectorRegistry()

# Get Git SHA from environment or default to 'unknown'
git_sha = os.getenv('GITHUB_SHA', 'unknown')
common_labels = {'git_sha': git_sha}

# Basic metrics
pages_total = Counter("crawler_pages_total", "Pages successfully fetched", registry=reg, labelnames=['git_sha'])
errors_total = Counter("crawler_errors_total", "HTTP or parse errors", registry=reg, labelnames=['git_sha'])
duration = Histogram("crawler_fetch_seconds", "Fetch duration", registry=reg, labelnames=['git_sha'])

# Enhanced metrics
queue_depth = Gauge("crawler_queue_depth", "URLs awaiting fetch", registry=reg, labelnames=['git_sha'])
bytes_pulled = Histogram(
    "crawler_response_bytes",
    "Size of fetched pages",
    buckets=[1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6],  # 1 KB → 1 MB
    registry=reg,
    labelnames=['git_sha']
)
http_status = Counter(
    "crawler_http_status_total",
    "Responses by HTTP code",
    ["code", "git_sha"],
    registry=reg
)

# Initialize metrics with Git SHA label
for metric in [pages_total, errors_total, queue_depth]:
    metric.labels(git_sha=git_sha)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(reg), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "ok"} 