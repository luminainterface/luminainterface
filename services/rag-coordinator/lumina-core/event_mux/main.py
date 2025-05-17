from fastapi import FastAPI
from prometheus_client import make_asgi_app, Counter

app = FastAPI()

# Example metric
EVENTS_PROCESSED = Counter('event_mux_events_total', 'Total number of events processed by event-mux')

@app.get("/health")
def health():
    return {"status": "ok"}

# Expose /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ... rest of your event-mux code ... 