from fastapi import FastAPI, WebSocket, HTTPException
from redis.asyncio import Redis
import json
import asyncio
from typing import Dict, Any
import os
from prometheus_client import Counter, Histogram
from loguru import logger

# Metrics
EVENT_COUNTER = Counter(
    'event_total',
    'Total number of events processed',
    ['type', 'destination']
)

EVENT_LATENCY = Histogram(
    'event_latency_seconds',
    'Event processing latency',
    ['type'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
)

app = FastAPI(title="Event Mux")

# Initialize Redis client
redis = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))

class EventMux:
    def __init__(self):
        self.redis = Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            encoding="utf-8",
            decode_responses=True
        )
        self.websocket_clients: Dict[str, WebSocket] = {}
        self.stream_name = "graph_stream"
        self.consumer_group = "event_mux"
        self.consumer_name = "event_mux_1"

    async def setup(self):
        """Initialize Redis streams and consumer group."""
        try:
            # Create consumer group if it doesn't exist
            await self.redis.xgroup_create(
                self.stream_name,
                self.consumer_group,
                mkstream=True
            )
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def process_event(self, event: Dict[str, Any]):
        """Process and distribute an event."""
        with EVENT_LATENCY.labels(type=event.get('type', 'unknown')).time():
            # Broadcast to WebSocket clients
            if self.websocket_clients:
                message = json.dumps(event)
                await asyncio.gather(*[
                    client.send_text(message)
                    for client in self.websocket_clients.values()
                ])
                EVENT_COUNTER.labels(
                    type=event.get('type', 'unknown'),
                    destination='websocket'
                ).inc()

            # TODO: Add Kafka producer here if needed
            # TODO: Add Prometheus push here if needed

    async def start_consumer(self):
        """Start consuming events from Redis stream."""
        while True:
            try:
                # Read new events
                events = await self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.stream_name: '>'},
                    count=100,
                    block=1000
                )

                for _, messages in events:
                    for msg_id, data in messages:
                        try:
                            event = json.loads(data['event'])
                            await self.process_event(event)
                            # Acknowledge processed message
                            await self.redis.xack(
                                self.stream_name,
                                self.consumer_group,
                                msg_id
                            )
                        except Exception as e:
                            logger.error(f"Error processing event: {e}")

            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)

    async def register_websocket(self, websocket: WebSocket, client_id: str):
        """Register a new WebSocket client."""
        await websocket.accept()
        self.websocket_clients[client_id] = websocket
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except Exception:
            pass
        finally:
            del self.websocket_clients[client_id]

@app.on_event("startup")
async def startup():
    """Initialize Event-Mux on startup."""
    await event_mux.setup()
    asyncio.create_task(event_mux.start_consumer())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    client_id = str(id(websocket))
    await event_mux.register_websocket(websocket, client_id)

@app.get("/health")
async def health() -> Dict[str, str]:
    try:
        await redis.ping()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) 