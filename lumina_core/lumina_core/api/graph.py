from fastapi import APIRouter, WebSocket, HTTPException
from redis.asyncio import Redis
import json
import os
from typing import Dict, Any, List
import networkx as nx
from loguru import logger

router = APIRouter()
redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# In-memory graph for fast lookups
graph = nx.Graph()

@router.get("/hierarchy")
async def get_hierarchy():
    """Get the hierarchical clustering of the graph."""
    try:
        # Read hierarchy from file
        with open("data/hierarchy.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Hierarchy not found")

@router.get("/subgraph/{cluster_id}")
async def get_subgraph(cluster_id: str):
    """Get nodes and edges for a specific cluster."""
    try:
        # Read subgraph from file
        with open(f"data/clusters/{cluster_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Cluster not found")

@router.get("/degree/{node_id}")
async def get_degree(node_id: str):
    """Get the degree of a node (constant-time lookup)."""
    try:
        return {"degree": graph.degree(node_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Node not found")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time graph updates."""
    await websocket.accept()
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception:
        pass

@router.get("/sse")
async def event_stream():
    """Server-Sent Events endpoint for graph updates."""
    async def event_generator():
        while True:
            # Read from Redis stream
            events = await redis.xread({"graph_stream": ">"}, count=1, block=1000)
            for _, messages in events:
                for _, data in messages:
                    yield f"data: {data['event']}\n\n"
    
    return EventSourceResponse(event_generator())

# Helper function to emit graph events
async def emit_graph_event(event_type: str, payload: Dict[str, Any]):
    """Emit a graph event to Redis stream."""
    event = {
        "type": event_type,
        "payload": payload,
        "ts": time.time()
    }
    await redis.xadd("graph_stream", {"event": json.dumps(event)}) 