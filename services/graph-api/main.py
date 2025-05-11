from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import redis
import json
import os
from typing import List, Optional, Dict
import logging
from prometheus_client import Counter, Histogram
import time
from qdrant_client import QdrantClient
import networkx as nx
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI(title="Graph API Service")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis client
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url)

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = QdrantClient(url=qdrant_url)

# Initialize graph
graph = nx.DiGraph()

# Prometheus metrics
graph_operations = Counter('graph_operations_total', 'Total number of graph operations')
graph_latency = Histogram('graph_operation_latency_seconds', 'Time spent on graph operations')

API_KEY = os.getenv("GRAPH_API_KEY", "changeme")

@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    if request.url.path.startswith("/health"):  # Allow health checks
        return await call_next(request)
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)

class Node(BaseModel):
    id: str
    type: str
    properties: Optional[Dict] = None

class Edge(BaseModel):
    source: str
    target: str
    type: str
    properties: Optional[Dict] = None

@app.get("/health")
async def health_check():
    try:
        redis_client.ping()
        qdrant_client.get_collections()
        return {"status": "healthy", "redis": "connected", "qdrant": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/nodes")
async def create_node(node: Node):
    with graph_latency.time():
        graph_operations.inc()
        try:
            graph.add_node(node.id, type=node.type, properties=node.properties or {})
            return {"status": "success", "node_id": node.id}
        except Exception as e:
            logger.error(f"Error creating node: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/edges")
async def create_edge(edge: Edge):
    with graph_latency.time():
        graph_operations.inc()
        try:
            graph.add_edge(edge.source, edge.target, type=edge.type, properties=edge.properties or {})
            return {"status": "success", "edge": f"{edge.source}->{edge.target}"}
        except Exception as e:
            logger.error(f"Error creating edge: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    with graph_latency.time():
        graph_operations.inc()
        try:
            if not graph.has_node(node_id):
                raise HTTPException(status_code=404, detail="Node not found")
            node_data = graph.nodes[node_id]
            return {"id": node_id, **node_data}
        except Exception as e:
            logger.error(f"Error retrieving node {node_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/nodes")
async def list_nodes():
    with graph_latency.time():
        graph_operations.inc()
        try:
            nodes = []
            for node_id, data in graph.nodes(data=True):
                nodes.append({"id": node_id, **data})
            return nodes
        except Exception as e:
            logger.error(f"Error listing nodes: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/edges")
async def list_edges():
    with graph_latency.time():
        graph_operations.inc()
        try:
            edges = []
            for source, target, data in graph.edges(data=True):
                edges.append({
                    "source": source,
                    "target": target,
                    **data
                })
            return edges
        except Exception as e:
            logger.error(f"Error listing edges: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200) 