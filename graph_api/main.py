from fastapi import FastAPI, HTTPException, APIRouter, Query, Body, BackgroundTasks, WebSocket
from typing import Dict, Any, List, Optional, Set
import os
import redis.asyncio as redis
from qdrant_client import QdrantClient
from prometheus_client import generate_latest, CollectorRegistry, Counter, Histogram
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from datetime import datetime
import json
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models as qdrant_models
import httpx
from functools import lru_cache
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

from models import (
    Node, Edge, GraphResponse, BatchOperation, TraversalConfig,
    BulkImportRequest, BulkExportRequest, ConceptSyncRequest,
    TrainingDataRequest, ChatRequest, ChatResponse,
    LearningPathOptimizationRequest, LearningPathOptimizationResponse,
    ConceptAnalysisRequest, ConceptAnalysisResponse,
    RecommendationRequest, RecommendationResponse,
    ClusteringRequest, ClusteringResponse,
    GraphAnalyticsRequest, GraphAnalyticsResponse,
    CrawlerRequest, ConceptUpdate, LearningPathRequest
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Graph API",
    description="Graph API for Lumina - Knowledge Graph Management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create versioned routers
v1_router = APIRouter(prefix="/v1", tags=["v1"])

# Initialize metrics
reg = CollectorRegistry()
request_total = Counter("graphapi_requests_total", "REST requests", ["path", "version"], registry=reg)
latency = Histogram("graphapi_request_seconds", "Latency", ["path", "version"], registry=reg)
graph_operations = Counter("graphapi_operations_total", "Graph operations", ["operation", "status"], registry=reg)

# Initialize sentence transformer for embeddings
model = None

# Initialize clients as None
redis_client = None
qdrant_client = None

# Service configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# Service URLs from environment
CRAWLER_URL = os.getenv("CRAWLER_URL", "http://crawler:8400")
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8000")
LEARNING_PATH_URL = os.getenv("LEARNING_PATH_URL", "http://learning-path-optimizer:8000")
DUAL_CHAT_URL = os.getenv("DUAL_CHAT_URL", "http://dual-chat-router:8000")

async def init_clients():
    """Initialize Redis and Qdrant clients."""
    global redis_client, qdrant_client, model
    
    try:
        # Initialize Redis
        redis_client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        await redis_client.ping()
        logger.info("✅ Redis connection established")
        
        # Initialize Qdrant
        qdrant_client = QdrantClient(url=QDRANT_URL)
        logger.info("✅ Qdrant connection established")
        
        # Initialize sentence transformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Sentence transformer model loaded")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize clients: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup."""
    logger.info("🚀 Starting graph-api service...")
    await init_clients()
    logger.info("✅ Graph API service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown."""
    if redis_client:
        await redis_client.close()
    logger.info("👋 Graph API service shut down")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Redis connection
        if not redis_client or not await redis_client.ping():
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": "Redis connection failed"}
            )
        
        # Check Qdrant connection
        if not qdrant_client:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": "Qdrant client not initialized"}
            )
        
        # Check model initialization
        if not model:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": "Sentence transformer not initialized"}
            )
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "dependencies": {
                "redis": "connected",
                "qdrant": "connected",
                "model": "loaded"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

async def get_redis_client():
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            await redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    return redis_client

def get_qdrant_client():
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "qdrant"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        # Create collection if it doesn't exist
        try:
            qdrant_client.get_collection("graph_nodes")
        except Exception:
            qdrant_client.create_collection(
                collection_name="graph_nodes",
                vectors_config=qdrant_models.VectorParams(
                    size=384,  # Default size for all-MiniLM-L6-v2
                    distance=qdrant_models.Distance.COSINE
                )
            )
    return qdrant_client

def get_model():
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# API version information
API_VERSIONS = {
    "v1": {
        "version": "1.0.0",
        "status": "current",
        "release_date": "2024-03-20",
        "deprecation_date": None,
        "sunset_date": None
    }
}

# Initialize HTTP client
@lru_cache()
def get_http_client():
    return httpx.AsyncClient(timeout=30.0)

# Helper functions
async def get_node(node_id: str) -> Optional[Node]:
    node_data = await get_redis_client().hget("graph:nodes", node_id)
    if node_data:
        return Node.parse_raw(node_data)
    return None

async def get_edge(edge_id: str) -> Optional[Edge]:
    edge_data = await get_redis_client().hget("graph:edges", edge_id)
    if edge_data:
        return Edge.parse_raw(edge_data)
    return None

async def save_node(node: Node) -> None:
    # Generate embedding if not present
    if not node.embedding and model:
        text = node.get_text_for_embedding()
        node.embedding = model.encode(text).tolist()
    
    # Save to Redis
    await get_redis_client().hset("graph:nodes", node.id, node.model_dump_json())
    await get_redis_client().sadd("graph:node_types", node.type)
    
    # Save to Qdrant if embedding exists
    if node.embedding:
        try:
            # Convert string ID to UUID for Qdrant
            point_id = uuid.UUID(node.id)
        except ValueError:
            # If not a valid UUID, generate a new one based on the string
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, node.id)
        
        get_qdrant_client().upsert(
            collection_name="graph_nodes",
            points=[
                qdrant_models.PointStruct(
                    id=str(point_id),
                    vector=node.embedding,
                    payload={"type": node.type, "original_id": node.id}
                )
            ]
        )

async def save_edge(edge: Edge) -> None:
    await get_redis_client().hset("graph:edges", edge.id, edge.model_dump_json())
    await get_redis_client().sadd(f"graph:node:{edge.source}:outgoing", edge.id)
    await get_redis_client().sadd(f"graph:node:{edge.target}:incoming", edge.id)
    await get_redis_client().sadd("graph:edge_types", edge.type)

async def delete_node(node_id: str) -> None:
    # Get all connected edges
    outgoing_edges = await get_redis_client().smembers(f"graph:node:{node_id}:outgoing")
    incoming_edges = await get_redis_client().smembers(f"graph:node:{node_id}:incoming")
    
    # Delete edges
    for edge_id in outgoing_edges | incoming_edges:
        await delete_edge(edge_id)
    
    # Delete node
    await get_redis_client().hdel("graph:nodes", node_id)
    await get_redis_client().delete(f"graph:node:{node_id}:outgoing")
    await get_redis_client().delete(f"graph:node:{node_id}:incoming")
    
    # Delete from Qdrant
    try:
        get_qdrant_client().delete(
            collection_name="graph_nodes",
            points_selector=qdrant_models.PointIdsList(
                points=[node_id]
            )
        )
    except Exception as e:
        logger.error(f"Error deleting node from Qdrant: {e}")

async def delete_edge(edge_id: str) -> None:
    edge = await get_edge(edge_id)
    if edge:
        await get_redis_client().hdel("graph:edges", edge_id)
        await get_redis_client().srem(f"graph:node:{edge.source}:outgoing", edge_id)
        await get_redis_client().srem(f"graph:node:{edge.target}:incoming", edge_id)

async def traverse_graph(
    start_node_id: str,
    config: TraversalConfig,
    visited_nodes: Optional[Set[str]] = None,
    visited_edges: Optional[Set[str]] = None,
    current_depth: int = 0
) -> tuple[List[Node], List[Edge]]:
    if visited_nodes is None:
        visited_nodes = set()
    if visited_edges is None:
        visited_edges = set()
    
    if current_depth > config.max_depth:
        return [], []
    
    start_node = await get_node(start_node_id)
    if not start_node or start_node_id in visited_nodes:
        return [], []
    
    visited_nodes.add(start_node_id)
    nodes = [start_node]
    edges = []
    
    # Get outgoing edges
    if config.direction in ["both", "outgoing"]:
        outgoing_edge_ids = await get_redis_client().smembers(f"graph:node:{start_node_id}:outgoing")
        for edge_id in outgoing_edge_ids:
            if edge_id in visited_edges:
                continue
            
            edge = await get_edge(edge_id)
            if not edge:
                continue
            
            if config.edge_types and edge.type not in config.edge_types:
                continue
            
            visited_edges.add(edge_id)
            edges.append(edge)
            
            target_node = await get_node(edge.target)
            if target_node and (not config.node_types or target_node.type in config.node_types):
                child_nodes, child_edges = await traverse_graph(
                    edge.target,
                    config,
                    visited_nodes,
                    visited_edges,
                    current_depth + 1
                )
                nodes.extend(child_nodes)
                edges.extend(child_edges)
    
    # Get incoming edges
    if config.direction in ["both", "incoming"]:
        incoming_edge_ids = await get_redis_client().smembers(f"graph:node:{start_node_id}:incoming")
        for edge_id in incoming_edge_ids:
            if edge_id in visited_edges:
                continue
            
            edge = await get_edge(edge_id)
            if not edge:
                continue
            
            if config.edge_types and edge.type not in config.edge_types:
                continue
            
            visited_edges.add(edge_id)
            edges.append(edge)
            
            source_node = await get_node(edge.source)
            if source_node and (not config.node_types or source_node.type in config.node_types):
                child_nodes, child_edges = await traverse_graph(
                    edge.source,
                    config,
                    visited_nodes,
                    visited_edges,
                    current_depth + 1
                )
                nodes.extend(child_nodes)
                edges.extend(child_edges)
    
    return nodes, edges

# API endpoints
@v1_router.post("/nodes", response_model=Node)
async def create_node(node: Node) -> Node:
    """Create a new node in the graph"""
    await save_node(node)
    graph_operations.labels(operation="create_node", status="success").inc()
    return node

@v1_router.put("/nodes/{node_id}", response_model=Node)
async def update_node(node_id: str, node: Node) -> Node:
    """Update an existing node in the graph"""
    if node_id != node.id:
        raise HTTPException(status_code=400, detail="Node ID mismatch")
    
    existing_node = await get_node(node_id)
    if not existing_node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    await save_node(node)
    graph_operations.labels(operation="update_node", status="success").inc()
    return node

@v1_router.delete("/nodes/{node_id}")
async def delete_node_endpoint(node_id: str):
    """Delete a node and all its connected edges from the graph"""
    node = await get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    await delete_node(node_id)
    graph_operations.labels(operation="delete_node", status="success").inc()
    return {"status": "success", "message": "Node deleted"}

@v1_router.post("/batch", response_model=GraphResponse)
async def batch_operation(operation: BatchOperation) -> GraphResponse:
    """Perform a batch operation on nodes and edges"""
    try:
        # Process nodes
        for node in operation.nodes:
            await save_node(node)
        
        # Process edges
        for edge in operation.edges:
            await save_edge(edge)
        
        graph_operations.labels(operation="batch_operation", status="success").inc()
        return GraphResponse(
            nodes=operation.nodes,
            edges=operation.edges,
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
    except Exception as e:
        graph_operations.labels(operation="batch_operation", status="error").inc()
        logger.error(f"Error during batch operation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/traverse/{node_id}", response_model=GraphResponse)
async def traverse_graph_endpoint(
    node_id: str,
    config: TraversalConfig = Body(...)
) -> GraphResponse:
    """Traverse the graph starting from a node"""
    try:
        nodes, edges = await traverse_graph(node_id, config)
        graph_operations.labels(operation="traverse", status="success").inc()
        return GraphResponse(
            nodes=nodes,
            edges=edges,
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
    except Exception as e:
        graph_operations.labels(operation="traverse", status="error").inc()
        logger.error(f"Error during graph traversal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.get("/search", response_model=GraphResponse)
async def search_graph(
    query: str = Query(..., description="Search query"),
    node_type: Optional[str] = Query(None, description="Filter by node type"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results")
) -> GraphResponse:
    """Search for nodes in the graph using vector similarity"""
    try:
        # Generate query embedding
        query_embedding = get_model().encode(query).tolist()
        
        # Search in Qdrant
        search_result = get_qdrant_client().search(
            collection_name="graph_nodes",
            query_vector=query_embedding,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="type",
                        match=qdrant_models.MatchValue(value=node_type)
                    )
                ]
            ) if node_type else None,
            limit=limit
        )
        
        # Get nodes from Redis
        nodes = []
        for hit in search_result:
            node = await get_node(hit.id)
            if node:
                nodes.append(node)
        
        # Get edges between found nodes
        edges = []
        node_ids = {node.id for node in nodes}
        for node_id in node_ids:
            outgoing_edge_ids = await get_redis_client().smembers(f"graph:node:{node_id}:outgoing")
            incoming_edge_ids = await get_redis_client().smembers(f"graph:node:{node_id}:incoming")
            
            for edge_id in outgoing_edge_ids | incoming_edge_ids:
                edge = await get_edge(edge_id)
                if edge and edge.source in node_ids and edge.target in node_ids:
                    edges.append(edge)
        
        graph_operations.labels(operation="search", status="success").inc()
        return GraphResponse(
            nodes=nodes,
            edges=edges,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "node_type": node_type
            }
        )
    except Exception as e:
        graph_operations.labels(operation="search", status="error").inc()
        logger.error(f"Error during graph search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    """Get Prometheus metrics"""
    return generate_latest(reg)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    path = request.url.path
    version = "v1" if path.startswith("/v1") else "v0"
    
    try:
        response = await call_next(request)
        request_total.labels(path=path, version=version).inc()
        latency.labels(path=path, version=version).observe(time.time() - start_time)
        return response
    except Exception as e:
        request_total.labels(path=path, version=version).inc()
        latency.labels(path=path, version=version).observe(time.time() - start_time)
        raise e

# Service integration endpoints
@v1_router.post("/integrate/crawler", response_model=Dict[str, Any])
async def trigger_crawler(request: CrawlerRequest) -> Dict[str, Any]:
    """Trigger the crawler to extract concepts from a URL"""
    try:
        async with get_http_client() as client:
            response = await client.post(
                f"{CRAWLER_URL}/crawl",
                json=request.dict()
            )
            response.raise_for_status()
            graph_operations.labels(operation="crawler_integration", status="success").inc()
            return response.json()
    except Exception as e:
        graph_operations.labels(operation="crawler_integration", status="error").inc()
        logger.error(f"Error triggering crawler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.post("/integrate/concept-dictionary/sync", response_model=Dict[str, Any])
async def sync_with_concept_dictionary(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Synchronize nodes with the concept dictionary"""
    try:
        # Get all nodes
        node_ids = await get_redis_client().hkeys("graph:nodes")
        nodes = []
        for node_id in node_ids:
            node = await get_node(node_id)
            if node:
                nodes.append(node)
        
        # Start background sync
        background_tasks.add_task(sync_nodes_with_concept_dictionary, nodes)
        
        graph_operations.labels(operation="concept_sync", status="success").inc()
        return {
            "status": "success",
            "message": f"Started synchronization of {len(nodes)} nodes",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        graph_operations.labels(operation="concept_sync", status="error").inc()
        logger.error(f"Error during concept sync: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@v1_router.post("/integrate/learning-path", response_model=GraphResponse)
async def get_learning_path(request: LearningPathRequest) -> GraphResponse:
    """Get an optimal learning path between concepts"""
    try:
        # Verify nodes exist
        start_node = await get_node(request.start_concept)
        target_node = await get_node(request.target_concept)
        
        if not start_node or not target_node:
            raise HTTPException(status_code=404, detail="Start or target concept not found")
        
        # Call learning path optimizer
        async with get_http_client() as client:
            response = await client.post(
                f"{LEARNING_PATH_URL}/optimize",
                json={
                    "start_concept": start_node.dict(),
                    "target_concept": target_node.dict(),
                    "constraints": request.constraints
                }
            )
            response.raise_for_status()
            path_data = response.json()
        
        # Get nodes and edges in the path
        nodes = [start_node]
        edges = []
        
        for step in path_data["path"]:
            node = await get_node(step["node_id"])
            edge = await get_edge(step["edge_id"])
            
            if node:
                nodes.append(node)
            if edge:
                edges.append(edge)
        
        nodes.append(target_node)
        
        graph_operations.labels(operation="learning_path", status="success").inc()
        return GraphResponse(
            nodes=nodes,
            edges=edges,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "path_metrics": path_data.get("metrics", {})
            }
        )
    except Exception as e:
        graph_operations.labels(operation="learning_path", status="error").inc()
        logger.error(f"Error getting learning path: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def sync_nodes_with_concept_dictionary(nodes: List[Node]):
    """Background task to sync nodes with the concept dictionary"""
    success_count = 0
    error_count = 0
    
    try:
        async with get_http_client() as client:
            for node in nodes:
                try:
                    # Update concept in dictionary
                    response = await client.post(
                        f"{CONCEPT_DICT_URL}/concepts/update",
                        json=ConceptUpdate(
                            concept_id=node.id,
                            properties=node.properties,
                            metadata=node.metadata
                        ).dict()
                    )
                    response.raise_for_status()
                    success_count += 1
                    
                    # Wait a bit to avoid overwhelming the service
                    await asyncio.sleep(0.1)
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error updating concept {node.id}: {str(e)}")
                    continue
            
            # Log final stats
            logger.info(f"Sync completed: {success_count} successful, {error_count} failed")
            graph_operations.labels(operation="concept_sync_background", status="success").inc()
    except Exception as e:
        graph_operations.labels(operation="concept_sync_background", status="error").inc()
        logger.error(f"Error in background sync: {str(e)}")
        raise  # Re-raise the exception to be handled by the caller

@v1_router.get("/metrics/summary")
async def metrics_summary():
    # Placeholder: return a simple summary
    return {"status": "ok", "metrics": {"example": 123}}

@app.get("/metrics/summary")
async def metrics_summary_root():
    return await metrics_summary()

@app.get("/sse")
async def sse():
    async def event_generator():
        while True:
            yield f"data: {{\"message\": \"heartbeat\"}}\n\n"
            await asyncio.sleep(5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@v1_router.get("/version")
async def get_version_info() -> Dict[str, Any]:
    """Get API version information"""
    return API_VERSIONS

@v1_router.post("/bulk/import", response_model=Dict[str, Any])
async def bulk_import(request: BulkImportRequest) -> Dict[str, Any]:
    """Import multiple nodes and edges in bulk"""
    try:
        # Process nodes
        for node in request.nodes:
            await save_node(node)
        
        # Process edges
        for edge in request.edges:
            await save_edge(edge)
        
        graph_operations.labels(operation="bulk_import", status="success").inc()
        return {
            "status": "success",
            "message": f"Imported {len(request.nodes)} nodes and {len(request.edges)} edges",
            "metadata": request.metadata
        }
    except Exception as e:
        graph_operations.labels(operation="bulk_import", status="error").inc()
        logger.error(f"Error during bulk import: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to Graph API v1", "docs_url": "/docs"}

# Include versioned routes
app.include_router(v1_router)

@v1_router.get("/nodes/{node_id}", response_model=Node)
async def get_node_endpoint(node_id: str) -> Node:
    """Get a node by ID"""
    node = await get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node

@v1_router.post("/edges", response_model=Edge)
async def create_edge(edge: Edge) -> Edge:
    """Create a new edge"""
    # Check if source and target nodes exist
    source_node = await get_node(edge.source)
    target_node = await get_node(edge.target)
    if not source_node or not target_node:
        raise HTTPException(status_code=404, detail="Source or target node not found")
    
    # Generate ID if not provided
    if not edge.id:
        edge.id = str(uuid.uuid4())
    
    # Set timestamps
    edge.created_at = datetime.now()
    edge.updated_at = datetime.now()
    
    # Save edge
    await save_edge(edge)
    return edge

@v1_router.get("/edges/{edge_id}", response_model=Edge)
async def get_edge_endpoint(edge_id: str) -> Edge:
    """Get an edge by ID"""
    edge = await get_edge(edge_id)
    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found")
    return edge

@v1_router.post("/traverse/{node_id}", response_model=GraphResponse)
async def traverse_graph_endpoint(
    node_id: str,
    config: TraversalConfig = Body(...)
) -> GraphResponse:
    """Traverse the graph starting from a node"""
    nodes, edges = await traverse_graph(node_id, config)
    return GraphResponse(nodes=nodes, edges=edges)

@v1_router.post("/integrate/concept/analyze", response_model=ConceptAnalysisResponse)
async def analyze_concept(request: ConceptAnalysisRequest) -> ConceptAnalysisResponse:
    """Analyze a concept and its relationships"""
    # Get the concept node
    concept = await get_node(request.concept_id)
    if not concept:
        raise HTTPException(status_code=404, detail="Concept not found")
    
    # Get all relationships
    outgoing_edges = await get_redis_client().smembers(f"graph:node:{request.concept_id}:outgoing")
    incoming_edges = await get_redis_client().smembers(f"graph:node:{request.concept_id}:incoming")
    
    relationships = []
    for edge_id in outgoing_edges | incoming_edges:
        edge = await get_edge(edge_id)
        if edge:
            relationships.append(edge)
    
    # Calculate metrics
    metrics = {
        "total_relationships": len(relationships),
        "outgoing_relationships": len(outgoing_edges),
        "incoming_relationships": len(incoming_edges)
    }
    
    return ConceptAnalysisResponse(
        concept=concept,
        relationships=relationships,
        metrics=metrics
    )

@v1_router.post("/integrate/learning-path/optimize", response_model=LearningPathOptimizationResponse)
async def optimize_learning_path(request: LearningPathOptimizationRequest) -> LearningPathOptimizationResponse:
    """Optimize learning path between two concepts"""
    # Check if concepts exist
    start_concept = await get_node(request.start_concept)
    target_concept = await get_node(request.target_concept)
    if not start_concept or not target_concept:
        raise HTTPException(status_code=404, detail="Start or target concept not found")
    
    # Find path using graph traversal
    config = TraversalConfig(max_depth=5, direction="outgoing")
    nodes, edges = await traverse_graph(request.start_concept, config)
    
    # Calculate path cost
    total_cost = sum(edge.properties.get("weight", 1.0) for edge in edges)
    
    return LearningPathOptimizationResponse(
        path=edges,
        total_cost=total_cost,
        concepts=nodes
    )

@v1_router.post("/integrate/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """Get recommendations based on context nodes"""
    recommendations = []
    confidence_scores = []
    
    # Get context nodes
    context_nodes = []
    for node_id in request.context_nodes:
        node = await get_node(node_id)
        if node:
            context_nodes.append(node)
    
    if not context_nodes:
        raise HTTPException(status_code=404, detail="No valid context nodes found")
    
    # Get similar nodes using vector search
    for node in context_nodes:
        if node.embedding:
            search_result = get_qdrant_client().search(
                collection_name="graph_nodes",
                query_vector=node.embedding,
                limit=5
            )
            for hit in search_result:
                recommended_node = await get_node(hit.id)
                if recommended_node and recommended_node.id not in request.context_nodes:
                    recommendations.append(recommended_node)
                    confidence_scores.append(hit.score)
    
    return RecommendationResponse(
        recommendations=recommendations,
        confidence_scores=confidence_scores
    )

@v1_router.post("/integrate/clustering", response_model=ClusteringResponse)
async def cluster_concepts(request: ClusteringRequest) -> ClusteringResponse:
    """Cluster concepts based on similarity"""
    # Get all nodes
    node_keys = await get_redis_client().hkeys("graph:nodes")
    nodes = []
    embeddings = []
    
    for key in node_keys:
        node = await get_node(key)
        if node and node.embedding:
            nodes.append(node)
            embeddings.append(node.embedding)
    
    if not nodes:
        raise HTTPException(status_code=404, detail="No nodes with embeddings found")
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Perform clustering (simple distance-based clustering)
    clusters = []
    used_indices = set()
    
    for i in range(len(nodes)):
        if i in used_indices:
            continue
        
        cluster = [nodes[i]]
        used_indices.add(i)
        
        # Find similar nodes
        for j in range(i + 1, len(nodes)):
            if j in used_indices:
                continue
            
            similarity = np.dot(embeddings_array[i], embeddings_array[j])
            if similarity >= request.similarity_threshold:
                cluster.append(nodes[j])
                used_indices.add(j)
        
        if len(cluster) >= request.min_cluster_size:
            clusters.append(cluster)
    
    # Calculate metrics
    metrics = {
        "total_clusters": len(clusters),
        "average_cluster_size": sum(len(c) for c in clusters) / len(clusters) if clusters else 0,
        "largest_cluster_size": max(len(c) for c in clusters) if clusters else 0
    }
    
    return ClusteringResponse(
        clusters=clusters,
        metrics=metrics
    )

@v1_router.post("/integrate/analytics", response_model=GraphAnalyticsResponse)
async def analyze_graph(request: GraphAnalyticsRequest) -> GraphAnalyticsResponse:
    """Analyze the graph structure"""
    # Get all nodes and edges
    node_keys = await get_redis_client().hkeys("graph:nodes")
    edge_keys = await get_redis_client().hkeys("graph:edges")
    
    nodes = []
    edges = []
    
    for key in node_keys:
        node = await get_node(key)
        if node:
            nodes.append(node)
    
    for key in edge_keys:
        edge = await get_edge(key)
        if edge:
            edges.append(edge)
    
    # Calculate metrics
    node_types = await get_redis_client().smembers("graph:node_types")
    edge_types = await get_redis_client().smembers("graph:edge_types")
    
    metrics = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "node_types": list(node_types),
        "edge_types": list(edge_types),
        "density": len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0
    }
    
    return GraphAnalyticsResponse(
        metrics=metrics,
        nodes=nodes,
        edges=edges
    )

@v1_router.get("/integrate/status")
async def integration_status() -> Dict[str, Any]:
    """Get integration status"""
    # Check Redis connection
    redis_status = "healthy"
    try:
        await get_redis_client().ping()
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    # Check Qdrant connection
    qdrant_status = "healthy"
    try:
        get_qdrant_client().get_collection("graph_nodes")
    except Exception as e:
        qdrant_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok" if redis_status == "healthy" and qdrant_status == "healthy" else "error",
        "components": {
            "redis": redis_status,
            "qdrant": qdrant_status
        }
    } 

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        redis_client = await get_redis_client()
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except Exception:
        await websocket.close()

@app.get("/hierarchy")
async def get_hierarchy():
    # Placeholder: return a simple hierarchy structure
    return {"root": {"children": []}}

@app.get("/redis-health")
async def redis_health_check() -> Dict[str, Any]:
    """Simple Redis health check endpoint"""
    try:
        redis_client = get_redis_client()
        redis_healthy = await redis_client.ping()
        return {
            "status": "healthy" if redis_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
    } 