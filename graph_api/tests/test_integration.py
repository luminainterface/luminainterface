import pytest
import httpx
import asyncio
from fastapi.testclient import TestClient
from datetime import datetime
import json
import numpy as np
import os
import redis.asyncio as redis
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import uuid

# Set environment variables for Redis and Qdrant
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["REDIS_DB"] = "0"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"

from main import app
from models import Node, Edge, BatchOperation, TraversalConfig

# Custom JSON encoder for datetime
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Initialize Redis and Qdrant clients
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def redis_client():
    client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True
    )
    yield client
    await client.close()

@pytest.fixture(scope="session")
def qdrant_client():
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )
    # Create collection if it doesn't exist
    try:
        client.get_collection("graph_nodes")
    except Exception:
        client.create_collection(
            collection_name="graph_nodes",
            vectors_config=qdrant_models.VectorParams(
                size=384,  # Default size for all-MiniLM-L6-v2
                distance=qdrant_models.Distance.COSINE
            )
        )
    return client

# Test client
@pytest.fixture(scope="module")
def client():
    return TestClient(app)

# Test data
@pytest.fixture
def sample_nodes():
    return [
        Node(
            id="node1",
            type="concept",
            properties={"name": "Python", "description": "Programming language"},
            embedding=[0.1] * 384,
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        Node(
            id="node2",
            type="concept",
            properties={"name": "FastAPI", "description": "Web framework"},
            embedding=[0.2] * 384,
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        Node(
            id="node3",
            type="concept",
            properties={"name": "Redis", "description": "In-memory database"},
            embedding=[0.3] * 384,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    ]

@pytest.fixture
def sample_edges():
    return [
        Edge(
            id="edge1",
            type="requires",
            source="node1",
            target="node2",
            properties={"weight": 0.8},
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        Edge(
            id="edge2",
            type="uses",
            source="node2",
            target="node3",
            properties={"weight": 0.6},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    ]

# Helper functions
async def clear_test_data(redis_client, qdrant_client):
    """Clear test data from Redis and Qdrant"""
    await redis_client.flushdb()
    try:
        qdrant_client.delete_collection("graph_nodes")
    except:
        pass

async def setup_test_data(redis_client, qdrant_client, nodes, edges):
    """Set up test data in Redis and Qdrant"""
    for node in nodes:
        await redis_client.hset("graph:nodes", node.id, node.model_dump_json())
        await redis_client.sadd("graph:node_types", node.type)
        
        # Add to Qdrant if embedding exists
        if node.embedding:
            try:
                point_id = uuid.UUID(node.id)
            except ValueError:
                point_id = uuid.uuid5(uuid.NAMESPACE_DNS, node.id)
            
            qdrant_client.upsert(
                collection_name="graph_nodes",
                points=[{
                    "id": str(point_id),
                    "vector": node.embedding,
                    "payload": {"type": node.type, "original_id": node.id}
                }]
            )
    
    for edge in edges:
        await redis_client.hset("graph:edges", edge.id, edge.model_dump_json())
        await redis_client.sadd(f"graph:node:{edge.source}:outgoing", edge.id)
        await redis_client.sadd(f"graph:node:{edge.target}:incoming", edge.id)
        await redis_client.sadd("graph:edge_types", edge.type)

# Test cases
@pytest.mark.asyncio
async def test_basic_crud_operations(client, redis_client, qdrant_client, sample_nodes, sample_edges):
    """Test basic CRUD operations"""
    await clear_test_data(redis_client, qdrant_client)
    
    # Test node creation
    for node in sample_nodes:
        response = client.post("/v1/nodes", json=json.loads(node.model_dump_json()))
        assert response.status_code == 200
        created_node = response.json()
        assert created_node["id"] == node.id
        assert created_node["type"] == node.type
    
    # Test node retrieval
    for node in sample_nodes:
        response = client.get(f"/v1/nodes/{node.id}")
        assert response.status_code == 200
        retrieved_node = response.json()
        assert retrieved_node["id"] == node.id

    # Test edge creation
    for edge in sample_edges:
        response = client.post("/v1/edges", json=json.loads(edge.model_dump_json()))
        assert response.status_code == 200
        created_edge = response.json()
        assert created_edge["source"] == edge.source
        assert created_edge["target"] == edge.target
    
    # Test edge retrieval
    for edge in sample_edges:
        response = client.get(f"/v1/edges/{edge.id}")
        assert response.status_code == 200
        retrieved_edge = response.json()
        assert retrieved_edge["id"] == edge.id

@pytest.mark.asyncio
async def test_graph_traversal(client, redis_client, qdrant_client, sample_nodes, sample_edges):
    """Test graph traversal functionality"""
    await clear_test_data(redis_client, qdrant_client)
    await setup_test_data(redis_client, qdrant_client, sample_nodes, sample_edges)
    
    # Test traversal from node1
    config = TraversalConfig(max_depth=2, direction="both")
    response = client.post(
        f"/v1/traverse/{sample_nodes[0].id}",
        json=json.loads(config.model_dump_json())
    )
    assert response.status_code == 200
    result = response.json()
    assert len(result["nodes"]) > 0
    assert len(result["edges"]) > 0

@pytest.mark.asyncio
async def test_search_functionality(client, redis_client, qdrant_client, sample_nodes, sample_edges):
    """Test search functionality"""
    await clear_test_data(redis_client, qdrant_client)
    await setup_test_data(redis_client, qdrant_client, sample_nodes, sample_edges)
    
    # Test vector search
    response = client.get("/v1/search?query=Python programming")
    assert response.status_code == 200
    result = response.json()
    assert len(result["nodes"]) > 0

@pytest.mark.asyncio
async def test_batch_operations(client, redis_client, qdrant_client, sample_nodes, sample_edges):
    """Test batch operations"""
    await clear_test_data(redis_client, qdrant_client)
    
    # Test batch import
    batch = BatchOperation(nodes=sample_nodes, edges=sample_edges)
    response = client.post("/v1/batch", json=json.loads(batch.model_dump_json()))
    assert response.status_code == 200
    result = response.json()
    assert len(result["nodes"]) == len(sample_nodes)
    assert len(result["edges"]) == len(sample_edges)

@pytest.mark.asyncio
async def test_concept_analysis(client, redis_client, qdrant_client, sample_nodes, sample_edges):
    """Test concept analysis functionality"""
    await clear_test_data(redis_client, qdrant_client)
    await setup_test_data(redis_client, qdrant_client, sample_nodes, sample_edges)
    
    # Test concept analysis
    response = client.post(
        "/v1/integrate/concept/analyze",
        json={"concept_id": sample_nodes[0].id}
    )
    assert response.status_code == 200
    result = response.json()
    assert "relationships" in result
    assert "metrics" in result

@pytest.mark.asyncio
async def test_learning_path_optimization(client, redis_client, qdrant_client, sample_nodes, sample_edges):
    """Test learning path optimization"""
    await clear_test_data(redis_client, qdrant_client)
    await setup_test_data(redis_client, qdrant_client, sample_nodes, sample_edges)
    
    # Test path optimization
    response = client.post(
        "/v1/integrate/learning-path/optimize",
        json={
            "start_concept": sample_nodes[0].id,
            "target_concept": sample_nodes[1].id
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "path" in result
    assert "total_cost" in result

@pytest.mark.asyncio
async def test_recommendations(client, redis_client, qdrant_client, sample_nodes, sample_edges):
    """Test recommendation system"""
    await clear_test_data(redis_client, qdrant_client)
    await setup_test_data(redis_client, qdrant_client, sample_nodes, sample_edges)
    
    # Test recommendations
    response = client.post(
        "/v1/integrate/recommendations",
        json={
            "user_id": "test_user",
            "context_nodes": [sample_nodes[0].id]
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "recommendations" in result
    assert "confidence_scores" in result

@pytest.mark.asyncio
async def test_clustering(client, redis_client, qdrant_client, sample_nodes, sample_edges):
    """Test concept clustering"""
    await clear_test_data(redis_client, qdrant_client)
    await setup_test_data(redis_client, qdrant_client, sample_nodes, sample_edges)
    
    # Test clustering
    response = client.post(
        "/v1/integrate/clustering",
        json={
            "min_cluster_size": 2,
            "similarity_threshold": 0.7
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "clusters" in result
    assert "metrics" in result

@pytest.mark.asyncio
async def test_graph_analytics(client, redis_client, qdrant_client, sample_nodes, sample_edges):
    """Test graph analytics"""
    await clear_test_data(redis_client, qdrant_client)
    await setup_test_data(redis_client, qdrant_client, sample_nodes, sample_edges)
    
    # Test analytics
    response = client.post(
        "/v1/integrate/analytics",
        json={"analysis_type": "full"}
    )
    assert response.status_code == 200
    result = response.json()
    assert "metrics" in result

@pytest.mark.asyncio
async def test_integration_status(client):
    """Test integration status endpoint"""
    response = client.get("/v1/integrate/status")
    assert response.status_code == 200
    result = response.json()
    assert "status" in result

@pytest.mark.asyncio
async def test_error_handling(client):
    """Test error handling"""
    # Test non-existent node
    response = client.get("/v1/nodes/nonexistent")
    assert response.status_code == 404
    result = response.json()
    assert "detail" in result

@pytest.mark.asyncio
async def test_concurrent_operations(client, redis_client, qdrant_client, sample_nodes, sample_edges):
    """Test concurrent operations"""
    await clear_test_data(redis_client, qdrant_client)
    await setup_test_data(redis_client, qdrant_client, sample_nodes, sample_edges)
    
    # Test concurrent node retrievals
    async def get_node(node_id):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:8000/v1/nodes/{node_id}")
            return response.status_code
    
    tasks = [get_node(node.id) for node in sample_nodes]
    results = await asyncio.gather(*tasks)
    assert all(status == 200 for status in results)

@pytest.mark.asyncio
async def test_cleanup(redis_client, qdrant_client):
    """Clean up test data"""
    await clear_test_data(redis_client, qdrant_client) 