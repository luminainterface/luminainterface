import httpx
import asyncio
import numpy as np
from typing import List, Dict
import json

async def test_concept_training():
    """Test the concept training functionality"""
    async with httpx.AsyncClient(base_url="http://localhost:8905") as client:
        # Test data
        concept_id = "test_concept_1"
        embeddings = np.random.randn(10, 768).tolist()  # 10 samples with 768-dim embeddings
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Binary classification
        
        # Test training
        response = await client.post(
            "/train",
            json={
                "concept_id": concept_id,
                "embeddings": embeddings,
                "labels": labels
            }
        )
        assert response.status_code == 200
        training_result = response.json()
        print("Training result:", json.dumps(training_result, indent=2))
        
        # Test getting metrics
        response = await client.get(f"/concepts/{concept_id}/metrics")
        assert response.status_code == 200
        metrics = response.json()
        print("Concept metrics:", json.dumps(metrics, indent=2))
        
        # Test network stats
        response = await client.get("/network/stats")
        assert response.status_code == 200
        stats = response.json()
        print("Network stats:", json.dumps(stats, indent=2))
        
        # Test layer growth
        response = await client.post(
            "/grow",
            json={
                "concept_id": concept_id,
                "layer_idx": 0,
                "new_size": 768  # Grow first layer
            }
        )
        assert response.status_code == 200
        growth_result = response.json()
        print("Growth result:", json.dumps(growth_result, indent=2))
        
        # Test health check
        response = await client.get("/health")
        assert response.status_code == 200
        health = response.json()
        print("Health check:", json.dumps(health, indent=2))

if __name__ == "__main__":
    asyncio.run(test_concept_training()) 