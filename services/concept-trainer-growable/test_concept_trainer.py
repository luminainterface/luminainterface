import httpx
import asyncio
import numpy as np
from typing import List, Dict
import json
from datetime import datetime

async def test_concept_training():
    """Test the concept training functionality"""
    async with httpx.AsyncClient(base_url="http://localhost:8710") as client:
        # Test data for training
        concept_id = "test_concept_1"
        payload_train = {
            "concept_id": concept_id,
            "vectors": [np.random.randn(768).tolist() for _ in range(10)],
            "labels": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "metadata": { "source": "test" },
            "timestamp": datetime.now().isoformat()
        }
        
        # Test training
        response = await client.post(
            "/api/v1/model/train",
            json=payload_train
        )
        print("/api/v1/model/train status:", response.status_code)
        print("/api/v1/model/train response:", response.text)
        assert response.status_code == 200
        training_result = response.json()
        print("Training result:", json.dumps(training_result, indent=2))
        
        # Test getting concept status
        response = await client.get(f"/api/v1/model/status/{concept_id}")
        print(f"/api/v1/model/status/{concept_id} status:", response.status_code)
        print(f"/api/v1/model/status/{concept_id} response:", response.text)
        assert response.status_code == 200
        status = response.json()
        print("Concept status:", json.dumps(status, indent=2))
        
        # Test model health
        response = await client.get("/api/v1/model/health")
        print("/api/v1/model/health status:", response.status_code)
        print("/api/v1/model/health response:", response.text)
        assert response.status_code == 200
        health = response.json()
        print("Model health:", json.dumps(health, indent=2))
        
        # Test model growth
        payload_grow = {
            "concept_id": concept_id,
            "target_size": 1024,  # Grow first layer
            "metadata": { "reason": "test_growth" }
        }
        response = await client.post(
            "/api/v1/model/grow",
            json=payload_grow
        )
        print("/api/v1/model/grow status:", response.status_code)
        print("/api/v1/model/grow response:", response.text)
        assert response.status_code == 200
        growth_result = response.json()
        print("Growth result:", json.dumps(growth_result, indent=2))
        
        # Test health check
        response = await client.get("/api/v1/health")
        print("/api/v1/health status:", response.status_code)
        print("/api/v1/health response:", response.text)
        assert response.status_code == 200
        health = response.json()
        print("Health check:", json.dumps(health, indent=2))

if __name__ == "__main__":
    asyncio.run(test_concept_training()) 