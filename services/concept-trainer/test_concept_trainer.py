import httpx
import asyncio
import numpy as np
import json

async def test_train_endpoint():
    async with httpx.AsyncClient(base_url="http://localhost:8813") as client:
        concept_id = "test_concept_trainer_1"
        nn_response = np.random.randn(768).tolist()
        mistral_response = np.random.randn(768).tolist()
        payload = {
            "concept_id": concept_id,
            "nn_response": nn_response,
            "mistral_response": mistral_response,
            "confidence_delta": 0.5,
            "feedback_score": 0.8
        }
        response = await client.post("/train", json=payload)
        print("/train status:", response.status_code)
        print(json.dumps(response.json(), indent=2))
        assert response.status_code == 200

async def test_train_batch_endpoint():
    async with httpx.AsyncClient(base_url="http://localhost:8813") as client:
        batch = {
            "vectors": [
                {"embedding": np.random.randn(768).tolist(), "metadata": {"concept_id": f"batch_concept_{i}"}} for i in range(5)
            ],
            "batch_id": "batch_test_1",
            "metadata": {}
        }
        response = await client.post("/train/batch", json=batch)
        print("/train/batch status:", response.status_code)
        print(json.dumps(response.json(), indent=2))
        assert response.status_code == 200 or response.status_code == 207

async def test_health_endpoint():
    async with httpx.AsyncClient(base_url="http://localhost:8813") as client:
        response = await client.get("/health")
        print("/health status:", response.status_code)
        print(json.dumps(response.json(), indent=2))
        assert response.status_code == 200

async def main():
    await test_train_endpoint()
    await test_train_batch_endpoint()
    await test_health_endpoint()

if __name__ == "__main__":
    asyncio.run(main()) 