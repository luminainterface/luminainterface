import asyncio
import os
import json
import time
import httpx
import redis.asyncio as redis
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "unified_knowledge")
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://localhost:8000")

TEST_CONCEPT = "integration test concept"

async def publish_crawl_request():
    r = redis.Redis.from_url(REDIS_URL)
    msg = {"concept": TEST_CONCEPT, "priority": "high", "metadata": {"source": "test"}}
    await r.publish("lumina.graph.crawl", json.dumps(msg))
    await r.close()

async def wait_for_qdrant_entry(timeout=30):
    client = QdrantClient(url=QDRANT_URL)
    start = time.time()
    while time.time() - start < timeout:
        hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=[0.0]*384,  # dummy vector, will be ignored by filter
            limit=1,
            query_filter=Filter(
                must=[FieldCondition(key="concept", match=MatchValue(value=TEST_CONCEPT))]
            )
        )
        if hits:
            return True
        await asyncio.sleep(2)
    return False

async def wait_for_concept_dict_entry(timeout=30):
    async with httpx.AsyncClient() as client:
        start = time.time()
        while time.time() - start < timeout:
            resp = await client.get(f"{CONCEPT_DICT_URL}/concepts/{TEST_CONCEPT}")
            if resp.status_code == 200:
                return True
            await asyncio.sleep(2)
    return False

async def test_crawler_flow():
    print("Publishing crawl request...")
    await publish_crawl_request()
    print("Waiting for Qdrant entry...")
    qdrant_ok = await wait_for_qdrant_entry()
    print(f"Qdrant entry found: {qdrant_ok}")
    print("Waiting for Concept Dictionary entry...")
    dict_ok = await wait_for_concept_dict_entry()
    print(f"Concept Dictionary entry found: {dict_ok}")
    assert qdrant_ok or dict_ok, "Concept was not stored in Qdrant or Concept Dictionary!"
    print("Test passed!")

if __name__ == "__main__":
    asyncio.run(test_crawler_flow()) 