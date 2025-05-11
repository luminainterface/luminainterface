import pytest, httpx, os, time

BASE = os.getenv("RAG_URL", "http://localhost:8000")

@pytest.mark.asyncio
async def test_rag():
    async with httpx.AsyncClient() as c:
        j = {"query": "What is artificial intelligence?", "top_k": 3}
        resp = await c.post(f"{BASE}/query", json=j, timeout=60)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "answer" in data and data["answer"]
        assert len(data["sources"]) > 0 