import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
from lumina_core.agents import get_agent

# Mock Wikipedia API responses
MOCK_WIKI_RESPONSE = {
    "extract": "Alan Turing was a British mathematician and computer scientist.",
    "content_urls": {
        "desktop": {
            "page": "https://en.wikipedia.org/wiki/Alan_Turing"
        }
    }
}

MOCK_WIKI_LINKS = {
    "links": [
        {"type": "article", "title": "Computer Science"},
        {"type": "article", "title": "Mathematics"}
    ]
}

@pytest.fixture
async def mock_wiki_client():
    """Mock the Wikipedia API client."""
    with patch("httpx.AsyncClient.get") as mock_get:
        # Mock article summary
        mock_get.return_value = AsyncMock(
            status_code=200,
            json=AsyncMock(return_value=MOCK_WIKI_RESPONSE)
        )
        yield mock_get

@pytest.mark.asyncio
async def test_full_flow(mock_wiki_client):
    """Test the complete wiki-qa workflow."""
    # 1. Test CrawlAgent
    crawl_agent = get_agent("crawl")
    crawl_result = await crawl_agent.run({
        "topic": "Alan Turing",
        "depth": 1
    })
    
    assert crawl_result["status"] == "success"
    assert len(crawl_result["articles"]) > 0
    assert "title" in crawl_result["articles"][0]
    assert "content" in crawl_result["articles"][0]
    
    # 2. Test SummariseAgent
    summarise_agent = get_agent("summarise")
    summarise_result = await summarise_agent.run({
        "articles": crawl_result["articles"]
    })
    
    assert summarise_result["status"] == "success"
    assert "summary" in summarise_result
    assert "facts" in summarise_result
    assert len(summarise_result["facts"]) > 0
    
    # 3. Test QAAgent
    qa_agent = get_agent("qa")
    qa_result = await qa_agent.run({
        "question": "Who was Alan Turing?"
    })
    
    assert qa_result["status"] == "success"
    assert "answer" in qa_result
    assert "sources" in qa_result
    assert len(qa_result["sources"]) > 0
    
    # 4. Test MasterChat integration
    async with AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/masterchat/plan",
            json={
                "mode": "wiki_qa",
                "question": "Who was Alan Turing?"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "Turing" in data["answer"]
        assert "sources" in data
``` 