import pytest
from lumina_core.agents import get_agent

@pytest.mark.asyncio
async def test_crawl_agent_basic():
    """Test basic functionality of the CrawlAgent."""
    agent = get_agent("crawl")
    
    # Test with valid input
    result = await agent.run({
        "topic": "Alan Turing",
        "depth": 1
    })
    
    assert result["status"] == "success"
    assert "articles" in result
    assert len(result["articles"]) > 0
    
    # Verify article structure
    article = result["articles"][0]
    assert "title" in article
    assert "content" in article
    assert "embedding" in article
    assert len(article["embedding"]) == 384  # Mock embedding size

@pytest.mark.asyncio
async def test_crawl_agent_error_handling():
    """Test error handling in the CrawlAgent."""
    agent = get_agent("crawl")
    
    # Test with missing topic
    result = await agent.run({
        "depth": 1
    })
    
    assert result["status"] == "error"
    assert "error" in result
    assert "Topic is required" in result["error"]

@pytest.mark.asyncio
async def test_crawl_agent_depth():
    """Test that depth parameter is respected."""
    agent = get_agent("crawl")
    
    # Test with different depths
    for depth in [1, 2]:
        result = await agent.run({
            "topic": "Alan Turing",
            "depth": depth
        })
        
        assert result["status"] == "success"
        assert "articles" in result
        assert len(result["articles"]) > 0 