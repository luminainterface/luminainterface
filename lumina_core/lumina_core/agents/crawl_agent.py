from typing import Dict, Any, List
import asyncio
from loguru import logger
from prometheus_client import Counter, Histogram
from .base import BaseAgent
from ..wiki.crawler import WikiCrawler
from ..utils.rate_limit import redis_rate_limit

# Prometheus metrics
CRAWL_REQUESTS = Counter(
    "crawl_requests_total",
    "Total number of crawl requests",
    ["status"]
)
CRAWL_DURATION = Histogram(
    "crawl_duration_seconds",
    "Time spent crawling",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

class CrawlAgent(BaseAgent):
    """Agent for crawling Wikipedia articles."""
    
    def __init__(self):
        super().__init__(
            name="CrawlAgent",
            description="Crawls Wikipedia articles for a given topic"
        )
    
    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the crawl agent.
        
        Args:
            payload: Dictionary containing:
                - topic: str, The topic to crawl
                - depth: int, Optional, How deep to crawl (default: 2)
                
        Returns:
            Dictionary containing:
                - status: str, "success" or "error"
                - articles: List[Dict], List of articles if successful
                - error: str, Error message if failed
        """
        # Validate input
        if "topic" not in payload:
            CRAWL_REQUESTS.labels(status="error").inc()
            return {
                "status": "error",
                "error": "Topic is required"
            }
        
        # Get parameters
        topic = payload["topic"]
        depth = payload.get("depth", 2)
        
        try:
            # Rate limit by topic
            async with redis_rate_limit(f"crawl:{topic}", 5, 60):
                with CRAWL_DURATION.time():
                    # Crawl articles
                    async with WikiCrawler() as crawler:
                        articles = await crawler.crawl(topic, depth)
                    
                    # Log success
                    CRAWL_REQUESTS.labels(status="success").inc()
                    
                    return {
                        "status": "success",
                        "articles": articles
                    }
                    
        except Exception as e:
            # Log error
            logger.error(f"Crawl failed: {str(e)}")
            CRAWL_REQUESTS.labels(status="error").inc()
            
            return {
                "status": "error",
                "error": str(e)
            } 