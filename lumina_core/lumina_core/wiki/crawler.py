import os
import asyncio
from typing import Dict, Any, List, Set
import httpx
from loguru import logger
from prometheus_client import Counter
from ..memory.qdrant_store import embed_text

# Prometheus metrics
pages_fetched_total = Counter(
    'pages_fetched_total',
    'Total number of Wikipedia pages fetched',
    ['status']
)

class WikiCrawler:
    """Wikipedia crawler that fetches articles and their embeddings."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.max_pages = int(os.getenv("MAX_PAGES", "10"))
        self.max_depth = int(os.getenv("MAX_DEPTH", "2"))
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page"
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def crawl(self, topic: str, depth: int) -> List[Dict[str, Any]]:
        """
        Crawl Wikipedia starting from a topic.
        
        Args:
            topic: The Wikipedia topic to start from
            depth: Maximum crawl depth
            
        Returns:
            List of dictionaries containing article data
        """
        visited = set()
        articles = []
        
        try:
            # Start with the initial topic
            queue = [(topic, 0)]  # (title, current_depth)
            
            while queue and len(articles) < self.max_pages:
                current_topic, current_depth = queue.pop(0)
                
                if current_topic in visited or current_depth > depth:
                    continue
                    
                visited.add(current_topic)
                
                # Fetch article content
                article = await self._fetch_article(current_topic)
                if article:
                    articles.append(article)
                    pages_fetched_total.labels(status="success").inc()
                    
                    # If we haven't reached max depth, get links
                    if current_depth < depth:
                        links = await self._get_links(current_topic)
                        for link in links:
                            if link not in visited:
                                queue.append((link, current_depth + 1))
                
        except Exception as e:
            logger.error(f"Error during crawl: {str(e)}")
            pages_fetched_total.labels(status="error").inc()
            
        return articles
    
    async def _fetch_article(self, title: str) -> Dict[str, Any]:
        """Fetch a single Wikipedia article."""
        try:
            # Get article content
            content_url = f"{self.base_url}/html/{title}"
            content_resp = await self.client.get(content_url)
            content_resp.raise_for_status()
            
            # Get article summary
            summary_url = f"{self.base_url}/summary/{title}"
            summary_resp = await self.client.get(summary_url)
            summary_resp.raise_for_status()
            summary = summary_resp.json()
            
            # Generate embedding
            text = summary.get("extract", "")
            embedding = await embed_text(text)
            
            return {
                "title": title,
                "url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "content": text,
                "embedding": embedding
            }
            
        except Exception as e:
            logger.error(f"Error fetching article {title}: {str(e)}")
            return None
    
    async def _get_links(self, title: str) -> Set[str]:
        """Get links from a Wikipedia article."""
        try:
            url = f"{self.base_url}/links/{title}"
            resp = await self.client.get(url)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract article titles from links
            links = set()
            for link in data.get("links", []):
                if link.get("type") == "article":
                    links.add(link.get("title"))
            
            return links
            
        except Exception as e:
            logger.error(f"Error getting links for {title}: {str(e)}")
            return set() 