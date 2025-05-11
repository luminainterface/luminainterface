from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import os
from typing import List, Optional, Dict
import logging
from prometheus_client import Counter, Histogram
import time
import httpx
import asyncio
from bs4 import BeautifulSoup
import aiohttp

# Initialize FastAPI app
app = FastAPI(title="Crawler Service")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis client
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url)

# Prometheus metrics
crawl_requests = Counter('crawl_requests_total', 'Total number of crawl requests')
crawl_latency = Histogram('crawl_latency_seconds', 'Time spent crawling')
pages_crawled = Counter('pages_crawled_total', 'Total number of pages crawled')

class CrawlRequest(BaseModel):
    url: str
    depth: Optional[int] = 1
    max_pages: Optional[int] = 10

class CrawlResponse(BaseModel):
    url: str
    title: str
    content: str
    links: List[str]

@app.get("/health")
async def health_check():
    try:
        redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

async def fetch_page(url: str) -> Optional[Dict]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title = soup.title.string if soup.title else ""
                    
                    # Extract main content (simplified)
                    content = soup.get_text(separator=' ', strip=True)
                    
                    # Extract links
                    links = [a.get('href') for a in soup.find_all('a', href=True)]
                    
                    return {
                        "url": url,
                        "title": title,
                        "content": content,
                        "links": links
                    }
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None

@app.post("/crawl")
async def crawl(request: CrawlRequest):
    with crawl_latency.time():
        crawl_requests.inc()
        try:
            # Fetch the initial page
            page_data = await fetch_page(request.url)
            if not page_data:
                raise HTTPException(status_code=404, detail="Failed to fetch page")
            
            # Store in Redis
            key = f"crawl:{request.url}"
            redis_client.setex(key, 3600, json.dumps(page_data))  # 1 hour TTL
            
            pages_crawled.inc()
            return CrawlResponse(**page_data)
            
        except Exception as e:
            logger.error(f"Error crawling {request.url}: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8400) 