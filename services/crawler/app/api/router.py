from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import logging
import os
import time
from redis import Redis
from redis.exceptions import RedisError
from neo4j import GraphDatabase
import requests

from ..core.crawler import Crawler
from ..core.smart_crawler import SmartCrawler
from ..core.graph_processor import GraphProcessor

logger = logging.getLogger(__name__)

router = APIRouter()
crawler: Optional[SmartCrawler] = None

# Rate limiting setup
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds

class RateLimiter:
    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url, decode_responses=True)

    async def check_rate_limit(self, request: Union[Request, SearchRequest]) -> bool:
        # Get client IP from FastAPI Request object
        if isinstance(request, Request):
            client_ip = request.client.host
        else:
            # For SearchRequest, use a default key
            client_ip = "default"
            
        current = int(time.time())
        window_key = f"rate_limit:{client_ip}:{current // RATE_WINDOW}"
        
        try:
            count = self.redis.incr(window_key)
            if count == 1:
                self.redis.expire(window_key, RATE_WINDOW)
            return count <= RATE_LIMIT
        except RedisError as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow request if Redis fails

class CrawlRequest(BaseModel):
    start_title: str
    max_depth: Optional[int] = 2
    max_links_per_page: Optional[int] = 10

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    start_nodes: Optional[List[str]] = None
    depth: int = 2

class CrawlResponse(BaseModel):
    status: str
    message: str

class SearchResult(BaseModel):
    id: str
    title: str
    summary: str
    url: str
    score: float

class SearchResponse(BaseModel):
    results: List[Dict]
    cache_hits: int = 0
    total_requests: int = 0
    neo4j_nodes: int = 0
    neo4j_relationships: int = 0

class SmartCrawlRequest(BaseModel):
    start_title: str
    max_pages: Optional[int] = None
    min_relevance_score: Optional[float] = 0.6
    max_depth: Optional[int] = 3
    max_links_per_page: Optional[int] = 15

class CrawlStats(BaseModel):
    total_pages: int
    average_priority: float
    max_priority: float
    active_crawls: int
    cache_hits: Dict[str, int]

@router.post("/crawl", response_model=CrawlResponse)
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start a crawl from a given Wikipedia page"""
    if not crawler:
        raise HTTPException(status_code=503, detail="Crawler not initialized")
        
    try:
        # Add crawl task to background tasks
        background_tasks.add_task(crawler.crawl, request.start_title)
        return CrawlResponse(
            status="accepted",
            message=f"Started crawling from {request.start_title}"
        )
    except Exception as e:
        logger.error(f"Error starting crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search", response_model=List[SearchResult])
async def search_similar(query: str, limit: int = 5):
    """Search for similar concepts"""
    if not crawler:
        raise HTTPException(status_code=503, detail="Crawler not initialized")
        
    try:
        results = await crawler.search_similar(query, limit)
        return [
            SearchResult(
                id=result["id"],
                title=result["payload"]["title"],
                summary=result["payload"]["summary"],
                url=result["payload"]["url"],
                score=result["score"]
            )
            for result in results
        ]
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process", response_model=SearchResponse)
async def process_graph(
    request: Request,
    search_request: SearchRequest,
    rate_limiter: RateLimiter = Depends(lambda: RateLimiter(os.getenv("REDIS_URL", "redis://localhost:6379")))
):
    """Process the graph and perform Wikipedia searches with rate limiting."""
    try:
        # Check rate limit using the FastAPI Request object
        if not await rate_limiter.check_rate_limit(request):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )

        graph_path = os.getenv("TRAINING_DATA_PATH")
        if not graph_path:
            raise HTTPException(
                status_code=500,
                detail="Training data path not configured"
            )

        processor = GraphProcessor(
            graph_path=graph_path,
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            neo4j_url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
            concept_dict_url=os.getenv("CONCEPT_DICT_URL", "http://localhost:8000")
        )
        
        try:
            processor.load_graph()
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to load graph data"
            )

        try:
            results = processor.process_graph(search_request.start_nodes, search_request.depth)
        except Exception as e:
            logger.error(f"Error processing graph: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process graph"
            )
        
        # Get Neo4j stats
        try:
            with processor.neo4j.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        except Exception as e:
            logger.error(f"Error getting Neo4j stats: {e}")
            node_count = 0
            rel_count = 0
        
        return SearchResponse(
            results=results,
            cache_hits=len(processor.processed_nodes) - len(results),
            total_requests=len(processor.processed_nodes),
            neo4j_nodes=node_count,
            neo4j_relationships=rel_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_graph: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint with Redis and Neo4j status."""
    try:
        # Check Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis = Redis.from_url(redis_url)
        redis.ping()
        
        # Check Neo4j
        neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        neo4j = GraphDatabase.driver(neo4j_url, auth=("neo4j", "password"))
        with neo4j.session() as session:
            session.run("RETURN 1")
        
        # Check Concept Dictionary
        concept_dict_url = os.getenv("CONCEPT_DICT_URL", "http://localhost:8000")
        response = requests.get(f"{concept_dict_url}/health")
        concept_dict_status = "healthy" if response.status_code == 200 else "unhealthy"
        
        return {
            "status": "healthy",
            "redis": "connected",
            "neo4j": "connected",
            "concept_dictionary": concept_dict_status,
            "graph_path": os.getenv("TRAINING_DATA_PATH", "not configured")
        }
    except RedisError as e:
        logger.error(f"Redis health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "redis": "disconnected",
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@router.post("/smart-crawl", response_model=CrawlResponse)
async def start_smart_crawl(request: SmartCrawlRequest, background_tasks: BackgroundTasks):
    """Start a smart crawl from a given Wikipedia page with adaptive depth and priority"""
    if not crawler:
        raise HTTPException(status_code=503, detail="Crawler not initialized")
        
    try:
        # Update crawler settings if provided
        if request.min_relevance_score is not None:
            crawler.min_relevance_score = request.min_relevance_score
        if request.max_depth is not None:
            crawler.max_depth = request.max_depth
        if request.max_links_per_page is not None:
            crawler.max_links_per_page = request.max_links_per_page
            
        # Add crawl task to background tasks
        background_tasks.add_task(crawler.crawl_smart, request.start_title, request.max_pages)
        return CrawlResponse(
            status="accepted",
            message=f"Started smart crawl from {request.start_title}"
        )
    except Exception as e:
        logger.error(f"Error starting smart crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=CrawlStats)
async def get_crawl_stats():
    """Get statistics about the current crawl process"""
    if not crawler:
        raise HTTPException(status_code=503, detail="Crawler not initialized")
        
    try:
        stats = await crawler.get_crawl_stats()
        return CrawlStats(**stats)
    except Exception as e:
        logger.error(f"Error getting crawl stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 