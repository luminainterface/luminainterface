from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import asyncio
from datetime import datetime, timedelta
import json
import uuid
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import requests
from redis import Redis
from neo4j import GraphDatabase
from fastapi.responses import JSONResponse
from lumina_core.common.bus import BusClient
import aiohttp

from .api.router import router, crawler
from .core.smart_crawler import SmartCrawler
from .core.graph_processor import GraphProcessor
from .core.ml_bridge import MLBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Smart Crawler Service",
    description="A service for crawling Wikipedia pages and building a knowledge graph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix="/api/v1")

# Global variables for background tasks
background_tasks = set()
PROCESS_INTERVAL = int(os.getenv("PROCESS_INTERVAL", "3600"))  # Default: 1 hour
INQUIRY_CHECK_INTERVAL = int(os.getenv("INQUIRY_CHECK_INTERVAL", "300"))  # Default: 5 minutes

ml_bridge = MLBridge()
RETRAIN_INTERVAL = int(os.getenv("RETRAIN_INTERVAL", "100"))  # Retrain after N samples
ml_sample_counter = 0

# MLBridge metrics
MLBRIDGE_SAMPLES = Counter("mlbridge_samples_total", "Total MLBridge training samples")
MLBRIDGE_RETRAINS = Counter("mlbridge_retrains_total", "Total MLBridge retrain events")
MLBRIDGE_LAST_RETRAIN = Gauge("mlbridge_last_retrain_timestamp", "Timestamp of last MLBridge retrain")
MLBRIDGE_LAST_ERROR = Gauge("mlbridge_last_error", "1 if last MLBridge operation errored, else 0")
MLBRIDGE_LAST_ERROR_MSG = None  # Store last error message (not a Prometheus metric)

GRAPH_API_KEY = os.getenv("GRAPH_API_KEY", "changeme")
CONCEPT_DICT_API_KEY = os.getenv("CONCEPT_DICT_API_KEY", "changeme")

# Add Prometheus metric for fetch time
CRAWLER_FETCH_SECONDS = Histogram("crawler_fetch_seconds", "Time spent fetching URLs in crawler")

# New: BusClient for ingest.queue
bus = BusClient(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))

async def process_inquiries():
    """Background task to process crawl requests with system priority and a 10s pause, with MLBridge hooks and robust debugging."""
    global ml_sample_counter, MLBRIDGE_LAST_ERROR_MSG
    while True:
        try:
            logger.info("Checking for system crawl requests...")
            request = await crawler.redis_client.get_next_crawl_request(source="system")
            if not request:
                logger.info("No system crawl found, checking for graph crawl requests...")
                request = await crawler.redis_client.get_next_crawl_request(source="graph")

            if request:
                concept = request.get('concept')
                inquiry_id = request.get('inquiry_id')
                source = request.get('source', 'unknown')
                logger.info(f"Processing {source} crawl for concept: {concept}")
                try:
                    # --- MLBridge: Prepare sample ---
                    crawl_sample = {
                        'vector': [0.0],  # TODO: Replace with actual vector
                        'usage': 0,       # TODO: Replace with actual usage
                        'last_crawled': 0, # TODO: Replace with actual timestamp
                        'label': 1.0 if source == 'system' else 0.5
                    }
                    output_sample = {
                        'vector': [0.0],  # TODO: Replace with actual output vector
                        'context': [0.0], # TODO: Replace with actual context
                        'prev_output': [0.0], # TODO: Replace with previous output
                        'label': 1.0 if source == 'system' else 0.5
                    }
                    crawl_priority_score = 0.5  # TODO: Replace with actual model score
                    output_score = 0.5          # TODO: Replace with actual model score

                    # Add samples to MLBridge
                    ml_bridge.add_priority_sample(crawl_sample, output_score=output_score)
                    ml_bridge.add_output_sample(output_sample, priority_score=crawl_priority_score)
                    ml_sample_counter += 1
                    MLBRIDGE_SAMPLES.inc()

                    # --- End MLBridge sample ---
                    success = await crawler.crawl(concept)
                    if success:
                        if inquiry_id:
                            await crawler.redis_client.update_inquiry_status(inquiry_id, "completed")
                        logger.info(f"Successfully processed {source} inquiry for concept: {concept}")
                    else:
                        if inquiry_id:
                            await crawler.redis_client.update_inquiry_status(inquiry_id, "failed")
                        await crawler.redis_client.add_to_dead_letter_queue(concept, "Crawl failed")
                        logger.error(f"Failed to crawl concept: {concept}")

                    # --- MLBridge: Retrain periodically ---
                    if ml_sample_counter % RETRAIN_INTERVAL == 0:
                        logger.info("[MLBridge] Retraining models after %d samples...", ml_sample_counter)
                        try:
                            ml_bridge.cross_train(epochs=5, lr=1e-3)
                            MLBRIDGE_RETRAINS.inc()
                            MLBRIDGE_LAST_RETRAIN.set(time.time())
                            MLBRIDGE_LAST_ERROR.set(0)
                            MLBRIDGE_LAST_ERROR_MSG = None
                        except Exception as ml_e:
                            logger.error(f"[MLBridge] Error during cross-training: {ml_e}", exc_info=True)
                            MLBRIDGE_LAST_ERROR.set(1)
                            MLBRIDGE_LAST_ERROR_MSG = str(ml_e)
                except Exception as e:
                    logger.error(f"Error processing {source} inquiry for concept {concept}: {e}", exc_info=True)
                    if inquiry_id:
                        await crawler.redis_client.update_inquiry_status(inquiry_id, "error")
                    await crawler.redis_client.add_to_dead_letter_queue(concept, str(e))
                    MLBRIDGE_LAST_ERROR.set(1)
                    MLBRIDGE_LAST_ERROR_MSG = str(e)
            else:
                logger.info("No crawl requests found. Waiting for next cycle.")
        except Exception as e:
            logger.error(f"Error in inquiry processing: {e}", exc_info=True)
            MLBRIDGE_LAST_ERROR.set(1)
            MLBRIDGE_LAST_ERROR_MSG = str(e)
        await asyncio.sleep(10)  # Natural 10 second pause between each crawl

async def process_graph_concepts():
    """Process concepts from the graph when no inquiries are pending"""
    try:
        graph_path = os.getenv("TRAINING_DATA_PATH", "/training_data/graph (1).json")
        if not os.path.exists(graph_path):
            logger.warning("Graph file not found at " + graph_path + " (using fallback).")
            graph_path = "/training_data/graph (1).json"
        if not os.path.exists(graph_path):
            logger.warning("Fallback graph file " + graph_path + " not found, cannot load graph.")
            return

        processor = GraphProcessor(crawler)
        if processor.load_graph(graph_path):
            concepts = processor.get_unprocessed_concepts()
            logger.info(f"Graph loaded. {len(concepts)} unprocessed concepts found.")
            for concept in concepts:
                logger.info(f"Adding graph concept to crawl queue: {concept}")
                await crawler.redis_client.add_to_crawl_queue(
                    concept=concept,
                    weight=0.3,  # Lower weight for graph-based concepts
                    source="graph"
                )
        else:
            logger.warning("Graph could not be loaded or was empty.")
    except Exception as e:
        logger.error(f"Error processing graph concepts: {e}")

async def start_background_tasks():
    """Start background tasks"""
    # Start inquiry processing task
    inquiry_task = asyncio.create_task(process_inquiries())
    background_tasks.add(inquiry_task)
    inquiry_task.add_done_callback(background_tasks.discard)

async def handle_ingest_queue(msg):
    data = msg.data
    if data.get("type") != "url":
        return
    url = data["payload"]
    fp = data["fp"]
    ts = int(time.time())
    try:
        with CRAWLER_FETCH_SECONDS.time():
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as resp:
                    html = await resp.text()
        # Publish to ingest.raw_html
        payload = {"url": url, "html": html, "fp": fp, "ts": ts}
        await bus.publish("ingest.raw_html", payload)
        logger.info(f"Fetched and published {url} (fp={fp}) to ingest.raw_html")
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")

async def start_ingest_queue_consumer():
    await bus.connect()
    await bus.consume(
        stream="ingest.queue",
        group="workers",
        consumer="crawler",
        handler=handle_ingest_queue,
        block_ms=1000,
        count=1
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the crawler on startup"""
    try:
        # Get configuration from environment variables
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        graph_api_url = os.getenv("GRAPH_API_URL", "http://graph-api:8200")
        concept_dict_url = os.getenv("CONCEPT_DICT_URL", "http://concept-dict:8000")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        # Initialize smart crawler
        global crawler
        crawler = SmartCrawler(
            redis_url=redis_url,
            qdrant_url=qdrant_url,
            graph_api_url=graph_api_url,
            concept_dict_url=concept_dict_url,
            embedding_model=embedding_model,
            max_depth=int(os.getenv("WIKI_SEARCH_DEPTH", "3")),
            max_links_per_page=int(os.getenv("WIKI_MAX_RESULTS", "15")),
            min_relevance_score=float(os.getenv("MIN_RELEVANCE_SCORE", "0.6")),
            max_concurrent_crawls=int(os.getenv("MAX_CONCURRENT_CRAWLS", "5"))
        )
        
        logger.info("Smart crawler service initialized successfully")

        # Start background tasks
        await start_background_tasks()
        asyncio.create_task(start_ingest_queue_consumer())
        
    except Exception as e:
        logger.error(f"Failed to initialize crawler service: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down crawler service") 
    # Cancel all background tasks
    for task in background_tasks:
        task.cancel()
    # Wait for all tasks to complete
    await asyncio.gather(*background_tasks, return_exceptions=True)

@app.get("/health")
async def health():
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

@app.get("/mlbridge/health")
async def mlbridge_health():
    """Health endpoint for MLBridge."""
    status = {
        "status": "ok",
        "priority_samples": len(ml_bridge.priority_training_data),
        "output_samples": len(ml_bridge.output_training_data),
        "retrain_count": MLBRIDGE_RETRAINS._value.get(),
        "last_retrain": MLBRIDGE_LAST_RETRAIN._value.get(),
        "last_error": MLBRIDGE_LAST_ERROR._value.get(),
        "last_error_msg": MLBRIDGE_LAST_ERROR_MSG,
    }
    return status

@app.get("/mlbridge/metrics")
def mlbridge_metrics():
    """Prometheus metrics endpoint for MLBridge."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST) 