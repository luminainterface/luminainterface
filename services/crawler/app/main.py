from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pydantic import BaseModel
import httpx
from sse_starlette.sse import EventSourceResponse
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http.models import VectorParams, Distance
from .core.embeddings import CustomOllamaEmbeddings

from .core.crawler import Crawler
from .core.training_crawler import TrainingCrawler
from .models.file_item import FileProcessingConfig, FileMetadata
from .core.ollama import ensure_ollama_model
from .core.redis_client import RedisClient
from .core.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Lumina Crawler Service",
    description="Service for crawling and processing training data files",
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

# Global variables for background tasks
background_tasks = set()
PROCESS_INTERVAL = int(os.getenv("PROCESS_INTERVAL", "3600"))  # Default: 1 hour
INQUIRY_CHECK_INTERVAL = int(os.getenv("INQUIRY_CHECK_INTERVAL", "300"))  # Default: 5 minutes

# Global crawler instance
crawler: Optional[Crawler] = None
# Singleton pattern for training crawler and its background task
training_crawler_singleton = None
training_crawler_task_started = False

# Global state
class CrawlerState:
    def __init__(self):
        self.crawler: Optional[TrainingCrawler] = None
        self.initialization_started: bool = False
        self.initialization_complete: bool = False
        self.last_error: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.initialization_timeout: timedelta = timedelta(minutes=5)  # 5 minute timeout

state = CrawlerState()

class CrawlerConfig(BaseModel):
    """Configuration for the crawler service."""
    redis_url: str = "redis://:02211998@redis:6379"
    qdrant_url: str = "http://qdrant:6333"
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "nomic-embed-text"
    training_data_path: str = "/app/training_data"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 32

async def ensure_ollama_model(model_name: str = "nomic-embed-text", base_url: str = "http://ollama:11434"):
    """Ensure the Ollama model is available."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        try:
            # Check if model exists
            response = await client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                if any(m.get("name") == model_name for m in models):
                    logger.info(f"Model {model_name} already exists")
                    return True
            
            # Pull the model with specific parameters
            logger.info(f"Pulling model {model_name}...")
            response = await client.post(
                "/api/pull",
                json={
                    "name": model_name,
                    "insecure": True  # Allow pulling from insecure registries
                }
            )
            if response.status_code == 200:
                logger.info(f"Successfully pulled model {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error ensuring Ollama model: {e}")
            return False

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint that reports crawler status."""
    current_time = datetime.utcnow()
    
    # If initialization hasn't started, return starting state
    if not state.initialization_started:
        return {
            "status": "starting",
            "message": "Crawler initialization not started",
            "uptime": 0
        }
    
    # If initialization has timed out
    if state.start_time and (current_time - state.start_time) > state.initialization_timeout:
        state.last_error = "Initialization timed out"
        return {
            "status": "error",
            "message": f"Initialization timed out after {state.initialization_timeout.total_seconds()} seconds",
            "error": state.last_error,
            "uptime": (current_time - state.start_time).total_seconds() if state.start_time else 0
        }
    
    # If initialization is in progress
    if not state.initialization_complete:
        return {
            "status": "initializing",
            "message": "Crawler is initializing",
            "uptime": (current_time - state.start_time).total_seconds() if state.start_time else 0
        }
    
    # If initialization failed
    if state.last_error:
        return {
            "status": "error",
            "message": "Crawler initialization failed",
            "error": state.last_error,
            "uptime": (current_time - state.start_time).total_seconds() if state.start_time else 0
        }
    
    # If crawler is initialized, get its status
    try:
        if not state.crawler:
            raise ValueError("Crawler instance is None")
            
        # Get crawler status
        status = await state.crawler.get_status()
        return {
            "status": "healthy" if status.get("is_healthy", False) else "degraded",
            "message": status.get("message", "Crawler is running"),
            "details": status,
            "uptime": (current_time - state.start_time).total_seconds() if state.start_time else 0
        }
    except Exception as e:
        state.last_error = str(e)
        return {
            "status": "error",
            "message": "Error getting crawler status",
            "error": str(e),
            "uptime": (current_time - state.start_time).total_seconds() if state.start_time else 0
        }

async def initialize_crawler():
    """Initialize the crawler in the background."""
    try:
        state.initialization_started = True
        state.start_time = datetime.utcnow()
        state.last_error = None
        
        # First ensure Ollama model is available
        logger.info("Ensuring Ollama model is available...")
        await ensure_ollama_model()
        
        # Initialize Redis client for health checks
        logger.info("Initializing Redis client...")
        redis_client = RedisClient()
        await redis_client.connect()
        
        # Initialize vector store for health checks
        logger.info("Initializing vector store...")
        vector_store = VectorStore(url="http://qdrant:6333")
        
        # Initialize the training data crawler singleton
        logger.info("Initializing training data crawler...")
        state.crawler = await TrainingCrawler.get_instance()
        
        # Start the crawler
        logger.info("Starting crawler process...")
        await state.crawler.start()
        
        state.initialization_complete = True
        logger.info("Crawler initialization completed successfully")
        
    except Exception as e:
        state.last_error = str(e)
        logger.error(f"Failed to initialize crawler: {e}", exc_info=True)
        # Don't set initialization_complete to True on error
        raise

@app.on_event("startup")
async def startup_event():
    """Startup event handler that initializes the crawler."""
    try:
        # Start initialization in the background
        asyncio.create_task(initialize_crawler())
    except Exception as e:
        logger.error(f"Error in startup event: {e}", exc_info=True)
        state.last_error = str(e)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler that stops the crawler."""
    try:
        if state.crawler:
            logger.info("Stopping crawler...")
            await state.crawler.stop()
            logger.info("Crawler stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping crawler: {e}", exc_info=True)
        raise

@app.post("/process")
async def process_file(file_path: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Process a single file."""
    if not crawler:
        raise HTTPException(status_code=503, detail="Crawler service not initialized")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    try:
        # Add file to crawl queue
        await crawler.redis.xadd(
            'crawl_queue',
            {
                'file_path': file_path,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        return {
            "status": "queued",
            "file_path": file_path,
            "message": "File added to processing queue"
        }
        
    except Exception as e:
        logger.error(f"Error queueing file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get detailed crawler status."""
    if not crawler:
        raise HTTPException(status_code=503, detail="Crawler service not initialized")
    
    try:
        return await crawler.get_status()
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/restart")
async def restart_crawler() -> Dict[str, Any]:
    """Restart the crawler service."""
    if not crawler:
        raise HTTPException(status_code=503, detail="Crawler service not initialized")
    
    try:
        await crawler.stop()
        await crawler.initialize()
        await crawler.start()
        return {"status": "restarted", "message": "Crawler service restarted successfully"}
    except Exception as e:
        logger.error(f"Error restarting crawler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training_crawler/health")
async def training_crawler_health() -> Dict[str, Any]:
    """Get training crawler health status."""
    if not training_crawler_singleton:
        raise HTTPException(status_code=503, detail="Training crawler not initialized")
    
    try:
        return {
            "status": "healthy",
            "initialized": True,
            "processed_files": len(training_crawler_singleton.processed_files),
            "failed_files": len(training_crawler_singleton.failed_files)
        }
    except Exception as e:
        logger.error(f"Training crawler health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/training_crawler/metrics")
async def training_crawler_metrics() -> Dict[str, Any]:
    """Get training crawler metrics."""
    if not training_crawler_singleton:
        raise HTTPException(status_code=503, detail="Training crawler not initialized")
    
    try:
        return {
            "processed_files": len(training_crawler_singleton.processed_files),
            "failed_files": len(training_crawler_singleton.failed_files),
            "queue_length": await training_crawler_singleton.redis.xlen('crawl_queue'),
            "dead_letter_length": await training_crawler_singleton.redis.xlen('crawl_dead_letter')
        }
    except Exception as e:
        logger.error(f"Error getting training crawler metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream_concepts")
async def stream_concepts_to_dictionary(batch_size: int = 100, poll_interval: float = 5.0):
    """
    Continuously stream new/updated concepts from Qdrant to the Concept Dictionary via both HTTP POST and Redis.
    """
    async def event_generator():
        last_sent_ts = None
        backoff = 1
        max_backoff = 60
        concept_dict_url = "http://concept-dictionary:8000/concepts/"  # Adjust port/path if needed
        redis_stream = "concept.new"
        while True:
            try:
                # Fetch new/updated concepts from Qdrant (implement this helper if needed)
                concepts = await crawler.qdrant.retrieve_new_concepts(last_sent_ts, limit=batch_size)
                if not concepts:
                    await asyncio.sleep(poll_interval)
                    continue
                for concept in concepts:
                    # Send via HTTP POST
                    try:
                        async with httpx.AsyncClient() as client:
                            resp = await client.put(f"{concept_dict_url}{concept['term']}", json=concept, timeout=10)
                            if resp.status_code == 200:
                                yield {"event": "http_success", "data": f"HTTP: {concept['term']} sent"}
                            else:
                                yield {"event": "http_error", "data": f"HTTP: {concept['term']} failed: {resp.text}"}
                    except Exception as e:
                        yield {"event": "http_error", "data": f"HTTP: {concept['term']} exception: {e}"}
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, max_backoff)
                        continue
                    # Send via Redis stream
                    try:
                        await crawler.redis.bus.publish(redis_stream, concept)
                        yield {"event": "redis_success", "data": f"Redis: {concept['term']} published"}
                    except Exception as e:
                        yield {"event": "redis_error", "data": f"Redis: {concept['term']} exception: {e}"}
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, max_backoff)
                        continue
                    # Reset backoff on success
                    backoff = 1
                    # Update last_sent_ts (assume concept has 'last_updated' or similar)
                    if 'last_updated' in concept:
                        last_sent_ts = concept['last_updated']
                await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                yield {"event": "fatal_error", "data": f"Exception: {e}"}
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
    return EventSourceResponse(event_generator())

class DictionaryCrawler:
    """Handles crawling and processing of dictionary concepts with legal compliance."""
    def __init__(self, ollama_url: str = "http://ollama:11434", model_name: str = "nomic-embed-text"):
        self.ollama = CustomOllamaEmbeddings(
            base_url=ollama_url,
            model=model_name
        )

class TrainingDataCrawler:
    """Handles crawling and processing of training data files."""
    def __init__(self):
        # Use nomic-embed-text model which produces 768-dim vectors
        self.ollama = CustomOllamaEmbeddings(
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            model=os.getenv("OLLAMA_MODEL", "nomic-embed-text")
        )

class Crawler:
    def __init__(self):
        self.redis = app_redis_client
        self.qdrant = app_qdrant_client
        # Use nomic-embed-text consistently across all crawlers
        self.embedding_model = CustomOllamaEmbeddings(
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            model=os.getenv("OLLAMA_MODEL", "nomic-embed-text")
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8400) 