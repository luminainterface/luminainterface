from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel
import httpx

from .core.crawler import Crawler
from .core.training_crawler import TrainingCrawler
from .models.file_item import FileProcessingConfig, FileMetadata

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
            
            # Pull the model
            logger.info(f"Pulling model {model_name}...")
            response = await client.post(
                "/api/pull",
                json={"name": model_name}
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

@app.on_event("startup")
async def startup_event():
    """Initialize the crawler service on startup."""
    global crawler, training_crawler_singleton, training_crawler_task_started
    
    try:
        logger.info("Starting crawler service initialization...")
        
        # Ensure Ollama model is available
        if not await ensure_ollama_model():
            raise Exception("Failed to ensure Ollama model availability")
        
        # Initialize training crawler singleton
        logger.info("Initializing training data crawler singleton...")
        training_crawler_singleton = await TrainingCrawler.get_instance()
        await training_crawler_singleton.initialize()
        
        # Start the crawler
        logger.info("Starting crawler...")
        crawler = Crawler(
            redis_url=os.getenv("REDIS_URL", "redis://:02211998@redis:6379"),
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
            training_data_path=os.getenv("TRAINING_DATA_PATH", "/app/training_data"),
            config=FileProcessingConfig(
                chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
                batch_size=int(os.getenv("BATCH_SIZE", "32"))
            )
        )
        await crawler.initialize()
        
        # Start the crawl worker in the background
        asyncio.create_task(crawler.start_crawl_worker())
        
        logger.info("Crawler service started")
    except Exception as e:
        logger.error(f"Failed to initialize crawler service: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the crawler service on shutdown."""
    global crawler
    if crawler:
        await crawler.stop()
        logger.info("Crawler service stopped")

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    if not crawler:
        raise HTTPException(status_code=503, detail="Crawler service not initialized")
    
    try:
        status = await crawler.get_status()
        return {
            "status": "healthy" if status["worker_running"] else "degraded",
            "details": status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 