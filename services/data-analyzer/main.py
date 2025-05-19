import os
import httpx
import logging
import numpy as np
import asyncio
import traceback
import json
from typing import Dict, List, Any, Optional, Set
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import time
from datetime import datetime, timedelta

# Environment variables
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8000")
OUTPUT_ENGINE_URL = os.getenv("OUTPUT_ENGINE_URL", "http://output-engine:9000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
CRAWLER_URL = os.getenv("CRAWLER_URL", "http://crawler:7000")
API_KEY = os.getenv("ANALYZER_API_KEY", "changeme")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "admin_key_change_me")  # For sync operations
DEBUG_NO_OLLAMA = os.getenv("DEBUG_NO_OLLAMA", "0") == "1"
REDIS_URL = os.getenv("REDIS_URL", "redis://:02211998@redis:6379")

# Queue configuration
QUEUE_KEY = "analyzer:request_queue"
QUEUE_PROCESSING_KEY = "analyzer:processing_queue"
QUEUE_RESULT_KEY = "analyzer:result_queue"
MAX_QUEUE_SIZE = 1000
MAX_PROCESSING_TIME = 300  # 5 minutes
QUEUE_CHECK_INTERVAL = 1  # 1 second
MAX_CONCURRENT_REQUESTS = 5

# Cache configuration
CACHE_TTL = 3600  # 1 hour
CACHE_PREFIX = "analyzer:cache:"
FREQUENT_ACCESS_THRESHOLD = 10  # Number of accesses to consider a concept "frequently accessed"

# Sync configuration
SYNC_LOCK_KEY = "analyzer:sync_lock"
SYNC_STATUS_KEY = "analyzer:sync_status"
SYNC_HISTORY_KEY = "analyzer:sync_history"
SYNC_LOCK_TIMEOUT = 300  # 5 minutes
SYNC_CHECK_INTERVAL = 10  # 10 seconds
MAX_SYNC_RETRIES = 3

# Setup logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("data_analyzer")
logger.setLevel(logging.DEBUG)

app = FastAPI(title="Data Analyzer Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis client
import aioredis
redis_client = None

async def get_redis():
    global redis_client
    if redis_client is None:
        redis_client = await aioredis.from_url(REDIS_URL)
    return redis_client

class QueueItem(BaseModel):
    request_id: str
    term: str
    definition: str
    priority: int = 1
    timestamp: float
    status: str = "queued"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AnalyzedConcept(BaseModel):
    term: str
    definition: str
    insights: List[str]
    narrative: str
    related_concepts: List[Dict[str, Any]] = []
    embedding: List[float] = []

class ConceptAnalysis(BaseModel):
    concepts: List[AnalyzedConcept]
    overall_narrative: str

class ConceptRequest(BaseModel):
    term: str
    definition: str
    priority: int = 1

class SyncStatus(BaseModel):
    status: str
    start_time: float
    end_time: Optional[float] = None
    services: Dict[str, str]  # service -> status
    errors: List[str] = []
    affected_concepts: int = 0
    sync_id: str

class SyncRequest(BaseModel):
    force: bool = False
    services: List[str] = ["concept-dict", "crawler", "ollama"]
    dry_run: bool = False

class ServiceStatus(BaseModel):
    service: str
    status: str
    last_sync: Optional[float] = None
    error_count: int = 0
    concept_count: Optional[int] = None

# API key security
api_key_header = APIKeyHeader(name="X-API-Key")
admin_key_header = APIKeyHeader(name="X-Admin-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

async def verify_admin_key(admin_key: str = Depends(admin_key_header)):
    if admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return admin_key

async def check_service_health(service_url: str) -> bool:
    """Check if a service is healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{service_url}/health")
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed for {service_url}: {e}")
        return False

async def get_service_concept_count(service_url: str) -> Optional[int]:
    """Get the number of concepts in a service."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{service_url}/stats")
            if response.status_code == 200:
                data = response.json()
                return data.get("concept_count")
    except Exception as e:
        logger.error(f"Failed to get concept count from {service_url}: {e}")
    return None

async def trigger_service_sync(service_url: str, sync_id: str) -> bool:
    """Trigger synchronization for a specific service."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{service_url}/sync",
                json={"sync_id": sync_id},
                headers={"X-Admin-Key": ADMIN_API_KEY}
            )
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to trigger sync for {service_url}: {e}")
        return False

async def acquire_sync_lock() -> bool:
    """Try to acquire the sync lock."""
    redis = await get_redis()
    return await redis.set(
        SYNC_LOCK_KEY,
        str(time.time()),
        ex=SYNC_LOCK_TIMEOUT,
        nx=True  # Only set if not exists
    )

async def release_sync_lock():
    """Release the sync lock."""
    redis = await get_redis()
    await redis.delete(SYNC_LOCK_KEY)

async def update_sync_status(sync_id: str, status: Dict[str, Any]):
    """Update the current sync status."""
    redis = await get_redis()
    await redis.set(
        SYNC_STATUS_KEY,
        json.dumps(status),
        ex=SYNC_LOCK_TIMEOUT
    )
    # Add to history
    await redis.lpush(SYNC_HISTORY_KEY, json.dumps({
        "sync_id": sync_id,
        "timestamp": time.time(),
        **status
    }))
    # Keep only last 10 sync records
    await redis.ltrim(SYNC_HISTORY_KEY, 0, 9)

async def perform_sync(sync_id: str, services: List[str], dry_run: bool = False) -> Dict[str, Any]:
    """Perform synchronization between services."""
    redis = await get_redis()
    start_time = time.time()
    status = {
        "status": "in_progress",
        "start_time": start_time,
        "services": {},
        "errors": [],
        "affected_concepts": 0,
        "sync_id": sync_id
    }
    
    try:
        # Update initial status
        await update_sync_status(sync_id, status)
        
        # Check service health
        service_urls = {
            "concept-dict": CONCEPT_DICT_URL,
            "crawler": CRAWLER_URL,
            "ollama": OLLAMA_URL
        }
        
        for service in services:
            if service not in service_urls:
                status["errors"].append(f"Unknown service: {service}")
                continue
                
            url = service_urls[service]
            is_healthy = await check_service_health(url)
            status["services"][service] = "healthy" if is_healthy else "unhealthy"
            
            if not is_healthy:
                status["errors"].append(f"Service {service} is unhealthy")
                continue
            
            if dry_run:
                # Just get concept counts
                count = await get_service_concept_count(url)
                if count is not None:
                    status["affected_concepts"] = max(status["affected_concepts"], count)
            else:
                # Trigger actual sync
                success = await trigger_service_sync(url, sync_id)
                status["services"][service] = "synced" if success else "failed"
                if not success:
                    status["errors"].append(f"Failed to sync {service}")
        
        # Update final status
        status["status"] = "completed" if not status["errors"] else "failed"
        status["end_time"] = time.time()
        await update_sync_status(sync_id, status)
        
        return status
        
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        status["status"] = "failed"
        status["end_time"] = time.time()
        status["errors"].append(str(e))
        await update_sync_status(sync_id, status)
        return status

@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection and start queue processor."""
    await get_redis()
    asyncio.create_task(process_queue())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup Redis connection."""
    if redis_client:
        await redis_client.close()

async def add_to_queue(request: ConceptRequest) -> str:
    """Add a request to the processing queue."""
    redis = await get_redis()
    request_id = f"req_{int(time.time() * 1000)}_{hash(request.term)}"
    
    queue_item = QueueItem(
        request_id=request_id,
        term=request.term,
        definition=request.definition,
        priority=request.priority,
        timestamp=time.time()
    )
    
    # Check queue size
    queue_size = await redis.llen(QUEUE_KEY)
    if queue_size >= MAX_QUEUE_SIZE:
        raise HTTPException(status_code=429, detail="Queue is full. Please try again later.")
    
    # Add to queue with priority score (higher priority = lower score)
    score = time.time() - (request.priority * 1000)  # Higher priority items get processed first
    await redis.zadd(QUEUE_KEY, {json.dumps(queue_item.dict()): score})
    
    return request_id

async def get_queue_status(request_id: str) -> Dict[str, Any]:
    """Get the status of a queued request."""
    redis = await get_redis()
    
    # Check processing queue
    processing_item = await redis.hget(QUEUE_PROCESSING_KEY, request_id)
    if processing_item:
        return json.loads(processing_item)
    
    # Check result queue
    result = await redis.hget(QUEUE_RESULT_KEY, request_id)
    if result:
        return json.loads(result)
    
    # Check main queue
    queue_items = await redis.zrange(QUEUE_KEY, 0, -1, withscores=True)
    for item, _ in queue_items:
        item_data = json.loads(item)
        if item_data["request_id"] == request_id:
            return item_data
    
    raise HTTPException(status_code=404, detail="Request not found in queue")

async def process_queue():
    """Background task to process the request queue."""
    while True:
        try:
            redis = await get_redis()
            
            # Get number of currently processing items
            processing_count = await redis.hlen(QUEUE_PROCESSING_KEY)
            if processing_count >= MAX_CONCURRENT_REQUESTS:
                await asyncio.sleep(QUEUE_CHECK_INTERVAL)
                continue
            
            # Get next item from queue
            items = await redis.zrange(QUEUE_KEY, 0, 0, withscores=True)
            if not items:
                await asyncio.sleep(QUEUE_CHECK_INTERVAL)
                continue
            
            item_data = json.loads(items[0][0])
            request_id = item_data["request_id"]
            
            # Move to processing queue
            await redis.zrem(QUEUE_KEY, items[0][0])
            item_data["status"] = "processing"
            item_data["processing_start"] = time.time()
            await redis.hset(QUEUE_PROCESSING_KEY, request_id, json.dumps(item_data))
            
            # Process in background
            asyncio.create_task(process_request(request_id, item_data))
            
        except Exception as e:
            logger.error(f"Error in queue processor: {e}")
            await asyncio.sleep(QUEUE_CHECK_INTERVAL)

async def process_request(request_id: str, item_data: Dict[str, Any]):
    """Process a single request from the queue."""
    redis = await get_redis()
    try:
        # Check cache first
        cache_key = f"{CACHE_PREFIX}{hash(item_data['term'])}"
        cached_result = await redis.get(cache_key)
        
        if cached_result:
            # Update access count for frequently accessed concepts
            access_key = f"{CACHE_PREFIX}access:{hash(item_data['term'])}"
            access_count = await redis.incr(access_key)
            await redis.expire(access_key, CACHE_TTL)
            
            result = json.loads(cached_result)
            item_data["status"] = "completed"
            item_data["result"] = result
            item_data["from_cache"] = True
        else:
            # Process the request
            result = await analyze_concept_internal(
                ConceptRequest(
                    term=item_data["term"],
                    definition=item_data["definition"]
                )
            )
            
            # Cache the result
            await redis.set(cache_key, json.dumps(result), ex=CACHE_TTL)
            
            item_data["status"] = "completed"
            item_data["result"] = result
            item_data["from_cache"] = False
        
        # Move to result queue
        await redis.hdel(QUEUE_PROCESSING_KEY, request_id)
        await redis.hset(QUEUE_RESULT_KEY, request_id, json.dumps(item_data))
        
        # Cleanup old results
        await cleanup_old_results()
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        item_data["status"] = "failed"
        item_data["error"] = str(e)
        await redis.hdel(QUEUE_PROCESSING_KEY, request_id)
        await redis.hset(QUEUE_RESULT_KEY, request_id, json.dumps(item_data))

async def cleanup_old_results():
    """Clean up old results from the result queue."""
    redis = await get_redis()
    try:
        # Get all results
        results = await redis.hgetall(QUEUE_RESULT_KEY)
        now = time.time()
        
        for request_id, result_data in results.items():
            result = json.loads(result_data)
            if now - result["timestamp"] > MAX_PROCESSING_TIME:
                await redis.hdel(QUEUE_RESULT_KEY, request_id)
    except Exception as e:
        logger.error(f"Error cleaning up old results: {e}")

async def analyze_concept_internal(request: ConceptRequest) -> Dict[str, Any]:
    """Internal function to analyze a concept without queue management."""
    try:
        logger.info(f"Received analysis request for concept: {request.term}")
        
        # Create a single concept for analysis
        concept = {
            "term": request.term,
            "definition": request.definition
        }
        
        # Generate insights and narrative in a single Ollama call
        prompt = f"""You are a precise and structured AI assistant. Your task is to analyze concepts and provide insights and a narrative in a very specific format.

        SYSTEM: You MUST follow the exact format below. Do not add any other text, sections, or explanations. The response must start with "INSIGHTS:" and contain exactly 3 insights, followed by "NARRATIVE:" and the narrative.

        Concept: {concept['term']}
        Definition: {concept['definition']}
        
        RESPONSE FORMAT (copy this exactly):
        INSIGHTS:
        - [First insight]
        - [Second insight]
        - [Third insight]
        
        NARRATIVE:
        [Your narrative here]

        CRITICAL FORMAT RULES:
        1. Start with "INSIGHTS:" (exactly as shown)
        2. List exactly 3 insights, each starting with "- " (exactly as shown)
        3. Then write "NARRATIVE:" (exactly as shown)
        4. Write your narrative after that
        5. Do not add any other sections, text, or explanations
        6. Do not add any markdown formatting
        7. Do not add any line breaks between insights
        8. Do not add any line breaks before or after NARRATIVE:"""
        
        print("[DEBUG] Requesting Ollama completion...")
        logger.debug("Requesting Ollama completion")
        response_text = await get_ollama_completion(prompt)
        print(f"[DEBUG] Raw Ollama response: {response_text}")
        logger.debug(f"Raw Ollama response (from analyze_concept): {response_text}")
        
        # Parse the response with more robust error handling
        try:
            # Split on NARRATIVE: and ensure we have exactly two parts
            parts = response_text.split("NARRATIVE:")
            if len(parts) != 2:
                raise ValueError("Response missing NARRATIVE section")
                
            # Extract and clean insights
            insights_part = parts[0].replace("INSIGHTS:", "").strip()
            insights = []
            for line in insights_part.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    insights.append(line[2:].strip())
            
            if len(insights) != 3:
                raise ValueError(f"Expected exactly 3 insights, got {len(insights)}")
            
            # Extract and clean narrative
            narrative = parts[1].strip()
            if not narrative:
                raise ValueError("Empty narrative in response")
            
            logger.debug(f"Successfully parsed response with {len(insights)} insights")
            
            # Create analyzed concept
            analyzed_concept = AnalyzedConcept(
                term=concept["term"],
                definition=concept["definition"],
                insights=insights,
                narrative=narrative,
                related_concepts=[],
                embedding=[]
            )
            
            # Create analysis with single concept
            analysis = ConceptAnalysis(
                concepts=[analyzed_concept],
                overall_narrative=narrative
            )
            
            # Send to output engine
            success = await send_to_output_engine(analysis)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to send to output engine")
            
            logger.info(f"Successfully analyzed concept: {request.term}")
            return {
                "status": "success",
                "message": "Concept analyzed and sent to output engine",
                "analysis": analysis.dict()
            }
        except ValueError as e:
            logger.error(f"Invalid response format: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_concept(request: ConceptRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Queue a concept analysis request."""
    try:
        request_id = await add_to_queue(request)
        return {
            "status": "queued",
            "request_id": request_id,
            "message": "Request queued for processing"
        }
    except Exception as e:
        logger.error(f"Error queueing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{request_id}")
async def get_status(request_id: str) -> Dict[str, Any]:
    """Get the status of a queued request."""
    try:
        return await get_queue_status(request_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting request status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue/stats")
async def get_queue_stats() -> Dict[str, Any]:
    """Get current queue statistics."""
    redis = await get_redis()
    try:
        queue_size = await redis.zcard(QUEUE_KEY)
        processing_count = await redis.hlen(QUEUE_PROCESSING_KEY)
        result_count = await redis.hlen(QUEUE_RESULT_KEY)
        
        return {
            "queue_size": queue_size,
            "processing_count": processing_count,
            "result_count": result_count,
            "max_queue_size": MAX_QUEUE_SIZE,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
        }
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

async def get_ollama_completion(prompt: str, max_retries: int = 3) -> str:
    print("[DEBUG] Entered get_ollama_completion")
    if DEBUG_NO_OLLAMA:
        # Return a stubbed response for testing
        return "INSIGHTS:\n- This is a stub insight 1\n- This is a stub insight 2\n- This is a stub insight 3\n\nNARRATIVE:\nThis is a stub narrative for testing."
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting Ollama completion (attempt {attempt + 1}/{max_retries})")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()["response"]
                print(f"[DEBUG] Raw Ollama response (from get_ollama_completion): {result}")
                logger.debug(f"Raw Ollama response: {result}")
                logger.debug(f"Ollama completion successful (attempt {attempt + 1})")
                return result
        except Exception as e:
            logger.error(f"Ollama completion failed (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Failed to get completion after {max_retries} attempts: {str(e)}")
            await asyncio.sleep(1)  # Wait before retry

async def send_to_output_engine(analysis: ConceptAnalysis) -> bool:
    """Send analyzed data to the output engine."""
    try:
        logger.debug("Sending analysis to output engine")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OUTPUT_ENGINE_URL}/process",
                json=analysis.dict(),
                headers={"X-API-Key": API_KEY}
            )
            response.raise_for_status()
            logger.debug("Successfully sent to output engine")
            return True
    except Exception as e:
        logger.error(f"Failed to send to output engine: {str(e)}")
        return False

@app.post("/sync/trigger")
async def trigger_sync(
    request: SyncRequest,
    admin_key: str = Depends(verify_admin_key)
) -> Dict[str, Any]:
    """Trigger synchronization between services."""
    try:
        # Generate sync ID
        sync_id = f"sync_{int(time.time() * 1000)}"
        
        # Try to acquire sync lock
        if not request.force and not await acquire_sync_lock():
            raise HTTPException(
                status_code=409,
                detail="Another sync operation is in progress"
            )
        
        try:
            # Start sync in background
            status = await perform_sync(sync_id, request.services, request.dry_run)
            return {
                "sync_id": sync_id,
                "status": status["status"],
                "message": "Sync operation started" if not request.dry_run else "Dry run completed",
                "services": status["services"],
                "affected_concepts": status["affected_concepts"]
            }
        finally:
            if not request.force:
                await release_sync_lock()
                
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to trigger sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sync/status/{sync_id}")
async def get_sync_status(
    sync_id: str,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Get the status of a sync operation."""
    redis = await get_redis()
    try:
        status_data = await redis.get(SYNC_STATUS_KEY)
        if not status_data:
            raise HTTPException(status_code=404, detail="Sync status not found")
        
        status = json.loads(status_data)
        if status["sync_id"] != sync_id:
            raise HTTPException(status_code=404, detail="Sync ID not found")
        
        return status
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to get sync status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sync/history")
async def get_sync_history(
    limit: int = 10,
    api_key: str = Depends(verify_api_key)
) -> List[Dict[str, Any]]:
    """Get sync operation history."""
    redis = await get_redis()
    try:
        history = await redis.lrange(SYNC_HISTORY_KEY, 0, limit - 1)
        return [json.loads(item) for item in history]
    except Exception as e:
        logger.error(f"Failed to get sync history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/services/status")
async def get_services_status(
    api_key: str = Depends(verify_api_key)
) -> List[ServiceStatus]:
    """Get the status of all services."""
    service_urls = {
        "concept-dict": CONCEPT_DICT_URL,
        "crawler": CRAWLER_URL,
        "ollama": OLLAMA_URL
    }
    
    statuses = []
    for service, url in service_urls.items():
        try:
            is_healthy = await check_service_health(url)
            concept_count = await get_service_concept_count(url)
            
            # Get last sync time from history
            redis = await get_redis()
            history = await redis.lrange(SYNC_HISTORY_KEY, 0, 0)
            last_sync = None
            if history:
                last_sync_data = json.loads(history[0])
                if last_sync_data["services"].get(service) == "synced":
                    last_sync = last_sync_data["timestamp"]
            
            statuses.append(ServiceStatus(
                service=service,
                status="healthy" if is_healthy else "unhealthy",
                last_sync=last_sync,
                concept_count=concept_count,
                error_count=len([s for s in statuses if s.service == service and s.status == "unhealthy"])
            ))
        except Exception as e:
            logger.error(f"Failed to get status for {service}: {e}")
            statuses.append(ServiceStatus(
                service=service,
                status="error",
                error_count=1
            ))
    
    return statuses

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500, log_level="debug") 