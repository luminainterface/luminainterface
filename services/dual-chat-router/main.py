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

# Initialize FastAPI app
app = FastAPI(title="Dual Chat Router Service")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis client
redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
redis_client = redis.from_url(redis_url)

# Service URLs (use Docker Compose service names by default)
output_engine_url = os.getenv("OUTPUT_ENGINE_URL", "http://output-engine:8010")
ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")

# Prometheus metrics
chat_requests = Counter('chat_requests_total', 'Total number of chat requests')
chat_latency = Histogram('chat_latency_seconds', 'Time spent processing chat requests')
nn_responses = Counter('nn_responses_total', 'Total number of NN responses')
llm_responses = Counter('llm_responses_total', 'Total number of LLM responses')

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict] = None

class ChatResponse(BaseModel):
    response: str
    source: str  # "nn" or "llm"
    confidence: float
    concepts_used: Optional[List[str]] = None

async def get_nn_response(query: str, session_id: Optional[str] = None) -> ChatResponse:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{output_engine_url}/respond",
            json={"query": query, "session_id": session_id}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="NN service error")
        data = response.json()
        return ChatResponse(
            response=data["response"],
            source="nn",
            confidence=data["confidence"],
            concepts_used=data.get("concepts_used", [])
        )

async def get_llm_response(query: str, session_id: Optional[str] = None) -> ChatResponse:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "mistral",
                "prompt": query,
                "stream": False
            }
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="LLM service error")
        data = response.json()
        return ChatResponse(
            response=data["response"],
            source="llm",
            confidence=1.0  # LLM doesn't provide confidence scores
        )

@app.get("/health")
async def health_check():
    errors = {}
    # Retry logic for dependencies
    for attempt in range(3):
        try:
            # Redis
            try:
                redis_client.ping()
            except Exception as e:
                errors['redis'] = str(e)
            # NN service
            try:
                async with httpx.AsyncClient() as client:
                    nn_response = await client.get(f"{output_engine_url}/health", timeout=2)
                    if nn_response.status_code != 200:
                        raise Exception(f"Status {nn_response.status_code}")
            except Exception as e:
                errors['output_engine'] = str(e)
            # LLM service
            try:
                async with httpx.AsyncClient() as client:
                    llm_response = await client.get(f"{ollama_url}/api/tags", timeout=2)
                    if llm_response.status_code != 200:
                        raise Exception(f"Status {llm_response.status_code}")
            except Exception as e:
                errors['ollama'] = str(e)
            if not errors:
                return {"status": "healthy", "redis": "connected", "nn": "connected", "llm": "connected"}
        except Exception as e:
            pass
        if attempt < 2:
            logger.warning(f"Health check failed, retrying in 2s... Errors: {errors}")
            await asyncio.sleep(2)
    logger.error(f"Health check failed after retries: {errors}")
    raise HTTPException(status_code=503, detail={"status": "unhealthy", "errors": errors})

@app.post("/chat")
async def chat(request: ChatRequest):
    with chat_latency.time():
        chat_requests.inc()
        try:
            # Get responses from both services concurrently
            nn_task = asyncio.create_task(get_nn_response(request.query, request.session_id))
            llm_task = asyncio.create_task(get_llm_response(request.query, request.session_id))
            
            nn_response, llm_response = await asyncio.gather(nn_task, llm_task)
            
            # Choose the better response based on confidence
            if nn_response.confidence >= 0.7:
                nn_responses.inc()
                return nn_response
            else:
                llm_responses.inc()
                return llm_response
                
        except Exception as e:
            logger.error(f"Error processing chat request: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300) 