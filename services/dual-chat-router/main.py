from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, ValidationError
import redis
import json
import os
from typing import List, Optional, Dict
import logging
from prometheus_client import Counter, Histogram
import time
import httpx
import asyncio
import traceback
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Initialize FastAPI app
app = FastAPI(title="Dual Chat Router Service")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Redis client
redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
redis_client = redis.from_url(redis_url)

# Service URLs (use Docker Compose service names by default)
output_engine_url = os.getenv("OUTPUT_ENGINE_URL", "http://output-engine:9000")
output_engine_api_key = os.getenv("OUTPUT_ENGINE_API_KEY", "changeme")
output_engine_base_url = f"{output_engine_url}?api_key={output_engine_api_key}"
ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")

# Debug log the environment variables
logger.debug(f"OUTPUT_ENGINE_URL: {output_engine_url}")
logger.debug(f"OUTPUT_ENGINE_API_KEY: {output_engine_api_key}")
logger.debug(f"OUTPUT_ENGINE_BASE_URL: {output_engine_base_url}")

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

# Add custom exceptions
class ServiceUnavailableError(Exception):
    pass

class ServiceTimeoutError(Exception):
    pass

class ServiceRateLimitError(Exception):
    pass

# Add retry decorator for transient failures
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ServiceUnavailableError, ServiceTimeoutError))
        )
async def get_llm_response_with_retry(text: str, session_id: Optional[str] = None) -> Dict:
    """Get response from LLM with retries"""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            payload = {
                "model": "mistral:7b-instruct",
                "prompt": text,
                "stream": False
            }
            r = await client.post(
                f"{ollama_url}/api/generate",
                json=payload
            )
            r.raise_for_status()
            return r.json()
    except httpx.TimeoutException:
        raise ServiceTimeoutError("LLM service request timed out")
    except httpx.ConnectError:
        raise ServiceUnavailableError("LLM service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error from LLM service: {str(e)}")
        raise

async def get_nn_response(text: str, session_id: Optional[str] = None) -> Dict:
    """Get response from NN service"""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            payload = {"query": text, "session_id": session_id}
            url = f"{output_engine_url}/respond?api_key={output_engine_api_key}"
            logger.debug(f"Making request to URL: {url}")
            logger.debug(f"Request payload: {payload}")
            r = await client.post(
                url,
                json=payload
            )
            logger.debug(f"Response status: {r.status_code}")
            logger.debug(f"Response headers: {r.headers}")
            logger.debug(f"Response body: {r.text}")
            r.raise_for_status()
            return r.json()
    except httpx.TimeoutException:
        raise ServiceTimeoutError("NN service request timed out")
    except httpx.ConnectError:
        raise ServiceUnavailableError("NN service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error from output engine: {str(e)}")
        raise

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
                    nn_response = await client.get(f"{output_engine_url}/health?api_key={output_engine_api_key}", timeout=2)
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
async def chat(request: dict = Body(...)):
    logger.info(f"/chat endpoint called with raw data: {request}")
    with chat_latency.time():
        chat_requests.inc()
        try:
            # Validate request
            try:
                chat_req = ChatRequest(**request)
            except Exception as ve:
                logger.error(f"Request validation error: {ve}\n{traceback.format_exc()}")
                raise HTTPException(
                    status_code=422,
                    detail={"error": "Invalid request format", "details": str(ve)}
                )

            logger.info(f"Processing chat request: {chat_req}")
            
            # Get responses from both services concurrently with retries
            try:
                # Always use get_nn_response which includes the API key
                nn_task = asyncio.create_task(get_nn_response(chat_req.query, chat_req.session_id))
                llm_task = asyncio.create_task(get_llm_response_with_retry(chat_req.query, chat_req.session_id))
            
                nn_response, llm_response = await asyncio.gather(nn_task, llm_task, return_exceptions=True)
                
                # Handle exceptions from either service
                if isinstance(nn_response, Exception):
                    logger.error(f"NN service error: {str(nn_response)}")
                    if isinstance(nn_response, ServiceRateLimitError):
                        raise HTTPException(status_code=429, detail="Service temporarily rate limited")
                    elif isinstance(nn_response, ServiceTimeoutError):
                        raise HTTPException(status_code=504, detail="NN service timeout")
                    elif isinstance(nn_response, ServiceUnavailableError):
                        raise HTTPException(status_code=503, detail="NN service unavailable")
                    else:
                        raise HTTPException(status_code=500, detail="NN service error")
                
                if isinstance(llm_response, Exception):
                    logger.error(f"LLM service error: {str(llm_response)}")
                    if isinstance(llm_response, ServiceTimeoutError):
                        raise HTTPException(status_code=504, detail="LLM service timeout")
                    elif isinstance(llm_response, ServiceUnavailableError):
                        raise HTTPException(status_code=503, detail="LLM service unavailable")
                    else:
                        raise HTTPException(status_code=500, detail="LLM service error")
            
            # Choose the better response based on confidence
                if nn_response.get("confidence", 0) >= 0.7:
                    nn_responses.inc()
                    logger.info(f"Returning NN response with confidence {nn_response.get('confidence')}")
                    return nn_response
                else:
                    llm_responses.inc()
                    logger.info(f"Returning LLM response with confidence {llm_response.get('confidence')}")
                    return llm_response
                    
            except asyncio.TimeoutError:
                logger.error("Request timed out waiting for services")
                raise HTTPException(status_code=504, detail="Request timed out")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing chat request: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail={"error": "Internal server error", "message": str(e)}
            )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Request validation error: {exc.errors()} | Body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": (await request.body()).decode()}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300) 