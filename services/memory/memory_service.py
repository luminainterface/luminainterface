#!/usr/bin/env python3
"""
Memory Service
Stores and retrieves conversation memory and context
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import redis
import json
from typing import Optional, Dict, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Memory Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoreRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = 3600  # 1 hour default
    
class RetrieveRequest(BaseModel):
    key: str

# Initialize Redis connection
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Connected to Redis")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None

# Fallback in-memory storage
memory_store = {}

@app.get("/health")
async def health_check():
    redis_status = "connected" if redis_client else "fallback_memory"
    return {
        "status": "healthy", 
        "service": "memory_service", 
        "version": "1.0.0",
        "redis_status": redis_status,
        "stored_keys": len(memory_store) if not redis_client else "unknown"
    }

@app.post("/store")
async def store_memory(request: StoreRequest):
    """Store key-value pair in memory"""
    try:
        value_str = json.dumps(request.value) if not isinstance(request.value, str) else request.value
        
        if redis_client:
            redis_client.setex(request.key, request.ttl, value_str)
        else:
            memory_store[request.key] = {
                "value": value_str,
                "timestamp": time.time(),
                "ttl": request.ttl
            }
        
        return {
            "status": "stored",
            "key": request.key,
            "ttl": request.ttl,
            "storage_backend": "redis" if redis_client else "memory"
        }
        
    except Exception as e:
        logger.error(f"Storage error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve_memory(request: RetrieveRequest):
    """Retrieve value by key"""
    try:
        if redis_client:
            value = redis_client.get(request.key)
            if value:
                try:
                    parsed_value = json.loads(value)
                except:
                    parsed_value = value
                return {"key": request.key, "value": parsed_value, "found": True}
            else:
                return {"key": request.key, "value": None, "found": False}
        else:
            if request.key in memory_store:
                entry = memory_store[request.key]
                # Check TTL
                if time.time() - entry["timestamp"] < entry["ttl"]:
                    try:
                        parsed_value = json.loads(entry["value"])
                    except:
                        parsed_value = entry["value"]
                    return {"key": request.key, "value": parsed_value, "found": True}
                else:
                    del memory_store[request.key]
            return {"key": request.key, "value": None, "found": False}
            
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8915)
