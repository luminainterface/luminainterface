#!/usr/bin/env python3
"""
Memory API Server

Provides a RESTful API for the Language Memory System to interface with LLM systems.
This allows any external LLM system to connect to the memory capabilities.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# FastAPI and server components
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import memory API
from src.memory_api import MemoryAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/logs/memory_api_server.log")
    ]
)
logger = logging.getLogger("memory_api_server")

# Metrics for monitoring (optional - requires prometheus-client)
try:
    from prometheus_client import Counter, Histogram, start_http_server
    METRICS_ENABLED = True
    
    # Define metrics
    REQUEST_COUNT = Counter('memory_api_requests_total', 'Total number of requests', ['endpoint', 'method', 'status'])
    REQUEST_LATENCY = Histogram('memory_api_request_latency_seconds', 'Request latency in seconds', ['endpoint'])
    SYNTHESIS_LATENCY = Histogram('memory_synthesis_latency_seconds', 'Synthesis latency in seconds')
    MEMORY_OPERATIONS = Counter('memory_operations_total', 'Memory operations by type', ['operation'])
    
except ImportError:
    METRICS_ENABLED = False
    logger.warning("prometheus-client not installed - metrics collection disabled")


# Initialize FastAPI
app = FastAPI(
    title="Language Memory API",
    description="REST API for Language Memory System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    # Record metrics if enabled
    if METRICS_ENABLED:
        endpoint = request.url.path
        REQUEST_LATENCY.labels(endpoint).observe(process_time)
        REQUEST_COUNT.labels(
            endpoint=endpoint,
            method=request.method,
            status=response.status_code
        ).inc()
    
    return response


# Pydantic models for API requests/responses
class ConversationRequest(BaseModel):
    message: str
    metadata: Optional[Dict[str, Any]] = Field(default=None, 
                                             description="Optional metadata about the conversation")

class TopicRequest(BaseModel):
    topic: str
    depth: Optional[int] = Field(default=3, ge=1, le=5, 
                              description="How deep to search for related memories")

class EnhanceRequest(BaseModel):
    message: str
    enhance_mode: Optional[str] = Field(default="contextual", 
                                    description="Enhancement mode: contextual, synthesized, or combined")

class TrainingExamplesRequest(BaseModel):
    topic: str
    count: Optional[int] = Field(default=3, ge=1, le=10, 
                              description="Number of examples to generate")

class APIResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Memory API instance (shared across requests)
memory_api = None

def get_memory_api():
    """Get or initialize the Memory API instance"""
    global memory_api
    if memory_api is None:
        memory_api = MemoryAPI()
        logger.info("Memory API initialized")
    return memory_api


# API routes
@app.get("/", tags=["General"])
async def root():
    """Root endpoint, provides API information"""
    return {
        "name": "Language Memory API",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs"
    }

@app.post("/memory/conversation", response_model=APIResponse, tags=["Memory"])
async def store_conversation(request: ConversationRequest, api: MemoryAPI = Depends(get_memory_api)):
    """Store a conversation message in memory"""
    if METRICS_ENABLED:
        MEMORY_OPERATIONS.labels(operation="store_conversation").inc()
    
    result = api.store_conversation(message=request.message, metadata=request.metadata)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
    
    return {
        "status": "success",
        "data": result,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/memory/retrieve", response_model=APIResponse, tags=["Memory"])
async def retrieve_memories(request: ConversationRequest, api: MemoryAPI = Depends(get_memory_api)):
    """Retrieve memories relevant to a message"""
    if METRICS_ENABLED:
        MEMORY_OPERATIONS.labels(operation="retrieve_memories").inc()
    
    result = api.retrieve_relevant_memories(message=request.message)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
    
    return {
        "status": "success",
        "data": result,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/memory/synthesize", response_model=APIResponse, tags=["Synthesis"])
async def synthesize_topic(request: TopicRequest, api: MemoryAPI = Depends(get_memory_api)):
    """Synthesize memories around a specific topic"""
    if METRICS_ENABLED:
        MEMORY_OPERATIONS.labels(operation="synthesize_topic").inc()
        with SYNTHESIS_LATENCY.time():
            result = api.synthesize_topic(topic=request.topic, depth=request.depth)
    else:
        result = api.synthesize_topic(topic=request.topic, depth=request.depth)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
    
    return {
        "status": "success",
        "data": result,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/memory/enhance", response_model=APIResponse, tags=["Integration"])
async def enhance_message(request: EnhanceRequest, api: MemoryAPI = Depends(get_memory_api)):
    """Enhance a message with memory context for LLM integration"""
    if METRICS_ENABLED:
        MEMORY_OPERATIONS.labels(operation="enhance_message").inc()
    
    result = api.enhance_message_with_memory(
        message=request.message,
        enhance_mode=request.enhance_mode
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
    
    return {
        "status": "success",
        "data": result,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/training/examples", response_model=APIResponse, tags=["Training"])
async def generate_training_examples(request: TrainingExamplesRequest, api: MemoryAPI = Depends(get_memory_api)):
    """Generate training examples for a specific topic"""
    if METRICS_ENABLED:
        MEMORY_OPERATIONS.labels(operation="generate_examples").inc()
    
    result = api.get_training_examples(topic=request.topic, count=request.count)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
    
    return {
        "status": "success",
        "data": result,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats", response_model=APIResponse, tags=["Monitoring"])
async def get_stats(api: MemoryAPI = Depends(get_memory_api)):
    """Get statistics about the memory system"""
    if METRICS_ENABLED:
        MEMORY_OPERATIONS.labels(operation="get_stats").inc()
    
    result = api.get_memory_stats()
    
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
    
    return {
        "status": "success",
        "data": result,
        "timestamp": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health", tags=["Monitoring"])
async def health_check(api: MemoryAPI = Depends(get_memory_api)):
    """Health check endpoint for monitoring"""
    # Check if memory API is initialized
    if not getattr(api, "initialized", False):
        raise HTTPException(status_code=503, detail="Memory API not initialized")
    
    # Get basic system stats to verify functionality
    try:
        stats = api.get_memory_stats()
        
        return {
            "status": "healthy",
            "memory_api_status": "initialized",
            "components": {
                "conversation_memory": hasattr(api, "conversation_memory") and api.conversation_memory is not None,
                "language_trainer": hasattr(api, "language_trainer") and api.language_trainer is not None,
                "memory_system": hasattr(api, "memory_system") and api.memory_system is not None
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


# Startup and shutdown events
@app.on_event("startup")
def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Memory API Server")
    
    # Create required directories
    os.makedirs("data/logs", exist_ok=True)
    
    # Start metrics server if enabled
    if METRICS_ENABLED:
        try:
            metrics_port = int(os.environ.get("METRICS_PORT", 8001))
            start_http_server(metrics_port)
            logger.info(f"Metrics server started on port {metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
    
    # Initialize memory API (lazy initialization via dependency)
    logger.info("Memory API will be initialized on first request")


@app.on_event("shutdown")
def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down Memory API Server")
    
    # Shutdown memory API if initialized
    global memory_api
    if memory_api and hasattr(memory_api.memory_system, "shutdown"):
        try:
            memory_api.memory_system.shutdown()
            logger.info("Memory system shutdown completed")
        except Exception as e:
            logger.error(f"Error during memory system shutdown: {str(e)}")


def main():
    """Run the Memory API server"""
    # Set default port
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run(
        "memory_api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main() 