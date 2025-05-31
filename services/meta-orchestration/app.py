#!/usr/bin/env python3
"""
Meta-Orchestration FastAPI Service
=================================

REST API service for meta-level AI orchestration and system benchmarking
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import logging
import uvicorn
from datetime import datetime

# Import the meta-orchestration controller
from meta_orchestration_controller import meta_controller

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Meta-Orchestration Controller",
    description="Advanced meta-level orchestration for coordinating multiple AI systems",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ResearchTaskRequest(BaseModel):
    topic: str = Field(..., description="Research topic", min_length=10)
    field: str = Field("general", description="Research field")
    priority: int = Field(1, description="Task priority (1-10)", ge=1, le=10)

class OrchestrationResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    execution_time: float
    timestamp: str

# Global stats
orchestration_stats = {
    "total_tasks": 0,
    "successful_tasks": 0,
    "failed_tasks": 0,
    "last_benchmark": None
}

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Meta-Orchestration Controller",
        "version": "1.0.0",
        "description": "Advanced meta-level orchestration for coordinating multiple AI systems",
        "endpoints": {
            "health": "/health - Health check",
            "orchestrate": "/orchestrate/research - Orchestrate research task",
            "benchmark": "/benchmark/system - Run system benchmark",
            "status": "/status - Get orchestration status",
            "services": "/services/health - Check all service health"
        },
        "capabilities": [
            "Multi-service coordination",
            "Real-world benchmarking",
            "Health monitoring",
            "Task orchestration",
            "System optimization"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "controller_initialized": True,
        "registered_services": len(meta_controller.services),
        "active_tasks": len(meta_controller.active_tasks)
    }

@app.post("/orchestrate/research", response_model=OrchestrationResponse)
async def orchestrate_research_task(request: ResearchTaskRequest, background_tasks: BackgroundTasks):
    """Orchestrate a complex research task across multiple services"""
    logger.info(f"🎯 Received orchestration request for: {request.topic}")
    
    orchestration_stats["total_tasks"] += 1
    
    try:
        # Execute orchestration
        result = await meta_controller.orchestrate_research_task(
            topic=request.topic,
            field=request.field
        )
        
        # Update stats
        if result.get("overall_status") == "success":
            orchestration_stats["successful_tasks"] += 1
        else:
            orchestration_stats["failed_tasks"] += 1
        
        return OrchestrationResponse(
            status="success",
            data=result,
            execution_time=result.get("execution_time", 0),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        orchestration_stats["failed_tasks"] += 1
        logger.error(f"❌ Orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")

@app.post("/benchmark/system")
async def run_system_benchmark(background_tasks: BackgroundTasks):
    """Run comprehensive system benchmark"""
    logger.info("🚀 Starting system benchmark via API")
    
    try:
        # Run benchmark
        benchmark_result = await meta_controller.run_system_benchmark()
        
        # Store result
        orchestration_stats["last_benchmark"] = benchmark_result
        
        return {
            "status": "success",
            "benchmark": benchmark_result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@app.get("/services/health")
async def check_all_service_health():
    """Check health of all registered services"""
    try:
        health_status = await meta_controller.health_check_all_services()
        
        return {
            "status": "success",
            "health_check": {
                "timestamp": datetime.now().isoformat(),
                "services": {name: {
                    "status": health.status,
                    "response_time": health.response_time,
                    "last_checked": health.last_checked
                } for name, health in health_status.items()},
                "summary": {
                    "total_services": len(health_status),
                    "healthy_services": sum(1 for h in health_status.values() if h.status == "healthy"),
                    "unhealthy_services": sum(1 for h in health_status.values() if h.status != "healthy")
                }
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/status")
async def get_orchestration_status():
    """Get current orchestration status and statistics"""
    try:
        status = await meta_controller.get_orchestration_status()
        
        return {
            "status": "operational",
            "orchestration": status,
            "statistics": orchestration_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/benchmark/last")
async def get_last_benchmark():
    """Get the last benchmark results"""
    if orchestration_stats["last_benchmark"]:
        return {
            "status": "success",
            "benchmark": orchestration_stats["last_benchmark"],
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "status": "no_benchmark",
            "message": "No benchmark has been run yet",
            "timestamp": datetime.now().isoformat()
        }

# Background task to run periodic health checks
async def periodic_health_check():
    """Run periodic health checks in the background"""
    while True:
        try:
            await meta_controller.health_check_all_services()
            logger.info("🔍 Periodic health check completed")
        except Exception as e:
            logger.error(f"❌ Periodic health check failed: {e}")
        
        # Wait 5 minutes before next check
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    """Initialize the meta-orchestration controller on startup"""
    logger.info("🚀 Starting Meta-Orchestration Controller Service")
    logger.info("✅ Meta-Orchestration Controller initialized")
    
    # Start periodic health checks
    asyncio.create_task(periodic_health_check())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8900) 