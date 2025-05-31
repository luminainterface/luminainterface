#!/usr/bin/env python3
"""
Enhanced Research Agent v3 - FastAPI Service
============================================

REST API service wrapper for the Enhanced Research Agent v3
Enables other agents and services to utilize the research capabilities
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import logging
import time
from datetime import datetime
import json
import uvicorn

# Import the Enhanced Research Agent
from enhanced_research_agent_v3 import EnhancedResearchAgentV3, ResearchRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Enhanced Research Agent v3",
    description="Ultimate research intelligence with 15-minute LoRA wait, RAG circular growth, and anti-hallucination",
    version="3.0.0",
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

# Global research agent instance
research_agent = None

# Pydantic models for API
class ResearchRequestAPI(BaseModel):
    topic: str = Field(..., description="Research topic", min_length=10)
    field: str = Field(..., description="Research field", min_length=3)
    target_quality: float = Field(9.0, description="Target quality (0-10)", ge=0, le=10)
    max_lora_wait_minutes: int = Field(15, description="Maximum LoRA wait time in minutes", ge=1, le=60)
    enable_rag_crawling: bool = Field(True, description="Enable RAG circular growth crawling")
    enable_anti_hallucination: bool = Field(True, description="Enable anti-hallucination measures")
    quality_threshold: float = Field(8.5, description="Quality threshold", ge=0, le=10)
    word_count_target: int = Field(2000, description="Target word count", ge=500, le=10000)
    academic_style: str = Field("comprehensive", description="Academic writing style")
    fact_check_level: str = Field("strict", description="Fact-checking level")

class ResearchResponse(BaseModel):
    status: str
    paper: Optional[Dict[str, Any]] = None
    generation_time: float
    knowledge_plan: Optional[Dict[str, Any]] = None
    anti_hallucination_metrics: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    systems_utilized: List[str]
    recommendations: List[str]
    error: Optional[str] = None

class QuickResearchRequest(BaseModel):
    query: str = Field(..., description="Quick research query", min_length=5)
    max_wait_seconds: int = Field(300, description="Maximum wait time in seconds", ge=30, le=1800)

class ResearchStatusResponse(BaseModel):
    active_research_tasks: int
    total_completed: int
    average_quality: float
    system_health: Dict[str, str]

# Global state tracking
research_stats = {
    "total_completed": 0,
    "active_tasks": 0,
    "quality_sum": 0.0,
    "completed_topics": []
}

@app.on_event("startup")
async def startup_event():
    """Initialize the research agent on startup"""
    global research_agent
    logger.info("🚀 Starting Enhanced Research Agent v3 Service")
    research_agent = EnhancedResearchAgentV3()
    logger.info("✅ Enhanced Research Agent v3 initialized")

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Enhanced Research Agent v3",
        "version": "3.0.0",
        "description": "Ultimate research intelligence with LoRA, RAG, and anti-hallucination",
        "endpoints": {
            "research": "/research/generate - Generate comprehensive research paper",
            "quick": "/research/quick - Quick research query",
            "status": "/research/status - Get system status",
            "health": "/health - Health check"
        },
        "capabilities": [
            "15-minute LoRA training with timeout",
            "RAG 2025 circular growth integration",
            "Anti-hallucination engine",
            "Multi-system parallel knowledge acquisition",
            "Comprehensive fact-checking",
            "Quality assurance pipeline"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_initialized": research_agent is not None,
        "active_tasks": research_stats["active_tasks"]
    }

@app.post("/research/generate", response_model=ResearchResponse)
async def generate_research_paper(request: ResearchRequestAPI, background_tasks: BackgroundTasks):
    """Generate comprehensive research paper using all systems"""
    if not research_agent:
        raise HTTPException(status_code=503, detail="Research agent not initialized")
    
    logger.info(f"📝 Research request received: {request.topic}")
    
    # Convert API request to internal request
    research_request = ResearchRequest(
        topic=request.topic,
        field=request.field,
        target_quality=request.target_quality,
        max_lora_wait_minutes=request.max_lora_wait_minutes,
        enable_rag_crawling=request.enable_rag_crawling,
        enable_anti_hallucination=request.enable_anti_hallucination,
        quality_threshold=request.quality_threshold,
        word_count_target=request.word_count_target,
        academic_style=request.academic_style,
        fact_check_level=request.fact_check_level
    )
    
    # Update stats
    research_stats["active_tasks"] += 1
    
    try:
        # Generate research paper
        result = await research_agent.generate_research_paper(research_request)
        
        # Update stats
        research_stats["active_tasks"] -= 1
        research_stats["total_completed"] += 1
        
        if result.get("status") == "success":
            quality = result.get("paper", {}).get("quality_metrics", {}).get("overall_quality", 0)
            research_stats["quality_sum"] += quality
            research_stats["completed_topics"].append({
                "topic": request.topic,
                "quality": quality,
                "timestamp": datetime.now().isoformat()
            })
        
        return ResearchResponse(**result)
    
    except Exception as e:
        research_stats["active_tasks"] -= 1
        logger.error(f"❌ Research generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research generation failed: {str(e)}")

@app.post("/research/quick")
async def quick_research(request: QuickResearchRequest):
    """Quick research query with time limit"""
    if not research_agent:
        raise HTTPException(status_code=503, detail="Research agent not initialized")
    
    logger.info(f"⚡ Quick research request: {request.query}")
    
    # Create quick research request
    quick_request = ResearchRequest(
        topic=request.query,
        field="general",
        target_quality=8.0,
        max_lora_wait_minutes=min(5, request.max_wait_seconds // 60),
        enable_rag_crawling=True,
        enable_anti_hallucination=True,
        word_count_target=1000
    )
    
    start_time = time.time()
    
    try:
        # Use asyncio.wait_for for timeout
        result = await asyncio.wait_for(
            research_agent.generate_research_paper(quick_request),
            timeout=request.max_wait_seconds
        )
        
        return {
            "status": "success",
            "query": request.query,
            "response_time": time.time() - start_time,
            "content": result.get("paper", {}).get("content", ""),
            "word_count": result.get("paper", {}).get("word_count", 0),
            "quality": result.get("paper", {}).get("quality_metrics", {}).get("overall_quality", 0)
        }
    
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "query": request.query,
            "response_time": time.time() - start_time,
            "message": f"Research exceeded {request.max_wait_seconds}s timeout"
        }
    except Exception as e:
        logger.error(f"❌ Quick research failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick research failed: {str(e)}")

@app.get("/research/status", response_model=ResearchStatusResponse)
async def get_research_status():
    """Get research system status"""
    if not research_agent:
        raise HTTPException(status_code=503, detail="Research agent not initialized")
    
    # Calculate average quality
    avg_quality = 0.0
    if research_stats["total_completed"] > 0:
        avg_quality = research_stats["quality_sum"] / research_stats["total_completed"]
    
    # Check system health (simplified)
    system_health = {
        "llm_gap_detector": "unknown",
        "rag_2025": "unknown", 
        "background_lora": "unknown",
        "enhanced_crawler": "unknown",
        "fact_checker": "unknown"
    }
    
    return ResearchStatusResponse(
        active_research_tasks=research_stats["active_tasks"],
        total_completed=research_stats["total_completed"],
        average_quality=avg_quality,
        system_health=system_health
    )

@app.get("/research/topics")
async def get_completed_topics(limit: int = 10):
    """Get list of recently completed research topics"""
    recent_topics = research_stats["completed_topics"][-limit:]
    return {
        "recent_topics": recent_topics,
        "total_topics": len(research_stats["completed_topics"])
    }

@app.post("/research/batch")
async def batch_research(topics: List[str], background_tasks: BackgroundTasks):
    """Start batch research for multiple topics"""
    if not research_agent:
        raise HTTPException(status_code=503, detail="Research agent not initialized")
    
    batch_id = f"batch_{int(time.time())}"
    
    async def process_batch():
        """Process batch research in background"""
        results = []
        for topic in topics:
            try:
                request = ResearchRequest(
                    topic=topic,
                    field="general",
                    target_quality=8.5,
                    max_lora_wait_minutes=10
                )
                result = await research_agent.generate_research_paper(request)
                results.append({"topic": topic, "status": "success", "result": result})
            except Exception as e:
                results.append({"topic": topic, "status": "failed", "error": str(e)})
        
        # Store results (in production, use database)
        logger.info(f"📊 Batch {batch_id} completed: {len(results)} topics processed")
    
    # Start batch processing in background
    background_tasks.add_task(process_batch)
    
    return {
        "batch_id": batch_id,
        "status": "started",
        "topics_count": len(topics),
        "estimated_completion": f"{len(topics) * 15} minutes"
    }

@app.get("/systems/utilized")
async def get_systems_utilized():
    """Get list of systems that can be utilized"""
    if not research_agent:
        raise HTTPException(status_code=503, detail="Research agent not initialized")
    
    return {
        "systems": research_agent._get_systems_utilized(),
        "architecture_integration": {
            "layer_1": "High-Rank Adapter - Strategic research steering",
            "layer_2": "Meta-Orchestration Controller - Research strategy selection", 
            "layer_3": "Enhanced Execution Suite - 8-phase orchestration",
            "layer_4": "LLM-Integrated Gap Detection - Fast chat + LoRA creation",
            "layer_5": "Research Paper Generation - Publication excellence",
            "layer_6": "V7 Base Logic Agent - Constraint satisfaction + AI failsafe",
            "infrastructure": "Redis, Qdrant, Neo4j, Ollama - Full backend support"
        }
    }

# For other agents to use programmatically
class ResearchAgentClient:
    """Client class for other agents to use the Enhanced Research Agent v3"""
    
    def __init__(self, base_url: str = "http://enhanced-research-agent:8999"):
        self.base_url = base_url
    
    async def generate_research(self, topic: str, field: str = "general", **kwargs) -> Dict[str, Any]:
        """Generate research paper"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/research/generate",
                json={
                    "topic": topic,
                    "field": field,
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def quick_research(self, query: str, max_wait_seconds: int = 300) -> Dict[str, Any]:
        """Quick research query"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/research/quick",
                json={
                    "query": query,
                    "max_wait_seconds": max_wait_seconds
                }
            )
            response.raise_for_status()
            return response.json()

if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8999,
        reload=False,
        log_level="info"
    ) 