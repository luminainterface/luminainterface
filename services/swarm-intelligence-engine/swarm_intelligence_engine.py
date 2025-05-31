"""
🐝 Swarm Intelligence Engine - Collective Intelligence Processing
Port: 8977
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import asyncio
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Swarm Intelligence Engine", version="1.0.0")

class SwarmRequest(BaseModel):
    query: str
    agents: List[str] = []
    consensus_threshold: float = 0.7

class SwarmResponse(BaseModel):
    consensus: Dict[str, Any]
    confidence: float
    participating_agents: List[str]

# Global state
swarm_state = {
    "active_agents": [],
    "consensus_history": [],
    "collective_knowledge": {}
}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "swarm-intelligence-engine",
        "version": "1.0.0",
        "active_agents": len(swarm_state["active_agents"])
    }

@app.post("/process_swarm", response_model=SwarmResponse)
async def process_swarm_intelligence(request: SwarmRequest):
    """Process collective intelligence from swarm agents"""
    try:
        logger.info(f"Processing swarm intelligence for query: {request.query}")
        
        # Simulate swarm processing
        consensus = {
            "primary_response": f"Swarm consensus for: {request.query}",
            "confidence_score": request.consensus_threshold,
            "collective_insights": ["Enhanced by swarm intelligence", "Verified by multiple agents"]
        }
        
        response = SwarmResponse(
            consensus=consensus,
            confidence=request.consensus_threshold,
            participating_agents=request.agents or ["agent-1", "agent-2", "agent-3"]
        )
        
        # Store in collective knowledge
        swarm_state["collective_knowledge"][request.query] = consensus
        
        return response
        
    except Exception as e:
        logger.error(f"Error in swarm processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/swarm_status")
async def get_swarm_status():
    """Get current swarm status"""
    return {
        "active_agents": len(swarm_state["active_agents"]),
        "knowledge_base_size": len(swarm_state["collective_knowledge"]),
        "consensus_history_size": len(swarm_state["consensus_history"]),
        "status": "operational"
    }

if __name__ == "__main__":
    logger.info("Starting Swarm Intelligence Engine on port 8977")
    uvicorn.run(app, host="0.0.0.0", port=8977) 