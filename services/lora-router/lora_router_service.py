#!/usr/bin/env python3
"""
Optimal LoRA Router Service
Routes queries to appropriate LoRA adapters
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Optimal LoRA Router", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequest(BaseModel):
    query: str
    domain: str = "general"
    
class RouteResponse(BaseModel):
    selected_lora: str
    confidence: float
    routing_reason: str
    domain: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "lora_router", "version": "1.0.0"}

@app.post("/route")
async def route_query(request: RouteRequest):
    """Route query to appropriate LoRA adapter"""
    try:
        # Simple routing logic based on domain and keywords
        query_lower = request.query.lower()
        
        if any(word in query_lower for word in ["math", "calculate", "number", "equation"]):
            selected_lora = "mathematical_reasoning_lora"
            confidence = 0.85
            reason = "Mathematical keywords detected"
        elif any(word in query_lower for word in ["code", "programming", "python", "function"]):
            selected_lora = "coding_specialist_lora"
            confidence = 0.90
            reason = "Programming keywords detected"
        elif any(word in query_lower for word in ["science", "research", "analysis", "study"]):
            selected_lora = "scientific_reasoning_lora"
            confidence = 0.80
            reason = "Scientific terminology detected"
        else:
            selected_lora = "general_purpose_lora"
            confidence = 0.60
            reason = "Default routing for general queries"
        
        return RouteResponse(
            selected_lora=selected_lora,
            confidence=confidence,
            routing_reason=reason,
            domain=request.domain
        )
        
    except Exception as e:
        logger.error(f"Routing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5030)
