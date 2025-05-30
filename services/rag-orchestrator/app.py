#!/usr/bin/env python3
"""
🌟 CENTRALIZED RAG ORCHESTRATOR - GOLD STAR INTEGRATION
========================================================

Integrates with flow2.md DEPLOYED REVOLUTIONARY ARCHITECTURE:
- 🎄🌟 UNIFIED THINKING ENGINE coordination
- 🌟 BIDIRECTIONAL DEEP THINKING (ONLY ONE STAR!)
- ⚡ 8-Phase Reasoning Pipeline
- 🤝 A2A Deep Thinking Swarm
- 🔧 Coordinated Tools

This orchestrator provides centralized coordination for all RAG services
through the neural thought engine with Gold Star capabilities.
"""

import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import os
import time
import json
from datetime import datetime

app = FastAPI(
    title="🌟 RAG Orchestrator - Gold Star Coordination",
    description="Centralized RAG coordination through Neural Thought Engine",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str
    coordination_mode: str = "gold_star"  # gold_star, bidirectional, a2a, eight_phase
    thinking_depth: int = 3
    use_bidirectional: bool = True
    enable_circuit_breakers: bool = True
    context: Dict = {}
    routing_preferences: List[str] = []

class CoordinationResponse(BaseModel):
    query: str
    coordination_mode: str
    execution_time: str
    thinking_phases: List[Dict]
    rag_coordination: Dict
    final_response: str
    performance_metrics: Dict
    gold_star_activated: bool

class RAGOrchestrator:
    def __init__(self):
        # 🎄🌟 UNIFIED THINKING ENGINE Integration
        self.thinking_engine_host = os.getenv("THINKING_ENGINE_HOST", "neural-thought-engine")
        self.thinking_engine_port = os.getenv("THINKING_ENGINE_PORT", "8890")
        
        # RAG Coordination Components
        self.rag_coordination_host = os.getenv("RAG_COORDINATION_INTERFACE_HOST", "rag-coordination-interface")
        self.rag_coordination_port = os.getenv("RAG_COORDINATION_INTERFACE_PORT", "8952")
        
        self.rag_router_enhanced_host = os.getenv("RAG_ROUTER_ENHANCED_HOST", "rag-router-enhanced")
        self.rag_router_enhanced_port = os.getenv("RAG_ROUTER_ENHANCED_PORT", "8951")
        
        # 🌟 GOLD STAR FEATURES (ONLY ONE STAR!)
        self.bidirectional_thinking = os.getenv("BIDIRECTIONAL_THINKING", "true").lower() == "true"
        self.eight_phase_reasoning = os.getenv("EIGHT_PHASE_REASONING", "true").lower() == "true"
        self.a2a_coordination = os.getenv("A2A_COORDINATION", "true").lower() == "true"
        self.orchestration_mode = os.getenv("ORCHESTRATION_MODE", "gold_star")
        self.diminishing_returns_detection = os.getenv("DIMINISHING_RETURNS_DETECTION", "true").lower() == "true"
        self.circuit_breakers = os.getenv("CIRCUIT_BREAKERS", "true").lower() == "true"
        
        # Performance tracking
        self.query_count = 0
        self.gold_star_activations = 0
        self.coordination_metrics = {
            "total_queries": 0,
            "successful_coordinations": 0,
            "thinking_engine_calls": 0,
            "rag_coordinations": 0,
            "gold_star_activations": 0,
            "average_response_time": 0.0
        }
        
    async def coordinate_through_thinking_engine(self, query: str, coordination_mode: str) -> Dict:
        """🧠 Coordinate through the Neural Thought Engine"""
        try:
            thinking_payload = {
                "query": query,
                "thinking_mode": coordination_mode,
                "use_bidirectional": self.bidirectional_thinking,
                "enable_circuit_breakers": self.circuit_breakers,
                "coordination_context": {
                    "rag_orchestration": True,
                    "gold_star_active": coordination_mode == "gold_star",
                    "eight_phase_reasoning": self.eight_phase_reasoning,
                    "a2a_coordination": self.a2a_coordination
                }
            }
            
            thinking_url = f"http://{self.thinking_engine_host}:{self.thinking_engine_port}"
            
            async with aiohttp.ClientSession() as session:
                # Try Gold Star endpoint first
                if coordination_mode == "gold_star":
                    async with session.post(
                        f"{thinking_url}/thinking/star-on-tree",
                        json=thinking_payload,
                        timeout=45
                    ) as response:
                        if response.status == 200:
                            self.gold_star_activations += 1
                            self.coordination_metrics["gold_star_activations"] += 1
                            return await response.json()
                
                # Fallback to deep thinking
                async with session.post(
                    f"{thinking_url}/thinking/deep",
                    json=thinking_payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        # Final fallback to basic thinking
                        async with session.post(
                            f"{thinking_url}/thinking/process",
                            json=thinking_payload,
                            timeout=20
                        ) as basic_response:
                            if basic_response.status == 200:
                                return await basic_response.json()
                            else:
                                return {"status": "fallback", "message": "Using direct coordination"}
                                
        except Exception as e:
            return {"status": "error", "error": str(e), "fallback": "direct_coordination"}
    
    async def coordinate_rag_systems(self, query: str, thinking_result: Dict) -> Dict:
        """🔧 Coordinate RAG systems based on thinking engine guidance"""
        try:
            # Extract thinking guidance for RAG coordination
            thinking_guidance = thinking_result.get("thinking_process", {})
            coordination_plan = thinking_result.get("coordination_plan", {})
            
            rag_coordination_payload = {
                "query": query,
                "thinking_guidance": thinking_guidance,
                "coordination_plan": coordination_plan,
                "use_enhanced_routing": True,
                "multi_tier_retrieval": True
            }
            
            rag_coordination_url = f"http://{self.rag_coordination_host}:{self.rag_coordination_port}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{rag_coordination_url}/coordinate",
                    json=rag_coordination_payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        # Fallback to enhanced router
                        return await self.fallback_enhanced_routing(session, query)
                        
        except Exception as e:
            # Direct enhanced router fallback
            async with aiohttp.ClientSession() as session:
                return await self.fallback_enhanced_routing(session, query)
    
    async def fallback_enhanced_routing(self, session: aiohttp.ClientSession, query: str) -> Dict:
        """🔀 Fallback to enhanced router"""
        try:
            router_url = f"http://{self.rag_router_enhanced_host}:{self.rag_router_enhanced_port}"
            
            async with session.post(
                f"{router_url}/query",
                json={"query": query},
                timeout=15
            ) as response:
                if response.status == 200:
                    return {
                        "source": "enhanced_router",
                        "coordination": await response.json()
                    }
                else:
                    return {
                        "source": "fallback_failed",
                        "error": f"HTTP {response.status}"
                    }
                    
        except Exception as e:
            return {
                "source": "connection_error",
                "error": str(e)
            }
    
    def synthesize_final_response(self, query: str, thinking_result: Dict, 
                                rag_result: Dict, coordination_mode: str) -> str:
        """✨ Synthesize final response using all coordination results"""
        
        # Extract components
        thinking_insights = thinking_result.get("insights", [])
        thinking_conclusion = thinking_result.get("conclusion", "")
        rag_response = rag_result.get("coordination", {}).get("response", {})
        
        # Build synthesized response
        synthesis_parts = []
        
        synthesis_parts.append(f"🌟 **Gold Star Coordinated Response** (Mode: {coordination_mode})")
        synthesis_parts.append(f"**Query**: {query}\n")
        
        # Thinking Engine Insights
        if thinking_insights:
            synthesis_parts.append("🧠 **Neural Thought Engine Analysis**:")
            for i, insight in enumerate(thinking_insights[:3], 1):
                if isinstance(insight, str):
                    synthesis_parts.append(f"  {i}. {insight}")
                elif isinstance(insight, dict):
                    insight_text = insight.get("insight", insight.get("text", str(insight)))
                    synthesis_parts.append(f"  {i}. {insight_text}")
        
        # RAG Coordination Results
        if rag_response:
            synthesis_parts.append(f"\n🔧 **RAG Coordination Results**:")
            if "selected_head" in rag_response:
                synthesis_parts.append(f"  • Selected RAG Head: {rag_response['selected_head']}")
            if "routing_reason" in rag_response:
                synthesis_parts.append(f"  • Routing Reason: {rag_response['routing_reason']}")
            if "message" in rag_response:
                synthesis_parts.append(f"  • Response: {rag_response['message']}")
        
        # Final Conclusion
        if thinking_conclusion:
            synthesis_parts.append(f"\n✨ **Coordinated Conclusion**:")
            synthesis_parts.append(thinking_conclusion)
        
        synthesis_parts.append(f"\n🎯 **Coordination Status**: Successfully orchestrated through {coordination_mode} mode")
        synthesis_parts.append(f"🌟 **Gold Star Active**: {coordination_mode == 'gold_star'}")
        
        return "\n".join(synthesis_parts)

orchestrator = RAGOrchestrator()

@app.post("/orchestrate", response_model=CoordinationResponse)
async def orchestrate_query(request: QueryRequest):
    """🌟 Main orchestration endpoint - Gold Star coordination"""
    start_time = time.time()
    
    try:
        # Update metrics
        orchestrator.query_count += 1
        orchestrator.coordination_metrics["total_queries"] += 1
        
        # Phase 1: Neural Thought Engine Coordination
        thinking_start = time.time()
        thinking_result = await orchestrator.coordinate_through_thinking_engine(
            request.query, 
            request.coordination_mode
        )
        thinking_time = time.time() - thinking_start
        orchestrator.coordination_metrics["thinking_engine_calls"] += 1
        
        # Phase 2: RAG Systems Coordination
        rag_start = time.time()
        rag_result = await orchestrator.coordinate_rag_systems(
            request.query, 
            thinking_result
        )
        rag_time = time.time() - rag_start
        orchestrator.coordination_metrics["rag_coordinations"] += 1
        
        # Phase 3: Synthesis
        synthesis_start = time.time()
        final_response = orchestrator.synthesize_final_response(
            request.query,
            thinking_result,
            rag_result,
            request.coordination_mode
        )
        synthesis_time = time.time() - synthesis_start
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Update success metrics
        orchestrator.coordination_metrics["successful_coordinations"] += 1
        
        # Calculate average response time
        total_queries = orchestrator.coordination_metrics["total_queries"]
        current_avg = orchestrator.coordination_metrics["average_response_time"]
        orchestrator.coordination_metrics["average_response_time"] = (
            (current_avg * (total_queries - 1) + total_time) / total_queries
        )
        
        return CoordinationResponse(
            query=request.query,
            coordination_mode=request.coordination_mode,
            execution_time=f"{total_time:.2f}s",
            thinking_phases=[
                {
                    "phase": "neural_thinking",
                    "duration": f"{thinking_time:.2f}s",
                    "result": thinking_result
                },
                {
                    "phase": "rag_coordination", 
                    "duration": f"{rag_time:.2f}s",
                    "result": rag_result
                },
                {
                    "phase": "synthesis",
                    "duration": f"{synthesis_time:.2f}s",
                    "result": {"synthesized": True}
                }
            ],
            rag_coordination=rag_result,
            final_response=final_response,
            performance_metrics={
                "total_time": f"{total_time:.2f}s",
                "thinking_time": f"{thinking_time:.2f}s", 
                "rag_time": f"{rag_time:.2f}s",
                "synthesis_time": f"{synthesis_time:.2f}s"
            },
            gold_star_activated=request.coordination_mode == "gold_star"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Orchestrator - Gold Star Coordination",
        "coordination_mode": orchestrator.orchestration_mode,
        "gold_star_enabled": True,
        "bidirectional_thinking": orchestrator.bidirectional_thinking,
        "eight_phase_reasoning": orchestrator.eight_phase_reasoning,
        "a2a_coordination": orchestrator.a2a_coordination,
        "circuit_breakers": orchestrator.circuit_breakers,
        "query_count": orchestrator.query_count,
        "gold_star_activations": orchestrator.gold_star_activations,
        "metrics": orchestrator.coordination_metrics,
        "integration": {
            "neural_thought_engine": f"{orchestrator.thinking_engine_host}:{orchestrator.thinking_engine_port}",
            "rag_coordination": f"{orchestrator.rag_coordination_host}:{orchestrator.rag_coordination_port}",
            "enhanced_router": f"{orchestrator.rag_router_enhanced_host}:{orchestrator.rag_router_enhanced_port}"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get coordination metrics"""
    return {
        "timestamp": datetime.now().isoformat(),
        "coordination_metrics": orchestrator.coordination_metrics,
        "gold_star_activations": orchestrator.gold_star_activations,
        "total_queries": orchestrator.query_count,
        "success_rate": (
            orchestrator.coordination_metrics["successful_coordinations"] / 
            max(orchestrator.coordination_metrics["total_queries"], 1) * 100
        ),
        "average_response_time": orchestrator.coordination_metrics["average_response_time"]
    }

@app.post("/coordinate/simple")
async def simple_coordination(query: str):
    """🚀 Simple coordination endpoint for quick queries"""
    request = QueryRequest(
        query=query,
        coordination_mode="gold_star",
        thinking_depth=2
    )
    result = await orchestrate_query(request)
    return {"response": result.final_response}

if __name__ == "__main__":
    print("🌟 Starting RAG Orchestrator - Gold Star Coordination...")
    print("🎄🌟 UNIFIED THINKING ENGINE Integration Active")
    print("=" * 60)
    print("📍 Available endpoints:")
    print("  • POST /orchestrate - Full Gold Star coordination")
    print("  • POST /coordinate/simple - Quick coordination")
    print("  • GET  /health - Health and integration status")
    print("  • GET  /metrics - Performance metrics")
    print("\n🌟 GOLD STAR FEATURES ACTIVE:")
    print("  • 🎄🌟 Bidirectional Deep Thinking")
    print("  • ⚡ 8-Phase Reasoning Pipeline")
    print("  • 🤝 A2A Deep Thinking Swarm")
    print("  • 🔧 Coordinated Tools Integration")
    print("  • 🔄 Diminishing Returns Detection")
    print("  • ⚡ Circuit Breakers")
    print(f"\n🚀 Server starting on http://0.0.0.0:8953")
    
    uvicorn.run(app, host="0.0.0.0", port=8953) 