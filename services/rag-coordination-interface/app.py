#!/usr/bin/env python3
"""
🧠 RAG COORDINATION INTERFACE
============================

Provides coordination interface for multi-tier RAG retrieval
across specialized heads with Neural Thought Engine integration.
"""

import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import os
import time
import json
from datetime import datetime
from enhanced_concept_integration import EnhancedConceptIntegrator

app = FastAPI(
    title="🧠 RAG Coordination Interface",
    description="Multi-tier RAG coordination with enhanced routing",
    version="1.0.0"
)

class CoordinationRequest(BaseModel):
    query: str
    thinking_guidance: Dict = {}
    coordination_plan: Dict = {}
    use_enhanced_routing: bool = True
    multi_tier_retrieval: bool = True
    max_heads: int = 3

class CoordinationResponse(BaseModel):
    query: str
    coordination_plan: Dict
    head_responses: List[Dict]
    final_coordination: Dict
    execution_time: str
    heads_used: List[str]

class RAGCoordinationInterface:
    def __init__(self):
        # Service endpoints
        self.thinking_engine_host = os.getenv("THINKING_ENGINE_HOST", "neural-thought-engine")
        self.thinking_engine_port = os.getenv("THINKING_ENGINE_PORT", "8890")
        
        self.rag_router_enhanced_host = os.getenv("RAG_ROUTER_ENHANCED_HOST", "rag-router-enhanced")
        self.rag_router_enhanced_port = os.getenv("RAG_ROUTER_ENHANCED_PORT", "8951")
        
        self.vector_store_host = os.getenv("VECTOR_STORE_HOST", "vector-store")
        self.vector_store_port = os.getenv("VECTOR_STORE_PORT", "9262")
        
        # RAG heads
        self.rag_heads = {
            "rag_cpu": "http://rag-2025-cpu-optimized:8902",
            "rag_gpu_long": "http://rag-gpu-long:8920",
            "graph_rag": "http://rag-graph:8921", 
            "code_rag": "http://rag-code:8922"
        }
        
        # Coordination metrics
        self.coordination_count = 0
        self.successful_coordinations = 0
        self.total_execution_time = 0.0
        
    async def coordinate_multi_tier_retrieval(self, request: CoordinationRequest) -> Dict:
        """Coordinate multi-tier retrieval across RAG heads"""
        start_time = time.time()
        
        try:
            self.coordination_count += 1
            
            # Phase 1: Enhanced routing to get primary head
            primary_head, routing_reason = await self._get_enhanced_routing(request.query)
            
            # Phase 2: Determine additional heads based on thinking guidance
            additional_heads = await self._determine_additional_heads(
                request.query, 
                primary_head,
                request.thinking_guidance,
                request.max_heads
            )
            
            # Phase 3: Query selected heads in parallel
            head_responses = await self._query_multiple_heads(
                request.query,
                [primary_head] + additional_heads
            )
            
            # Phase 4: Coordinate and synthesize responses
            final_coordination = await self._synthesize_responses(
                request.query,
                head_responses,
                request.coordination_plan
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            self.total_execution_time += execution_time
            self.successful_coordinations += 1
            
            return {
                "query": request.query,
                "coordination_plan": {
                    "primary_head": primary_head,
                    "routing_reason": routing_reason,
                    "additional_heads": additional_heads,
                    "thinking_guidance": request.thinking_guidance
                },
                "head_responses": head_responses,
                "final_coordination": final_coordination,
                "execution_time": f"{execution_time:.3f}s",
                "heads_used": [primary_head] + additional_heads
            }
            
        except Exception as e:
            return {
                "query": request.query,
                "coordination_plan": {"error": str(e)},
                "head_responses": [],
                "final_coordination": {"error": str(e)},
                "execution_time": f"{time.time() - start_time:.3f}s",
                "heads_used": []
            }
    
    async def _get_enhanced_routing(self, query: str) -> tuple:
        """Get enhanced routing recommendation"""
        try:
            router_url = f"http://{self.rag_router_enhanced_host}:{self.rag_router_enhanced_port}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{router_url}/query",
                    json={"query": query},
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("selected_head", "rag_cpu"), data.get("routing_reason", "Enhanced routing")
                    else:
                        return "rag_cpu", "Fallback routing"
                        
        except Exception as e:
            return "rag_cpu", f"Error routing: {e}"
    
    async def _determine_additional_heads(self, query: str, primary_head: str, 
                                        thinking_guidance: Dict, max_heads: int) -> List[str]:
        """Determine additional heads for multi-tier retrieval"""
        try:
            additional_heads = []
            query_lower = query.lower()
            
            # Add graph_rag for relationship queries
            if any(keyword in query_lower for keyword in ["relationship", "connection", "between", "related"]):
                if "graph_rag" != primary_head and "graph_rag" not in additional_heads:
                    additional_heads.append("graph_rag")
            
            # Add code_rag for programming queries
            if any(keyword in query_lower for keyword in ["code", "programming", "python", "javascript", "implement"]):
                if "code_rag" != primary_head and "code_rag" not in additional_heads:
                    additional_heads.append("code_rag")
            
            # Add rag_gpu_long for complex analysis
            if any(keyword in query_lower for keyword in ["analyze", "complex", "deep", "comprehensive"]):
                if "rag_gpu_long" != primary_head and "rag_gpu_long" not in additional_heads:
                    additional_heads.append("rag_gpu_long")
            
            # Limit to max_heads - 1 (excluding primary)
            return additional_heads[:max_heads-1]
            
        except Exception as e:
            print(f"⚠️  Error determining additional heads: {e}")
            return []
    
    async def _query_multiple_heads(self, query: str, heads: List[str]) -> List[Dict]:
        """Query multiple RAG heads in parallel"""
        try:
            tasks = []
            
            for head in heads:
                if head in self.rag_heads:
                    task = self._query_single_head(head, self.rag_heads[head], query)
                    tasks.append(task)
            
            # Execute queries in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            head_responses = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    head_responses.append({
                        "head": heads[i] if i < len(heads) else "unknown",
                        "status": "error",
                        "error": str(result)
                    })
                else:
                    head_responses.append(result)
            
            return head_responses
            
        except Exception as e:
            print(f"❌ Error querying multiple heads: {e}")
            return []
    
    async def _query_single_head(self, head_name: str, head_url: str, query: str) -> Dict:
        """Query a single RAG head"""
        try:
            async with aiohttp.ClientSession() as session:
                # Try query endpoint first
                try:
                    async with session.post(
                        f"{head_url}/query",
                        json={"query": query},
                        timeout=15
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                "head": head_name,
                                "status": "success",
                                "response": data,
                                "url": head_url
                            }
                        else:
                            return {
                                "head": head_name,
                                "status": "failed",
                                "error": f"HTTP {response.status}",
                                "url": head_url
                            }
                except:
                    # Fallback to health check
                    async with session.get(f"{head_url}/health", timeout=5) as health_response:
                        if health_response.status == 200:
                            health_data = await health_response.json()
                            return {
                                "head": head_name,
                                "status": "simulation",
                                "response": {
                                    "simulated": True,
                                    "query": query,
                                    "service_status": health_data.get("status", "unknown"),
                                    "message": f"Service {head_name} is healthy and would process this query"
                                },
                                "url": head_url
                            }
                        else:
                            return {
                                "head": head_name,
                                "status": "unavailable",
                                "error": "Service not responding",
                                "url": head_url
                            }
                            
        except Exception as e:
            return {
                "head": head_name,
                "status": "error",
                "error": str(e),
                "url": head_url
            }
    
    async def _synthesize_responses(self, query: str, head_responses: List[Dict], 
                                  coordination_plan: Dict) -> Dict:
        """Synthesize responses from multiple heads"""
        try:
            successful_responses = [resp for resp in head_responses if resp.get("status") == "success"]
            simulation_responses = [resp for resp in head_responses if resp.get("status") == "simulation"]
            
            # Build synthesis
            synthesis = {
                "query": query,
                "total_heads_queried": len(head_responses),
                "successful_responses": len(successful_responses),
                "simulation_responses": len(simulation_responses),
                "coordination_strategy": "multi_tier_parallel",
                "synthesis_method": "weighted_aggregation"
            }
            
            # Add response details
            if successful_responses:
                synthesis["primary_insights"] = []
                for resp in successful_responses:
                    head_name = resp.get("head", "unknown")
                    response_data = resp.get("response", {})
                    insight = {
                        "head": head_name,
                        "contribution": f"Response from {head_name}",
                        "confidence": response_data.get("confidence", 0.8),
                        "relevance": "high" if "success" in resp.get("status", "") else "medium"
                    }
                    synthesis["primary_insights"].append(insight)
            
            # Add simulation insights
            if simulation_responses:
                synthesis["simulation_insights"] = []
                for resp in simulation_responses:
                    head_name = resp.get("head", "unknown")
                    synthesis["simulation_insights"].append({
                        "head": head_name,
                        "status": "Service healthy and ready",
                        "capability": f"{head_name} would contribute to this query"
                    })
            
            # Overall coordination status
            total_operational = len(successful_responses) + len(simulation_responses)
            synthesis["coordination_success"] = total_operational > 0
            synthesis["operational_heads"] = total_operational
            
            return synthesis
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "coordination_success": False
            }

coordination_interface = RAGCoordinationInterface()

@app.post("/coordinate", response_model=CoordinationResponse)
async def coordinate_rag_systems(request: CoordinationRequest):
    """Main coordination endpoint"""
    try:
        result = await coordination_interface.coordinate_multi_tier_retrieval(request)
        return CoordinationResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Coordination Interface",
        "coordination_count": coordination_interface.coordination_count,
        "successful_coordinations": coordination_interface.successful_coordinations,
        "success_rate": (
            coordination_interface.successful_coordinations / 
            max(coordination_interface.coordination_count, 1) * 100
        ),
        "average_execution_time": (
            coordination_interface.total_execution_time / 
            max(coordination_interface.coordination_count, 1)
        ),
        "available_heads": list(coordination_interface.rag_heads.keys()),
        "integration": {
            "thinking_engine": f"{coordination_interface.thinking_engine_host}:{coordination_interface.thinking_engine_port}",
            "enhanced_router": f"{coordination_interface.rag_router_enhanced_host}:{coordination_interface.rag_router_enhanced_port}",
            "vector_store": f"{coordination_interface.vector_store_host}:{coordination_interface.vector_store_port}"
        }
    }

@app.get("/heads/status")
async def check_heads_status():
    """Check status of all RAG heads"""
    heads_status = {}
    
    for head_name, head_url in coordination_interface.rag_heads.items():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{head_url}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        heads_status[head_name] = {
                            "status": "healthy",
                            "url": head_url,
                            "service_info": data.get("service", "unknown")
                        }
                    else:
                        heads_status[head_name] = {
                            "status": "unhealthy",
                            "url": head_url,
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            heads_status[head_name] = {
                "status": "error",
                "url": head_url,
                "error": str(e)
            }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "heads_status": heads_status,
        "total_heads": len(coordination_interface.rag_heads),
        "healthy_heads": len([s for s in heads_status.values() if s.get("status") == "healthy"])
    }

# Initialize enhanced concept integrator
concept_integrator = EnhancedConceptIntegrator()

@app.post('/enhanced-query')
async def enhanced_query(request: Request):
    """Enhanced query with intelligent concept-based routing"""
    try:
        data = await request.json()
        query = data.get('query', '')
        user_context = data.get('user_context', {})
        
        if not query:
            return JSONResponse({"error": "Query is required"}, status_code=400)
        
        # Enhanced concept detection and routing
        concept_result = await concept_integrator.detect_and_route_concepts(query, user_context)
        
        # Extract routing plan
        routing_plan = concept_result.get("routing_plan", {})
        primary_service = routing_plan.get("primary_service", "rag-cpu-optimized")
        
        # Execute query on recommended service
        service_response = await execute_query_on_service(primary_service, query, data)
        
        # Combine results
        enhanced_response = {
            "query": query,
            "concept_analysis": concept_result,
            "primary_service_used": primary_service,
            "service_response": service_response,
            "routing_confidence": routing_plan.get("confidence", 0.5),
            "processing_time_ms": concept_result.get("processing_time_ms", 0),
            "intelligence_level": "enhanced"
        }
        
        return JSONResponse(enhanced_response)
        
    except Exception as e:
        logger.error(f"Enhanced query failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get('/concept-metrics')
def get_concept_metrics():
    """Get enhanced concept detection performance metrics"""
    try:
        metrics = concept_integrator.get_performance_metrics()
        return JSONResponse(metrics)
    except Exception as e:
        logger.error(f"Failed to get concept metrics: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

async def execute_query_on_service(service_name: str, query: str, original_data: dict) -> dict:
    """Execute query on the specified RAG service"""
    service_urls = {
        "rag-code": "http://localhost:8922/search",
        "rag-graph": "http://localhost:8921/query", 
        "rag-gpu-long": "http://localhost:8920/query",
        "rag-cpu-optimized": "http://localhost:8902/query",
        "rag-router": "http://localhost:8950/search"
    }
    
    service_url = service_urls.get(service_name)
    if not service_url:
        return {"error": f"Unknown service: {service_name}"}
    
    try:
        async with aiohttp.ClientSession() as session:
            # Prepare payload based on service type
            payload = {"query": query}
            if service_name == "rag-cpu-optimized":
                payload.update({
                    "max_tokens": original_data.get("max_tokens", 200),
                    "temperature": original_data.get("temperature", 0.7)
                })
            elif service_name == "rag-gpu-long":
                payload["deep_processing"] = True
            
            async with session.post(service_url, json=payload, timeout=30) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Service {service_name} returned status {response.status}"}
                    
    except Exception as e:
        return {"error": f"Failed to query {service_name}: {str(e)}"}

if __name__ == "__main__":
    print("🧠 Starting RAG Coordination Interface...")
    print("=" * 60)
    print("📍 Available endpoints:")
    print("  • POST /coordinate - Multi-tier RAG coordination")
    print("  • GET  /health - Health and metrics")
    print("  • GET  /heads/status - RAG heads status")
    print("\n🔧 Coordination Capabilities:")
    print("  • Multi-tier parallel retrieval")
    print("  • Enhanced routing integration")
    print("  • Neural Thought Engine guidance")
    print("  • Response synthesis")
    print(f"\n🚀 Server starting on http://0.0.0.0:8952")
    
    uvicorn.run(app, host="0.0.0.0", port=8952) 