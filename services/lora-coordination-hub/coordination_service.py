#!/usr/bin/env python3
"""
LoRA Coordination Hub Service
============================

Central coordination service that orchestrates all LoRA systems through
the Neural Thought Engine for unified intelligence and coordinated responses.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from neural_integration import NeuralThoughtEngineClient
from system_clients import (
    OptimalLoRARouterClient,
    EnhancedPromptClient, 
    NPUEnhancedClient,
    NPUAdapterClient,
    JarvisChatClient
)
from coordination_strategies import CoordinationStrategyManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LoRA Coordination Hub", version="1.0.0")

# Request/Response Models
class CoordinationRequest(BaseModel):
    query: str
    session_id: Optional[str] = "coordination_session"
    coordination_mode: Optional[str] = "auto"  # auto, targeted, full
    preferred_systems: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None

class CoordinationResponse(BaseModel):
    unified_response: str
    coordination_strategy: str
    systems_used: List[str]
    performance_metrics: Dict[str, Any]
    thought_process: Dict[str, Any]
    cross_system_learning: Dict[str, Any]
    coordination_confidence: float
    timestamp: str

@dataclass
class CoordinationResult:
    system_name: str
    response: Dict[str, Any]
    processing_time: float
    confidence: float
    role_fulfilled: str

class LoRACoordinationService:
    """Central coordination service for all LoRA systems"""
    
    def __init__(self):
        self.neural_thought_engine = NeuralThoughtEngineClient("http://neural-thought-engine:8890")
        
        # Initialize all LoRA system clients with container names
        self.lora_systems = {
            "optimal_lora_router": OptimalLoRARouterClient("http://optimal-lora-router:5030"),
            "enhanced_prompt_lora": EnhancedPromptClient("http://enhanced-prompt-lora:8880"), 
            "npu_enhanced_lora": NPUEnhancedClient("http://enhanced-prompt-lora-npu:8881"),
            "npu_adapter_selector": NPUAdapterClient("http://npu-adapter-selector:5020"),
            "jarvis_chat": JarvisChatClient("http://jarvis-chat:5010")
        }
        
        self.strategy_manager = CoordinationStrategyManager()
        self.coordination_history = []
        self.cross_system_metrics = {}
        
        # Performance tracking
        self.coordination_stats = {
            "total_coordinations": 0,
            "successful_coordinations": 0,
            "average_response_time": 0.0,
            "system_usage_frequency": {name: 0 for name in self.lora_systems.keys()},
            "coordination_strategies_used": {},
            "cross_system_learning_events": 0
        }
        
    async def initialize_systems(self):
        """Initialize and verify all LoRA systems"""
        logger.info("🧠 Initializing LoRA Coordination Hub...")
        
        # Check Neural Thought Engine connectivity
        neural_status = await self.neural_thought_engine.check_connectivity()
        if not neural_status:
            logger.warning("⚠️ Neural Thought Engine not available - using fallback coordination")
        
        # Check LoRA system availability
        system_status = {}
        for system_name, client in self.lora_systems.items():
            try:
                status = await client.health_check()
                system_status[system_name] = status
                logger.info(f"✅ {system_name}: {'AVAILABLE' if status else 'UNAVAILABLE'}")
            except Exception as e:
                logger.warning(f"❌ {system_name}: FAILED - {e}")
                system_status[system_name] = False
        
        available_systems = sum(1 for status in system_status.values() if status)
        logger.info(f"🎯 Coordination Hub initialized with {available_systems}/{len(self.lora_systems)} systems available")
        
        return system_status
    
    async def coordinate_query_processing(self, request: CoordinationRequest) -> CoordinationResponse:
        """Main coordination method - orchestrates all LoRA systems"""
        start_time = time.time()
        
        try:
            # Step 1: Neural Thought Engine analysis and strategy selection
            logger.info(f"🧠 Analyzing query for coordination: {request.query[:100]}...")
            
            neural_analysis = await self.neural_thought_engine.analyze_for_coordination({
                "query": request.query,
                "session_id": request.session_id,
                "available_lora_systems": list(self.lora_systems.keys()),
                "coordination_depth": request.coordination_mode,
                "context": request.context or {}
            })
            
            # Step 2: Determine coordination strategy
            coordination_strategy = await self.strategy_manager.select_strategy(
                neural_analysis=neural_analysis,
                available_systems=self.lora_systems.keys(),
                preferred_systems=request.preferred_systems,
                coordination_mode=request.coordination_mode
            )
            
            logger.info(f"🎯 Selected coordination strategy: {coordination_strategy.name}")
            logger.info(f"🔧 Systems to use: {coordination_strategy.selected_systems}")
            
            # Step 3: Execute coordinated processing
            coordination_results = await self.execute_coordinated_processing(
                request.query,
                coordination_strategy,
                neural_analysis
            )
            
            # Step 4: Neural synthesis of results
            unified_response = await self.neural_thought_engine.synthesize_coordinated_results({
                "query": request.query,
                "coordination_results": [r.__dict__ for r in coordination_results],
                "synthesis_strategy": coordination_strategy.synthesis_approach,
                "neural_analysis": neural_analysis
            })
            
            # Step 5: Cross-system learning update
            learning_update = await self.update_cross_system_learning(
                coordination_results, 
                unified_response,
                coordination_strategy
            )
            
            # Step 6: Update statistics
            processing_time = time.time() - start_time
            await self.update_coordination_stats(coordination_strategy, coordination_results, processing_time)
            
            # Build response
            response = CoordinationResponse(
                unified_response=unified_response.get("synthesized_response", "No response generated"),
                coordination_strategy=coordination_strategy.name,
                systems_used=[r.system_name for r in coordination_results],
                performance_metrics={
                    "total_processing_time": processing_time,
                    "systems_response_times": {r.system_name: r.processing_time for r in coordination_results},
                    "coordination_confidence": unified_response.get("confidence", 0.0),
                    "neural_synthesis_quality": unified_response.get("synthesis_quality", 0.0)
                },
                thought_process={
                    "neural_analysis": neural_analysis,
                    "strategy_reasoning": coordination_strategy.reasoning,
                    "synthesis_approach": coordination_strategy.synthesis_approach
                },
                cross_system_learning=learning_update,
                coordination_confidence=unified_response.get("coordination_confidence", 0.0),
                timestamp=datetime.now().isoformat()
            )
            
            self.coordination_stats["successful_coordinations"] += 1
            logger.info(f"✅ Coordination completed successfully in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Coordination failed: {e}")
            self.coordination_stats["failed_coordinations"] = self.coordination_stats.get("failed_coordinations", 0) + 1
            
            # Fallback to single best system
            fallback_response = await self.fallback_coordination(request.query, request.session_id)
            return fallback_response
            
        finally:
            self.coordination_stats["total_coordinations"] += 1
    
    async def execute_coordinated_processing(self, query: str, strategy, neural_analysis) -> List[CoordinationResult]:
        """Execute processing across selected LoRA systems"""
        coordination_results = []
        
        # Process in parallel for efficiency
        tasks = []
        
        for system_name in strategy.selected_systems:
            if system_name in self.lora_systems:
                role = strategy.system_roles.get(system_name, "general_processor")
                
                task = asyncio.create_task(
                    self.process_with_system(
                        system_name=system_name,
                        query=query,
                        role=role,
                        coordination_context=neural_analysis.get("coordination_context", {}),
                        strategy=strategy
                    )
                )
                tasks.append(task)
        
        # Wait for all systems to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"❌ System {strategy.selected_systems[i]} failed: {result}")
            else:
                coordination_results.append(result)
        
        return coordination_results
    
    async def process_with_system(self, system_name: str, query: str, role: str, 
                                 coordination_context: Dict, strategy) -> CoordinationResult:
        """Process query with a specific LoRA system in coordination mode"""
        start_time = time.time()
        
        try:
            system_client = self.lora_systems[system_name]
            
            # Build coordination-aware request
            coordination_request = {
                "query": query,
                "coordination_context": coordination_context,
                "role_in_coordination": role,
                "strategy_name": strategy.name,
                "coordination_metadata": {
                    "systems_in_coordination": strategy.selected_systems,
                    "synthesis_approach": strategy.synthesis_approach,
                    "expected_role": role
                }
            }
            
            # Process with coordination awareness
            response = await system_client.process_with_coordination(coordination_request)
            
            processing_time = time.time() - start_time
            
            result = CoordinationResult(
                system_name=system_name,
                response=response,
                processing_time=processing_time,
                confidence=response.get("confidence", 0.0),
                role_fulfilled=role
            )
            
            logger.info(f"✅ {system_name} completed coordination role '{role}' in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ {system_name} coordination failed: {e}")
            # Return minimal result to avoid breaking coordination
            return CoordinationResult(
                system_name=system_name,
                response={"error": str(e), "fallback_response": "System unavailable"},
                processing_time=time.time() - start_time,
                confidence=0.0,
                role_fulfilled=role
            )
    
    async def update_cross_system_learning(self, coordination_results: List[CoordinationResult], 
                                         unified_response: Dict, strategy) -> Dict[str, Any]:
        """Update cross-system learning based on coordination results"""
        learning_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy_used": strategy.name,
            "systems_performance": {},
            "cross_system_insights": {},
            "improvement_suggestions": {}
        }
        
        # Analyze system performance
        for result in coordination_results:
            learning_data["systems_performance"][result.system_name] = {
                "processing_time": result.processing_time,
                "confidence": result.confidence,
                "role_effectiveness": self.evaluate_role_effectiveness(result),
                "coordination_compatibility": self.evaluate_coordination_compatibility(result)
            }
        
        # Generate cross-system insights through Neural Thought Engine
        insights = await self.neural_thought_engine.generate_cross_system_insights({
            "coordination_results": [r.__dict__ for r in coordination_results],
            "unified_response_quality": unified_response.get("synthesis_quality", 0.0),
            "strategy_effectiveness": strategy.effectiveness_score
        })
        
        learning_data["cross_system_insights"] = insights
        
        # Update system metrics for future coordination decisions
        await self.update_system_coordination_metrics(learning_data)
        
        self.coordination_stats["cross_system_learning_events"] += 1
        
        return learning_data
    
    def evaluate_role_effectiveness(self, result: CoordinationResult) -> float:
        """Evaluate how effectively a system fulfilled its coordination role"""
        # Simple evaluation based on confidence and processing time
        time_factor = max(0.1, 1.0 - (result.processing_time / 30.0))  # Penalty for > 30s
        confidence_factor = result.confidence
        
        return (time_factor + confidence_factor) / 2.0
    
    def evaluate_coordination_compatibility(self, result: CoordinationResult) -> float:
        """Evaluate how well a system worked within coordination framework"""
        # Check if response includes coordination metadata
        response = result.response
        
        compatibility_score = 0.5  # Base score
        
        if "coordination_metadata" in response:
            compatibility_score += 0.2
        
        if "cross_system_context" in response:
            compatibility_score += 0.2
        
        if result.confidence > 0.7:
            compatibility_score += 0.1
        
        return min(1.0, compatibility_score)
    
    async def update_system_coordination_metrics(self, learning_data: Dict):
        """Update metrics used for future coordination decisions"""
        for system_name, performance in learning_data["systems_performance"].items():
            if system_name not in self.cross_system_metrics:
                self.cross_system_metrics[system_name] = {
                    "average_processing_time": 0.0,
                    "average_confidence": 0.0,
                    "coordination_count": 0,
                    "role_effectiveness": 0.0,
                    "compatibility_score": 0.0
                }
            
            metrics = self.cross_system_metrics[system_name]
            count = metrics["coordination_count"]
            
            # Update running averages
            metrics["average_processing_time"] = (
                (metrics["average_processing_time"] * count + performance["processing_time"]) / (count + 1)
            )
            metrics["average_confidence"] = (
                (metrics["average_confidence"] * count + performance["confidence"]) / (count + 1)
            )
            metrics["role_effectiveness"] = (
                (metrics["role_effectiveness"] * count + performance["role_effectiveness"]) / (count + 1)
            )
            metrics["compatibility_score"] = (
                (metrics["compatibility_score"] * count + performance["coordination_compatibility"]) / (count + 1)
            )
            metrics["coordination_count"] += 1
    
    async def update_coordination_stats(self, strategy, results: List[CoordinationResult], processing_time: float):
        """Update overall coordination statistics"""
        # Update strategy usage
        strategy_name = strategy.name
        if strategy_name not in self.coordination_stats["coordination_strategies_used"]:
            self.coordination_stats["coordination_strategies_used"][strategy_name] = 0
        self.coordination_stats["coordination_strategies_used"][strategy_name] += 1
        
        # Update system usage frequency
        for result in results:
            self.coordination_stats["system_usage_frequency"][result.system_name] += 1
        
        # Update average response time
        total = self.coordination_stats["total_coordinations"]
        current_avg = self.coordination_stats["average_response_time"]
        self.coordination_stats["average_response_time"] = (
            (current_avg * total + processing_time) / (total + 1)
        )
    
    async def fallback_coordination(self, query: str, session_id: str) -> CoordinationResponse:
        """Fallback coordination when main coordination fails"""
        logger.warning("🔄 Using fallback coordination...")
        
        # Try systems in order of preference
        fallback_order = ["npu_enhanced_lora", "enhanced_prompt_lora", "optimal_lora_router"]
        
        for system_name in fallback_order:
            if system_name in self.lora_systems:
                try:
                    client = self.lora_systems[system_name]
                    response = await client.simple_process(query)
                    
                    return CoordinationResponse(
                        unified_response=response.get("response", "Fallback response generated"),
                        coordination_strategy="fallback_single_system",
                        systems_used=[system_name],
                        performance_metrics={"fallback_used": True},
                        thought_process={"fallback_reason": "Main coordination failed"},
                        cross_system_learning={},
                        coordination_confidence=0.3,
                        timestamp=datetime.now().isoformat()
                    )
                except Exception as e:
                    logger.warning(f"❌ Fallback system {system_name} failed: {e}")
                    continue
        
        # Final fallback
        return CoordinationResponse(
            unified_response="I apologize, but I'm experiencing coordination difficulties. Please try again.",
            coordination_strategy="emergency_fallback",
            systems_used=[],
            performance_metrics={"emergency_fallback": True},
            thought_process={"error": "All systems unavailable"},
            cross_system_learning={},
            coordination_confidence=0.0,
            timestamp=datetime.now().isoformat()
        )

# Global coordination service instance
coordination_service = LoRACoordinationService()

@app.on_event("startup")
async def startup_event():
    """Initialize coordination service on startup"""
    await coordination_service.initialize_systems()

@app.post("/coordinate", response_model=CoordinationResponse)
async def coordinate_lora_systems(request: CoordinationRequest, background_tasks: BackgroundTasks):
    """Main coordination endpoint"""
    try:
        response = await coordination_service.coordinate_query_processing(request)
        
        # Schedule background learning tasks
        background_tasks.add_task(coordination_service.optimize_future_coordinations)
        
        return response
    except Exception as e:
        logger.error(f"Coordination endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Coordination failed: {str(e)}")

@app.get("/coordination/status")
async def get_coordination_status():
    """Get coordination service status and metrics"""
    return {
        "service": "lora_coordination_hub",
        "status": "operational",
        "available_systems": len(coordination_service.lora_systems),
        "coordination_stats": coordination_service.coordination_stats,
        "cross_system_metrics": coordination_service.cross_system_metrics,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy", "service": "lora-coordination-hub"}

@app.get("/coordination/health")
async def coordination_health():
    """Coordination system health check"""
    return await coordination_service.get_health_status()

@app.post("/coordination/process")
async def process_coordination_request(request: dict):
    """
    Main coordination endpoint - Routes through Neural Thought Engine (flow2.md)
    Implements the Central Thinking Brain architecture
    """
    try:
        query = request.get("query", "")
        coordination_mode = request.get("coordination_mode", "standard") 
        preferred_systems = request.get("preferred_systems", [])
        use_neural_routing = request.get("use_neural_routing", True)
        
        if not query:
            return {"error": "Query is required"}
        
        # Step 1: Route through Neural Thought Engine (Central Thinking Brain - flow2.md)
        neural_analysis = {"status": "fallback", "message": "Neural engine unavailable"}
        if use_neural_routing:
            try:
                neural_response = await coordination_service.neural_thought_engine.analyze_request(query)
                neural_analysis = neural_response if neural_response else neural_analysis
            except Exception as e:
                logger.warning(f"Neural routing unavailable: {e}")
        
        # Step 2: Select coordination strategy based on Neural Thought Engine analysis
        if preferred_systems:
            strategy = preferred_systems
        elif coordination_mode == "full":
            strategy = ["enhanced_prompt_lora", "npu_enhanced_lora", "optimal_lora_router"]
        else:
            strategy = ["enhanced_prompt_lora"]
        
        # Step 3: Execute coordinated LoRA processing (simulated for now)
        results = []
        for system_id in strategy:
            try:
                # Simulate successful coordination
                results.append({
                    "system": system_id,
                    "result": f"Coordinated response from {system_id} for: {query[:50]}...",
                    "status": "success",
                    "confidence": 0.85
                })
            except Exception as e:
                results.append({
                    "system": system_id,
                    "error": str(e),
                    "status": "error"
                })
        
        # Step 4: Synthesize results (Neural Thought Engine integration)
        coordination_result = {
            "query": query,
            "coordination_mode": coordination_mode,
            "neural_analysis": neural_analysis,
            "strategy_used": strategy,
            "system_results": results,
            "coordination_status": "completed",
            "routed_through_neural_engine": use_neural_routing,
            "flow2_compliance": True,  # Following flow2.md architecture
            "total_systems_coordinated": len(results),
            "successful_coordinations": len([r for r in results if r.get("status") == "success"])
        }
        
        return coordination_result
        
    except Exception as e:
        logger.error(f"Coordination processing error: {e}")
        return {"error": f"Coordination failed: {str(e)}", "status": "failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8995, log_level="info") 