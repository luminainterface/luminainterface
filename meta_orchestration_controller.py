#!/usr/bin/env python3
"""
🎯 META-ORCHESTRATION CONTROLLER - STRATEGIC LOGIC
Layer 2 of the Ultimate AI Orchestration Architecture v10

This service provides strategic logic through:
- Deep context analysis
- Dynamic strategy selection
- 7 orchestration strategies
- Adaptive parameter tuning
"""

import os
import sys
import json
import time
import redis
import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestrationStrategy(Enum):
    """7 Orchestration Strategies"""
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_MAXIMIZED = "quality_maximized"
    CONCEPT_FOCUSED = "concept_focused"
    RESEARCH_INTENSIVE = "research_intensive"
    CREATIVE_SYNTHESIS = "creative_synthesis"
    VERIFICATION_HEAVY = "verification_heavy"
    ADAPTIVE_LEARNING = "adaptive_learning"

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str
    weight: float
    parameters: Dict[str, Any]
    description: str

class MetaOrchestrationController:
    """🎯 Meta-Orchestration Controller - Strategic Logic"""
    
    def __init__(self):
        self.redis_client = self._setup_redis()
        self.neural_engine_host = os.getenv('NEURAL_ENGINE_HOST', 'localhost')
        self.neural_engine_port = int(os.getenv('NEURAL_ENGINE_PORT', 8890))
        self.rag_coordination_host = os.getenv('RAG_COORDINATION_HOST', 'localhost')
        self.rag_coordination_port = int(os.getenv('RAG_COORDINATION_PORT', 8952))
        self.multi_concept_detector_host = os.getenv('MULTI_CONCEPT_DETECTOR_HOST', 'localhost')
        self.multi_concept_detector_port = int(os.getenv('MULTI_CONCEPT_DETECTOR_PORT', 8860))
        
        # Strategic orchestration parameters
        self.concept_detection_importance = float(os.getenv('CONCEPT_DETECTION_IMPORTANCE', 0.8))
        self.verification_thoroughness = float(os.getenv('VERIFICATION_THOROUGHNESS', 0.7))
        self.speed_vs_quality_balance = float(os.getenv('SPEED_VS_QUALITY_BALANCE', 0.6))
        self.research_depth_preference = float(os.getenv('RESEARCH_DEPTH_PREFERENCE', 0.5))
        
        # Initialize orchestration strategies
        self.strategies = self._initialize_strategies()
        self.current_strategy_weights = self._get_default_strategy_weights()
        
        logger.info("🎯 Meta-Orchestration Controller initialized with strategic logic")
    
    def _setup_redis(self) -> redis.Redis:
        """Setup Redis connection"""
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            redis_password = os.getenv('REDIS_PASSWORD', '')
            
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True
            )
            client.ping()
            logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
            return client
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    def _initialize_strategies(self) -> Dict[OrchestrationStrategy, StrategyConfig]:
        """Initialize the 7 orchestration strategies"""
        return {
            OrchestrationStrategy.SPEED_OPTIMIZED: StrategyConfig(
                name="Speed Optimized",
                weight=0.2,
                parameters={
                    "timeout_reduction": 0.5,
                    "parallel_processing": True,
                    "cache_priority": "high",
                    "verification_level": "basic"
                },
                description="Fast response priority with minimal overhead"
            ),
            OrchestrationStrategy.QUALITY_MAXIMIZED: StrategyConfig(
                name="Quality Maximized",
                weight=0.3,
                parameters={
                    "verification_stages": 3,
                    "consensus_threshold": 0.8,
                    "refinement_cycles": 2,
                    "quality_gates": True
                },
                description="Thorough analysis priority with multiple verification stages"
            ),
            OrchestrationStrategy.CONCEPT_FOCUSED: StrategyConfig(
                name="Concept Focused",
                weight=0.15,
                parameters={
                    "concept_detection_weight": 0.9,
                    "concept_validation": True,
                    "concept_enhancement": True,
                    "multi_concept_analysis": True
                },
                description="Enhanced concept detection and analysis priority"
            ),
            OrchestrationStrategy.RESEARCH_INTENSIVE: StrategyConfig(
                name="Research Intensive",
                weight=0.1,
                parameters={
                    "knowledge_depth": "deep",
                    "source_diversity": "high",
                    "fact_checking": True,
                    "research_expansion": True
                },
                description="Deep knowledge retrieval and research priority"
            ),
            OrchestrationStrategy.CREATIVE_SYNTHESIS: StrategyConfig(
                name="Creative Synthesis",
                weight=0.1,
                parameters={
                    "creativity_boost": 0.8,
                    "novel_combinations": True,
                    "lateral_thinking": True,
                    "innovation_weight": 0.7
                },
                description="Innovation and creative thinking priority"
            ),
            OrchestrationStrategy.VERIFICATION_HEAVY: StrategyConfig(
                name="Verification Heavy",
                weight=0.1,
                parameters={
                    "verification_rounds": 4,
                    "cross_validation": True,
                    "accuracy_threshold": 0.95,
                    "error_detection": "aggressive"
                },
                description="Accuracy and verification priority"
            ),
            OrchestrationStrategy.ADAPTIVE_LEARNING: StrategyConfig(
                name="Adaptive Learning",
                weight=0.05,
                parameters={
                    "learning_rate": 0.6,
                    "pattern_adaptation": True,
                    "feedback_integration": True,
                    "continuous_improvement": True
                },
                description="Continuous improvement and adaptation priority"
            )
        }
    
    def _get_default_strategy_weights(self) -> Dict[str, float]:
        """Get default strategy weights"""
        return {strategy.value: config.weight for strategy, config in self.strategies.items()}
    
    async def analyze_context(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """🧠 Perform deep context analysis"""
        try:
            if context is None:
                context = {}
            
            analysis = {
                "query_complexity": self._assess_query_complexity(query),
                "required_strategies": [],
                "recommended_weights": {},
                "processing_requirements": {},
                "estimated_resources": {}
            }
            
            # Assess query complexity
            complexity = analysis["query_complexity"]
            
            # Determine required strategies based on complexity
            if complexity["technical_depth"] > 0.7:
                analysis["required_strategies"].append("research_intensive")
            if complexity["creative_elements"] > 0.6:
                analysis["required_strategies"].append("creative_synthesis")
            if complexity["verification_needs"] > 0.8:
                analysis["required_strategies"].append("verification_heavy")
            if complexity["concept_density"] > 0.7:
                analysis["required_strategies"].append("concept_focused")
            if complexity["speed_requirement"] > 0.8:
                analysis["required_strategies"].append("speed_optimized")
            
            # Calculate recommended weights
            total_strategies = len(analysis["required_strategies"])
            if total_strategies > 0:
                base_weight = 0.8 / total_strategies
                for strategy in analysis["required_strategies"]:
                    analysis["recommended_weights"][strategy] = base_weight
                
                # Distribute remaining weight to quality_maximized
                analysis["recommended_weights"]["quality_maximized"] = 0.2
            else:
                # Default distribution
                analysis["recommended_weights"] = self.current_strategy_weights.copy()
            
            # Estimate processing requirements
            analysis["processing_requirements"] = {
                "estimated_time": complexity["overall_complexity"] * 30,  # seconds
                "memory_requirement": complexity["overall_complexity"] * 512,  # MB
                "cpu_intensity": complexity["overall_complexity"],
                "parallel_processing": complexity["overall_complexity"] > 0.6
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Context analysis failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def _assess_query_complexity(self, query: str) -> Dict[str, float]:
        """Assess query complexity across multiple dimensions"""
        try:
            # Simple heuristic-based complexity assessment
            query_lower = query.lower()
            word_count = len(query.split())
            
            # Technical depth indicators
            technical_keywords = ["algorithm", "implementation", "architecture", "system", "performance", "optimization"]
            technical_score = sum(1 for keyword in technical_keywords if keyword in query_lower) / len(technical_keywords)
            
            # Creative elements indicators
            creative_keywords = ["creative", "innovative", "design", "brainstorm", "imagine", "generate"]
            creative_score = sum(1 for keyword in creative_keywords if keyword in query_lower) / len(creative_keywords)
            
            # Verification needs indicators
            verification_keywords = ["verify", "check", "validate", "confirm", "accurate", "correct"]
            verification_score = sum(1 for keyword in verification_keywords if keyword in query_lower) / len(verification_keywords)
            
            # Concept density (based on word count and complexity indicators)
            concept_density = min(word_count / 50.0, 1.0)  # Normalize to 0-1
            
            # Speed requirement indicators
            speed_keywords = ["quick", "fast", "immediate", "urgent", "asap"]
            speed_score = sum(1 for keyword in speed_keywords if keyword in query_lower) / len(speed_keywords)
            
            # Overall complexity
            overall_complexity = (technical_score + creative_score + verification_score + concept_density) / 4
            
            return {
                "technical_depth": technical_score,
                "creative_elements": creative_score,
                "verification_needs": verification_score,
                "concept_density": concept_density,
                "speed_requirement": speed_score,
                "overall_complexity": overall_complexity,
                "word_count": word_count
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Complexity assessment failed: {e}")
            return {
                "technical_depth": 0.5,
                "creative_elements": 0.5,
                "verification_needs": 0.5,
                "concept_density": 0.5,
                "speed_requirement": 0.5,
                "overall_complexity": 0.5,
                "word_count": 0
            }
    
    async def select_optimal_strategy(self, context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """⚡ Select optimal orchestration strategy"""
        try:
            # Get recommended weights from context analysis
            recommended_weights = context_analysis.get("recommended_weights", self.current_strategy_weights)
            
            # Apply current strategic adjustments from high-rank adapter
            current_adjustments = self.redis_client.get("strategic_adjustments")
            if current_adjustments:
                try:
                    adjustments = json.loads(current_adjustments)
                    # Merge adjustments with recommended weights
                    for strategy, weight in adjustments.get("strategy_weights", {}).items():
                        if strategy in recommended_weights:
                            recommended_weights[strategy] = weight
                except json.JSONDecodeError:
                    logger.warning("⚠️ Failed to parse strategic adjustments")
            
            # Select primary strategy (highest weight)
            primary_strategy = max(recommended_weights.items(), key=lambda x: x[1])
            
            # Build strategy configuration
            strategy_config = {
                "primary_strategy": primary_strategy[0],
                "primary_weight": primary_strategy[1],
                "all_weights": recommended_weights,
                "strategy_parameters": {},
                "execution_plan": []
            }
            
            # Get parameters for primary strategy
            primary_enum = None
            for enum_strategy in OrchestrationStrategy:
                if enum_strategy.value == primary_strategy[0]:
                    primary_enum = enum_strategy
                    break
            
            if primary_enum and primary_enum in self.strategies:
                strategy_config["strategy_parameters"] = self.strategies[primary_enum].parameters.copy()
            
            # Build execution plan based on strategy
            execution_plan = await self._build_execution_plan(strategy_config, context_analysis)
            strategy_config["execution_plan"] = execution_plan
            
            # Store strategy selection
            self.redis_client.setex(
                f"selected_strategy:{datetime.now().isoformat()}",
                3600,  # 1 hour TTL
                json.dumps(strategy_config)
            )
            
            return {
                "status": "strategy_selected",
                "strategy_config": strategy_config,
                "context_analysis": context_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy selection failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _build_execution_plan(self, strategy_config: Dict[str, Any], context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build detailed execution plan based on strategy"""
        try:
            plan = []
            primary_strategy = strategy_config["primary_strategy"]
            
            # Phase 1: Initialization
            plan.append({
                "phase": 1,
                "name": "Initialization",
                "description": "Initialize orchestration with selected strategy",
                "services": ["neural-thought-engine"],
                "parameters": {"strategy": primary_strategy}
            })
            
            # Phase 2: Context Processing
            if primary_strategy in ["concept_focused", "research_intensive"]:
                plan.append({
                    "phase": 2,
                    "name": "Enhanced Concept Detection",
                    "description": "Deep concept analysis and detection",
                    "services": ["multi-concept-detector", "rag-coordination-enhanced"],
                    "parameters": {"concept_priority": "high"}
                })
            else:
                plan.append({
                    "phase": 2,
                    "name": "Basic Context Processing",
                    "description": "Standard context analysis",
                    "services": ["neural-thought-engine"],
                    "parameters": {"processing_mode": "standard"}
                })
            
            # Phase 3: Strategy-specific processing
            if primary_strategy == "speed_optimized":
                plan.append({
                    "phase": 3,
                    "name": "Fast Processing",
                    "description": "Optimized for speed",
                    "services": ["rag-cpu-optimized"],
                    "parameters": {"timeout": 5, "parallel": True}
                })
            elif primary_strategy == "quality_maximized":
                plan.append({
                    "phase": 3,
                    "name": "Quality Processing",
                    "description": "Multi-stage quality enhancement",
                    "services": ["rag-orchestrator", "swarm-intelligence-engine"],
                    "parameters": {"verification_stages": 3}
                })
            elif primary_strategy == "research_intensive":
                plan.append({
                    "phase": 3,
                    "name": "Deep Research",
                    "description": "Comprehensive knowledge retrieval",
                    "services": ["rag-gpu-long", "rag-graph"],
                    "parameters": {"depth": "maximum"}
                })
            
            # Phase 4: Verification (if needed)
            if primary_strategy in ["verification_heavy", "quality_maximized"]:
                plan.append({
                    "phase": 4,
                    "name": "Verification",
                    "description": "Multi-stage verification and validation",
                    "services": ["consensus-manager", "emergence-detector"],
                    "parameters": {"verification_level": "high"}
                })
            
            return plan
            
        except Exception as e:
            logger.error(f"❌ Execution plan building failed: {e}")
            return []
    
    async def coordinate_execution(self, execution_plan: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """🎛️ Coordinate execution according to plan"""
        try:
            results = {
                "status": "coordinating",
                "phases_completed": 0,
                "total_phases": len(execution_plan),
                "phase_results": [],
                "overall_result": None
            }
            
            for phase in execution_plan:
                phase_start = time.time()
                
                try:
                    # Execute phase
                    phase_result = await self._execute_phase(phase, query)
                    phase_result["execution_time"] = time.time() - phase_start
                    
                    results["phase_results"].append(phase_result)
                    results["phases_completed"] += 1
                    
                    logger.info(f"✅ Phase {phase['phase']} completed: {phase['name']}")
                    
                except Exception as phase_error:
                    logger.error(f"❌ Phase {phase['phase']} failed: {phase_error}")
                    results["phase_results"].append({
                        "phase": phase["phase"],
                        "status": "failed",
                        "error": str(phase_error),
                        "execution_time": time.time() - phase_start
                    })
            
            # Compile overall result
            successful_phases = [r for r in results["phase_results"] if r.get("status") == "success"]
            if successful_phases:
                results["overall_result"] = "success"
                results["status"] = "completed"
            else:
                results["overall_result"] = "failed"
                results["status"] = "failed"
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Execution coordination failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _execute_phase(self, phase: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Execute a single phase of the plan"""
        try:
            phase_result = {
                "phase": phase["phase"],
                "name": phase["name"],
                "status": "executing",
                "services_called": [],
                "results": []
            }
            
            # Call each service in the phase
            for service in phase["services"]:
                try:
                    service_result = await self._call_service(service, query, phase["parameters"])
                    phase_result["services_called"].append(service)
                    phase_result["results"].append(service_result)
                except Exception as service_error:
                    logger.warning(f"⚠️ Service {service} failed: {service_error}")
                    phase_result["results"].append({
                        "service": service,
                        "status": "failed",
                        "error": str(service_error)
                    })
            
            phase_result["status"] = "success"
            return phase_result
            
        except Exception as e:
            return {
                "phase": phase.get("phase", 0),
                "status": "failed",
                "error": str(e)
            }
    
    async def _call_service(self, service: str, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific service"""
        try:
            # Service endpoint mapping
            service_ports = {
                "neural-thought-engine": self.neural_engine_port,
                "multi-concept-detector": self.multi_concept_detector_port,
                "rag-coordination-enhanced": self.rag_coordination_port,
                "rag-orchestrator": 8953,
                "rag-cpu-optimized": 8902,
                "rag-gpu-long": 8920,
                "rag-graph": 8921,
                "swarm-intelligence-engine": 8977,
                "consensus-manager": 8978,
                "emergence-detector": 8979
            }
            
            port = service_ports.get(service)
            if not port:
                return {"status": "unknown_service", "service": service}
            
            # Make service call
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query": query,
                    "parameters": parameters,
                    "timestamp": datetime.now().isoformat()
                }
                
                async with session.post(
                    f"http://localhost:{port}/process",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"status": "success", "service": service, "result": result}
                    else:
                        return {"status": "error", "service": service, "status_code": response.status}
                        
        except Exception as e:
            return {"status": "failed", "service": service, "error": str(e)}

# FastAPI app
app = FastAPI(title="🎯 Meta-Orchestration Controller", version="10.0.0")

# Global controller instance
controller = None

@app.on_event("startup")
async def startup_event():
    global controller
    controller = MetaOrchestrationController()
    logger.info("🎯 Meta-Orchestration Controller startup complete")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        controller.redis_client.ping()
        return {"status": "healthy", "service": "meta-orchestration-controller", "version": "10.0.0"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.post("/update_strategy")
async def update_strategy(request: Request):
    """Update orchestration strategy from high-rank adapter"""
    try:
        strategy_update = await request.json()
        
        # Store strategic adjustments
        controller.redis_client.setex(
            "strategic_adjustments",
            3600,  # 1 hour TTL
            json.dumps(strategy_update)
        )
        
        # Update current strategy weights
        if "strategy_weights" in strategy_update:
            controller.current_strategy_weights.update(strategy_update["strategy_weights"])
        
        logger.info("✅ Strategy updated from high-rank adapter")
        return {"status": "strategy_updated", "update": strategy_update}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/orchestrate")
async def orchestrate_query(request: Request):
    """🎛️ Main orchestration endpoint"""
    try:
        data = await request.json()
        query = data.get("query", "")
        context = data.get("context", {})
        
        # Step 1: Analyze context
        context_analysis = await controller.analyze_context(query, context)
        
        # Step 2: Select optimal strategy
        strategy_selection = await controller.select_optimal_strategy(context_analysis)
        
        # Step 3: Coordinate execution
        if strategy_selection.get("status") == "strategy_selected":
            execution_plan = strategy_selection["strategy_config"]["execution_plan"]
            execution_result = await controller.coordinate_execution(execution_plan, query)
            
            return {
                "status": "orchestrated",
                "query": query,
                "context_analysis": context_analysis,
                "strategy_selection": strategy_selection,
                "execution_result": execution_result
            }
        else:
            return {"status": "strategy_selection_failed", "error": strategy_selection}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies")
async def get_strategies():
    """Get available orchestration strategies"""
    try:
        return {
            "strategies": [
                {
                    "name": strategy.value,
                    "config": {
                        "display_name": config.name,
                        "weight": config.weight,
                        "description": config.description,
                        "parameters": config.parameters
                    }
                }
                for strategy, config in controller.strategies.items()
            ],
            "current_weights": controller.current_strategy_weights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get controller status and metrics"""
    try:
        return {
            "service": "meta-orchestration-controller",
            "version": "10.0.0",
            "status": "operational",
            "strategies_count": len(controller.strategies),
            "current_strategy_weights": controller.current_strategy_weights,
            "orchestration_parameters": {
                "concept_detection_importance": controller.concept_detection_importance,
                "verification_thoroughness": controller.verification_thoroughness,
                "speed_vs_quality_balance": controller.speed_vs_quality_balance,
                "research_depth_preference": controller.research_depth_preference
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8999))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info") 