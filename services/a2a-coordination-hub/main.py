from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import asyncio
import aiohttp
import sympy as sp
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced A2A Coordination Hub - Intelligent Orchestrator", version="2.0.0")

class ProcessingPhase(Enum):
    BASELINE = "baseline"
    ENHANCED = "enhanced" 
    ORCHESTRATED = "orchestrated"
    COMPREHENSIVE = "comprehensive"

class ValidationResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"

class AgentCoordinationRequest(BaseModel):
    agents: List[str]
    task: str
    coordination_type: str = "intelligent"
    enable_routing: bool = True
    enable_validation: bool = True

class IntelligentQueryRequest(BaseModel):
    query: str
    enable_mathematical_validation: bool = True
    max_services: Optional[int] = None
    preferred_phase: Optional[str] = None

class CoordinationResponse(BaseModel):
    coordination_id: str
    agents: List[str]
    status: str
    processing_phase: str
    services_used: List[str]
    confidence_score: float
    validation_results: Dict[str, Any]
    result: Dict[str, Any]

class IntelligentResponse(BaseModel):
    query: str
    response: str
    confidence: float
    processing_phase: str
    services_used: List[str]
    validation_applied: bool
    mathematical_corrections: List[str]
    processing_time: float

# Enhanced service configuration from router.md
SERVICE_TIERS = {
    "baseline": [
        {"name": "rag_coordination_interface", "port": 8952, "endpoint": "/coordinate"},
        {"name": "rag_orchestrator", "port": 8953, "endpoint": "/orchestrate"},
        {"name": "rag_router_enhanced", "port": 8951, "endpoint": "/query"},
        {"name": "rag_cpu_optimized", "port": 8902, "endpoint": "/process"},
        {"name": "ollama", "port": 11434, "endpoint": "/api/generate"}
    ],
    "enhanced": [
        {"name": "multi_concept_detector", "port": 8860, "endpoint": "/detect"},
        {"name": "enhanced_prompt_lora", "port": 8880, "endpoint": "/enhance"},
        {"name": "rag_gpu_long", "port": 8920, "endpoint": "/process"},
        {"name": "rag_graph", "port": 8921, "endpoint": "/query"},
        {"name": "rag_code", "port": 8922, "endpoint": "/process"},
        {"name": "lora_coordination_hub", "port": 8995, "endpoint": "/coordinate"},
        {"name": "optimal_lora_router", "port": 5030, "endpoint": "/route"}
    ],
    "orchestrated": [
        {"name": "high_rank_adapter", "port": 9000, "endpoint": "/adapt"},
        {"name": "meta_orchestration_controller", "port": 8999, "endpoint": "/orchestrate"},
        {"name": "enhanced_execution_suite", "port": 8998, "endpoint": "/execute"},
        {"name": "swarm_intelligence_engine", "port": 8977, "endpoint": "/optimize"},
        {"name": "multi_agent_system", "port": 8970, "endpoint": "/coordinate"},
        {"name": "ultimate_architecture_summary", "port": 9001, "endpoint": "/summarize"}
    ],
    "comprehensive": [
        {"name": "concept_training_worker", "port": 8851, "endpoint": "/train"},
        {"name": "enhanced_crawler_nlp", "port": 8850, "endpoint": "/crawl"}
    ]
}

# Global state
coordination_state = {
    "active_agents": [],
    "coordination_sessions": {},
    "communication_channels": [],
    "service_health": {},
    "processing_metrics": {
        "total_queries": 0,
        "successful_orchestrations": 0,
        "mathematical_corrections": 0,
        "average_confidence": 0.0
    }
}

class IntelligentOrchestrator:
    def __init__(self):
        self.session = None
        
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self.session
    
    def calculate_confidence(self, query: str) -> tuple[float, ProcessingPhase]:
        """Calculate confidence score and determine processing phase"""
        base_confidence = 0.7
        query_lower = query.lower()
        
        # Domain-specific confidence factors
        if re.search(r'\d+\s*[\+\-\*\/÷]\s*\d+|calculate|solve|equation', query_lower):
            base_confidence = 0.85  # Math queries
        elif re.search(r'research|analyze|study|literature|impact|historical', query_lower):
            base_confidence = 0.35  # Research queries
        elif re.search(r'secret|conversation|meeting|tesla.*einstein|time travel', query_lower):
            base_confidence = 0.45  # Speculative queries
        elif re.search(r'explain|describe|relationship|between|create', query_lower):
            base_confidence = 0.55  # Explanatory queries
        
        # Apply modifiers
        word_count = len(query.split())
        if word_count > 15:
            base_confidence -= 0.1
        if re.search(r'exactly|precisely|specific', query_lower):
            base_confidence -= 0.15
        
        base_confidence -= word_count / 100
        base_confidence = max(0.2, min(0.95, base_confidence))
        
        # Determine phase
        if base_confidence >= 0.7:
            phase = ProcessingPhase.BASELINE
        elif base_confidence >= 0.5:
            phase = ProcessingPhase.ENHANCED
        elif base_confidence >= 0.35:
            phase = ProcessingPhase.ORCHESTRATED
        else:
            phase = ProcessingPhase.COMPREHENSIVE
        
        return base_confidence, phase
    
    async def call_service(self, service_config: dict, query: str, context: str = "") -> dict:
        """Call a specific service with proper payload"""
        session = await self.get_session()
        
        try:
            url = f"http://localhost:{service_config['port']}{service_config['endpoint']}"
            
            # Service-specific payloads
            if service_config["name"] == "ollama":
                payload = {
                    "model": "phi3:3.8b",
                    "prompt": f"Context: {context}\n\nQuery: {query}\n\nResponse:",
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 300}
                }
            else:
                payload = {
                    "query": query,
                    "context": context,
                    "mode": "adaptive_confidence"
                }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "service": service_config["name"],
                        "status": "success",
                        "response": result,
                        "endpoint": service_config["endpoint"]
                    }
                else:
                    return {
                        "service": service_config["name"],
                        "status": "error",
                        "error": f"HTTP {response.status}",
                        "endpoint": service_config["endpoint"]
                    }
        except Exception as e:
            return {
                "service": service_config["name"],
                "status": "error", 
                "error": str(e),
                "endpoint": service_config["endpoint"]
            }
    
    def validate_mathematical_response(self, query: str, response: str) -> dict:
        """SymPy-powered mathematical validation"""
        corrections = []
        validation_status = ValidationResult.PASSED
        
        try:
            # Basic arithmetic patterns
            arithmetic_patterns = [
                (r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', lambda a, b, r: float(a) + float(b) == float(r)),
                (r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', lambda a, b, r: float(a) - float(b) == float(r)),
                (r'(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', lambda a, b, r: float(a) * float(b) == float(r)),
                (r'(\d+(?:\.\d+)?)\s*[\/÷]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)', lambda a, b, r: abs(float(a) / float(b) - float(r)) < 0.001),
            ]
            
            for pattern, validator in arithmetic_patterns:
                matches = re.finditer(pattern, response)
                for match in matches:
                    a, b, result = match.groups()
                    if not validator(a, b, result):
                        validation_status = ValidationResult.FAILED
                        
                        # Calculate correct answer using SymPy
                        if '÷' in match.group() or '/' in match.group():
                            correct = sp.Rational(a) / sp.Rational(b)
                        elif '*' in match.group():
                            correct = sp.Rational(a) * sp.Rational(b)
                        elif '+' in match.group():
                            correct = sp.Rational(a) + sp.Rational(b)
                        elif '-' in match.group():
                            correct = sp.Rational(a) - sp.Rational(b)
                        
                        corrections.append(f"Mathematical error detected: {match.group()} → Corrected: {a} {match.group().split('=')[0].split()[-1]} {b} = {correct}")
            
            # Check for algebraic solutions
            algebra_pattern = r'x\s*=\s*(\d+(?:\.\d+)?)'
            if 'solve' in query.lower() and 'x' in query:
                try:
                    # Extract equation from query
                    equation_match = re.search(r'(\d*x?\s*[\+\-]\s*\d+)\s*=\s*(\d+)', query)
                    if equation_match:
                        left_side, right_side = equation_match.groups()
                        
                        # Use SymPy to solve
                        x = sp.Symbol('x')
                        equation = sp.Eq(sp.sympify(left_side.replace('x', '*x') if 'x' in left_side else left_side), int(right_side))
                        solution = sp.solve(equation, x)
                        
                        if solution:
                            correct_x = solution[0]
                            
                            # Check if response has correct solution
                            response_match = re.search(algebra_pattern, response)
                            if response_match:
                                response_x = float(response_match.group(1))
                                if abs(response_x - float(correct_x)) > 0.001:
                                    validation_status = ValidationResult.FAILED
                                    corrections.append(f"Algebra error: x = {response_x} → Corrected: x = {correct_x}")
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"Mathematical validation error: {e}")
        
        return {
            "status": validation_status,
            "corrections": corrections,
            "mathematical_engine": "SymPy",
            "validation_applied": True
        }
    
    async def process_with_orchestration(self, query: str, enable_validation: bool = True) -> dict:
        """Process query through intelligent orchestration"""
        start_time = datetime.now()
        
        # Calculate confidence and determine phase
        confidence, phase = self.calculate_confidence(query)
        
        # Determine services to use
        services_to_use = SERVICE_TIERS["baseline"].copy()
        if phase in [ProcessingPhase.ENHANCED, ProcessingPhase.ORCHESTRATED, ProcessingPhase.COMPREHENSIVE]:
            services_to_use.extend(SERVICE_TIERS["enhanced"])
        if phase in [ProcessingPhase.ORCHESTRATED, ProcessingPhase.COMPREHENSIVE]:
            services_to_use.extend(SERVICE_TIERS["orchestrated"])
        if phase == ProcessingPhase.COMPREHENSIVE:
            services_to_use.extend(SERVICE_TIERS["comprehensive"])
        
        # Execute services in parallel
        tasks = [self.call_service(service, query, "") for service in services_to_use]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        service_names = []
        
        for result in results:
            if isinstance(result, dict) and result.get("status") == "success":
                successful_results.append(result)
                service_names.append(result["service"])
        
        # Synthesize response
        final_response = "No successful service responses"
        if successful_results:
            # Priority: Ollama > RAG services > Others
            ollama_result = next((r for r in successful_results if r["service"] == "ollama"), None)
            if ollama_result:
                response_data = ollama_result["response"]
                final_response = response_data.get("response", response_data.get("content", str(response_data)))
            else:
                # Use best available result
                best_result = successful_results[0]
                response_data = best_result["response"]
                final_response = response_data.get("response", response_data.get("content", str(response_data)))
        
        # Apply mathematical validation
        validation_results = {"applied": False}
        mathematical_corrections = []
        
        if enable_validation:
            validation_results = self.validate_mathematical_response(query, final_response)
            mathematical_corrections = validation_results.get("corrections", [])
            
            # Apply corrections to response
            if mathematical_corrections:
                final_response += f"\n\n🔧 MATHEMATICAL CORRECTIONS APPLIED:\n" + "\n".join(mathematical_corrections)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update metrics
        coordination_state["processing_metrics"]["total_queries"] += 1
        if successful_results:
            coordination_state["processing_metrics"]["successful_orchestrations"] += 1
        if mathematical_corrections:
            coordination_state["processing_metrics"]["mathematical_corrections"] += 1
        
        return {
            "query": query,
            "response": final_response,
            "confidence": confidence,
            "processing_phase": phase.value,
            "services_used": service_names,
            "validation_applied": enable_validation,
            "mathematical_corrections": mathematical_corrections,
            "processing_time": processing_time,
            "successful_services": len(successful_results),
            "total_services_attempted": len(services_to_use)
        }

# Initialize orchestrator
orchestrator = IntelligentOrchestrator()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enhanced-a2a-coordination-hub",
        "version": "2.0.0",
        "orchestration_ready": True,
        "mathematical_validation": True
    }

@app.get("/")
async def root():
    return {
        "message": "Enhanced A2A Coordination Hub - Intelligent Orchestrator", 
        "port": 8891,
        "features": ["confidence-driven orchestration", "mathematical validation", "18+ microservices"],
        "processing_phases": ["baseline", "enhanced", "orchestrated", "comprehensive"]
    }

@app.post("/coordinate")
async def coordinate_agents(request: AgentCoordinationRequest):
    """Enhanced agent coordination with intelligent orchestration"""
    try:
        coordination_id = f"coord_{datetime.now().timestamp()}"
        
        # If routing enabled, process through orchestration
        if request.enable_routing:
            orchestration_result = await orchestrator.process_with_orchestration(
                request.task, 
                request.enable_validation
            )
            
            result = {
                "coordination_established": True,
                "participating_agents": request.agents,
                "task_assigned": request.task,
                "orchestration_applied": True,
                "confidence_score": orchestration_result["confidence"],
                "processing_phase": orchestration_result["processing_phase"],
                "services_utilized": orchestration_result["services_used"],
                "mathematical_validation": orchestration_result["validation_applied"],
                "processing_time": orchestration_result["processing_time"],
                "intelligent_response": orchestration_result["response"]
            }
        else:
            # Simple coordination without orchestration
        result = {
            "coordination_established": True,
            "participating_agents": request.agents,
            "task_assigned": request.task,
                "orchestration_applied": False,
            "communication_channel": f"channel_{coordination_id}"
        }
        
        coordination_state["coordination_sessions"][coordination_id] = {
            "agents": request.agents,
            "task": request.task,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "orchestration_enabled": request.enable_routing
        }
        
        return CoordinationResponse(
            coordination_id=coordination_id,
            agents=request.agents,
            status="coordinated",
            processing_phase=result.get("processing_phase", "standard"),
            services_used=result.get("services_utilized", []),
            confidence_score=result.get("confidence_score", 1.0),
            validation_results={"mathematical_validation": request.enable_validation},
            result=result
        )
    except Exception as e:
        logger.error(f"Coordination error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intelligent_query")
async def intelligent_query(request: IntelligentQueryRequest):
    """Direct intelligent query processing with orchestration"""
    try:
        result = await orchestrator.process_with_orchestration(
            request.query,
            request.enable_mathematical_validation
        )
        
        return IntelligentResponse(**result)
        
    except Exception as e:
        logger.error(f"Intelligent query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get orchestration and processing metrics"""
    return {
        "processing_metrics": coordination_state["processing_metrics"],
        "active_sessions": len(coordination_state["coordination_sessions"]),
        "system_status": "operational",
        "service_tiers": {
            "baseline": len(SERVICE_TIERS["baseline"]),
            "enhanced": len(SERVICE_TIERS["enhanced"]),
            "orchestrated": len(SERVICE_TIERS["orchestrated"]),
            "comprehensive": len(SERVICE_TIERS["comprehensive"])
        },
        "total_available_services": sum(len(tier) for tier in SERVICE_TIERS.values()),
        "last_updated": datetime.now().isoformat()
    }

@app.get("/service_health")
async def check_service_health():
    """Check health of all orchestration services"""
    health_checks = {}
    
    for tier_name, services in SERVICE_TIERS.items():
        tier_health = {}
        for service in services:
            try:
                # Quick health check
                session = await orchestrator.get_session()
                url = f"http://localhost:{service['port']}/health"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    tier_health[service["name"]] = "healthy" if response.status == 200 else "unhealthy"
            except:
                tier_health[service["name"]] = "unavailable"
        
        health_checks[tier_name] = tier_health
    
    return {
        "service_health": health_checks,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("Starting Enhanced A2A Coordination Hub - Intelligent Orchestrator on port 8891")
    uvicorn.run(app, host="0.0.0.0", port=8891) 