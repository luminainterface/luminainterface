#!/usr/bin/env python3
"""
Meta-Orchestration Controller
============================

Advanced meta-level orchestration for coordinating multiple AI systems
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import aiohttp
from enhanced_real_world_benchmark import EnhancedRealWorldBenchmark

logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    """Service health status"""
    name: str
    status: str
    response_time: float
    last_checked: str

@dataclass
class OrchestrationTask:
    """Meta-orchestration task definition"""
    task_id: str
    task_type: str
    services_required: List[str]
    priority: int
    payload: Dict[str, Any]
    created_at: str

class MetaOrchestrationController:
    """
    Meta-level orchestration controller for coordinating AI services
    """
    
    def __init__(self):
        self.services = {
            "neural-thought-engine": "http://neural-thought-engine:8890",
            "a2a-coordination-hub": "http://a2a-coordination-hub:8880",
            "consensus-manager": "http://consensus-manager:8883",
            "enhanced-research-agent-v3": "http://enhanced-research-agent-v3:8999",
            "rag-coordination-interface": "http://rag-coordination-interface:8952",
            "lora-coordination-hub": "http://lora-coordination-hub:8995",
            "swarm-intelligence-engine": "http://swarm-intelligence-engine:8882"
        }
        self.benchmark = EnhancedRealWorldBenchmark()
        self.active_tasks = {}
        self.service_health = {}
        
    async def check_service_health(self, service_name: str, service_url: str) -> ServiceHealth:
        """Check health of a specific service"""
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{service_url}/health", timeout=5) as response:
                    response_time = time.time() - start_time
                    if response.status == 200:
                        status = "healthy"
                    else:
                        status = f"unhealthy_status_{response.status}"
        except Exception as e:
            response_time = time.time() - start_time
            status = f"error_{str(e)[:50]}"
            
        return ServiceHealth(
            name=service_name,
            status=status,
            response_time=response_time,
            last_checked=datetime.now().isoformat()
        )
    
    async def health_check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all registered services"""
        logger.info("🔍 Performing health check on all services")
        
        health_tasks = []
        for service_name, service_url in self.services.items():
            task = self.check_service_health(service_name, service_url)
            health_tasks.append(task)
        
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        health_status = {}
        for i, result in enumerate(health_results):
            service_name = list(self.services.keys())[i]
            if isinstance(result, ServiceHealth):
                health_status[service_name] = result
                self.service_health[service_name] = result
            else:
                # Handle exceptions
                health_status[service_name] = ServiceHealth(
                    name=service_name,
                    status=f"exception_{str(result)[:50]}",
                    response_time=5.0,
                    last_checked=datetime.now().isoformat()
                )
        
        healthy_count = sum(1 for h in health_status.values() if h.status == "healthy")
        logger.info(f"✅ Health check complete: {healthy_count}/{len(self.services)} services healthy")
        
        return health_status
    
    async def orchestrate_research_task(self, topic: str, field: str = "general") -> Dict[str, Any]:
        """Orchestrate a complex research task across multiple services"""
        logger.info(f"🎯 Orchestrating research task: {topic}")
        
        task_id = f"research_{int(time.time())}"
        start_time = time.time()
        
        # Step 1: Coordinate with A2A hub for service allocation
        try:
            async with aiohttp.ClientSession() as session:
                coordination_payload = {
                    "task_type": "research_coordination",
                    "topic": topic,
                    "services_needed": ["enhanced-research-agent-v3", "rag-coordination-interface"]
                }
                
                async with session.post(
                    f"{self.services['a2a-coordination-hub']}/coordinate",
                    json=coordination_payload,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        coordination_result = await response.json()
                    else:
                        coordination_result = {"status": "coordination_failed"}
        except Exception as e:
            coordination_result = {"status": "error", "error": str(e)}
        
        # Step 2: Execute research via Enhanced Research Agent v3
        try:
            async with aiohttp.ClientSession() as session:
                research_payload = {
                    "topic": topic,
                    "field": field,
                    "target_quality": 9.0,
                    "max_lora_wait_minutes": 10
                }
                
                async with session.post(
                    f"{self.services['enhanced-research-agent-v3']}/research/generate",
                    json=research_payload,
                    timeout=300
                ) as response:
                    if response.status == 200:
                        research_result = await response.json()
                    else:
                        research_result = {"status": "research_failed"}
        except Exception as e:
            research_result = {"status": "error", "error": str(e)}
        
        # Step 3: Get consensus on results
        try:
            async with aiohttp.ClientSession() as session:
                consensus_payload = {
                    "decision_type": "research_quality",
                    "data": research_result,
                    "threshold": 0.8
                }
                
                async with session.post(
                    f"{self.services['consensus-manager']}/consensus",
                    json=consensus_payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        consensus_result = await response.json()
                    else:
                        consensus_result = {"status": "consensus_failed"}
        except Exception as e:
            consensus_result = {"status": "error", "error": str(e)}
        
        execution_time = time.time() - start_time
        
        orchestration_result = {
            "task_id": task_id,
            "topic": topic,
            "execution_time": execution_time,
            "coordination": coordination_result,
            "research": research_result,
            "consensus": consensus_result,
            "overall_status": "success" if all(r.get("status") == "success" for r in [coordination_result, research_result] if "status" in r) else "partial_success",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"🎉 Research orchestration complete: {task_id}")
        return orchestration_result
    
    async def run_system_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive system benchmark"""
        logger.info("🚀 Starting system-wide benchmark")
        
        start_time = time.time()
        
        # Run benchmark suite
        benchmark_result = await self.benchmark.run_full_benchmark()
        
        # Check service health
        health_status = await self.health_check_all_services()
        
        # Calculate system metrics
        healthy_services = sum(1 for h in health_status.values() if h.status == "healthy")
        avg_response_time = sum(h.response_time for h in health_status.values()) / len(health_status)
        
        system_report = {
            "benchmark": benchmark_result,
            "service_health": {
                "healthy_count": healthy_services,
                "total_count": len(health_status),
                "health_percentage": (healthy_services / len(health_status)) * 100,
                "average_response_time": avg_response_time,
                "individual_health": {h.name: h.status for h in health_status.values()}
            },
            "meta_orchestration": {
                "active_tasks": len(self.active_tasks),
                "total_execution_time": time.time() - start_time,
                "system_load": "optimal" if healthy_services >= len(self.services) * 0.8 else "degraded"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ System benchmark completed in {time.time() - start_time:.2f}s")
        return system_report
    
    async def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status"""
        return {
            "active_tasks": len(self.active_tasks),
            "registered_services": len(self.services),
            "last_health_check": max(
                (h.last_checked for h in self.service_health.values()),
                default="never"
            ),
            "system_status": "operational",
            "timestamp": datetime.now().isoformat()
        }

# Initialize global controller
meta_controller = MetaOrchestrationController()

# Export for FastAPI app
__all__ = ['MetaOrchestrationController', 'meta_controller'] 