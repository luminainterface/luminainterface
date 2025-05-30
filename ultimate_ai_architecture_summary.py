#!/usr/bin/env python3
"""
🎯 ULTIMATE ARCHITECTURE SUMMARY - SYSTEM OVERVIEW & COORDINATION
Monitoring & Management for the Ultimate AI Orchestration Architecture v10

This service provides:
- Complete system overview
- Architecture layer monitoring
- Service health tracking
- Performance metrics aggregation
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
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ServiceStatus:
    """Service status information"""
    name: str
    status: str
    port: int
    health: bool
    response_time: float
    last_check: datetime

class UltimateArchitectureSummary:
    """🎯 Ultimate Architecture Summary - System Overview"""
    
    def __init__(self):
        self.redis_client = self._setup_redis()
        
        # Layer 1: High-Rank Adapter
        self.high_rank_adapter_host = os.getenv('HIGH_RANK_ADAPTER_HOST', 'localhost')
        self.high_rank_adapter_port = int(os.getenv('HIGH_RANK_ADAPTER_PORT', 9000))
        
        # Layer 2: Meta-Orchestration Controller
        self.meta_orchestration_host = os.getenv('META_ORCHESTRATION_HOST', 'localhost')
        self.meta_orchestration_port = int(os.getenv('META_ORCHESTRATION_PORT', 8999))
        
        # Layer 3: Enhanced Execution Suite
        self.enhanced_execution_host = os.getenv('ENHANCED_EXECUTION_HOST', 'localhost')
        self.enhanced_execution_port = int(os.getenv('ENHANCED_EXECUTION_PORT', 8998))
        
        # Service registry
        self.services_registry = self._initialize_services_registry()
        self.service_status_cache = {}
        
        logger.info("🎯 Ultimate Architecture Summary initialized")
    
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
    
    def _initialize_services_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the complete services registry"""
        return {
            # Layer 1: High-Rank Adapter
            "high-rank-adapter": {
                "layer": 1,
                "name": "🌟 High-Rank Adapter",
                "port": 9000,
                "description": "Ultimate Strategic Steering",
                "category": "strategic_steering"
            },
            
            # Layer 2: Meta-Orchestration Controller
            "meta-orchestration-controller": {
                "layer": 2,
                "name": "🎯 Meta-Orchestration Controller",
                "port": 8999,
                "description": "Strategic Logic",
                "category": "strategic_logic"
            },
            
            # Layer 3: Enhanced Execution Suite
            "enhanced-execution-suite": {
                "layer": 3,
                "name": "⚡ Enhanced Execution Suite",
                "port": 8998,
                "description": "8-Phase Orchestration",
                "category": "enhanced_execution"
            },
            
            # Central Unified Thinking Engine
            "neural-thought-engine": {
                "layer": 0,
                "name": "🧠 Neural Thought Engine",
                "port": 8890,
                "description": "Central Brain with Bidirectional Thinking",
                "category": "thinking_engine"
            },
            
            # Neural Coordination & A2A
            "a2a-coordination-hub": {
                "layer": 4,
                "name": "🤝 A2A Coordination Hub",
                "port": 8891,
                "description": "Agent-to-Agent Communication",
                "category": "neural_coordination"
            },
            "swarm-intelligence-engine": {
                "layer": 4,
                "name": "🐝 Swarm Intelligence Engine",
                "port": 8977,
                "description": "Collective Intelligence",
                "category": "neural_coordination"
            },
            "neural-memory-bridge": {
                "layer": 4,
                "name": "🤖 Neural Memory Bridge",
                "port": 8892,
                "description": "Advanced Memory Management",
                "category": "neural_coordination"
            },
            
            # RAG² Enhanced Knowledge
            "rag-coordination-enhanced": {
                "layer": 5,
                "name": "🎯 RAG Coordination Enhanced",
                "port": 8952,
                "description": "Concept Detection Integration",
                "category": "rag_enhanced"
            },
            "rag-router-enhanced": {
                "layer": 5,
                "name": "🔀 RAG Router Enhanced",
                "port": 8951,
                "description": "Smart Query Distribution",
                "category": "rag_enhanced"
            },
            "rag-orchestrator": {
                "layer": 5,
                "name": "📋 RAG Orchestrator",
                "port": 8953,
                "description": "Central RAG Coordination",
                "category": "rag_enhanced"
            },
            "rag-gpu-long": {
                "layer": 5,
                "name": "🔥 RAG GPU Long",
                "port": 8920,
                "description": "Complex Analysis Processing",
                "category": "rag_enhanced"
            },
            "rag-graph": {
                "layer": 5,
                "name": "🕸️ RAG Graph",
                "port": 8921,
                "description": "Graph Knowledge Retrieval",
                "category": "rag_enhanced"
            },
            "rag-code": {
                "layer": 5,
                "name": "💻 RAG Code",
                "port": 8922,
                "description": "Code Processing",
                "category": "rag_enhanced"
            },
            "rag-cpu-optimized": {
                "layer": 5,
                "name": "⚡ RAG CPU Optimized",
                "port": 8902,
                "description": "Fast Processing",
                "category": "rag_enhanced"
            },
            
            # LoRA² Enhanced Generation
            "lora-coordination-hub": {
                "layer": 6,
                "name": "🎯 LoRA Coordination Hub",
                "port": 8995,
                "description": "Central LoRA Orchestration",
                "category": "lora_enhanced"
            },
            "enhanced-prompt-lora": {
                "layer": 6,
                "name": "⚡ Enhanced Prompt LoRA",
                "port": 8880,
                "description": "Advanced Prompt Enhancement",
                "category": "lora_enhanced"
            },
            "optimal-lora-router": {
                "layer": 6,
                "name": "🚀 Optimal LoRA Router",
                "port": 5030,
                "description": "Smart LoRA Routing",
                "category": "lora_enhanced"
            },
            "quality-adapter-manager": {
                "layer": 6,
                "name": "🎭 Quality Adapter Manager",
                "port": 8996,
                "description": "Quality Control",
                "category": "lora_enhanced"
            },
            
            # Coordinated Tools & Concept Detection
            "multi-concept-detector": {
                "layer": 7,
                "name": "🎯 Multi-Concept Detector",
                "port": 8860,
                "description": "Enhanced Concept Detection",
                "category": "coordinated_tools"
            },
            "concept-training-worker": {
                "layer": 7,
                "name": "🧠 Concept Training Worker",
                "port": 8851,
                "description": "Advanced Concept Learning",
                "category": "coordinated_tools"
            },
            "enhanced-crawler-nlp": {
                "layer": 7,
                "name": "🔍 Enhanced Crawler NLP",
                "port": 8850,
                "description": "Advanced Web Crawling",
                "category": "coordinated_tools"
            },
            
            # Advanced Processing Services
            "multi-agent-system": {
                "layer": 8,
                "name": "🤖 Multi-Agent System",
                "port": 8970,
                "description": "Advanced Agent Coordination",
                "category": "advanced_processing"
            },
            "consensus-manager": {
                "layer": 8,
                "name": "🎭 Consensus Manager",
                "port": 8978,
                "description": "Decision Consensus",
                "category": "advanced_processing"
            },
            "emergence-detector": {
                "layer": 8,
                "name": "🌊 Emergence Detector",
                "port": 8979,
                "description": "Pattern Emergence Detection",
                "category": "advanced_processing"
            },
            
            # Infrastructure Services
            "vector-store": {
                "layer": 9,
                "name": "🗂️ Vector Store",
                "port": 9262,
                "description": "Enhanced Vector Storage",
                "category": "infrastructure"
            },
            "transcript-ingest": {
                "layer": 9,
                "name": "📝 Transcript Ingest",
                "port": 9264,
                "description": "Conversation Logging",
                "category": "infrastructure"
            }
        }
    
    async def get_complete_architecture_overview(self) -> Dict[str, Any]:
        """🎯 Get complete architecture overview"""
        try:
            overview = {
                "architecture_info": {
                    "name": "🌟 ULTIMATE AI ORCHESTRATION ARCHITECTURE v10",
                    "version": "10.0.0",
                    "description": "Revolutionary 3-Tier Strategic Steering System",
                    "total_services": len(self.services_registry),
                    "architecture_layers": 3,
                    "orchestration_phases": 8,
                    "status": "PRODUCTION READY"
                },
                "layer_summary": await self._get_layer_summary(),
                "service_health": await self._check_all_services_health(),
                "performance_metrics": await self._get_performance_metrics(),
                "strategic_insights": await self._get_strategic_insights(),
                "system_recommendations": await self._get_system_recommendations()
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"❌ Architecture overview failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_layer_summary(self) -> Dict[str, Any]:
        """Get summary of architecture layers"""
        layers = {
            "layer_1": {
                "name": "🧠 HIGH-RANK ADAPTER - ULTIMATE STRATEGIC STEERING",
                "services": [s for s in self.services_registry.values() if s["layer"] == 1],
                "status": "operational",
                "description": "Strategic steering through transcript analysis and meta-reasoning"
            },
            "layer_2": {
                "name": "🎯 META-ORCHESTRATION CONTROLLER - STRATEGIC LOGIC",
                "services": [s for s in self.services_registry.values() if s["layer"] == 2],
                "status": "operational", 
                "description": "7 orchestration strategies with adaptive parameter tuning"
            },
            "layer_3": {
                "name": "⚡ ENHANCED EXECUTION SUITE - 8-PHASE ORCHESTRATION",
                "services": [s for s in self.services_registry.values() if s["layer"] == 3],
                "status": "operational",
                "description": "Neural coordination with LoRA² enhancement"
            },
            "supporting_layers": {
                "thinking_engine": [s for s in self.services_registry.values() if s["category"] == "thinking_engine"],
                "neural_coordination": [s for s in self.services_registry.values() if s["category"] == "neural_coordination"],
                "rag_enhanced": [s for s in self.services_registry.values() if s["category"] == "rag_enhanced"],
                "lora_enhanced": [s for s in self.services_registry.values() if s["category"] == "lora_enhanced"],
                "coordinated_tools": [s for s in self.services_registry.values() if s["category"] == "coordinated_tools"],
                "advanced_processing": [s for s in self.services_registry.values() if s["category"] == "advanced_processing"],
                "infrastructure": [s for s in self.services_registry.values() if s["category"] == "infrastructure"]
            }
        }
        
        return layers
    
    async def _check_all_services_health(self) -> Dict[str, Any]:
        """Check health of all services"""
        health_summary = {
            "total_services": len(self.services_registry),
            "healthy_services": 0,
            "unhealthy_services": 0,
            "unknown_services": 0,
            "service_details": {},
            "overall_health": "unknown"
        }
        
        # Check each service
        for service_id, service_info in self.services_registry.items():
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{service_info['port']}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            health_summary["healthy_services"] += 1
                            status = "healthy"
                        else:
                            health_summary["unhealthy_services"] += 1
                            status = "unhealthy"
                        
                        health_summary["service_details"][service_id] = {
                            "name": service_info["name"],
                            "status": status,
                            "response_time": response_time,
                            "port": service_info["port"],
                            "last_check": datetime.now().isoformat()
                        }
                        
            except Exception as e:
                health_summary["unknown_services"] += 1
                health_summary["service_details"][service_id] = {
                    "name": service_info["name"],
                    "status": "unknown",
                    "error": str(e),
                    "port": service_info["port"],
                    "last_check": datetime.now().isoformat()
                }
        
        # Calculate overall health
        if health_summary["healthy_services"] == health_summary["total_services"]:
            health_summary["overall_health"] = "excellent"
        elif health_summary["healthy_services"] >= health_summary["total_services"] * 0.8:
            health_summary["overall_health"] = "good"
        elif health_summary["healthy_services"] >= health_summary["total_services"] * 0.6:
            health_summary["overall_health"] = "fair"
        else:
            health_summary["overall_health"] = "poor"
        
        return health_summary
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # Get metrics from Redis
            metrics_keys = self.redis_client.keys('performance_metrics:*')
            orchestration_keys = self.redis_client.keys('orchestration:*')
            
            metrics = {
                "orchestrations_count": len(orchestration_keys),
                "recent_performance": {},
                "average_response_time": 0.0,
                "success_rate": 0.0,
                "system_load": "moderate"
            }
            
            # Calculate recent performance
            recent_metrics = []
            for key in metrics_keys[-10:]:  # Last 10 metrics
                metric_data = self.redis_client.get(key)
                if metric_data:
                    try:
                        recent_metrics.append(json.loads(metric_data))
                    except json.JSONDecodeError:
                        continue
            
            if recent_metrics:
                avg_response_time = sum(m.get('response_time', 0.0) for m in recent_metrics) / len(recent_metrics)
                success_count = sum(1 for m in recent_metrics if m.get('status') == 'success')
                success_rate = success_count / len(recent_metrics)
                
                metrics["average_response_time"] = avg_response_time
                metrics["success_rate"] = success_rate
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Performance metrics failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_strategic_insights(self) -> Dict[str, Any]:
        """Get strategic insights from high-rank adapter"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{self.high_rank_adapter_host}:{self.high_rank_adapter_port}/status",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"status": "unavailable", "reason": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _get_system_recommendations(self) -> List[str]:
        """Get system-wide recommendations"""
        recommendations = []
        
        # Check service health
        health = await self._check_all_services_health()
        
        if health["overall_health"] == "poor":
            recommendations.append("🚨 Critical: Multiple services are down - immediate attention required")
        elif health["overall_health"] == "fair":
            recommendations.append("⚠️ Warning: Some services need attention")
        
        if health["unknown_services"] > 0:
            recommendations.append(f"🔍 Investigate {health['unknown_services']} unresponsive services")
        
        # Performance recommendations
        performance = await self._get_performance_metrics()
        
        if performance["success_rate"] < 0.8:
            recommendations.append("📊 Performance: Consider optimizing orchestration strategies")
        
        if performance["average_response_time"] > 30.0:
            recommendations.append("⚡ Speed: Response times are high - optimize processing pipeline")
        
        # Strategic recommendations
        strategic = await self._get_strategic_insights()
        if strategic.get("status") == "operational":
            current_analysis = strategic.get("current_analysis", {})
            if current_analysis.get("transcript_status") == "no_transcripts":
                recommendations.append("📝 Data: No conversation transcripts available for analysis")
        
        if not recommendations:
            recommendations.append("✅ System operating optimally - no immediate actions required")
        
        return recommendations
    
    def generate_html_dashboard(self, overview: Dict[str, Any]) -> str:
        """Generate HTML dashboard"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🌟 Ultimate AI Orchestration Architecture v10 - Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #ffffff; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; padding: 20px; background: linear-gradient(45deg, #00d4ff, #ff6b6b); border-radius: 10px; margin-bottom: 20px; }}
                .layer {{ background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #00d4ff; }}
                .service {{ background: #3a3a3a; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .healthy {{ border-left: 4px solid #4CAF50; }}
                .unhealthy {{ border-left: 4px solid #f44336; }}
                .unknown {{ border-left: 4px solid #ff9800; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center; }}
                .recommendations {{ background: #2a2a2a; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .status-indicator {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }}
                .status-healthy {{ background-color: #4CAF50; }}
                .status-unhealthy {{ background-color: #f44336; }}
                .status-unknown {{ background-color: #ff9800; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌟 ULTIMATE AI ORCHESTRATION ARCHITECTURE v10</h1>
                    <h2>Complete System Dashboard</h2>
                    <p>Total Services: {overview['architecture_info']['total_services']} | Layers: {overview['architecture_info']['architecture_layers']} | Phases: {overview['architecture_info']['orchestration_phases']}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <h3>🏥 System Health</h3>
                        <h2>{overview['service_health']['overall_health'].title()}</h2>
                        <p>{overview['service_health']['healthy_services']}/{overview['service_health']['total_services']} Healthy</p>
                    </div>
                    <div class="metric-card">
                        <h3>📊 Success Rate</h3>
                        <h2>{overview['performance_metrics']['success_rate']:.1%}</h2>
                        <p>Recent Performance</p>
                    </div>
                    <div class="metric-card">
                        <h3>⚡ Response Time</h3>
                        <h2>{overview['performance_metrics']['average_response_time']:.1f}s</h2>
                        <p>Average Response</p>
                    </div>
                    <div class="metric-card">
                        <h3>🚀 Orchestrations</h3>
                        <h2>{overview['performance_metrics']['orchestrations_count']}</h2>
                        <p>Total Executed</p>
                    </div>
                </div>
                
                <div class="layer">
                    <h2>🧠 Layer 1: High-Rank Adapter - Ultimate Strategic Steering</h2>
                    <p>Strategic steering through transcript analysis and meta-reasoning</p>
        """
        
        # Add service health details
        for service_id, service_detail in overview['service_health']['service_details'].items():
            status_class = service_detail['status']
            status_indicator = f"status-{status_class}"
            
            html += f"""
                    <div class="service {status_class}">
                        <span class="status-indicator {status_indicator}"></span>
                        <strong>{service_detail['name']}</strong> - Port {service_detail['port']} 
                        ({service_detail['status']})
                        {f"- {service_detail.get('response_time', 0):.3f}s" if 'response_time' in service_detail else ""}
                    </div>
            """
        
        html += """
                </div>
                
                <div class="recommendations">
                    <h2>🎯 System Recommendations</h2>
                    <ul>
        """
        
        for rec in overview['system_recommendations']:
            html += f"<li>{rec}</li>"
        
        html += """
                    </ul>
                </div>
                
                <div style="text-align: center; margin-top: 40px; color: #888;">
                    <p>🌟 Ultimate AI Orchestration Architecture v10 - Production Ready</p>
                    <p>Generated at {}</p>
                </div>
            </div>
        </body>
        </html>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return html

# FastAPI app
app = FastAPI(title="🎯 Ultimate Architecture Summary", version="10.0.0")

# Global summary instance
summary = None

@app.on_event("startup")
async def startup_event():
    global summary
    summary = UltimateArchitectureSummary()
    logger.info("🎯 Ultimate Architecture Summary startup complete")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        summary.redis_client.ping()
        return {"status": "healthy", "service": "ultimate-architecture-summary", "version": "10.0.0"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.get("/")
async def get_dashboard():
    """🎯 Main dashboard endpoint (HTML)"""
    try:
        overview = await summary.get_complete_architecture_overview()
        html_dashboard = summary.generate_html_dashboard(overview)
        return HTMLResponse(content=html_dashboard)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/overview")
async def get_api_overview():
    """🎯 API overview endpoint (JSON)"""
    try:
        overview = await summary.get_complete_architecture_overview()
        return JSONResponse(content=overview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health-check")
async def api_health_check():
    """Check health of all services"""
    try:
        health = await summary._check_all_services_health()
        return JSONResponse(content=health)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/services")
async def get_services_registry():
    """Get complete services registry"""
    try:
        return JSONResponse(content=summary.services_registry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv('PORT', 9001))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info") 