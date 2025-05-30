#!/usr/bin/env python3
"""
🌟 HIGH-RANK ADAPTER - ULTIMATE STRATEGIC STEERING
Layer 1 of the Ultimate AI Orchestration Architecture v10

This service provides strategic steering through:
- Transcript analysis & pattern recognition
- Strategic evolution & self-reflection  
- Meta-reasoning with performance optimization
- 5 steering mechanisms for ultimate coordination
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
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StrategicInsight:
    """Strategic insight data structure"""
    pattern_type: str
    confidence: float
    impact_score: float
    recommendation: str
    timestamp: datetime

class HighRankAdapter:
    """🌟 High-Rank Adapter - Ultimate Strategic Steering"""
    
    def __init__(self, offline_mode=False):
        if offline_mode:
            self.redis_client = None
            logger.info("🔧 High-Rank Adapter running in OFFLINE MODE")
        else:
            self.redis_client = self._setup_redis()
        
        self.offline_mode = offline_mode
        self.meta_orchestration_host = os.getenv('META_ORCHESTRATION_HOST', 'localhost')
        self.meta_orchestration_port = int(os.getenv('META_ORCHESTRATION_PORT', 8999))
        self.enhanced_execution_host = os.getenv('ENHANCED_EXECUTION_HOST', 'localhost')
        self.enhanced_execution_port = int(os.getenv('ENHANCED_EXECUTION_PORT', 8998))
        
        # Strategic steering parameters
        self.transcript_influence = float(os.getenv('TRANSCRIPT_INFLUENCE', 0.8))
        self.pattern_sensitivity = float(os.getenv('PATTERN_SENSITIVITY', 0.7))
        self.evolution_aggressiveness = float(os.getenv('EVOLUTION_AGGRESSIVENESS', 0.6))
        self.self_reflection_depth = float(os.getenv('SELF_REFLECTION_DEPTH', 0.9))
        self.quality_prioritization = float(os.getenv('QUALITY_PRIORITIZATION', 0.85))
        
        # Strategic insights storage
        self.insights_cache = []
        self.performance_trends = []
        
        logger.info("🌟 High-Rank Adapter initialized with ultimate strategic steering")
    
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
            if not self.offline_mode:
                raise
            return None
    
    async def analyze_conversation_transcripts(self) -> Dict[str, Any]:
        """🔍 Analyze conversation transcripts for strategic insights"""
        try:
            # Get recent transcripts from Redis
            transcripts = self.redis_client.lrange('conversation_transcripts', 0, -1)
            
            if not transcripts:
                return {"status": "no_transcripts", "insights": []}
            
            insights = []
            patterns = {
                "user_satisfaction": 0.0,
                "response_quality": 0.0,
                "concept_detection_effectiveness": 0.0,
                "orchestration_efficiency": 0.0,
                "strategic_alignment": 0.0
            }
            
            # Analyze patterns in transcripts
            for transcript_json in transcripts[-10:]:  # Last 10 conversations
                try:
                    transcript = json.loads(transcript_json)
                    
                    # Pattern analysis
                    if 'user_feedback' in transcript:
                        patterns["user_satisfaction"] += transcript.get('user_feedback', {}).get('satisfaction', 0.5)
                    
                    if 'response_metrics' in transcript:
                        metrics = transcript['response_metrics']
                        patterns["response_quality"] += metrics.get('quality_score', 0.5)
                        patterns["concept_detection_effectiveness"] += metrics.get('concept_detection_score', 0.5)
                        patterns["orchestration_efficiency"] += metrics.get('efficiency_score', 0.5)
                    
                except json.JSONDecodeError:
                    continue
            
            # Calculate average patterns
            transcript_count = min(len(transcripts), 10)
            if transcript_count > 0:
                for key in patterns:
                    patterns[key] = patterns[key] / transcript_count
            
            # Generate strategic insights
            for pattern_type, score in patterns.items():
                if score < 0.6:  # Below acceptable threshold
                    insight = StrategicInsight(
                        pattern_type=pattern_type,
                        confidence=0.8,
                        impact_score=1.0 - score,
                        recommendation=self._generate_recommendation(pattern_type, score),
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
            
            return {
                "status": "analyzed",
                "patterns": patterns,
                "insights": [
                    {
                        "type": insight.pattern_type,
                        "confidence": insight.confidence,
                        "impact": insight.impact_score,
                        "recommendation": insight.recommendation,
                        "timestamp": insight.timestamp.isoformat()
                    } for insight in insights
                ],
                "transcript_count": transcript_count
            }
            
        except Exception as e:
            logger.error(f"❌ Transcript analysis failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def _generate_recommendation(self, pattern_type: str, score: float) -> str:
        """Generate strategic recommendations based on patterns"""
        recommendations = {
            "user_satisfaction": f"Increase focus on user engagement. Current score: {score:.2f}. Recommend enhancing response personalization.",
            "response_quality": f"Improve response quality mechanisms. Current score: {score:.2f}. Recommend activating quality verification modules.",
            "concept_detection_effectiveness": f"Enhance concept detection accuracy. Current score: {score:.2f}. Recommend retraining concept models.",
            "orchestration_efficiency": f"Optimize orchestration flow. Current score: {score:.2f}. Recommend load balancing adjustments.",
            "strategic_alignment": f"Improve strategic alignment. Current score: {score:.2f}. Recommend meta-orchestration parameter tuning."
        }
        return recommendations.get(pattern_type, f"Improve {pattern_type}. Score: {score:.2f}")
    
    async def perform_self_reflection(self) -> Dict[str, Any]:
        """🪞 Perform deep self-reflection on system performance"""
        try:
            # Get performance metrics from Redis
            metrics_keys = self.redis_client.keys('performance_metrics:*')
            recent_metrics = []
            
            for key in metrics_keys[-20:]:  # Last 20 metrics
                metric_data = self.redis_client.get(key)
                if metric_data:
                    try:
                        recent_metrics.append(json.loads(metric_data))
                    except json.JSONDecodeError:
                        continue
            
            if not recent_metrics:
                return {"status": "no_metrics", "reflection": "Insufficient data for self-reflection"}
            
            # Calculate performance trends
            quality_trend = [m.get('quality_score', 0.5) for m in recent_metrics]
            efficiency_trend = [m.get('efficiency_score', 0.5) for m in recent_metrics]
            
            # Analyze trends
            quality_improving = len(quality_trend) > 1 and quality_trend[-1] > quality_trend[0]
            efficiency_improving = len(efficiency_trend) > 1 and efficiency_trend[-1] > efficiency_trend[0]
            
            avg_quality = sum(quality_trend) / len(quality_trend) if quality_trend else 0.5
            avg_efficiency = sum(efficiency_trend) / len(efficiency_trend) if efficiency_trend else 0.5
            
            # Generate self-reflection insights
            reflection = {
                "overall_performance": (avg_quality + avg_efficiency) / 2,
                "quality_trend": "improving" if quality_improving else "declining",
                "efficiency_trend": "improving" if efficiency_improving else "declining",
                "avg_quality": avg_quality,
                "avg_efficiency": avg_efficiency,
                "recommendations": []
            }
            
            # Generate strategic recommendations
            if avg_quality < 0.7:
                reflection["recommendations"].append("Focus on quality enhancement - activate advanced verification")
            if avg_efficiency < 0.7:
                reflection["recommendations"].append("Optimize orchestration flow - reduce unnecessary processing")
            if not quality_improving and not efficiency_improving:
                reflection["recommendations"].append("Major strategic adjustment needed - consider architecture evolution")
            
            # Store reflection results
            self.redis_client.setex(
                f"self_reflection:{datetime.now().isoformat()}",
                3600,  # 1 hour TTL
                json.dumps(reflection)
            )
            
            return {"status": "reflected", "reflection": reflection}
            
        except Exception as e:
            logger.error(f"❌ Self-reflection failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def generate_strategic_steering(self) -> Dict[str, Any]:
        """⚡ Generate strategic steering commands for meta-orchestration"""
        try:
            # Analyze current system state
            transcript_analysis = await self.analyze_conversation_transcripts()
            self_reflection = await self.perform_self_reflection()
            
            # Generate steering parameters
            steering_params = {
                "strategy_weights": {
                    "speed_optimized": 0.2,
                    "quality_maximized": 0.3,
                    "concept_focused": 0.15,
                    "research_intensive": 0.1,
                    "creative_synthesis": 0.1,
                    "verification_heavy": 0.1,
                    "adaptive_learning": 0.05
                },
                "concept_detection_priority": 0.8,
                "verification_thoroughness": 0.7,
                "speed_vs_quality_balance": 0.6,
                "research_depth_preference": 0.5
            }
            
            # Adjust based on analysis
            if transcript_analysis.get("status") == "analyzed":
                patterns = transcript_analysis.get("patterns", {})
                
                # If quality is low, increase quality-focused strategies
                if patterns.get("response_quality", 0.5) < 0.6:
                    steering_params["strategy_weights"]["quality_maximized"] = 0.5
                    steering_params["strategy_weights"]["verification_heavy"] = 0.2
                    steering_params["verification_thoroughness"] = 0.9
                
                # If concept detection is poor, focus on concept strategies
                if patterns.get("concept_detection_effectiveness", 0.5) < 0.6:
                    steering_params["strategy_weights"]["concept_focused"] = 0.3
                    steering_params["concept_detection_priority"] = 0.9
            
            # Apply self-reflection insights
            if self_reflection.get("status") == "reflected":
                reflection = self_reflection.get("reflection", {})
                
                if reflection.get("avg_quality", 0.5) < 0.6:
                    steering_params["speed_vs_quality_balance"] = 0.3  # Favor quality
                if reflection.get("avg_efficiency", 0.5) < 0.6:
                    steering_params["strategy_weights"]["speed_optimized"] = 0.4  # Favor speed
            
            # Send steering to meta-orchestration
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        f"http://{self.meta_orchestration_host}:{self.meta_orchestration_port}/update_strategy",
                        json=steering_params,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            logger.info("✅ Strategic steering sent to meta-orchestration")
                        else:
                            logger.warning(f"⚠️ Meta-orchestration responded with status {response.status}")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Meta-orchestration steering timeout")
                except Exception as e:
                    logger.warning(f"⚠️ Meta-orchestration steering failed: {e}")
            
            return {
                "status": "steering_generated",
                "steering_params": steering_params,
                "analysis_results": {
                    "transcript_analysis": transcript_analysis,
                    "self_reflection": self_reflection
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Strategic steering generation failed: {e}")
            return {"status": "error", "message": str(e)}

    def analyze_conversation_patterns(self, transcript_data: List[Dict] = None) -> Dict[str, Any]:
        """🔍 Analyze conversation patterns (works offline)"""
        if self.offline_mode or transcript_data:
            # Use provided data or generate mock data for testing
            test_patterns = {
                "user_satisfaction": 0.85,
                "response_quality": 0.78,
                "concept_detection_effectiveness": 0.82,
                "orchestration_efficiency": 0.75,
                "strategic_alignment": 0.88
            }
            
            if transcript_data:
                # Simple pattern analysis on provided data
                satisfaction_keywords = ["good", "great", "excellent", "perfect", "thanks"]
                total_score = 0
                for entry in transcript_data:
                    user_text = entry.get("user", "").lower()
                    assistant_text = entry.get("assistant", "").lower()
                    
                    # Simple sentiment analysis
                    if any(keyword in user_text for keyword in satisfaction_keywords):
                        total_score += 0.9
                    elif len(assistant_text) > 20:  # Detailed response
                        total_score += 0.7
                    else:
                        total_score += 0.5
                
                avg_score = total_score / len(transcript_data) if transcript_data else 0.5
                test_patterns["user_satisfaction"] = min(avg_score, 1.0)
            
            return test_patterns
        else:
            # Use Redis-based analysis (existing method)
            return asyncio.run(self.analyze_conversation_transcripts())
    
    def generate_strategic_steering(self, transcript_data: List[Dict] = None, context: Dict = None) -> Dict[str, Any]:
        """⚡ Generate strategic steering parameters (works offline)"""
        if self.offline_mode or transcript_data is not None:
            # Offline mode - generate strategic parameters based on context
            patterns = self.analyze_conversation_patterns(transcript_data)
            
            # Generate strategic parameters based on patterns and context
            complexity = context.get("complexity", "medium") if context else "medium"
            domain = context.get("domain", "general") if context else "general"
            
            steering_params = {
                "strategy_weights": {
                    "speed_optimized": 0.3 if complexity == "low" else 0.2,
                    "quality_maximized": 0.4 if complexity == "high" else 0.3,
                    "concept_focused": 0.35 if domain == "technical" else 0.25,
                    "research_intensive": 0.4 if complexity == "high" else 0.2,
                    "creative_synthesis": 0.3,
                    "verification_heavy": 0.35 if domain == "science" else 0.25,
                    "adaptive_learning": 0.3
                },
                "orchestration_params": {
                    "depth_multiplier": 1.5 if complexity == "high" else 1.0,
                    "quality_threshold": 0.8 if patterns.get("response_quality", 0.5) < 0.7 else 0.7,
                    "concept_detection_sensitivity": 0.85,
                    "rag_coordination_weight": 0.7,
                    "lora_enhancement_factor": 1.2
                },
                "meta_reasoning": {
                    "self_reflection_enabled": True,
                    "pattern_recognition_depth": 0.8,
                    "strategic_evolution_rate": 0.6,
                    "performance_optimization": True
                },
                "execution_directives": {
                    "prioritize_accuracy": complexity == "high",
                    "enable_deep_analysis": domain in ["science", "technical"],
                    "activate_creative_mode": domain == "creative",
                    "enforce_verification": True
                }
            }
            
            # Adjust based on performance patterns
            if patterns.get("user_satisfaction", 0.5) < 0.7:
                steering_params["strategy_weights"]["quality_maximized"] += 0.1
                steering_params["orchestration_params"]["quality_threshold"] = 0.85
            
            if patterns.get("orchestration_efficiency", 0.5) < 0.7:
                steering_params["strategy_weights"]["speed_optimized"] += 0.1
                steering_params["orchestration_params"]["depth_multiplier"] *= 0.9
            
            return steering_params
        else:
            # Use Redis-based analysis (existing method)
            return asyncio.run(self.generate_strategic_steering())

# FastAPI app
app = FastAPI(title="🌟 High-Rank Adapter", version="10.0.0")

# Global adapter instance
adapter = None

@app.on_event("startup")
async def startup_event():
    global adapter
    adapter = HighRankAdapter()
    logger.info("🌟 High-Rank Adapter startup complete")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        adapter.redis_client.ping()
        return {"status": "healthy", "service": "high-rank-adapter", "version": "10.0.0"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.post("/analyze_transcripts")
async def analyze_transcripts():
    """🔍 Analyze conversation transcripts endpoint"""
    try:
        result = await adapter.analyze_conversation_transcripts()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/self_reflect")
async def self_reflect():
    """🪞 Perform self-reflection endpoint"""
    try:
        result = await adapter.perform_self_reflection()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_steering")
async def generate_steering():
    """⚡ Generate strategic steering endpoint"""
    try:
        result = await adapter.generate_strategic_steering()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get adapter status and metrics"""
    try:
        # Get current insights
        transcript_analysis = await adapter.analyze_conversation_transcripts()
        self_reflection = await adapter.perform_self_reflection()
        
        return {
            "service": "high-rank-adapter",
            "version": "10.0.0",
            "status": "operational",
            "strategic_parameters": {
                "transcript_influence": adapter.transcript_influence,
                "pattern_sensitivity": adapter.pattern_sensitivity,
                "evolution_aggressiveness": adapter.evolution_aggressiveness,
                "self_reflection_depth": adapter.self_reflection_depth,
                "quality_prioritization": adapter.quality_prioritization
            },
            "current_analysis": {
                "transcript_status": transcript_analysis.get("status"),
                "reflection_status": self_reflection.get("status")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv('PORT', 9000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info") 