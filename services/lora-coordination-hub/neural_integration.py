#!/usr/bin/env python3
"""
Neural Thought Engine Integration
=================================

Client for integrating with the Neural Thought Engine for coordinated
LoRA system orchestration and decision making.
"""

import aiohttp
import logging
import asyncio
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class NeuralThoughtEngineClient:
    """Client for Neural Thought Engine coordination integration"""
    
    def __init__(self, base_url: str = "http://neural-thought-engine:8890"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30)
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session
    
    async def check_connectivity(self) -> bool:
        """Check if Neural Thought Engine is available"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Neural Thought Engine connectivity check failed: {e}")
            return False
    
    async def analyze_for_coordination(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request Neural Thought Engine to analyze query for LoRA coordination strategy
        
        Args:
            request_data: {
                "query": str,
                "session_id": str,
                "available_lora_systems": List[str],
                "coordination_depth": str,
                "context": Dict
            }
        
        Returns:
            Analysis results with coordination strategy recommendations
        """
        try:
            session = await self._get_session()
            
            # Enhanced analysis request for coordination
            analysis_request = {
                "text": request_data["query"],
                "analysis_type": "lora_coordination",
                "context": {
                    "session_id": request_data.get("session_id", "default"),
                    "available_systems": request_data.get("available_lora_systems", []),
                    "coordination_depth": request_data.get("coordination_depth", "auto"),
                    "user_context": request_data.get("context", {})
                },
                "request_coordination_plan": True,
                "include_system_analysis": True
            }
            
            # Try new coordination endpoint first
            async with session.post(
                f"{self.base_url}/coordinate/analyze", 
                json=analysis_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._process_coordination_analysis(result, request_data)
                else:
                    logger.warning(f"Coordination analysis endpoint returned {response.status}")
                    
        except Exception as e:
            logger.warning(f"Coordination analysis failed: {e}")
        
        # Fallback to basic thinking endpoint
        return await self._fallback_analysis(request_data)
    
    def _process_coordination_analysis(self, result: Dict, request_data: Dict) -> Dict[str, Any]:
        """Process the coordination analysis results from Neural Thought Engine"""
        
        # Extract coordination strategy from neural analysis
        analysis = {
            "query_complexity": result.get("complexity_score", 0.5),
            "detected_domains": result.get("domains", []),
            "coordination_strategy": result.get("coordination_strategy", "auto"),
            "recommended_systems": result.get("recommended_systems", []),
            "system_roles": result.get("system_roles", {}),
            "synthesis_approach": result.get("synthesis_approach", "neural_weighted"),
            "coordination_context": {
                "neural_insights": result.get("neural_insights", {}),
                "thought_process": result.get("thought_process", {}),
                "query_analysis": result.get("query_analysis", {})
            },
            "confidence": result.get("coordination_confidence", 0.7)
        }
        
        return analysis
    
    async def _fallback_analysis(self, request_data: Dict) -> Dict[str, Any]:
        """Fallback analysis when coordination endpoint is not available"""
        logger.info("Using fallback analysis for coordination...")
        
        try:
            session = await self._get_session()
            
            # Use basic thinking endpoint
            thinking_request = {
                "user_query": request_data["query"],
                "session_id": request_data.get("session_id", "fallback"),
                "context": request_data.get("context", {})
            }
            
            async with session.post(
                f"{self.base_url}/thinking/process",
                json=thinking_request
            ) as response:
                if response.status == 200:
                    thinking_result = await response.json()
                    return self._convert_thinking_to_coordination(thinking_result, request_data)
                    
        except Exception as e:
            logger.warning(f"Fallback analysis failed: {e}")
        
        # Final fallback - basic analysis
        return self._basic_coordination_analysis(request_data)
    
    def _convert_thinking_to_coordination(self, thinking_result: Dict, request_data: Dict) -> Dict[str, Any]:
        """Convert thinking process results to coordination analysis"""
        
        query = request_data["query"].lower()
        available_systems = request_data.get("available_lora_systems", [])
        
        # Simple rule-based coordination strategy
        if any(word in query for word in ["creative", "story", "character", "narrative"]):
            strategy = "creative_coordination"
            recommended = ["jarvis_chat", "enhanced_prompt_lora"]
        elif any(word in query for word in ["code", "programming", "function", "algorithm"]):
            strategy = "technical_coordination" 
            recommended = ["optimal_lora_router", "npu_enhanced_lora"]
        elif any(word in query for word in ["complex", "analyze", "research", "deep"]):
            strategy = "full_coordination"
            recommended = available_systems[:3]  # Use top 3 systems
        else:
            strategy = "standard_coordination"
            recommended = ["npu_enhanced_lora", "enhanced_prompt_lora"]
        
        return {
            "query_complexity": len(query.split()) / 50.0,  # Simple complexity score
            "detected_domains": thinking_result.get("domains", ["general"]),
            "coordination_strategy": strategy,
            "recommended_systems": [s for s in recommended if s in available_systems],
            "system_roles": self._assign_fallback_roles(recommended),
            "synthesis_approach": "neural_weighted",
            "coordination_context": {
                "fallback_mode": True,
                "thinking_result": thinking_result
            },
            "confidence": 0.6
        }
    
    def _assign_fallback_roles(self, systems: List[str]) -> Dict[str, str]:
        """Assign roles to systems in fallback mode"""
        roles = {}
        
        role_priority = [
            "primary_processor",
            "quality_enhancer", 
            "specialist_processor",
            "performance_optimizer",
            "fallback_processor"
        ]
        
        for i, system in enumerate(systems):
            if i < len(role_priority):
                roles[system] = role_priority[i]
            else:
                roles[system] = "general_processor"
                
        return roles
    
    def _basic_coordination_analysis(self, request_data: Dict) -> Dict[str, Any]:
        """Most basic coordination analysis when all else fails"""
        available_systems = request_data.get("available_lora_systems", [])
        
        return {
            "query_complexity": 0.5,
            "detected_domains": ["general"],
            "coordination_strategy": "simple_coordination",
            "recommended_systems": available_systems[:2] if available_systems else [],
            "system_roles": {
                available_systems[0]: "primary_processor" if available_systems else "",
                available_systems[1]: "quality_enhancer" if len(available_systems) > 1 else ""
            },
            "synthesis_approach": "simple_weighted",
            "coordination_context": {"basic_fallback": True},
            "confidence": 0.3
        }
    
    async def synthesize_coordinated_results(self, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request Neural Thought Engine to synthesize results from multiple LoRA systems
        
        Args:
            synthesis_data: {
                "query": str,
                "coordination_results": List[Dict],
                "synthesis_strategy": str,
                "neural_analysis": Dict
            }
        
        Returns:
            Synthesized response combining all LoRA system outputs
        """
        try:
            session = await self._get_session()
            
            synthesis_request = {
                "query": synthesis_data["query"],
                "system_results": synthesis_data["coordination_results"],
                "synthesis_strategy": synthesis_data.get("synthesis_strategy", "neural_weighted"),
                "original_analysis": synthesis_data.get("neural_analysis", {}),
                "request_unified_response": True
            }
            
            # Try new synthesis endpoint
            async with session.post(
                f"{self.base_url}/coordinate/synthesize",
                json=synthesis_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    logger.warning(f"Synthesis endpoint returned {response.status}")
                    
        except Exception as e:
            logger.warning(f"Neural synthesis failed: {e}")
        
        # Fallback synthesis
        return await self._fallback_synthesis(synthesis_data)
    
    async def _fallback_synthesis(self, synthesis_data: Dict) -> Dict[str, Any]:
        """Fallback synthesis when neural endpoint is unavailable"""
        coordination_results = synthesis_data["coordination_results"]
        
        if not coordination_results:
            return {
                "synthesized_response": "No coordination results to synthesize.",
                "confidence": 0.0,
                "synthesis_quality": 0.0,
                "coordination_confidence": 0.0
            }
        
        # Simple synthesis - pick best result or combine top results
        best_result = max(coordination_results, key=lambda r: r.get("confidence", 0.0))
        
        if len(coordination_results) == 1:
            response = best_result.get("response", {}).get("response", "No response available")
        else:
            # Combine top 2 results
            top_results = sorted(coordination_results, key=lambda r: r.get("confidence", 0.0), reverse=True)[:2]
            responses = []
            for result in top_results:
                if "response" in result.get("response", {}):
                    responses.append(result["response"]["response"])
            
            if responses:
                response = f"Based on coordinated analysis:\n\n{responses[0]}"
                if len(responses) > 1:
                    response += f"\n\nAdditional insight: {responses[1]}"
            else:
                response = "Coordination synthesis failed - no valid responses."
        
        return {
            "synthesized_response": response,
            "confidence": best_result.get("confidence", 0.5),
            "synthesis_quality": 0.6,  # Fallback quality
            "coordination_confidence": 0.4,
            "synthesis_method": "fallback_best_result"
        }
    
    async def generate_cross_system_insights(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights for cross-system learning optimization
        
        Args:
            learning_data: {
                "coordination_results": List[Dict],
                "unified_response_quality": float,
                "strategy_effectiveness": float
            }
        
        Returns:
            Insights for improving future coordination
        """
        try:
            session = await self._get_session()
            
            insights_request = {
                "coordination_results": learning_data["coordination_results"],
                "response_quality": learning_data.get("unified_response_quality", 0.5),
                "strategy_effectiveness": learning_data.get("strategy_effectiveness", 0.5),
                "request_learning_insights": True
            }
            
            async with session.post(
                f"{self.base_url}/learning/insights",
                json=insights_request
            ) as response:
                if response.status == 200:
                    return await response.json()
                    
        except Exception as e:
            logger.warning(f"Cross-system insights generation failed: {e}")
        
        # Fallback insights
        return await self._generate_fallback_insights(learning_data)
    
    async def _generate_fallback_insights(self, learning_data: Dict) -> Dict[str, Any]:
        """Generate basic insights when neural endpoint is unavailable"""
        coordination_results = learning_data["coordination_results"]
        
        if not coordination_results:
            return {"insights": [], "recommendations": []}
        
        # Basic analysis
        avg_confidence = sum(r.get("confidence", 0) for r in coordination_results) / len(coordination_results)
        avg_processing_time = sum(r.get("processing_time", 0) for r in coordination_results) / len(coordination_results)
        
        insights = []
        recommendations = []
        
        if avg_confidence < 0.5:
            insights.append("Low average confidence across systems suggests need for better coordination strategy")
            recommendations.append("Consider using more specialized systems for domain-specific queries")
        
        if avg_processing_time > 10.0:
            insights.append("High processing times indicate potential optimization opportunities")
            recommendations.append("Implement parallel processing optimization for faster coordination")
        
        best_system = max(coordination_results, key=lambda r: r.get("confidence", 0))["system_name"]
        insights.append(f"System '{best_system}' performed best in this coordination")
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "performance_metrics": {
                "avg_confidence": avg_confidence,
                "avg_processing_time": avg_processing_time,
                "best_performing_system": best_system
            }
        }
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close() 