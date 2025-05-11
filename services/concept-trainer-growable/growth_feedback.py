import httpx
import json
import logging
import redis
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrowthFeedbackManager:
    def __init__(self, 
                 graph_api_url: str,
                 redis_host: str = "redis",
                 redis_port: int = 6379):
        self.graph_api_url = graph_api_url
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.http_client = httpx.AsyncClient()
        
    async def analyze_growth_context(self, concept_id: str, growth_metadata: Dict) -> Dict:
        """Analyze the semantic context around a growing concept"""
        try:
            # Get concept context from Graph API
            response = await self.http_client.get(
                f"{self.graph_api_url}/concept/context/{concept_id}"
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get concept context: {response.text}")
                return {"status": "error", "message": "Failed to get concept context"}
                
            context = response.json()
            
            # Analyze neighbor maturity
            neighbors = context.get("neighbors", [])
            mature_neighbors = [
                n for n in neighbors 
                if n.get("maturity", 0) > 0.7
            ]
            
            # Identify potential gaps
            gaps = self._identify_context_gaps(context, growth_metadata)
            
            # Prepare feedback
            feedback = {
                "concept_id": concept_id,
                "timestamp": datetime.now().isoformat(),
                "neighbor_analysis": {
                    "total_neighbors": len(neighbors),
                    "mature_neighbors": len(mature_neighbors),
                    "maturity_ratio": len(mature_neighbors) / len(neighbors) if neighbors else 0
                },
                "context_gaps": gaps,
                "growth_metadata": growth_metadata
            }
            
            # Emit feedback event
            await self._emit_feedback_event(feedback)
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error analyzing growth context: {e}")
            return {"status": "error", "message": str(e)}
            
    def _identify_context_gaps(self, context: Dict, growth_metadata: Dict) -> List[Dict]:
        """Identify potential gaps in the concept's semantic context"""
        gaps = []
        
        # Check for missing high-level concepts
        if context.get("depth", 0) < 2:
            gaps.append({
                "type": "hierarchy_gap",
                "description": "Concept lacks parent concepts",
                "suggestion": "Consider adding higher-level abstractions"
            })
            
        # Check for missing related concepts
        neighbors = context.get("neighbors", [])
        if len(neighbors) < 3:
            gaps.append({
                "type": "relation_gap",
                "description": "Concept has few semantic connections",
                "suggestion": "Explore related concepts in the domain"
            })
            
        # Check for maturity imbalance
        mature_neighbors = [n for n in neighbors if n.get("maturity", 0) > 0.7]
        if mature_neighbors and len(mature_neighbors) / len(neighbors) < 0.3:
            gaps.append({
                "type": "maturity_gap",
                "description": "Concept surrounded by immature neighbors",
                "suggestion": "Consider growing related concepts first"
            })
            
        return gaps
        
    async def _emit_feedback_event(self, feedback: Dict):
        """Emit feedback event to Redis"""
        try:
            event = {
                "type": "growth_context_feedback",
                "timestamp": datetime.now().isoformat(),
                "data": feedback
            }
            
            # Publish to Redis
            await self.redis_client.publish(
                "lumina.growth.feedback",
                json.dumps(event)
            )
            
            logger.info(f"Emitted growth feedback event: {event}")
            
        except Exception as e:
            logger.error(f"Error emitting feedback event: {e}")
            
    async def close(self):
        """Clean up resources"""
        await self.http_client.aclose()
        self.redis_client.close() 