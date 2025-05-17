import asyncio
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel
import aiohttp
from datetime import datetime

from shared.log_config import setup_logging
from .concept_scorer import ConceptScorer

logger = setup_logging('concept-planner')

class ActionPlan(BaseModel):
    """Represents a planned action for concept maintenance."""
    concept_id: str
    concept_label: str
    action_type: str
    priority: float
    reason: str
    parameters: Dict[str, str]
    created_at: datetime
    metadata: Dict[str, float]

class ConceptPlanner:
    """Plans actions based on concept states and importance scores."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph_api_url = config['graph_api_url']
        self.concept_scorer = ConceptScorer(config)
        self.action_thresholds = config['action_thresholds']
        self.min_confidence = config['min_confidence']
        self.max_actions = config['max_actions']
        
    async def generate_action_plans(self) -> List[ActionPlan]:
        """Generate action plans based on concept states."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get all concepts with scores
                concepts = await self._get_scored_concepts(session)
                
                # Generate plans for concepts needing attention
                plans = []
                for concept in concepts:
                    plan = await self._plan_concept_actions(concept, session)
                    if plan:
                        plans.append(plan)
                
                # Sort by priority and limit
                plans.sort(key=lambda x: x.priority, reverse=True)
                plans = plans[:self.max_actions]
                
                # Log planned actions
                await self._log_action_plans(plans)
                
                return plans
                
        except Exception as e:
            logger.error(f"Error generating action plans: {str(e)}", exc_info=True)
            return []
    
    async def _get_scored_concepts(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Get all concepts with their current scores."""
        try:
            async with session.get(f"{self.graph_api_url}/nodes?type=concept") as response:
                if response.status != 200:
                    return []
                
                concepts = await response.json()
                
                # Add scores to concepts
                for concept in concepts:
                    score = await self.concept_scorer.get_concept_score(concept['id'])
                    if score:
                        concept['importance_score'] = score
                    else:
                        concept['importance_score'] = 0.0
                
                return concepts
                
        except Exception as e:
            logger.error(f"Error getting scored concepts: {str(e)}", exc_info=True)
            return []
    
    async def _plan_concept_actions(self, concept: Dict,
                                  session: aiohttp.ClientSession) -> Optional[ActionPlan]:
        """Plan actions for a single concept based on its state."""
        try:
            # Get concept metrics
            metrics = await self._get_concept_metrics(concept['id'], session)
            
            # Check confidence threshold
            if metrics['confidence'] < self.min_confidence:
                return await self._create_retrain_plan(concept, metrics)
            
            # Check usage vs importance
            if metrics['usage_score'] < self.action_thresholds['low_usage'] and \
               concept['importance_score'] > self.action_thresholds['high_importance']:
                return await self._create_test_plan(concept, metrics)
            
            # Check knowledge gaps
            if await self._has_knowledge_gaps(concept, session):
                return await self._create_crawl_plan(concept, metrics)
            
            return None
            
        except Exception as e:
            logger.error(f"Error planning concept actions: {str(e)}", exc_info=True)
            return None
    
    async def _get_concept_metrics(self, concept_id: str,
                                 session: aiohttp.ClientSession) -> Dict[str, float]:
        """Get detailed metrics for a concept."""
        try:
            async with session.get(
                f"{self.graph_api_url}/nodes/{concept_id}/metrics"
            ) as response:
                if response.status != 200:
                    return {
                        'confidence': 0.0,
                        'usage_score': 0.0,
                        'retrieval_success': 0.0,
                        'action_success': 0.0
                    }
                
                return await response.json()
                
        except Exception as e:
            logger.error(f"Error getting concept metrics: {str(e)}", exc_info=True)
            return {
                'confidence': 0.0,
                'usage_score': 0.0,
                'retrieval_success': 0.0,
                'action_success': 0.0
            }
    
    async def _has_knowledge_gaps(self, concept: Dict,
                                session: aiohttp.ClientSession) -> bool:
        """Check if a concept has knowledge gaps."""
        try:
            # Get related concepts
            async with session.get(
                f"{self.graph_api_url}/relationships?from={concept['id']}&type=relates_to"
            ) as response:
                if response.status != 200:
                    return False
                
                relationships = await response.json()
                
                # Check for gaps in relationship coverage
                total_relationships = len(relationships)
                expected_relationships = self.action_thresholds['min_relationships']
                
                return total_relationships < expected_relationships
                
        except Exception as e:
            logger.error(f"Error checking knowledge gaps: {str(e)}", exc_info=True)
            return False
    
    async def _create_retrain_plan(self, concept: Dict,
                                 metrics: Dict[str, float]) -> ActionPlan:
        """Create a plan to retrain a concept."""
        return ActionPlan(
            concept_id=concept['id'],
            concept_label=concept['properties']['label'],
            action_type='retrain_concept',
            priority=concept['importance_score'] * 0.8,
            reason=f"Low confidence ({metrics['confidence']:.2f})",
            parameters={
                'min_samples': str(self.action_thresholds['min_samples']),
                'target_confidence': str(self.min_confidence)
            },
            created_at=datetime.utcnow(),
            metadata=metrics
        )
    
    async def _create_test_plan(self, concept: Dict,
                              metrics: Dict[str, float]) -> ActionPlan:
        """Create a plan to test concept retrieval."""
        return ActionPlan(
            concept_id=concept['id'],
            concept_label=concept['properties']['label'],
            action_type='test_retrieval',
            priority=concept['importance_score'] * 0.6,
            reason=f"Low usage ({metrics['usage_score']:.2f}) despite high importance",
            parameters={
                'num_queries': '10',
                'min_success_rate': str(self.action_thresholds['min_success_rate'])
            },
            created_at=datetime.utcnow(),
            metadata=metrics
        )
    
    async def _create_crawl_plan(self, concept: Dict,
                               metrics: Dict[str, float]) -> ActionPlan:
        """Create a plan to crawl for more concept data."""
        return ActionPlan(
            concept_id=concept['id'],
            concept_label=concept['properties']['label'],
            action_type='crawl_concept',
            priority=concept['importance_score'] * 0.4,
            reason="Knowledge gap detected",
            parameters={
                'max_urls': '50',
                'min_quality': str(self.action_thresholds['min_quality'])
            },
            created_at=datetime.utcnow(),
            metadata=metrics
        )
    
    async def _log_action_plans(self, plans: List[ActionPlan]):
        """Log generated action plans."""
        try:
            for plan in plans:
                logger.info(
                    f"Generated action plan",
                    extra={
                        "concept": plan.concept_label,
                        "action": plan.action_type,
                        "priority": plan.priority,
                        "reason": plan.reason
                    }
                )
        except Exception as e:
            logger.error(f"Error logging action plans: {str(e)}", exc_info=True)

# Example usage:
# config = {
#     'graph_api_url': 'http://graph-api:8000',
#     'min_confidence': 0.7,
#     'max_actions': 10,
#     'action_thresholds': {
#         'low_usage': 0.3,
#         'high_importance': 0.8,
#         'min_relationships': 5,
#         'min_samples': 100,
#         'min_success_rate': 0.7,
#         'min_quality': 0.8
#     }
# }
# planner = ConceptPlanner(config) 