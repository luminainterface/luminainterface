import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import aiohttp
from collections import defaultdict

from shared.log_config import setup_logging

logger = setup_logging('concept-scorer')

class ConceptScore(BaseModel):
    """Represents the importance score and metadata for a concept."""
    concept_id: str
    label: str
    importance_score: float
    source_weight: float
    frequency_score: float
    usage_score: float
    last_updated: datetime
    metadata: Dict[str, float]

class ConceptScorer:
    """Manages concept importance scoring and updates."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph_api_url = config['graph_api_url']
        self.source_weights = config['source_weights']
        self.frequency_weighting = config['frequency_weighting']
        self.action_success_boost = config['action_success_boost']
        self.score_cache = {}
        self.score_ttl = timedelta(minutes=config.get('score_cache_ttl', 30))
        
    async def calculate_concept_scores(self) -> Dict[str, ConceptScore]:
        """Calculate importance scores for all concepts."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get all concepts from graph
                async with session.get(f"{self.graph_api_url}/nodes?type=concept") as response:
                    if response.status != 200:
                        logger.error(f"Error fetching concepts: {await response.text()}")
                        return {}
                    
                    concepts = await response.json()
                    
                    # Calculate scores for each concept
                    scores = {}
                    for concept in concepts:
                        score = await self._calculate_concept_score(concept, session)
                        if score:
                            scores[concept['id']] = score
                    
                    # Update scores in graph
                    await self._update_concept_scores(scores, session)
                    
                    return scores
                    
        except Exception as e:
            logger.error(f"Error calculating concept scores: {str(e)}", exc_info=True)
            return {}
    
    async def _calculate_concept_score(self, concept: Dict, session: aiohttp.ClientSession) -> Optional[ConceptScore]:
        """Calculate importance score for a single concept."""
        try:
            # Get source documents
            source_weight = await self._get_source_weight(concept, session)
            
            # Calculate frequency score
            frequency_score = await self._calculate_frequency_score(concept, session)
            
            # Calculate usage score
            usage_score = await self._calculate_usage_score(concept, session)
            
            # Calculate final importance score
            importance_score = (
                source_weight * self.source_weights['source_weight'] +
                frequency_score * self.source_weights['frequency_weight'] +
                usage_score * self.source_weights['usage_weight']
            )
            
            return ConceptScore(
                concept_id=concept['id'],
                label=concept['properties']['label'],
                importance_score=importance_score,
                source_weight=source_weight,
                frequency_score=frequency_score,
                usage_score=usage_score,
                last_updated=datetime.utcnow(),
                metadata={
                    "source_weight": source_weight,
                    "frequency_score": frequency_score,
                    "usage_score": usage_score
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating score for concept {concept.get('id')}: {str(e)}", exc_info=True)
            return None
    
    async def _get_source_weight(self, concept: Dict, session: aiohttp.ClientSession) -> float:
        """Calculate source weight based on document types."""
        try:
            # Get source documents for concept
            async with session.get(
                f"{self.graph_api_url}/relationships?to={concept['id']}&type=contains"
            ) as response:
                if response.status != 200:
                    return 0.0
                
                relationships = await response.json()
                
                # Calculate weighted average of source weights
                total_weight = 0.0
                count = 0
                
                for rel in relationships:
                    source_type = rel['from']['properties'].get('source', 'unknown')
                    weight = self.source_weights.get(source_type, 0.0)
                    total_weight += weight
                    count += 1
                
                return total_weight / count if count > 0 else 0.0
                
        except Exception as e:
            logger.error(f"Error calculating source weight: {str(e)}", exc_info=True)
            return 0.0
    
    async def _calculate_frequency_score(self, concept: Dict, session: aiohttp.ClientSession) -> float:
        """Calculate frequency score based on concept occurrence."""
        try:
            if not self.frequency_weighting:
                return 0.0
            
            # Get all occurrences of concept
            async with session.get(
                f"{self.graph_api_url}/relationships?to={concept['id']}&type=contains"
            ) as response:
                if response.status != 200:
                    return 0.0
                
                relationships = await response.json()
                
                # Calculate normalized frequency score
                total_docs = await self._get_total_documents(session)
                if total_docs == 0:
                    return 0.0
                
                return len(relationships) / total_docs
                
        except Exception as e:
            logger.error(f"Error calculating frequency score: {str(e)}", exc_info=True)
            return 0.0
    
    async def _calculate_usage_score(self, concept: Dict, session: aiohttp.ClientSession) -> float:
        """Calculate usage score based on successful retrievals and actions."""
        try:
            # Get successful retrievals
            async with session.get(
                f"{self.graph_api_url}/relationships?to={concept['id']}&type=retrieved"
            ) as response:
                if response.status != 200:
                    return 0.0
                
                retrievals = await response.json()
                
                # Get successful actions
                async with session.get(
                    f"{self.graph_api_url}/relationships?to={concept['id']}&type=used_in_action"
                ) as action_response:
                    if action_response.status != 200:
                        return 0.0
                    
                    actions = await action_response.json()
                
                # Calculate usage score
                retrieval_score = len(retrievals) * self.action_success_boost
                action_score = len(actions) * self.action_success_boost
                
                return min(1.0, retrieval_score + action_score)
                
        except Exception as e:
            logger.error(f"Error calculating usage score: {str(e)}", exc_info=True)
            return 0.0
    
    async def _get_total_documents(self, session: aiohttp.ClientSession) -> int:
        """Get total number of documents in the system."""
        try:
            async with session.get(f"{self.graph_api_url}/nodes?type=document") as response:
                if response.status != 200:
                    return 0
                
                documents = await response.json()
                return len(documents)
                
        except Exception as e:
            logger.error(f"Error getting total documents: {str(e)}", exc_info=True)
            return 0
    
    async def _update_concept_scores(self, scores: Dict[str, ConceptScore], session: aiohttp.ClientSession):
        """Update concept scores in the graph."""
        try:
            for concept_id, score in scores.items():
                # Update concept node with new score
                update_payload = {
                    "properties": {
                        "importance_score": score.importance_score,
                        "source_weight": score.source_weight,
                        "frequency_score": score.frequency_score,
                        "usage_score": score.usage_score,
                        "last_updated": score.last_updated.isoformat(),
                        "score_metadata": score.metadata
                    }
                }
                
                async with session.patch(
                    f"{self.graph_api_url}/nodes/{concept_id}",
                    json=update_payload
                ) as response:
                    if response.status != 200:
                        logger.error(f"Error updating concept score: {await response.text()}")
            
            logger.info(f"Updated scores for {len(scores)} concepts")
            
        except Exception as e:
            logger.error(f"Error updating concept scores: {str(e)}", exc_info=True)
    
    async def get_concept_score(self, concept_id: str) -> Optional[float]:
        """Get cached importance score for a concept."""
        try:
            # Check cache first
            if concept_id in self.score_cache:
                cache_entry = self.score_cache[concept_id]
                if datetime.utcnow() - cache_entry['timestamp'] < self.score_ttl:
                    return cache_entry['score']
            
            # Calculate new score if not in cache or expired
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.graph_api_url}/nodes/{concept_id}") as response:
                    if response.status != 200:
                        return None
                    
                    concept = await response.json()
                    score = await self._calculate_concept_score(concept, session)
                    
                    if score:
                        # Update cache
                        self.score_cache[concept_id] = {
                            'score': score.importance_score,
                            'timestamp': datetime.utcnow()
                        }
                        return score.importance_score
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting concept score: {str(e)}", exc_info=True)
            return None

# Example usage:
# config = {
#     'graph_api_url': 'http://graph-api:8000',
#     'source_weights': {
#         'source_weight': 0.4,
#         'frequency_weight': 0.3,
#         'usage_weight': 0.3
#     },
#     'frequency_weighting': True,
#     'action_success_boost': 0.2,
#     'score_cache_ttl': 30
# }
# scorer = ConceptScorer(config) 