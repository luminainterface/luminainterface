import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import aiohttp
import numpy as np
from datetime import datetime

from shared.log_config import setup_logging
from .concept_scorer import ConceptScorer

logger = setup_logging('concept-retriever')

class RetrievalResult(BaseModel):
    """Represents a retrieval result with concept-weighted scoring."""
    document_id: str
    content: str
    similarity_score: float
    concept_scores: Dict[str, float]
    matched_concepts: List[str]
    metadata: Dict[str, str]
    timestamp: datetime

class ConceptRetriever:
    """Implements concept-weighted retrieval using importance scores."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.graph_api_url = config['graph_api_url']
        self.vector_api_url = config['vector_api_url']
        self.concept_scorer = ConceptScorer(config)
        self.min_similarity = config['min_similarity']
        self.concept_weight = config['concept_weight']
        self.max_results = config['max_results']
        
    async def retrieve(self, query: str, context: Optional[Dict] = None) -> List[RetrievalResult]:
        """Retrieve documents using concept-weighted scoring."""
        try:
            # Extract concepts from query
            query_concepts = await self._extract_query_concepts(query)
            
            # Get concept scores
            concept_scores = {}
            for concept in query_concepts:
                score = await self.concept_scorer.get_concept_score(concept['id'])
                if score:
                    concept_scores[concept['id']] = score
            
            # Perform vector search with concept weighting
            results = await self._vector_search(query, query_concepts, concept_scores)
            
            # Post-process results
            processed_results = await self._process_results(results, query_concepts)
            
            # Log retrieval metrics
            await self._log_retrieval_metrics(query, processed_results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in concept-weighted retrieval: {str(e)}", exc_info=True)
            return []
    
    async def _extract_query_concepts(self, query: str) -> List[Dict]:
        """Extract relevant concepts from the query."""
        try:
            async with aiohttp.ClientSession() as session:
                # Use concept extractor to find relevant concepts
                async with session.post(
                    f"{self.graph_api_url}/extract_concepts",
                    json={"text": query}
                ) as response:
                    if response.status != 200:
                        return []
                    
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"Error extracting query concepts: {str(e)}", exc_info=True)
            return []
    
    async def _vector_search(self, query: str, concepts: List[Dict],
                           concept_scores: Dict[str, float]) -> List[Dict]:
        """Perform vector search with concept weighting."""
        try:
            async with aiohttp.ClientSession() as session:
                # Prepare search payload with concept weights
                search_payload = {
                    "query": query,
                    "concepts": [
                        {
                            "id": concept['id'],
                            "weight": concept_scores.get(concept['id'], 0.0)
                        }
                        for concept in concepts
                    ],
                    "min_similarity": self.min_similarity,
                    "max_results": self.max_results
                }
                
                # Perform weighted search
                async with session.post(
                    f"{self.vector_api_url}/search",
                    json=search_payload
                ) as response:
                    if response.status != 200:
                        return []
                    
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}", exc_info=True)
            return []
    
    async def _process_results(self, results: List[Dict],
                             query_concepts: List[Dict]) -> List[RetrievalResult]:
        """Process and enrich search results."""
        try:
            processed_results = []
            
            for result in results:
                # Get matched concepts
                matched_concepts = await self._get_matched_concepts(
                    result['document_id'],
                    query_concepts
                )
                
                # Calculate concept-weighted score
                concept_scores = {}
                for concept in matched_concepts:
                    score = await self.concept_scorer.get_concept_score(concept['id'])
                    if score:
                        concept_scores[concept['id']] = score
                
                # Create retrieval result
                processed_result = RetrievalResult(
                    document_id=result['document_id'],
                    content=result['content'],
                    similarity_score=result['similarity_score'],
                    concept_scores=concept_scores,
                    matched_concepts=[c['label'] for c in matched_concepts],
                    metadata=result.get('metadata', {}),
                    timestamp=datetime.utcnow()
                )
                
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}", exc_info=True)
            return []
    
    async def _get_matched_concepts(self, document_id: str,
                                  query_concepts: List[Dict]) -> List[Dict]:
        """Get concepts that match between document and query."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get document concepts
                async with session.get(
                    f"{self.graph_api_url}/relationships?from={document_id}&type=contains"
                ) as response:
                    if response.status != 200:
                        return []
                    
                    doc_concepts = await response.json()
                    
                    # Find matching concepts
                    matched = []
                    for query_concept in query_concepts:
                        for doc_concept in doc_concepts:
                            if self._concepts_match(query_concept, doc_concept):
                                matched.append(doc_concept)
                    
                    return matched
                    
        except Exception as e:
            logger.error(f"Error getting matched concepts: {str(e)}", exc_info=True)
            return []
    
    def _concepts_match(self, concept1: Dict, concept2: Dict) -> bool:
        """Check if two concepts match based on similarity."""
        try:
            # Check exact match first
            if concept1['id'] == concept2['id']:
                return True
            
            # Check semantic similarity
            similarity = concept1.get('similarity', {}).get(concept2['id'], 0.0)
            return similarity > self.min_similarity
            
        except Exception as e:
            logger.error(f"Error checking concept match: {str(e)}", exc_info=True)
            return False
    
    async def _log_retrieval_metrics(self, query: str, results: List[RetrievalResult]):
        """Log retrieval metrics for analysis."""
        try:
            metrics = {
                "query": query,
                "num_results": len(results),
                "avg_similarity": np.mean([r.similarity_score for r in results]) if results else 0.0,
                "avg_concept_score": np.mean([
                    np.mean(list(r.concept_scores.values()))
                    for r in results
                ]) if results else 0.0,
                "num_matched_concepts": np.mean([
                    len(r.matched_concepts)
                    for r in results
                ]) if results else 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("Retrieval metrics", extra=metrics)
            
        except Exception as e:
            logger.error(f"Error logging retrieval metrics: {str(e)}", exc_info=True)

# Example usage:
# config = {
#     'graph_api_url': 'http://graph-api:8000',
#     'vector_api_url': 'http://vector-api:8000',
#     'min_similarity': 0.6,
#     'concept_weight': 0.4,
#     'max_results': 10
# }
# retriever = ConceptRetriever(config) 