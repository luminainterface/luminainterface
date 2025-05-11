from typing import List, Dict, Optional
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class ConceptClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_concept(self, concept_id: str) -> Optional[Dict]:
        """Get concept details by ID"""
        try:
            response = await self.client.get(f"{self.base_url}/concepts/{concept_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting concept {concept_id}: {str(e)}")
            return None
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def add_concept(self, concept_id: str, data: Dict) -> bool:
        """Add or update a concept"""
        try:
            response = await self.client.post(
                f"{self.base_url}/concepts",
                json={
                    "concept_id": concept_id,
                    **data
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error adding concept {concept_id}: {str(e)}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_concepts(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for concepts by query string"""
        try:
            response = await self.client.get(
                f"{self.base_url}/concepts/search",
                params={
                    "q": query,
                    "limit": limit
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error searching concepts with query '{query}': {str(e)}")
            return []
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_related_concepts(self, concept_id: str, relation_type: Optional[str] = None) -> List[Dict]:
        """Get concepts related to the given concept"""
        try:
            params = {"relation_type": relation_type} if relation_type else {}
            response = await self.client.get(
                f"{self.base_url}/concepts/{concept_id}/related",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting related concepts for {concept_id}: {str(e)}")
            return [] 