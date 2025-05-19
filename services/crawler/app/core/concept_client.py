from typing import List, Dict, Optional, Literal
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime

logger = logging.getLogger(__name__)

ConceptStatus = Literal["pending", "training", "trained", "error"]

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
            response = await self.client.put(
                f"{self.base_url}/concepts/{concept_id}",
                json=data
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_next_untrained_concept(self) -> Optional[Dict]:
        """Get the next concept that needs training"""
        try:
            response = await self.client.get(
                f"{self.base_url}/concepts/queue/next",
                params={"status": "pending"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting next untrained concept: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def update_concept_status(
        self, 
        concept_id: str, 
        status: ConceptStatus,
        embedding: Optional[List[float]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update concept status and optionally its embedding"""
        try:
            update_data = {
                "status": status,
                "last_updated": datetime.utcnow().isoformat()
            }
            if embedding is not None:
                update_data["embedding"] = embedding
            if error_message is not None:
                update_data["error_message"] = error_message

            response = await self.client.patch(
                f"{self.base_url}/concepts/{concept_id}/status",
                json=update_data
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error updating concept {concept_id} status: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_concepts_by_status(
        self, 
        status: ConceptStatus,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get concepts by their training status"""
        try:
            response = await self.client.get(
                f"{self.base_url}/concepts/status/{status}",
                params={"limit": limit, "offset": offset}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting concepts with status {status}: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def retry_failed_concepts(self, max_retries: int = 3) -> bool:
        """Reset failed concepts to pending status for retry"""
        try:
            response = await self.client.post(
                f"{self.base_url}/concepts/retry-failed",
                json={"max_retries": max_retries}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error retrying failed concepts: {str(e)}")
            return False 