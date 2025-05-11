from typing import List, Dict, Optional
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class GraphClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def add_node(self, node_id: str, properties: Dict) -> bool:
        """Add a node to the graph with given properties"""
        try:
            response = await self.client.post(
                f"{self.base_url}/nodes",
                json={
                    "node_id": node_id,
                    "properties": properties
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error adding node {node_id}: {str(e)}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def add_edge(self, from_id: str, to_id: str, edge_type: str, properties: Optional[Dict] = None) -> bool:
        """Add an edge between two nodes"""
        try:
            response = await self.client.post(
                f"{self.base_url}/edges",
                json={
                    "from_id": from_id,
                    "to_id": to_id,
                    "edge_type": edge_type,
                    "properties": properties or {}
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error adding edge from {from_id} to {to_id}: {str(e)}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_node(self, node_id: str) -> Optional[Dict]:
        """Get a node's properties by ID"""
        try:
            response = await self.client.get(f"{self.base_url}/nodes/{node_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting node {node_id}: {str(e)}")
            return None
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[Dict]:
        """Get neighboring nodes of a given node"""
        try:
            params = {"edge_type": edge_type} if edge_type else {}
            response = await self.client.get(
                f"{self.base_url}/nodes/{node_id}/neighbors",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting neighbors for node {node_id}: {str(e)}")
            return [] 