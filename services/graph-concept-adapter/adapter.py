import os
import httpx
import logging
from typing import Dict, Any, Optional, List
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class GraphConceptAdapter:
    def __init__(self):
        # Graph API configuration
        self.graph_api_url = os.getenv("GRAPH_API_URL", "http://graph-api:8200")
        
        # Neo4j configuration
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        # Initialize Neo4j driver
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # New configuration
        self.GRAPH_API_KEY = os.getenv("GRAPH_API_KEY", "changeme")
        self.CONCEPT_DICT_API_KEY = os.getenv("CONCEPT_DICT_API_KEY", "changeme")

    async def add_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Add a node to the graph, trying Graph API first, then falling back to Neo4j."""
        try:
            # Try Graph API first
            response = await self.http_client.post(
                f"{self.graph_api_url}/nodes",
                json={"id": node_id, "properties": properties},
                headers={"X-API-Key": self.GRAPH_API_KEY}
            )
            response.raise_for_status()
            logger.info(f"Graph API node added: {node_id}")
            return True
        except Exception as e:
            logger.error(f"Graph API node add failed, trying Neo4j: {str(e)}")
            try:
                # Fall back to Neo4j
                with self.neo4j_driver.session() as session:
                    query = """
                    MERGE (n:Concept {id: $id})
                    SET n += $properties
                    """
                    session.run(query, id=node_id, properties=properties)
                logger.info(f"Neo4j node added: {node_id}")
                return True
            except Exception as neo4j_error:
                logger.error(f"Neo4j node add failed: {str(neo4j_error)}")
                return False

    async def add_edge(self, source_id: str, target_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """Add an edge to the graph, trying Graph API first, then falling back to Neo4j."""
        if properties is None:
            properties = {}
            
        try:
            # Try Graph API first
            response = await self.http_client.post(
                f"{self.graph_api_url}/edges",
                json={
                    "source": source_id,
                    "target": target_id,
                    "type": relationship_type,
                    "properties": properties
                },
                headers={"X-API-Key": self.GRAPH_API_KEY}
            )
            response.raise_for_status()
            logger.info(f"Graph API edge added: {source_id} -> {target_id} [{relationship_type}]")
            return True
        except Exception as e:
            logger.error(f"Graph API edge add failed, trying Neo4j: {str(e)}")
            try:
                # Fall back to Neo4j
                with self.neo4j_driver.session() as session:
                    query = """
                    MATCH (source:Concept {id: $source_id})
                    MATCH (target:Concept {id: $target_id})
                    MERGE (source)-[r:$relationship_type]->(target)
                    SET r += $properties
                    """
                    session.run(
                        query,
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type=relationship_type,
                        properties=properties
                    )
                logger.info(f"Neo4j edge added: {source_id} -> {target_id} [{relationship_type}]")
                return True
            except Exception as neo4j_error:
                logger.error(f"Neo4j edge add failed: {str(neo4j_error)}")
                return False

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node from the graph, trying Graph API first, then falling back to Neo4j."""
        try:
            # Try Graph API first
            response = await self.http_client.get(f"{self.graph_api_url}/nodes/{node_id}", headers={"X-API-Key": self.GRAPH_API_KEY})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Graph API node get failed, trying Neo4j: {str(e)}")
            try:
                # Fall back to Neo4j
                with self.neo4j_driver.session() as session:
                    query = """
                    MATCH (n:Concept {id: $id})
                    RETURN n
                    """
                    result = session.run(query, id=node_id)
                    record = result.single()
                    if record:
                        node = record["n"]
                        return {
                            "id": node["id"],
                            "properties": dict(node)
                        }
                    return None
            except Exception as neo4j_error:
                logger.error(f"Neo4j node get failed: {str(neo4j_error)}")
                return None

    async def get_edges(self, node_id: str) -> List[Dict[str, Any]]:
        """Get edges for a node, trying Graph API first, then falling back to Neo4j."""
        try:
            # Try Graph API first
            response = await self.http_client.get(f"{self.graph_api_url}/nodes/{node_id}/edges", headers={"X-API-Key": self.GRAPH_API_KEY})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Graph API edges get failed, trying Neo4j: {str(e)}")
            try:
                # Fall back to Neo4j
                with self.neo4j_driver.session() as session:
                    query = """
                    MATCH (n:Concept {id: $id})-[r]->(target)
                    RETURN r, target
                    """
                    result = session.run(query, id=node_id)
                    edges = []
                    for record in result:
                        edge = record["r"]
                        target = record["target"]
                        edges.append({
                            "source": node_id,
                            "target": target["id"],
                            "type": type(edge).__name__,
                            "properties": dict(edge)
                        })
                    return edges
            except Exception as neo4j_error:
                logger.error(f"Neo4j edges get failed: {str(neo4j_error)}")
                return []

    def close(self):
        """Close connections."""
        self.neo4j_driver.close()
        self.http_client.close() 