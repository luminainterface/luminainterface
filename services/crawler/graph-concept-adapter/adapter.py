import os
import httpx
import logging
from typing import Dict, Any, Optional
from neo4j import GraphDatabase

GRAPH_API_URL = os.getenv("GRAPH_API_URL", "http://graph-api:8200")
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8526")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

logger = logging.getLogger("graph_concept_adapter")
logging.basicConfig(level=logging.INFO)

class GraphConceptAdapter:
    def __init__(self):
        self.graph_api_url = GRAPH_API_URL
        self.concept_dict_url = CONCEPT_DICT_URL
        self.neo4j_url = NEO4J_URL
        self.neo4j_user = NEO4J_USER
        self.neo4j_password = NEO4J_PASSWORD
        self.graph_api_key = os.getenv("GRAPH_API_KEY", "changeme")
        self.neo4j_driver = GraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_user, self.neo4j_password))

    async def process_and_send(self, raw_concept: Dict[str, Any]) -> bool:
        """Validate, translate, and send concept to graph-api, concept-dictionary, and Neo4j."""
        try:
            # --- Normalize for Graph API ---
            graph_payload = {
                "id": raw_concept.get("title"),
                "type": "concept",
                "properties": {
                    "name": raw_concept.get("title"),
                    "description": raw_concept.get("summary"),
                    "url": raw_concept.get("url"),
                    "categories": raw_concept.get("categories", []),
                    "links": raw_concept.get("links", []),
                    "source": raw_concept.get("source", "wikipedia"),
                    "crawl_timestamp": raw_concept.get("timestamp"),
                    "embedding": raw_concept.get("embedding"),
                }
            }
            # Validate required fields
            if not graph_payload["id"] or not graph_payload["type"]:
                logger.error(f"Missing required fields in graph_payload: {graph_payload}")
                return False
            logger.info(f"Sending to Graph API /nodes: {graph_payload}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    graph_resp = await client.post(
                        f"{self.graph_api_url}/nodes",
                        json=graph_payload,
                        headers={"X-API-Key": self.graph_api_key}
                    )
                    logger.info(f"Graph API /nodes response: {graph_resp.status_code} {graph_resp.text}")
                    graph_resp.raise_for_status()
                    logger.info(f"Graph API node added: {graph_resp.json()}")
                except Exception as e:
                    logger.error(f"Graph API node add failed, trying Neo4j: {e}")
                    self._add_node_neo4j(graph_payload["properties"])

                # --- Normalize for Concept Dictionary ---
                dict_payload = {
                    "term": raw_concept.get("title"),
                    "definition": raw_concept.get("summary"),
                    "embedding": raw_concept.get("embedding"),
                    "sources": [raw_concept.get("source", "wikipedia")],
                    "usage_count": 0
                }
                logger.info(f"Sending to Concept Dictionary /concepts: {dict_payload}")
                try:
                    dict_resp = await client.put(
                        f"{self.concept_dict_url}/concepts/{dict_payload['term']}",
                        json=dict_payload,
                        headers={"X-API-Key": self.graph_api_key}
                    )
                    dict_resp.raise_for_status()
                    logger.info(f"Concept Dictionary /concepts response: {dict_resp.status_code} {dict_resp.text}")
                    logger.info(f"Concept Dictionary entry added: {dict_resp.json()}")
                except Exception as e:
                    logger.warning(f"Concept Dictionary endpoint (/concepts) not reachable or error: {e}. Skipping concept dictionary update.")
                    # (We do not re-raise so that the crawler continues processing.)

            return True
        except Exception as e:
            logger.error(f"Adapter error: {e}")
            return False

    async def add_edge(self, from_id: str, to_id: str, edge_type: str = "RELATED_TO", properties: Optional[Dict] = None) -> bool:
        """Add an edge between two nodes via graph-api, fallback to Neo4j if needed."""
        edge_payload = {
            "source": from_id,
            "target": to_id,
            "type": edge_type,
            "properties": properties or {}
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    edge_resp = await client.post(
                        f"{self.graph_api_url}/edges",
                        json=edge_payload,
                        headers={"X-API-Key": self.graph_api_key}
                    )
                    edge_resp.raise_for_status()
                    logger.info(f"Graph API edge added: {edge_resp.json()}")
                    return True
                except Exception as e:
                    logger.error(f"Graph API edge add failed, trying Neo4j: {e}")
                    self._add_edge_neo4j(from_id, to_id, edge_type, properties)
                    return True
        except Exception as e:
            logger.error(f"Adapter edge error: {e}")
            return False

    def _add_node_neo4j(self, properties: Dict[str, Any]):
        """Directly add a node to Neo4j."""
        try:
            with self.neo4j_driver.session() as session:
                session.run(
                    """
                    MERGE (n:Concept {name: $name})
                    SET n += $props
                    """,
                    name=properties.get("name"),
                    props=properties
                )
            logger.info(f"Neo4j node added: {properties.get('name')}")
        except Exception as e:
            logger.error(f"Neo4j node add failed: {e}")

    def _add_edge_neo4j(self, from_id: str, to_id: str, edge_type: str, properties: Optional[Dict]):
        """Directly add an edge to Neo4j."""
        try:
            with self.neo4j_driver.session() as session:
                session.run(
                    f"""
                    MATCH (a:Concept {{name: $from_name}}), (b:Concept {{name: $to_name}})
                    MERGE (a)-[r:{edge_type}]->(b)
                    SET r += $props
                    """,
                    from_name=from_id,
                    to_name=to_id,
                    props=properties or {}
                )
            logger.info(f"Neo4j edge added: {from_id} -> {to_id} [{edge_type}]")
        except Exception as e:
            logger.error(f"Neo4j edge add failed: {e}") 