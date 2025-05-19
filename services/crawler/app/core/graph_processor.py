import json
import logging
from typing import Dict, List, Set, Optional
import wikipediaapi
from pathlib import Path
import redis
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
from neo4j import GraphDatabase
import requests
from datetime import datetime, timedelta
from .crawler import Crawler
from .wiki_client import WikiClient
from .vector_store import VectorStore
from .graph_client import GraphClient
from .concept_client import ConceptClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../graph-concept-adapter')))
from adapter import GraphConceptAdapter

logger = logging.getLogger(__name__)

class GraphProcessor:
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.concept_dict: Dict[str, List[str]] = {}
        self.last_processed_nodes: Dict[str, datetime] = {}
        self.processed_nodes: set = set()
        self.priority_queue: Dict[str, float] = {}  # Node ID -> Priority Score
        self.vector_cache: Dict[str, List[float]] = {}
        self.concept_dict_url = os.getenv("CONCEPT_DICT_URL", "http://localhost:8526")
        self.adapter = GraphConceptAdapter()
        self.concept_client = ConceptClient(self.concept_dict_url)
        
    def load_graph(self, file_path: str) -> bool:
        """Load graph data from JSON file (supports nodes with 'id' and 'summary' fields)"""
        try:
            logger.info(f"Attempting to load graph from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'nodes' in data:
                    logger.info(f"Graph file contains {len(data['nodes'])} nodes.")
                    for node in data['nodes']:
                        node_id = node.get('id')
                        if node_id:
                            # Use node_id as both concept and title for compatibility
                            if node_id not in self.concept_dict:
                                self.concept_dict[node_id] = []
                            self.concept_dict[node_id].append(node_id)
                            # Initialize priority score
                            self.priority_queue[node_id] = self._calculate_initial_priority(node)
                            # Optionally cache summary or other metadata
                            if 'summary' in node:
                                self.vector_cache[node_id] = node['summary']
            logger.info(f"Loaded {len(self.concept_dict)} concepts from graph file")
            return True
        except Exception as e:
            logger.error(f"Error loading graph file: {str(e)}")
            return False

    def _calculate_initial_priority(self, node: Dict) -> float:
        """Calculate initial priority score for a node"""
        base_priority = 0.5  # Default priority for graph nodes
        
        # Adjust priority based on node properties
        if node.get('source') == 'system':
            base_priority += 0.5  # System nodes get higher priority
        if node.get('importance_score'):
            base_priority += float(node['importance_score']) * 0.3
        if node.get('usage_count', 0) > 0:
            base_priority += min(node['usage_count'] * 0.1, 0.3)  # Cap usage boost
            
        return min(base_priority, 1.0)  # Ensure priority is between 0 and 1

    def should_process_node(self, node_id: str) -> bool:
        """Check if a node should be processed based on last processing time and priority"""
        if node_id not in self.last_processed_nodes:
            return True
        
        last_processed = self.last_processed_nodes[node_id]
        process_interval = timedelta(seconds=int(self.crawler.config.get('PROCESS_INTERVAL', 3600)))
        
        # Adjust interval based on priority
        priority = self.priority_queue.get(node_id, 0.5)
        adjusted_interval = process_interval * (1 - priority)  # Higher priority = shorter interval
        
        return datetime.now() - last_processed > adjusted_interval

    async def process_node(self, node_id: str, title: str, weight: float = 0.5) -> bool:
        """Process a single node and update its processing timestamp"""
        if node_id in self.processed_nodes:
            logger.info(f"Skipping node {node_id} - already processed")
            return True

        try:
            # Process the node using crawler
            success = await self.crawler.crawl(title)
            if success:
                # Update processing status
                self.last_processed_nodes[node_id] = datetime.now()
                self.processed_nodes.add(node_id)
                
                logger.info(f"Successfully processed node {node_id} ({title})")
                return True
            logger.error(f"Failed to process node {node_id} ({title})")
            return False
        except Exception as e:
            logger.error(f"Error processing node {node_id}: {str(e)}")
            return False

    async def process_graph(self, start_from_concepts: bool = True) -> bool:
        """Process the entire graph, prioritizing system nodes and high-priority concepts"""
        if not self.concept_dict:
            logger.error("No graph data loaded")
            return False

        try:
            # Sort nodes by priority
            sorted_nodes = sorted(
                self.priority_queue.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Process nodes in priority order
            for node_id, priority in sorted_nodes:
                concept, title = node_id.split('_', 1)
                if node_id not in self.processed_nodes:
                    logger.info(f"Processing node {node_id} with priority {priority}")
                    await self.process_node(node_id, title)

            logger.info(f"Graph processing completed. Processed {len(self.processed_nodes)} nodes")
            return True
        except Exception as e:
            logger.error(f"Error processing graph: {str(e)}")
            return False

    def get_processing_stats(self) -> Dict:
        """Get statistics about the graph processing"""
        return {
            "total_concepts": len(self.concept_dict),
            "total_nodes": sum(len(titles) for titles in self.concept_dict.values()),
            "processed_nodes": len(self.processed_nodes),
            "priority_distribution": {
                "high": len([p for p in self.priority_queue.values() if p > 0.7]),
                "medium": len([p for p in self.priority_queue.values() if 0.3 <= p <= 0.7]),
                "low": len([p for p in self.priority_queue.values() if p < 0.3])
            },
            "vector_cache_size": len(self.vector_cache),
            "last_processed": {
                node_id: timestamp.isoformat()
                for node_id, timestamp in self.last_processed_nodes.items()
            }
        }

    def update_node_priority(self, node_id: str, new_priority: float) -> None:
        """Update the priority of a node"""
        self.priority_queue[node_id] = min(max(new_priority, 0.0), 1.0)
        logger.info(f"Updated priority for node {node_id} to {new_priority}")

    def _register_concepts(self, concepts: List[str]) -> None:
        """Register concepts in the concept dictionary."""
        try:
            for concept in concepts:
                concept_data = {
                    "term": concept,
                    "definition": "",
                    "sources": ["wikipedia"],
                    "usage_count": 0
                }
                # Use RedisClient directly instead of adapter or ConceptClient
                asyncio.run(self.crawler.redis.add_concept(concept, concept_data))
            logger.info(f"Registered {len(concepts)} concepts in dictionary via Redis")
        except Exception as e:
            logger.error(f"Error registering concepts: {e}")

    def _store_in_neo4j(self, node: str, wiki_data: Dict, connections: List[str]) -> None:
        """Store node data and connections using the adapter."""
        try:
            # Prepare node payload
            node_payload = {
                "title": wiki_data.get("title", node),
                "summary": wiki_data.get("summary", ""),
                "url": wiki_data.get("url", ""),
                "categories": wiki_data.get("categories", []),
                "links": wiki_data.get("links", []),
                "source": "wikipedia",
                "timestamp": datetime.now().isoformat(),
                "embedding": wiki_data.get("embedding", None),
            }
            # Send node to adapter
            import asyncio
            asyncio.run(self.adapter.process_and_send(node_payload))
            # Add edges for all connections
            for connected in connections:
                asyncio.run(self.adapter.add_edge(node_payload["title"], connected, "RELATED_TO"))
        except Exception as e:
            logger.error(f"Error storing in adapter: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search_wikipedia(self, query: str) -> Optional[Dict]:
        """Search Wikipedia for a given query with caching and retries."""
        # Check cache first
        cache_key = f"wiki:{query}"
        cached_result = self.redis.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

        try:
            page = self.wiki.page(query)
            if page.exists():
                result = {
                    'title': page.title,
                    'summary': page.summary,
                    'url': page.fullurl,
                    'links': list(page.links.keys())
                }
                # Cache the result for 24 hours
                self.redis.setex(cache_key, 86400, json.dumps(result))
                return result
            return None
        except Exception as e:
            logger.error(f"Error searching Wikipedia for {query}: {e}")
            return None

    def process_node(self, node: str, depth: int = 2) -> Optional[Dict]:
        """Process a single node and its connections up to specified depth."""
        if node in self.processed_nodes or depth <= 0:
            return None

        self.processed_nodes.add(node)
        
        # Check if node data is cached
        cache_key = f"node:{node}:{depth}"
        cached_result = self.redis.get(cache_key)
        if cached_result:
            return json.loads(cached_result)

        try:
            wiki_data = self.search_wikipedia(node)
            if not wiki_data:
                return None

            result = {
                'node': node,
                'wiki_data': wiki_data,
                'connections': []
            }

            # Get connections from graph
            connected_nodes = self.connections.get(node, [])
            
            # Store in Neo4j
            self._store_in_neo4j(node, wiki_data, connected_nodes)

            # Process connected nodes
            for connected_node in connected_nodes:
                if connected_node not in self.processed_nodes:
                    connected_data = self.process_node(connected_node, depth - 1)
                    if connected_data:
                        result['connections'].append(connected_data)

            # Cache the result for 12 hours
            self.redis.setex(cache_key, 43200, json.dumps(result))
            return result
        except Exception as e:
            logger.error(f"Error processing node {node}: {e}")
            return None

    def process_graph(self, start_nodes: List[str] = None, depth: int = 2) -> List[Dict]:
        """Process the entire graph starting from given nodes."""
        if not start_nodes:
            # Get concepts from concept dictionary
            try:
                response = requests.get(f"{self.concept_dict_url}/api/v1/concepts")
                if response.status_code == 200:
                    concepts = response.json().get("concepts", [])
                    start_nodes = concepts[:5]  # Start with first 5 concepts
                else:
                    start_nodes = list(self.connections.keys())[:5]
            except Exception as e:
                logger.error(f"Error getting concepts from dictionary: {e}")
                start_nodes = list(self.connections.keys())[:5]

        results = []
        for node in start_nodes:
            if node not in self.processed_nodes:
                try:
                    node_data = self.process_node(node, depth)
                    if node_data:
                        results.append(node_data)
                except Exception as e:
                    logger.error(f"Error processing start node {node}: {e}")
                    continue

        return results

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.vector_cache.clear()
        logger.info("Vector cache cleared")

    def get_unprocessed_concepts(self) -> List[str]:
        """Get concepts that haven't been processed yet"""
        unprocessed = []
        for concept, titles in self.concept_dict.items():
            for title in titles:
                node_id = f"{concept}_{title}"
                if node_id not in self.processed_nodes:
                    unprocessed.append(title)
        logger.info(f"get_unprocessed_concepts: {len(unprocessed)} concepts unprocessed.")
        return unprocessed

    def get_vector(self, concept: str) -> Optional[List[float]]:
        """Get vector for a concept from cache or storage"""
        # Check cache first
        if concept in self.vector_cache:
            return self.vector_cache[concept]
        
        # Try to get from storage
        try:
            vector_data = self.crawler.redis_client.get_vector(concept)
            if vector_data and 'vector' in vector_data:
                # Cache the vector
                self.vector_cache[concept] = vector_data['vector']
                return vector_data['vector']
        except Exception as e:
            logger.error(f"Error retrieving vector for {concept}: {e}")
        
        return None 