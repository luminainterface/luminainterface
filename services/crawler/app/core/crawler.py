from typing import List, Dict, Optional, Set
import asyncio
import logging
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime
import uuid
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../graph-concept-adapter')))
from adapter import GraphConceptAdapter

from .wiki_client import WikiClient
from .vector_store import VectorStore
from .graph_client import GraphClient
from .concept_client import ConceptClient
from .redis_client import RedisClient

logger = logging.getLogger(__name__)

class Crawler:
    def __init__(
        self,
        redis_url: str,
        qdrant_url: str,
        graph_api_url: str,
        concept_dict_url: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_depth: int = 2,
        max_links_per_page: int = 10,
        cache_ttl: int = 86400  # 24 hours
    ):
        self.wiki_client = WikiClient()
        self.vector_store = VectorStore(qdrant_url)
        self.graph_client = GraphClient(graph_api_url)
        self.concept_client = ConceptClient(concept_dict_url)
        self.adapter = GraphConceptAdapter()
        self.redis_client = RedisClient(redis_url)
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.max_depth = max_depth
        self.max_links = max_links_per_page
        self.cache_ttl = cache_ttl
        self.concept_dict_url = concept_dict_url
        
        # Initialize vector store collection
        try:
            self.vector_store.init_collection(vector_size=self.embedding_model.get_sentence_embedding_dimension())
            logger.info("Vector store collection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store collection: {e}")
            raise
        
    def _generate_id(self, title: str) -> str:
        """Generate a consistent ID for a Wikipedia page"""
        # Use the title to seed the UUID generation for consistency
        namespace = uuid.NAMESPACE_DNS
        return str(uuid.uuid5(namespace, title.lower()))
        
    async def _process_page(self, title: str, depth: int = 0, visited: Optional[Set[str]] = None) -> Optional[str]:
        """Process a single Wikipedia page and its links"""
        if visited is None:
            visited = set()
        if depth > self.max_depth or title in visited:
            return None
        visited.add(title)
        page_id = self._generate_id(title)
        
        # Check cache first
        cached_data = await self.redis_client.get_cache(f"page:{page_id}")
        if cached_data:
            logger.info(f"Cache hit for page {title}")
            return page_id
            
        # Fetch and process page
        page = self.wiki_client.get_page(title)
        if not page:
            return None
            
        # Get page content
        summary = self.wiki_client.get_summary(page)
        full_text = self.wiki_client.get_full_text(page)
        
        # Generate embeddings
        summary_embedding = self.embedding_model.encode(summary)
        
        # Store in vector database
        metadata = {
            "title": title,
            "summary": summary,
            "url": page.fullurl,
            "last_updated": datetime.utcnow().isoformat()
        }
        self.vector_store.upsert_vectors(
            vectors=[summary_embedding],
            metadata=[metadata],
            ids=[page_id]
        )
        
        # Prepare concept data
        concept_data = {
            "title": title,
            "summary": summary,
            "content": full_text,
            "url": page.fullurl,
            "timestamp": datetime.utcnow().isoformat(),
            "embedding": summary_embedding.tolist()
        }
        
        # Store in concept dictionary
        try:
            await self.concept_client.add_concept(title, concept_data)
        except Exception as e:
            logger.error(f"Failed to store concept {title}: {e}")
            return None
            
        # Publish to ingest.crawl stream
        try:
            await self.redis_client.publish_crawl_result(title, {
                "url": page.fullurl,
                "title": title,
                "page_id": page_id,
                "summary": summary,
                "metadata": metadata
            })
        except Exception as e:
            logger.error(f"Failed to publish crawl result for {title}: {e}")
            
        # Cache the processed state
        await self.redis_client.set_cache(
            f"page:{page_id}",
            {"processed": True, "timestamp": datetime.utcnow().isoformat()},
            self.cache_ttl
        )
        return page_id
        
    async def crawl(self, start_title: str) -> bool:
        """Start crawling from a given Wikipedia page"""
        try:
            logger.info(f"Starting crawl from {start_title}")
            page_id = await self._process_page(start_title)
            if page_id:
                logger.info(f"Successfully crawled {start_title} with ID {page_id}")
                return True
            else:
                logger.error(f"Failed to crawl {start_title} - no page ID returned")
                return False
        except Exception as e:
            import traceback
            logger.error(f"Error during crawl of {start_title}: {str(e)}\nTraceback: {traceback.format_exc()}")
            return False
            
    async def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar concepts using the vector store"""
        try:
            query_embedding = self.embedding_model.encode(query)
            return self.vector_store.search_similar(query_embedding, limit)
        except Exception as e:
            logger.error(f"Error searching with query '{query}': {str(e)}")
            return [] 