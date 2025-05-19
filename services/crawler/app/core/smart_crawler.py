from typing import List, Dict, Optional, Set, Tuple
import asyncio
import logging
from datetime import datetime
import numpy as np
import time
from .crawler import Crawler
from .redis_client import RedisClient
from .vector_store import VectorStore
from .concept_client import ConceptClient
from .metrics import MetricsCollector, CRAWLER_REQUESTS, CRAWLER_PAGES
from lumina_core.common.retry import with_retry, Skip, MaxRetriesExceeded
from lumina_core.common.bus import BusClient

logger = logging.getLogger(__name__)

class SmartCrawler(Crawler):
    def __init__(
        self,
        redis_url: str,
        qdrant_url: str,
        graph_api_url: str,
        concept_dict_url: str,
        embedding_model: str = "nomic-embed-text",
        max_depth: int = 3,
        max_links_per_page: int = 15,
        cache_ttl: int = 86400,  # 24 hours
        min_relevance_score: float = 0.6,
        max_concurrent_crawls: int = 5,
        dead_letter_stream: str = "crawler.dlq"
    ):
        super().__init__(
            redis_url=redis_url,
            qdrant_url=qdrant_url,
            graph_api_url=graph_api_url,
            concept_dict_url=concept_dict_url,
            embedding_model=embedding_model,
            max_depth=max_depth,
            max_links_per_page=max_links_per_page,
            cache_ttl=cache_ttl
        )
        self.min_relevance_score = min_relevance_score
        self.max_concurrent_crawls = max_concurrent_crawls
        self.active_crawls = set()
        self.crawl_priorities = {}  # Store priority scores for URLs
        self.metrics = MetricsCollector()
        self.dead_letter_stream = dead_letter_stream
        self.bus = BusClient(redis_url=redis_url)
        
    async def connect(self):
        """Connect to Redis and create consumer groups"""
        await self.bus.connect()
        try:
            # Create consumer group for ingest.raw_html
            await self.bus.create_group("ingest.raw_html", "crawler")
            # Create consumer group for ingest.raw_pdf
            await self.bus.create_group("ingest.raw_pdf", "crawler")
            # Create consumer group for concept.new
            await self.bus.create_group("concept.new", "crawler")
        except Exception as e:
            logger.info(f"Groups may exist: {e}")
            
    async def _update_queue_metrics(self):
        """Update queue-related metrics from Redis"""
        try:
            # Get queue depth
            queue_depth = await self.redis_client.get_queue_length("ingest.queue")
            
            # Get consumer lag
            consumer_lag = await self.redis_client.get_consumer_lag("ingest.queue", "crawler")
            
            # Update metrics
            self.metrics.update_queue_metrics(queue_depth, consumer_lag)
        except Exception as e:
            logger.error(f"Error updating queue metrics: {e}")
            self.metrics.record_error("queue_metrics_update")
            
    async def calculate_priority(self, url: str, title: str, content: str, depth: int) -> float:
        """Calculate priority score for a URL based on multiple factors"""
        try:
            start_time = time.time()
            
            # Base priority decreases with depth
            base_priority = 1.0 / (depth + 1)
            
            # Check if URL is already in concept dictionary
            concept_exists = await self.concept_client.get_concept(title) is not None
            concept_factor = 0.5 if concept_exists else 1.0  # Higher priority for new concepts
            
            # Calculate content relevance using embedding similarity
            if content:
                content_embedding = self.embedding_model.encode(content[:1000])  # Use first 1000 chars
                # Get average similarity to existing concepts
                similar_concepts = self.vector_store.search_similar(content_embedding, limit=5)
                if similar_concepts:
                    relevance_score = np.mean([c["score"] for c in similar_concepts])
                else:
                    relevance_score = 0.0
            else:
                relevance_score = 0.0
                
            # Combine factors
            priority = base_priority * concept_factor * (1 + relevance_score)
            
            # Store priority for future reference
            self.crawl_priorities[url] = priority
            
            # Record processing time
            self.metrics.record_process_time(time.time() - start_time)
            
            return priority
            
        except Exception as e:
            logger.error(f"Error calculating priority for {url}: {e}")
            self.metrics.record_error("priority_calculation")
            return 0.0
            
    @with_retry("ingest.raw_html", dead_letter_stream="crawler.dlq")
    async def _process_page_smart(self, title: str, depth: int = 0, visited: Optional[Set[str]] = None) -> Optional[Tuple[str, float]]:
        """Enhanced page processing with priority calculation and retry logic"""
        if visited is None:
            visited = set()
            
        if depth > self.max_depth or title in visited:
            raise Skip("max_depth_or_visited")
            
        visited.add(title)
        page_id = self._generate_id(title)
        start_time = time.time()
        
        try:
            # Check cache first
            cached_data = await self.redis_client.get_cache(f"page:{page_id}")
            if cached_data:
                logger.info(f"Cache hit for page {title}")
                self.metrics.record_processing("url", self.crawl_priorities.get(title, 0.0), True)
                return page_id, self.crawl_priorities.get(title, 0.0)
                
            # Fetch and process page
            page = self.wiki_client.get_page(title)
            if not page:
                raise Skip("page_not_found")
                
            # Get page content
            summary = self.wiki_client.get_summary(page)
            full_text = self.wiki_client.get_full_text(page)
            
            # Calculate priority
            priority = await self.calculate_priority(
                page.fullurl,
                title,
                full_text,
                depth
            )
            
            # Only process if priority meets minimum threshold
            if priority < self.min_relevance_score:
                raise Skip(f"low_priority:{priority}")
                
            # Process page content
            summary_embedding = self.embedding_model.encode(summary)
            
            # Store in vector database with priority metadata
            metadata = {
                "title": title,
                "summary": summary,
                "url": page.fullurl,
                "priority": priority,
                "depth": depth,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            self.vector_store.upsert_vectors(
                vectors=[summary_embedding],
                metadata=[metadata],
                ids=[page_id]
            )
            
            # Store in concept dictionary
            concept_data = {
                "title": title,
                "summary": summary,
                "content": full_text,
                "url": page.fullurl,
                "priority": priority,
                "depth": depth,
                "timestamp": datetime.utcnow().isoformat(),
                "embedding": summary_embedding.tolist()
            }
            
            try:
                await self.redis_client.add_concept(title, concept_data)
            except Exception as e:
                logger.error(f"Failed to store concept {title}: {e}")
                self.metrics.record_error("concept_store")
                raise
                
            # Publish to ingest.raw_html stream for content extraction
            try:
                await self.bus.publish("ingest.raw_html", {
                    "url": page.fullurl,
                    "title": title,
                    "html": page.html(),
                    "page_id": page_id,
                    "priority": priority,
                    "metadata": metadata,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to publish to ingest.raw_html: {e}")
                self.metrics.record_error("stream_publish")
                raise
                
            # Cache the processed state
            await self.redis_client.set_cache(
                f"page:{page_id}",
                {
                    "processed": True,
                    "priority": priority,
                    "timestamp": datetime.utcnow().isoformat()
                },
                self.cache_ttl
            )
            
            # Record metrics
            self.metrics.record_processing("url", priority, False)
            self.metrics.record_fetch_time(time.time() - start_time)
            CRAWLER_PAGES.labels(status="success").inc()
            
            return page_id, priority
            
        except Skip as s:
            logger.info(f"Skipping {title}: {str(s)}")
            self.metrics.record_skip(str(s))
            raise
        except Exception as e:
            logger.error(f"Error processing page {title}: {e}")
            self.metrics.record_error("page_processing")
            CRAWLER_PAGES.labels(status="error").inc()
            raise
            
    @with_retry("concept.new", dead_letter_stream="crawler.dlq")
    async def process_concept(self, msg: Dict[str, Any]):
        """Process a new concept from the concept.new stream"""
        try:
            concept_id = msg["cid"]
            if concept_id in self.processed_concepts:
                raise Skip("already_processed")
                
            # Update crawl priorities based on concept
            if "title" in msg and "priority" in msg:
                self.crawl_priorities[msg["title"]] = msg["priority"]
                
            # Record metrics
            self.metrics.record_concept_processing(msg.get("priority", 0.0))
            
        except Skip as s:
            logger.info(f"Skipping concept {concept_id}: {str(s)}")
            self.metrics.record_skip(str(s))
            raise
        except Exception as e:
            logger.error(f"Error processing concept {concept_id}: {e}")
            self.metrics.record_error("concept_processing")
            raise
            
    async def start(self):
        """Start consuming from input streams"""
        await self.connect()
        
        while True:
            try:
                # Process concept.new stream
                await self.bus.consume(
                    stream="concept.new",
                    group="crawler",
                    consumer="worker",
                    handler=self.process_concept,
                    block_ms=1000,
                    count=10
                )
                
                # Update queue metrics
                await self._update_queue_metrics()
                
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)

    async def crawl_smart(self, start_title: str, max_pages: Optional[int] = None) -> bool:
        """Start smart crawling from a given Wikipedia page with adaptive depth"""
        try:
            logger.info(f"Starting smart crawl from {start_title}")
            CRAWLER_REQUESTS.labels(type="smart").inc()
            
            visited = set()
            to_visit = [(start_title, 0)]  # (title, depth)
            processed = 0
            
            while to_visit and (max_pages is None or processed < max_pages):
                # Update queue metrics
                await self._update_queue_metrics()
                
                # Sort by priority
                to_visit.sort(key=lambda x: self.crawl_priorities.get(x[0], 0.0), reverse=True)
                
                # Process next batch of pages
                current_batch = []
                while to_visit and len(current_batch) < self.max_concurrent_crawls:
                    title, depth = to_visit.pop(0)
                    if title not in visited:
                        current_batch.append((title, depth))
                
                if not current_batch:
                    break
                    
                # Update active crawls count
                self.metrics.update_active_crawls(len(current_batch))
                
                # Process batch concurrently
                tasks = [self._process_page_smart(title, depth, visited) for title, depth in current_batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for (title, depth), result in zip(current_batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing {title}: {result}")
                        self.metrics.record_error("batch_processing")
                        continue
                        
                    if result is None:
                        continue
                        
                    page_id, priority = result
                    processed += 1
                    
                    # Get links for next level
                    if depth < self.max_depth:
                        page = self.wiki_client.get_page(title)
                        if page:
                            links = self.wiki_client.get_links(page)
                            # Add new links to visit with their priorities
                            for link in links[:self.max_links_per_page]:
                                if link not in visited:
                                    to_visit.append((link, depth + 1))
                                    
            # Update final metrics
            self.metrics.update_active_crawls(0)
            if self.crawl_priorities:
                avg_priority = np.mean(list(self.crawl_priorities.values()))
                self.metrics.update_priority_avg(avg_priority)
                
            logger.info(f"Completed smart crawl from {start_title}. Processed {processed} pages.")
            return True
            
        except Exception as e:
            logger.error(f"Error during smart crawl of {start_title}: {e}")
            self.metrics.record_error("crawl_execution")
            return False
            
    async def get_crawl_stats(self) -> Dict:
        """Get statistics about the crawl process"""
        try:
            # Update queue metrics before getting stats
            await self._update_queue_metrics()
            
            total_pages = len(self.crawl_priorities)
            avg_priority = np.mean(list(self.crawl_priorities.values())) if total_pages > 0 else 0.0
            max_priority = max(self.crawl_priorities.values()) if total_pages > 0 else 0.0
            
            return {
                "total_pages": total_pages,
                "average_priority": avg_priority,
                "max_priority": max_priority,
                "active_crawls": len(self.active_crawls),
                "cache_hits": await self.redis_client.get_cache_stats()
            }
        except Exception as e:
            logger.error(f"Error getting crawl stats: {e}")
            self.metrics.record_error("stats_collection")
            return {} 