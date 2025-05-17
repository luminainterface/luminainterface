from typing import List, Dict, Optional, Set, Any
import asyncio
import logging
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime
import uuid
import sys
import os
import json

from .wiki_client import WikiClient
from .vector_store import VectorStore
from .graph_client import GraphClient
from .concept_client import ConceptClient
from .redis_client import RedisClient
from .file_processor import FileProcessor
from langchain_community.embeddings import OllamaEmbeddings
from ..models.file_item import FileProcessingConfig

logger = logging.getLogger(__name__)

class Crawler:
    """Main crawler class that manages file processing and worker coordination."""
    
    def __init__(
        self,
        redis_url: str = "redis://:02211998@redis:6379",
        qdrant_url: str = "http://qdrant:6333",
        graph_api_url: str = "http://graph-api:8526",
        concept_dict_url: str = "http://concept-dictionary:8526",
        ollama_url: str = "http://ollama:11434",
        ollama_model: str = "nomic-embed-text",
        training_data_path: str = "/app/training_data",
        max_depth: int = 2,
        max_links_per_page: int = 10,
        cache_ttl: int = 86400,  # 24 hours
        config: Optional[FileProcessingConfig] = None
    ):
        # Initialize clients
        self.redis = RedisClient(redis_url=redis_url)
        self.vector_store = VectorStore(qdrant_url)
        self.graph_client = GraphClient(graph_api_url)
        self.concept_client = ConceptClient(concept_dict_url)
        self.wiki_client = WikiClient()
        
        # Initialize embedding model
        self.embedding_model = OllamaEmbeddings(
            base_url=ollama_url,
            model=ollama_model
        )
        
        # Initialize file processor
        self.file_processor = FileProcessor(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            redis_client=self.redis,
            config=config
        )
        
        # Set paths and configuration
        self.training_data_path = training_data_path
        self.max_depth = max_depth
        self.max_links_per_page = max_links_per_page
        self.cache_ttl = cache_ttl
        self.concept_dict_url = concept_dict_url
        
        # Initialize state
        self.processed_files: Set[str] = set()
        self.failed_files: Set[str] = set()
        self._worker_running = False
        self._initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Try to import and initialize adapter, but don't fail if not available
        try:
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../graph-concept-adapter')))
            from adapter import GraphConceptAdapter
            self.adapter = GraphConceptAdapter()
            self.logger.info("Graph concept adapter initialized")
        except ImportError:
            self.adapter = None
            self.logger.info("Graph concept adapter not available, continuing without it")
        except Exception as e:
            self.adapter = None
            self.logger.warning(f"Failed to initialize graph concept adapter: {e}, continuing without it")
            
        # Initialize vector store collection
        try:
            # OllamaEmbeddings does not expose embedding dimension; set manually for the model
            embedding_dimension = 768  # nomic-embed-text uses 768-dim vectors
            self.vector_store.init_collection(vector_size=embedding_dimension)
            self.logger.info("Vector store collection initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store collection: {e}")
            raise

    async def initialize(self):
        """Initialize the crawler and its dependencies."""
        if not self._initialized:
            try:
                # Connect to Redis
                await self.redis.connect()
                # Initialize Redis streams
                await self._initialize_streams()
                self._initialized = True
                self.logger.info("Crawler initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize crawler: {str(e)}")
                raise

    async def _initialize_streams(self):
        """Initialize required Redis streams."""
        try:
            # Create consumer groups for each stream if they don't exist
            streams = {
                'crawl_queue': 'workers',
                'crawl_results': 'workers',
                'crawl_dead_letter': 'workers',
                'ingest_queue': 'workers'
            }
            
            for stream, group in streams.items():
                try:
                    # Create stream if it doesn't exist
                    await self.redis.xadd(stream, {'init': 'true'})
                    # Create consumer group
                    await self.redis.xgroup_create(stream, group, mkstream=True)
                    self.logger.info(f"Initialized stream {stream} with group {group}")
                except Exception as e:
                    if "BUSYGROUP" not in str(e):  # Ignore if group already exists
                        self.logger.error(f"Error initializing stream {stream}: {e}")
                        raise
        except Exception as e:
            self.logger.error(f"Error initializing streams: {e}")
            raise

    async def start(self):
        """Start the crawler service and worker."""
        if not self._initialized:
            await self.initialize()
        
        if not self._worker_running:
            self._worker_running = True
            asyncio.create_task(self.crawl_worker())
            self.logger.info("Crawler service started")
            
            # Start background task for incremental processing
            asyncio.create_task(self.process_training_data_incremental())
            self.logger.info("Training data processing started")

    async def stop(self):
        """Stop the crawler service and worker."""
        self._worker_running = False
        self.logger.info("Crawler service stopped")

    async def crawl_worker(self):
        """Background worker that processes items from the queue."""
        if not self._worker_running:
            self.logger.warning("Crawl worker is already running")
            return

        self.logger.info("Starting crawl worker...")
        
        try:
            while self._worker_running:
                try:
                    # Get items from Redis stream with a timeout
                    items = await asyncio.wait_for(
                        self.redis.xreadgroup(
                            group='workers',
                            consumer='worker-1',
                            streams={'crawl_queue': '>'},
                            count=10,
                            block=5000  # 5 second timeout
                        ),
                        timeout=5.0
                    )
                    
                    if not items:
                        await asyncio.sleep(1)
                        continue

                    for stream, messages in items:
                        for msg_id, data in messages:
                            try:
                                # Process the item
                                success = await self.process_queue_item(msg_id, data)
                                
                                if success:
                                    # Acknowledge the message
                                    await self.redis.xack('crawl_queue', 'workers', msg_id)
                                else:
                                    # Move to dead letter queue
                                    await self.redis.xadd(
                                        'crawl_dead_letter',
                                        {
                                            **data,
                                            'error': 'Processing failed',
                                            'timestamp': datetime.utcnow().isoformat()
                                        }
                                    )
                                    await self.redis.xack('crawl_queue', 'workers', msg_id)
                                    
                            except Exception as e:
                                self.logger.error(f"Error processing queue item: {str(e)}", exc_info=True)
                                await asyncio.sleep(1)

                except asyncio.TimeoutError:
                    # Normal timeout, just continue
                    continue
                except Exception as e:
                    self.logger.error(f"Error in crawl worker loop: {str(e)}", exc_info=True)
                    await asyncio.sleep(1)

        finally:
            self._worker_running = False
            self.logger.info("Crawl worker stopped")

    async def process_queue_item(self, msg_id: str, data: Dict[str, Any]) -> bool:
        """Process a single queue item."""
        try:
            file_path = data.get('file_path')
            if not file_path:
                self.logger.error(f"Missing file_path in queue item: {data}")
                return False

            # Process the file
            result = await self.file_processor.process_file(file_path)
            
            if result.status == "success":
                # Add to processed files
                self.processed_files.add(file_path)
                
                # Publish result
                await self.redis.xadd(
                    'crawl_results',
                    {
                        'file_path': file_path,
                        'vectors_count': len(result.vectors),
                        'status': 'success',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
                return True
            else:
                # Add to failed files
                self.failed_files.add(file_path)
                
                # Publish error
                await self.redis.xadd(
                    'crawl_results',
                    {
                        'file_path': file_path,
                        'error': result.error,
                        'status': 'error',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing queue item: {str(e)}", exc_info=True)
            return False

    async def process_training_data_incremental(self, delay: float = 1.0):
        """Incrementally process files in the training data directory."""
        self.logger.info(f"Starting incremental training data processing from {self.training_data_path}")
        
        while self._worker_running:
            try:
                # Find unprocessed files
                for root, _, files in os.walk(self.training_data_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file_path not in self.processed_files and file_path not in self.failed_files:
                            try:
                                # Add to crawl queue
                                await self.redis.xadd(
                                    'crawl_queue',
                                    {
                                        'file_path': file_path,
                                        'timestamp': datetime.utcnow().isoformat()
                                    }
                                )
                                self.logger.info(f"Added {file_path} to crawl queue")
                            except Exception as e:
                                self.logger.error(f"Error queueing file {file_path}: {str(e)}")
                                self.failed_files.add(file_path)
                
                # Log progress
                self.logger.info(
                    f"Training data processing status - "
                    f"Processed: {len(self.processed_files)}, "
                    f"Failed: {len(self.failed_files)}"
                )
                
                # Wait before next check
                await asyncio.sleep(delay)
                
            except Exception as e:
                self.logger.error(f"Error in incremental processing: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    async def get_status(self) -> Dict[str, Any]:
        """Get crawler status."""
        return {
            "initialized": self._initialized,
            "worker_running": self._worker_running,
            "processed_files": len(self.processed_files),
            "failed_files": len(self.failed_files),
            "queue_length": await self.redis.xlen('crawl_queue'),
            "dead_letter_length": await self.redis.xlen('crawl_dead_letter')
        }

    # Priority constants for different data sources
    GIT_PRIORITY = 1.0
    PDF_PRIORITY = 0.8
    GRAPH_PRIORITY = 0.4

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
        cached_data = await self.redis.get_cache(f"page:{page_id}")
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
            await self.redis.publish_crawl_result(title, {
                "url": page.fullurl,
                "title": title,
                "page_id": page_id,
                "summary": summary,
                "metadata": metadata
            })
        except Exception as e:
            logger.error(f"Failed to publish crawl result for {title}: {e}")
            
        # Cache the processed state
        await self.redis.set_cache(
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

    async def process_training_data(self):
        """Process training data from all sources in priority order."""
        try:
            # Process git content first (highest priority)
            if os.path.exists(self.git_training_path):
                logger.info("Processing git training data...")
                await self._process_git_content()
            
            # Process PDFs next (medium priority)
            if os.path.exists(self.pdf_training_path):
                logger.info("Processing PDF training data...")
                await self._process_pdf_content()
            
            # Process graph (1).json immediately after PDFs
            if os.path.exists(self.graph_training_path):
                logger.info("Processing graph (1).json training data...")
                await self._process_graph_content()
            else:
                logger.warning(f"Graph training file not found at {self.graph_training_path}")
                
        except Exception as e:
            logger.error(f"Error processing training data: {e}")
            raise

    async def _process_git_content(self):
        """Process git content with highest priority."""
        try:
            for root, _, files in os.walk(self.git_training_path):
                for file in files:
                    if file.endswith(('.md', '.txt', '.py')):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Extract concepts and add to crawl queue with git priority
                            concepts = self._extract_concepts(content)
                            for concept in concepts:
                                await self.redis.add_to_crawl_queue(
                                    concept=concept,
                                    weight=self.GIT_PRIORITY,
                                    source="git"
                                )
        except Exception as e:
            logger.error(f"Error processing git content: {e}")
            raise

    async def _process_pdf_content(self):
        """Process PDF content with medium priority."""
        try:
            for file in os.listdir(self.pdf_training_path):
                if file.endswith('.pdf'):
                    file_path = os.path.join(self.pdf_training_path, file)
                    logger.info(f"Processing PDF file: {file}")
                    
                    try:
                        # Process the PDF using async method
                        pdf_data = await self.pdf_processor.process_pdf(file_path)
                        
                        # Store the extracted text in Redis for caching
                        cache_key = f"pdf:{os.path.basename(file_path)}"
                        await self.redis.set_cache(
                            cache_key,
                            pdf_data,
                            self.cache_ttl
                        )
                        
                        # Add concepts to crawl queue
                        for concept in pdf_data.get('concepts', []):
                            await self.redis.add_to_crawl_queue(
                                concept=concept,
                                weight=self.PDF_PRIORITY,
                                source="pdf",
                                metadata={
                                    "pdf_title": pdf_data['metadata'].get('title'),
                                    "pdf_author": pdf_data['metadata'].get('author'),
                                    "pdf_path": pdf_data['metadata'].get('file_path')
                                }
                            )
                        
                        # Store the PDF text in vector store
                        if pdf_data.get('text'):
                            # Run CPU-bound embedding generation in thread pool
                            loop = asyncio.get_event_loop()
                            text_embedding = await loop.run_in_executor(
                                None,
                                self.embedding_model.encode,
                                pdf_data['text']
                            )
                            
                            await self.vector_store.upsert_vectors(
                                vectors=[text_embedding],
                                metadata=[{
                                    "title": pdf_data['metadata'].get('title'),
                                    "type": "pdf",
                                    "author": pdf_data['metadata'].get('author'),
                                    "file_path": pdf_data['metadata'].get('file_path'),
                                    "processed_at": pdf_data.get('processed_at')
                                }],
                                ids=[f"pdf_{os.path.basename(file_path)}"]
                            )
                        
                        logger.info(f"Successfully processed PDF: {file}")
                        
                    except Exception as e:
                        logger.error(f"Error processing PDF file {file}: {str(e)}")
                        continue
                    
        except Exception as e:
            logger.error(f"Error processing PDF content: {e}")
            raise

    async def _process_graph_content(self):
        """Process graph content with lowest priority."""
        try:
            if os.path.exists(self.graph_training_path):
                logger.info(f"Loading graph data from {self.graph_training_path}")
                with open(self.graph_training_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'nodes' in data:
                        logger.info(f"Found {len(data['nodes'])} nodes in graph (1).json")
                        for node in data['nodes']:
                            node_id = node.get('id')
                            if node_id:
                                await self.redis.add_to_crawl_queue(
                                    concept=node_id,
                                    weight=self.GRAPH_PRIORITY,
                                    source="graph (1)",
                                    metadata={
                                        "graph_file": "graph (1).json",
                                        "node_type": node.get('type', 'unknown'),
                                        "processed_at": datetime.utcnow().isoformat()
                                    }
                                )
                        logger.info("Successfully processed all nodes from graph (1).json")
                    else:
                        logger.warning("No nodes found in graph (1).json")
        except Exception as e:
            logger.error(f"Error processing graph (1).json: {e}")
            raise

    def _extract_concepts(self, content: str) -> List[str]:
        """Extract concepts from text content."""
        # TODO: Implement more sophisticated concept extraction
        # For now, just split on whitespace and filter out short words
        words = content.split()
        return [word for word in words if len(word) > 3] 