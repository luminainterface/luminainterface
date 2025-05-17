import os
import asyncio
import logging
import uuid
import json
import mimetypes
from datetime import datetime
from typing import Dict, Optional, List, Set, Tuple, Union, Any
from langchain_community.embeddings import OllamaEmbeddings
from lumina_core.common.bus import BusClient
from .redis_client import RedisClient
from lumina_core.common.qdrant import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import hashlib
import time

class TrainingCrawler:
    """Processes training data in stages from organized directories."""
    
    # Class-level variables for singleton pattern
    _instance_lock = asyncio.Lock()
    _instance = None
    
    # Define crawl stages and their priorities
    CRAWL_STAGES = [
        ('dictionary', 1),  # Highest priority
        ('json', 2),
        ('jsonl', 3),
        ('wikipedia', 4),
        ('txt', 5),        # Added txt directory
        ('images', 6)      # Added images directory
    ]
    
    # Define file type handlers for each stage
    STAGE_HANDLERS = {
        'dictionary': {
            '.json': '_process_dictionary_json',
            '.jsonl': '_process_dictionary_jsonl'
        },
        'json': {
            '.json': '_process_general_json',
            '.jsonl': '_process_general_jsonl'
        },
        'jsonl': {
            '.jsonl': '_process_jsonl_file'
        },
        'wikipedia': {
            '.json': '_process_wikipedia_graph',
            '.jsonl': '_process_wikipedia_jsonl'
        },
        'txt': {
            '.txt': '_process_text_file'  # Added text file handler
        },
        'images': {
            '.jpg': '_process_image_file',  # Added image handlers
            '.jpeg': '_process_image_file',
            '.png': '_process_image_file',
            '.gif': '_process_image_file'
        }
    }
    
    def __init__(
        self,
        redis_url: str = "redis://:02211998@redis:6379",
        qdrant_url: str = "http://qdrant:6333",
        training_data_path: str = "/app/training_data",
        ollama_url: str = "http://ollama:11434",
        ollama_model: str = "nomic-embed-text",
        max_workers: int = 1,  # Limit to single worker
        processing_delay: float = 5.0  # Increased delay between processing
    ):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Remove the singleton initialization from here since it's now a class variable
        self.bus = BusClient(redis_url=redis_url)
        self.redis = RedisClient(redis_url=redis_url)
        self.qdrant = QdrantClient(url=qdrant_url)
        
        # Initialize Ollama model with retry
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                self.embedding_model = OllamaEmbeddings(
                    base_url=ollama_url,
                    model=ollama_model
                )
                # Test the model
                test_embedding = self.embedding_model.embed_query("test")
                if test_embedding:
                    self.logger.info(f"Successfully initialized Ollama model: {ollama_model}")
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Failed to initialize Ollama model (attempt {attempt + 1}/{max_retries}): {e}")
                    self.logger.info(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to initialize Ollama model after {max_retries} attempts: {e}")
                    raise
        self.training_data_path = training_data_path
        self.processed_files = set()
        self.failed_files = set()
        self.current_stage_index = 0
        self.current_directory = None
        self.directory_files = []
        self.directory_index = 0
        self.max_workers = max_workers
        self.processing_delay = processing_delay
        self.is_processing = False  # Lock to prevent concurrent processing
        
        # Track progress through stages
        self.stage_progress = {stage: False for stage, _ in self.CRAWL_STAGES}
        
        # Initialize mimetypes with explicit mappings
        mimetypes.init()
        mimetypes.add_type('application/json', '.json')
        mimetypes.add_type('application/jsonl', '.jsonl')
        mimetypes.add_type('application/x-jsonlines', '.jsonl')
        mimetypes.add_type('text/plain', '.txt')
        mimetypes.add_type('image/jpeg', '.jpg')
        mimetypes.add_type('image/jpeg', '.jpeg')
        mimetypes.add_type('image/png', '.png')
        mimetypes.add_type('image/gif', '.gif')
        
        # Register file type handlers
        self._register_file_handlers()
    
    def _register_file_handlers(self):
        """Register file type handlers for each stage."""
        self.file_handlers = {
            'dictionary': {
                '.json': self._process_dictionary_json,
                '.jsonl': self._process_dictionary_jsonl
            },
            'json': {
                '.json': self._process_general_json,
                '.jsonl': self._process_general_jsonl
            },
            'jsonl': {
                '.jsonl': self._process_jsonl_file
            },
            'wikipedia': {
                '.json': self._process_wikipedia_graph,
                '.jsonl': self._process_wikipedia_jsonl
            },
            'txt': {
                '.txt': self._process_text_file
            },
            'images': {
                '.jpg': self._process_image_file,
                '.jpeg': self._process_image_file,
                '.png': self._process_image_file,
                '.gif': self._process_image_file
            }
        }
    
    @classmethod
    async def get_instance(cls, **kwargs):
        """Get or create a singleton instance of TrainingCrawler."""
        if cls._instance is None:
            async with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
                    await cls._instance.initialize()
        return cls._instance
        
    async def initialize(self):
        """Initialize connections and create Qdrant collections."""
        try:
            await self.bus.connect()
            await self.redis.connect()
            await self.qdrant.connect()
            
            # Create collections for different types of data
            collections = {
                "embeddings": {
                    "vectors_config": {
                        "size": 768,
                        "distance": "Cosine"
                    }
                },
                "graph_nodes": {
                    "vectors_config": {
                        "size": 768,
                        "distance": "Cosine"
                    }
                },
                "concepts": {
                    "vectors_config": {
                        "size": 768,
                        "distance": "Cosine"
                    }
                }
            }
            
            for name, config in collections.items():
                try:
                    # Check if collection exists first
                    try:
                        collections = await self.qdrant.get_collections()
                        collection_names = [c.name for c in collections.collections]
                        if name in collection_names:
                            self.logger.info(f"Collection {name} already exists")
                            continue
                    except Exception as e:
                        self.logger.warning(f"Error checking collections: {e}")
                    
                    # Create collection if it doesn't exist
                    await self.qdrant.create_collection(
                        collection_name=name,
                        **config
                    )
                    self.logger.info(f"Created collection: {name}")
                    
                    # Create payload indexes for better search performance
                    try:
                        await self.qdrant.create_payload_index(
                            collection_name=name,
                            field_name="embedding_source",
                            field_schema="keyword"
                        )
                        await self.qdrant.create_payload_index(
                            collection_name=name,
                            field_name="processed_at",
                            field_schema="datetime"
                        )
                        if name == "concepts":
                            await self.qdrant.create_payload_index(
                                collection_name=name,
                                field_name="term",
                                field_schema="keyword"
                            )
                        self.logger.info(f"Created indexes for collection: {name}")
                    except Exception as e:
                        self.logger.warning(f"Error creating indexes for {name}: {e}")
                        
                except Exception as e:
                    self.logger.error(f"Error setting up collection {name}: {e}")
                    raise
            
            self.logger.info("TrainingCrawler initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize TrainingCrawler: {e}")
            raise

    async def _get_file_type(self, file_path: str) -> Optional[str]:
        """Get the file type based on extension and content."""
        try:
            _, ext = os.path.splitext(file_path.lower())
            
            # Direct extension check first
            if ext == '.json':
                # Verify it's actually JSON
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    return '.json'
                except:
                    pass
            elif ext == '.jsonl':
                # Verify it's actually JSONL
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            json.loads(first_line)
                            return '.jsonl'
                except:
                    pass
            
            # If we get here, try to determine type from content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1024).strip()  # Read first 1KB
                    if content.startswith('{') or content.startswith('['):
                        try:
                            json.loads(content)
                            return '.json'
                        except:
                            pass
                    elif content and all(line.strip().startswith('{') or line.strip().startswith('[') 
                                       for line in content.split('\n')[:3] if line.strip()):
                        try:
                            for line in content.split('\n')[:3]:
                                if line.strip():
                                    json.loads(line.strip())
                            return '.jsonl'
                        except:
                            pass
            except:
                pass
            
            self.logger.warning(f"Could not determine file type for {file_path}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining file type for {file_path}: {e}")
            return None

    async def _process_dictionary_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Process dictionary JSON files with structured format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            concepts = []
            if isinstance(data, dict):
                for term, definition in data.items():
                    concept = {
                        "term": term,
                        "source": file_path,
                        "type": "dictionary_entry"
                    }
                    
                    if isinstance(definition, str):
                        concept["definition"] = definition
                    elif isinstance(definition, dict):
                        concept.update({
                            "definition": definition.get("definition", ""),
                            "examples": definition.get("examples", []),
                            "related": definition.get("related", []),
                            "categories": definition.get("categories", []),
                            "metadata": definition.get("metadata", {})
                        })
                    
                    concepts.append(concept)
            
            return concepts
        except Exception as e:
            self.logger.error(f"Error processing dictionary JSON file {file_path}: {e}")
            return []

    async def _process_dictionary_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Process dictionary JSONL files with one concept per line."""
        try:
            concepts = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if isinstance(entry, dict) and "term" in entry:
                            entry["source"] = file_path
                            entry["type"] = "dictionary_entry"
                            concepts.append(entry)
                    except json.JSONDecodeError:
                        continue
            return concepts
        except Exception as e:
            self.logger.error(f"Error processing dictionary JSONL file {file_path}: {e}")
            return []

    async def _process_general_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Process general JSON files with flexible structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entries = []
            if isinstance(data, dict):
                # Handle dictionary structure
                for key, value in data.items():
                    entry = {
                        "key": key,
                        "value": value,
                        "source": file_path,
                        "type": "json_entry"
                    }
                    entries.append(entry)
            elif isinstance(data, list):
                # Handle list structure
                for item in data:
                    if isinstance(item, dict):
                        item["source"] = file_path
                        item["type"] = "json_entry"
                        entries.append(item)
            
            return entries
        except Exception as e:
            self.logger.error(f"Error processing general JSON file {file_path}: {e}")
            return []

    async def _process_general_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Process general JSONL files with one entry per line."""
        try:
            entries = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if isinstance(entry, dict):
                            entry["source"] = file_path
                            entry["type"] = "jsonl_entry"
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue
            return entries
        except Exception as e:
            self.logger.error(f"Error processing general JSONL file {file_path}: {e}")
            return []

    async def _process_jsonl_file(self, file_path: str) -> List[Dict]:
        """Process JSONL files line by line."""
        try:
            entries = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if isinstance(entry, dict):
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue
            return entries
        except Exception as e:
            self.logger.error(f"Error processing JSONL file {file_path}: {e}")
            return []

    async def _process_wikipedia_graph(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process Wikipedia graph data into nodes and edges."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            nodes = []
            edges = []
            
            if isinstance(data, dict):
                # Process nodes
                for node_id, node_data in data.get("nodes", {}).items():
                    if isinstance(node_data, dict):
                        nodes.append({
                            "id": node_id,
                            "title": node_data.get("title", ""),
                            "content": node_data.get("content", ""),
                            "weight": node_data.get("weight", 1.0)
                        })
                
                # Process edges
                for edge in data.get("edges", []):
                    if isinstance(edge, dict):
                        edges.append({
                            "source": edge.get("source"),
                            "target": edge.get("target"),
                            "weight": edge.get("weight", 1.0),
                            "type": edge.get("type", "related")
                        })
            
            return nodes, edges
        except Exception as e:
            self.logger.error(f"Error processing Wikipedia graph {file_path}: {e}")
            return [], []

    async def _process_wikipedia_jsonl(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process Wikipedia JSONL files with graph data."""
        try:
            nodes = []
            edges = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if isinstance(entry, dict):
                            if "type" in entry:
                                if entry["type"] == "node":
                                    nodes.append({
                                        "id": entry.get("id"),
                                        "title": entry.get("title", ""),
                                        "content": entry.get("content", ""),
                                        "weight": entry.get("weight", 1.0)
                                    })
                                elif entry["type"] == "edge":
                                    edges.append({
                                        "source": entry.get("source"),
                                        "target": entry.get("target"),
                                        "weight": entry.get("weight", 1.0),
                                        "type": entry.get("edge_type", "related")
                                    })
                    except json.JSONDecodeError:
                        continue
            return nodes, edges
        except Exception as e:
            self.logger.error(f"Error processing Wikipedia JSONL file {file_path}: {e}")
            return [], []

    def _generate_cache_key(self, text: str, stage: str) -> str:
        """Generate a cache key for text content."""
        # Create a hash of the text content
        content_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{stage}:{content_hash}"
        
    async def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        try:
            cached_data = await self.redis.get_cache(cache_key)
            if cached_data and 'embedding' in cached_data:
                self.logger.info(f"Cache hit for key: {cache_key}")
                return cached_data['embedding']
            self.logger.debug(f"Cache miss for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting cached embedding: {e}")
            return None
            
    async def _cache_embedding(self, cache_key: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Cache embedding with metadata."""
        try:
            cache_data = {
                'embedding': embedding,
                'metadata': metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
            await self.redis.set_cache(cache_key, cache_data, 86400)  # 24 hour TTL
            self.logger.info(f"Cached embedding for key: {cache_key}")
        except Exception as e:
            self.logger.error(f"Error caching embedding: {e}")

    async def process_file(self, file_path: str) -> bool:
        """Process a file based on its directory and type."""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return False

            # Determine the stage based on directory
            rel_path = os.path.relpath(file_path, self.training_data_path)
            stage = rel_path.split(os.sep)[0]
            
            if stage not in [s[0] for s in self.CRAWL_STAGES]:
                self.logger.warning(f"Unknown stage directory: {stage}")
                return False

            # Get file type
            file_type = await self._get_file_type(file_path)
            if not file_type:
                self.logger.warning(f"Could not determine file type for {file_path}")
                return False

            # Get appropriate handler
            handlers = self.file_handlers.get(stage, {})
            handler = handlers.get(file_type)
            if not handler:
                self.logger.warning(f"No handler for {file_type} in {stage} stage")
                return False

            self.logger.info(f"Processing {stage} file: {file_path} with {handler.__name__}")

            # Process based on stage and file type
            try:
                # First, try to get the content
                content = await handler(file_path)
                if not content:
                    self.logger.warning(f"No content extracted from {file_path}")
                    return False

                # Process based on stage
                if stage == "dictionary":
                    if not isinstance(content, list):
                        self.logger.error(f"Expected list of concepts, got {type(content)}")
                        return False
                        
                    # Store concepts in Qdrant
                    points = []
                    for concept in content:
                        try:
                            # Generate embedding for the concept definition
                            text_to_embed = concept.get("definition", "") or concept.get("term", "")
                            if not text_to_embed:
                                continue
                                
                            # Generate a unique cache key for this concept
                            term = concept.get("term", "")
                            cache_key = f"embedding:{stage}:{term}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
                            
                            # Try to get from cache first
                            embedding = await self._get_cached_embedding(cache_key)
                            if not embedding:
                                self.logger.info(f"Generating new embedding for concept: {term[:50]}...")
                                try:
                                    embedding = await asyncio.to_thread(
                                        self.embedding_model.embed_query,
                                        text_to_embed
                                    )
                                    if not embedding:
                                        self.logger.error("Failed to generate embedding")
                                        continue
                                    # Cache the new embedding
                                    await self._cache_embedding(cache_key, embedding, {
                                        'source': file_path,
                                        'type': 'dictionary_entry',
                                        'term': term
                                    })
                                except Exception as e:
                                    self.logger.error(f"Error generating embedding: {e}")
                                    continue
                            
                            # Create point with unique ID based on term
                            point_id = f"concept:{term}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
                            points.append({
                                "id": point_id,
                                "vector": embedding,
                                "payload": {
                                    **concept,
                                    "embedding_source": "dictionary",
                                    "processed_at": datetime.utcnow().isoformat(),
                                    "term": term,
                                    "definition": text_to_embed
                                }
                            })
                        except Exception as e:
                            self.logger.error(f"Error processing concept: {e}")
                            continue
                    
                    if points:
                        try:
                            # First check if any points already exist
                            existing_ids = []
                            for point in points:
                                try:
                                    result = await self.qdrant.retrieve(
                                        collection_name="concepts",
                                        ids=[point["id"]]
                                    )
                                    if result and len(result) > 0:
                                        existing_ids.append(point["id"])
                                except Exception as e:
                                    self.logger.error(f"Error checking existing point: {e}")
                                    continue
                            
                            # Filter out existing points
                            new_points = [p for p in points if p["id"] not in existing_ids]
                            
                            if new_points:
                                self.logger.info(f"Storing {len(new_points)} new concepts in Qdrant")
                                await self.qdrant.upsert(
                                    collection_name="concepts",
                                    points=new_points
                                )
                                self.logger.info(f"Successfully stored {len(new_points)} concepts from {file_path}")
                            else:
                                self.logger.info(f"All {len(points)} concepts already exist in Qdrant")
                            
                            return True
                        except Exception as e:
                            self.logger.error(f"Error storing concepts in Qdrant: {e}")
                            # Try to get more details about the error
                            try:
                                collection_info = await self.qdrant.get_collection("concepts")
                                self.logger.info(f"Collection info: {collection_info}")
                            except Exception as e2:
                                self.logger.error(f"Error getting collection info: {e2}")
                            return False
                    return False
                    
                elif stage in ["json", "jsonl"]:
                    if not isinstance(content, list):
                        self.logger.error(f"Expected list of entries, got {type(content)}")
                        return False
                        
                    # Store entries in Qdrant
                    points = []
                    for entry in content:
                        try:
                            # Generate embedding for the entry text
                            text_to_embed = entry.get("text", "") or entry.get("value", "") or str(entry)
                            if not text_to_embed:
                                continue
                                
                            # Generate a unique cache key for this entry
                            entry_id = entry.get("id", str(uuid.uuid4()))
                            cache_key = f"embedding:{stage}:{entry_id}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
                            
                            # Try to get from cache first
                            embedding = await self._get_cached_embedding(cache_key)
                            if not embedding:
                                self.logger.info(f"Generating new embedding for {stage} entry...")
                                try:
                                    embedding = await asyncio.to_thread(
                                        self.embedding_model.embed_query,
                                        text_to_embed
                                    )
                                    if not embedding:
                                        self.logger.error("Failed to generate embedding")
                                        continue
                                    # Cache the new embedding
                                    await self._cache_embedding(cache_key, embedding, {
                                        'source': file_path,
                                        'type': f'{stage}_entry',
                                        'entry_id': entry_id
                                    })
                                except Exception as e:
                                    self.logger.error(f"Error generating embedding: {e}")
                                    continue
                            
                            point_id = str(uuid.uuid4())
                            points.append({
                                "id": point_id,
                                "vector": embedding,
                                "payload": {
                                    **entry,
                                    "embedding_source": stage,
                                    "processed_at": datetime.utcnow().isoformat()
                                }
                            })
                        except Exception as e:
                            self.logger.error(f"Error processing entry: {e}")
                            continue
                    
                    if points:
                        try:
                            await self.qdrant.upsert(
                                collection_name="embeddings",
                                points=points
                            )
                            self.logger.info(f"Stored {len(points)} entries from {file_path}")
                            return True
                        except Exception as e:
                            self.logger.error(f"Error storing entries in Qdrant: {e}")
                            return False
                    return False
                    
                elif stage == "wikipedia":
                    if not isinstance(content, tuple) or len(content) != 2:
                        self.logger.error(f"Expected tuple of (nodes, edges), got {type(content)}")
                        return False
                        
                    nodes, edges = content
                    if not isinstance(nodes, list) or not isinstance(edges, list):
                        self.logger.error("Invalid graph data format")
                        return False
                        
                    success = True
                    
                    # Store nodes in Qdrant
                    if nodes:
                        node_points = []
                        for node in nodes:
                            try:
                                text_to_embed = node.get("content", "")
                                if not text_to_embed:
                                    continue
                                    
                                # Generate a unique cache key for this node
                                node_id = node.get("id", str(uuid.uuid4()))
                                cache_key = f"embedding:{stage}:{node_id}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
                                
                                # Try to get from cache first
                                embedding = await self._get_cached_embedding(cache_key)
                                if not embedding:
                                    self.logger.info(f"Generating new embedding for Wikipedia node: {node.get('title', '')[:50]}...")
                                    try:
                                        embedding = await asyncio.to_thread(
                                            self.embedding_model.embed_query,
                                            text_to_embed
                                        )
                                        if not embedding:
                                            self.logger.error("Failed to generate embedding")
                                            continue
                                        # Cache the new embedding
                                        await self._cache_embedding(cache_key, embedding, {
                                            'source': file_path,
                                            'type': 'graph_node',
                                            'node_id': node_id
                                        })
                                    except Exception as e:
                                        self.logger.error(f"Error generating embedding: {e}")
                                        continue
                                
                                node_points.append({
                                    "id": node_id,
                                    "vector": embedding,
                                    "payload": {
                                        **node,
                                        "type": "graph_node",
                                        "embedding_source": "wikipedia",
                                        "processed_at": datetime.utcnow().isoformat()
                                    }
                                })
                            except Exception as e:
                                self.logger.error(f"Error processing Wikipedia node: {e}")
                                success = False
                                continue
                        
                        if node_points:
                            try:
                                await self.qdrant.upsert(
                                    collection_name="graph_nodes",
                                    points=node_points
                                )
                                self.logger.info(f"Stored {len(node_points)} graph nodes from {file_path}")
                            except Exception as e:
                                self.logger.error(f"Error storing nodes in Qdrant: {e}")
                                success = False
                    
                    # Store edges in Redis for graph traversal
                    if edges:
                        try:
                            edge_key = f"graph:edges:{os.path.basename(file_path)}"
                            await self.redis.set_cache(edge_key, edges, 86400)  # 24 hour TTL
                            self.logger.info(f"Stored {len(edges)} graph edges from {file_path}")
                        except Exception as e:
                            self.logger.error(f"Error storing graph edges: {e}")
                            success = False
                    
                    return success
                    
                elif stage == "txt":
                    if not isinstance(content, list):
                        self.logger.error(f"Expected list of text chunks, got {type(content)}")
                        return False
                        
                    # Store entries in Qdrant
                    points = []
                    for i, entry in enumerate(content):
                        try:
                            text_to_embed = entry.get("text", "")
                            if not text_to_embed:
                                continue
                                
                            # Generate a unique cache key for this chunk
                            chunk_id = f"{os.path.basename(file_path)}:chunk_{i}"
                            cache_key = f"embedding:{stage}:{chunk_id}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
                            
                            # Try to get from cache first
                            embedding = await self._get_cached_embedding(cache_key)
                            if not embedding:
                                self.logger.info(f"Generating new embedding for text chunk {i+1}/{len(content)}...")
                                try:
                                    embedding = await asyncio.to_thread(
                                        self.embedding_model.embed_query,
                                        text_to_embed
                                    )
                                    if not embedding:
                                        self.logger.error("Failed to generate embedding")
                                        continue
                                    # Cache the new embedding
                                    await self._cache_embedding(cache_key, embedding, {
                                        'source': file_path,
                                        'type': 'text_chunk',
                                        'chunk_id': chunk_id
                                    })
                                except Exception as e:
                                    self.logger.error(f"Error generating embedding: {e}")
                                    continue
                            
                            point_id = str(uuid.uuid4())
                            points.append({
                                "id": point_id,
                                "vector": embedding,
                                "payload": {
                                    **entry,
                                    "embedding_source": "text",
                                    "processed_at": datetime.utcnow().isoformat()
                                }
                            })
                        except Exception as e:
                            self.logger.error(f"Error processing text chunk: {e}")
                            continue
                    
                    if points:
                        try:
                            await self.qdrant.upsert(
                                collection_name="embeddings",
                                points=points
                            )
                            self.logger.info(f"Stored {len(points)} text chunks from {file_path}")
                            return True
                        except Exception as e:
                            self.logger.error(f"Error storing text chunks in Qdrant: {e}")
                            return False
                    return False
                    
                elif stage == "images":
                    if not isinstance(content, list):
                        self.logger.error(f"Expected list of image metadata, got {type(content)}")
                        return False
                        
                    # Store entries in Qdrant
                    points = []
                    for entry in content:
                        try:
                            # For images, we use the file path as the text to embed
                            text_to_embed = entry.get("source", "")
                            if not text_to_embed:
                                continue
                                
                            # Generate a unique cache key for this image
                            image_id = os.path.basename(text_to_embed)
                            cache_key = f"embedding:{stage}:{image_id}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
                            
                            # Try to get from cache first
                            embedding = await self._get_cached_embedding(cache_key)
                            if not embedding:
                                self.logger.info(f"Generating new embedding for image metadata: {image_id}...")
                                try:
                                    embedding = await asyncio.to_thread(
                                        self.embedding_model.embed_query,
                                        text_to_embed
                                    )
                                    if not embedding:
                                        self.logger.error("Failed to generate embedding")
                                        continue
                                    # Cache the new embedding
                                    await self._cache_embedding(cache_key, embedding, {
                                        'source': file_path,
                                        'type': 'image_metadata',
                                        'image_id': image_id
                                    })
                                except Exception as e:
                                    self.logger.error(f"Error generating embedding: {e}")
                                    continue
                            
                            point_id = str(uuid.uuid4())
                            points.append({
                                "id": point_id,
                                "vector": embedding,
                                "payload": {
                                    **entry,
                                    "embedding_source": "image",
                                    "processed_at": datetime.utcnow().isoformat()
                                }
                            })
                        except Exception as e:
                            self.logger.error(f"Error processing image metadata: {e}")
                            continue
                    
                    if points:
                        try:
                            await self.qdrant.upsert(
                                collection_name="embeddings",
                                points=points
                            )
                            self.logger.info(f"Stored {len(points)} image entries from {file_path}")
                            return True
                        except Exception as e:
                            self.logger.error(f"Error storing image entries in Qdrant: {e}")
                            return False
                    return False
                
                return False
                
            except Exception as e:
                self.logger.error(f"Error in stage processing for {file_path}: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return False

    async def _get_next_directory(self) -> Optional[str]:
        """Get the next directory to process in the current stage."""
        try:
            current_stage, _ = self.CRAWL_STAGES[self.current_stage_index]
            stage_path = os.path.join(self.training_data_path, current_stage)
            
            if not os.path.exists(stage_path):
                self.logger.warning(f"Stage directory not found: {stage_path}")
                return None
            
            # Get all subdirectories in the current stage
            directories = [d for d in os.listdir(stage_path) 
                         if os.path.isdir(os.path.join(stage_path, d))]
            
            # If we haven't started processing this stage yet
            if self.current_directory is None:
                self.current_directory = directories[0] if directories else None
                self.directory_index = 0
                return self.current_directory
            
            # Move to next directory
            self.directory_index += 1
            if self.directory_index < len(directories):
                self.current_directory = directories[self.directory_index]
                return self.current_directory
            
            # If we've processed all directories in this stage
            self.current_directory = None
            self.directory_index = 0
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting next directory: {e}")
            return None

    async def _get_directory_files(self, directory: str) -> List[str]:
        """Get all processable files in a directory."""
        try:
            current_stage, _ = self.CRAWL_STAGES[self.current_stage_index]
            dir_path = os.path.join(self.training_data_path, current_stage, directory)
            
            if not os.path.exists(dir_path):
                return []
            
            files = []
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if not os.path.isfile(file_path):
                    continue
                    
                file_type = await self._get_file_type(file_path)
                if file_type and file_type in self.STAGE_HANDLERS[current_stage]:
                    if file_path not in self.processed_files and file_path not in self.failed_files:
                        files.append(file_path)
            
            return sorted(files)
            
        except Exception as e:
            self.logger.error(f"Error getting directory files: {e}")
            return []

    async def _move_to_next_stage(self) -> bool:
        """Move to the next stage in the crawl sequence."""
        try:
            # Mark current stage as completed
            current_stage, _ = self.CRAWL_STAGES[self.current_stage_index]
            self.stage_progress[current_stage] = True
            
            # Move to next stage
            self.current_stage_index += 1
            if self.current_stage_index >= len(self.CRAWL_STAGES):
                # Check if all stages are completed
                if all(self.stage_progress.values()):
                    self.logger.info("All stages completed successfully")
                    return False
                else:
                    # Reset to first stage if some stages are incomplete
                    self.current_stage_index = 0
                    self.logger.info("Completed all stages, starting over")
            
            next_stage, _ = self.CRAWL_STAGES[self.current_stage_index]
            self.logger.info(f"Moving to stage: {next_stage}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error moving to next stage: {e}")
            return False

    async def crawl_next_file(self, max_retries: int = 3) -> bool:
        """Process the next file in the current stage."""
        try:
            # Get current stage
            current_stage, priority = self.CRAWL_STAGES[self.current_stage_index]
            stage_path = os.path.join(self.training_data_path, current_stage)
            
            self.logger.info(f"Processing stage: {current_stage} (priority: {priority})")
            
            if not os.path.exists(stage_path):
                self.logger.warning(f"Stage directory not found: {stage_path}")
                await self._move_to_next_stage()
                return False

            # Get all files in current stage that haven't been processed
            files = []
            for root, _, filenames in os.walk(stage_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if file_path not in self.processed_files and file_path not in self.failed_files:
                        file_type = await self._get_file_type(file_path)
                        if file_type and file_type in self.file_handlers[current_stage]:
                            files.append(file_path)

            if not files:
                self.logger.info(f"No more files to process in {current_stage} stage")
                await self._move_to_next_stage()
                return False

            # Process next file with retries
            file_path = files[0]
            self.logger.info(f"Attempting to process file: {file_path}")
            
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    # Check if file still exists
                    if not os.path.exists(file_path):
                        self.logger.error(f"File no longer exists: {file_path}")
                        self.failed_files.add(file_path)
                        return False
                    
                    # Process the file
                    success = await self.process_file(file_path)
                    
                    if success:
                        self.processed_files.add(file_path)
                        self.logger.info(f"Successfully processed {file_path}")
                        return True
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            self.logger.warning(
                                f"Failed to process {file_path} (attempt {retry_count}/{max_retries}), "
                                f"retrying after delay..."
                            )
                            await asyncio.sleep(self.processing_delay)
                        else:
                            self.logger.error(
                                f"Failed to process {file_path} after {max_retries} attempts"
                            )
                            self.failed_files.add(file_path)
                            return False
                            
                except Exception as e:
                    retry_count += 1
                    last_error = e
                    self.logger.error(f"Error processing {file_path} (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        await asyncio.sleep(self.processing_delay)
                    else:
                        self.logger.error(f"Max retries exceeded for {file_path}: {last_error}")
                        self.failed_files.add(file_path)
                        return False

            return False
            
        except Exception as e:
            self.logger.error(f"Error in crawl_next_file: {e}")
            return False

    async def crawl_training_data_incremental(self, delay: float = 1.0):
        """Process files one at a time with delay between each."""
        self.logger.info(f"Starting staged processing from {self.training_data_path}")
        
        # Initialize processing state
        self.is_processing = False
        self.current_stage_index = 0
        self.current_directory = None
        self.directory_index = 0
        self.directory_files = []
        self.stage_progress = {stage: False for stage, _ in self.CRAWL_STAGES}
        
        while True:
            try:
                # Check if already processing
                if self.is_processing:
                    await asyncio.sleep(delay)
                    continue
                
                self.is_processing = True
                try:
                    # Get current stage info
                    current_stage, priority = self.CRAWL_STAGES[self.current_stage_index]
                    stage_path = os.path.join(self.training_data_path, current_stage)
                    
                    self.logger.info(f"Processing stage: {current_stage} (priority: {priority})")
                    
                    if not os.path.exists(stage_path):
                        self.logger.warning(f"Stage directory not found: {stage_path}")
                        await self._move_to_next_stage()
                        self.is_processing = False
                        await asyncio.sleep(delay * 2)
                        continue
                    
                    # Get all files in current stage that haven't been processed
                    files_to_process = []
                    for root, _, filenames in os.walk(stage_path):
                        for filename in filenames:
                            file_path = os.path.join(root, filename)
                            if file_path not in self.processed_files and file_path not in self.failed_files:
                                file_type = await self._get_file_type(file_path)
                                if file_type and file_type in self.file_handlers[current_stage]:
                                    files_to_process.append(file_path)
                    
                    if not files_to_process:
                        self.logger.info(f"No more files to process in {current_stage} stage")
                        await self._move_to_next_stage()
                        self.is_processing = False
                        await asyncio.sleep(delay * 2)
                        continue
                    
                    # Process files in batches to avoid overwhelming the system
                    batch_size = 5
                    for i in range(0, len(files_to_process), batch_size):
                        batch = files_to_process[i:i + batch_size]
                        self.logger.info(f"Processing batch {i//batch_size + 1} of {(len(files_to_process) + batch_size - 1)//batch_size}")
                        
                        for file_path in batch:
                            try:
                                self.logger.info(f"Processing file: {file_path}")
                                success = await self.process_file(file_path)
                                
                                if success:
                                    self.processed_files.add(file_path)
                                    self.logger.info(f"Successfully processed {file_path}")
                                else:
                                    self.failed_files.add(file_path)
                                    self.logger.error(f"Failed to process {file_path}")
                                
                                # Add delay between files
                                await asyncio.sleep(delay)
                                
                            except Exception as e:
                                self.logger.error(f"Error processing {file_path}: {e}")
                                self.failed_files.add(file_path)
                                await asyncio.sleep(delay)
                        
                        # Add delay between batches
                        await asyncio.sleep(delay * 2)
                    
                    # Log progress after each stage
                    total_files = len(self.processed_files) + len(self.failed_files)
                    if total_files > 0:
                        self.logger.info(
                            f"Stage {current_stage} progress - "
                            f"Processed: {len(self.processed_files)}, "
                            f"Failed: {len(self.failed_files)}, "
                            f"Total: {total_files}, "
                            f"Success rate: {(len(self.processed_files) / total_files) * 100:.1f}%"
                        )
                    
                    # Move to next stage
                    await self._move_to_next_stage()
                    
                finally:
                    self.is_processing = False
                
                # Add delay between stages
                await asyncio.sleep(delay * 3)
                
            except Exception as e:
                self.logger.error(f"Error in crawl loop: {e}")
                self.is_processing = False
                await asyncio.sleep(delay * 5)  # Longer delay on error

    async def _process_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process text files and generate embeddings."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            entries = []
            
            for i, chunk in enumerate(chunks):
                entry = {
                    "text": chunk,
                    "source": file_path,
                    "type": "text_chunk",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                entries.append(entry)
            
            return entries
        except Exception as e:
            self.logger.error(f"Error processing text file {file_path}: {e}")
            return []

    async def _process_image_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process image files and extract metadata."""
        try:
            # For now, just extract basic metadata
            # TODO: Add image analysis and feature extraction
            stat = os.stat(file_path)
            entry = {
                "source": file_path,
                "type": "image",
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            return [entry]
        except Exception as e:
            self.logger.error(f"Error processing image file {file_path}: {e}")
            return [] 