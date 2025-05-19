import os
import asyncio
import logging
import uuid
import json
import mimetypes
from datetime import datetime
from typing import Dict, Optional, List, Set, Tuple, Union, Any
from .embeddings import CustomOllamaEmbeddings
from lumina_core.common.bus import BusClient
from .redis_client import RedisClient
from .qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import hashlib
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TrainingCrawler:
    """Processes training data in phases: phase1 > phase2 > phase3 > phase4 > phase5."""
    
    # Class-level variables for singleton pattern
    _instance_lock = asyncio.Lock()
    _instance = None
    
    # Define crawl phases and their priorities
    CRAWL_PHASES = [
        ("phase 1", 1),
        ("phase 2", 2),
        ("phase 3", 3),
        ("phase 4", 4),
        ("phase 5", 5)
    ]
    
    # Define file type handlers for each phase (all phases use the same handlers)
    PHASE_FILE_HANDLERS = {
        ".json": "_process_general_json",
        ".jsonl": "_process_general_jsonl",
        ".txt": "_process_text_file",
        ".jpg": "_process_image_file",
        ".jpeg": "_process_image_file",
        ".png": "_process_image_file",
        ".gif": "_process_image_file"
    }
    
    def __init__(
        self,
        redis_url: str = "redis://:02211998@redis:6379",
        qdrant_url: str = "http://qdrant:6333",
        training_data_path: str = "/app/training_data",
        ollama_url: str = "http://ollama:11434",
        ollama_model: str = "nomic-embed-text",
        max_workers: int = 1,
        processing_delay: float = 5.0,
        batch_size: int = 5,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Initialize text splitter for text files with smaller chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Store batch size
        self.batch_size = batch_size
        
        # Initialize clients
        self.bus = BusClient(redis_url=redis_url)
        self.redis = RedisClient(redis_url=redis_url)
        self.qdrant = QdrantClient(url=qdrant_url)
        
        # Initialize Ollama model with retry and proper configuration
        max_retries = 3
        retry_delay = 5
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Initialize with only required options
                self.embedding_model = CustomOllamaEmbeddings(
                    base_url=ollama_url,
                    model=ollama_model
                )
                # Test the model with a simple query
                test_embedding = self.embedding_model.embed_query("test")
                if not test_embedding:
                    raise ValueError("Failed to generate test embedding")
                if len(test_embedding) != 768:  # Verify embedding dimension
                    raise ValueError(f"Invalid embedding dimension: {len(test_embedding)}")
                self.logger.info(f"Successfully initialized Ollama model: {ollama_model}")
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    self.logger.warning(f"Failed to initialize Ollama model (attempt {attempt + 1}/{max_retries}): {e}")
                    self.logger.info(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to initialize Ollama model after {max_retries} attempts: {e}")
                    raise RuntimeError(f"Failed to initialize Ollama model: {last_error}")
        
        # Initialize other instance variables
        self.training_data_path = training_data_path
        self.processed_files = set()
        self.failed_files = set()
        self.current_phase_index = 0
        self.max_workers = max_workers
        self.processing_delay = processing_delay
        self.is_processing = False
        self.batch_size = batch_size
        self.phase_progress = {phase: False for phase, _ in self.CRAWL_PHASES}
        
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
        """Register file type handlers for all phases."""
        self.file_handlers = {ext: getattr(self, handler) for ext, handler in self.PHASE_FILE_HANDLERS.items()}
    
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
            
            # Create collections for different types of data with minimal required config
            collections = {
                "embeddings": {
                    "size": 768,
                    "distance": Distance.COSINE
                },
                "graph_nodes": {
                    "size": 768,
                    "distance": Distance.COSINE
                },
                "concepts": {
                    "size": 768,
                    "distance": Distance.COSINE
                }
            }
            
            for name, config in collections.items():
                try:
                    # Check if collection exists using get_collections (synchronous method)
                    collections_list = self.qdrant._client.get_collections()
                    collection_exists = any(c.name == name for c in collections_list.collections)
                    
                    if collection_exists:
                        # Get collection info to check vector size
                        collection_info = self.qdrant._client.get_collection(name)
                        current_size = collection_info.config.params.vectors.size
                        
                        if current_size != config["size"]:
                            self.logger.info(f"Collection {name} has wrong vector size ({current_size} vs {config['size']}). Deleting and recreating...")
                            try:
                                self.qdrant._client.delete_collection(name)
                                # Wait longer to ensure deletion is complete
                                await asyncio.sleep(2)
                            except Exception as e:
                                self.logger.error(f"Error deleting collection {name}: {e}")
                                raise
                            collection_exists = False
                        else:
                            self.logger.info(f"Collection {name} already exists with correct vector size")
                    
                    if not collection_exists:
                        # Create collection with minimal config
                        self.logger.info(f"Creating collection: {name}")
                        try:
                            # Use synchronous create_collection
                            self.qdrant._client.create_collection(
                                collection_name=name,
                                vectors_config=VectorParams(
                                    size=config["size"],
                                    distance=config["distance"]
                                )
                            )
                            # Wait longer to ensure creation is complete
                            await asyncio.sleep(2)
                            
                            # Verify collection was created successfully
                            collection_info = self.qdrant._client.get_collection(name)
                            if collection_info.config.params.vectors.size != config["size"]:
                                raise ValueError(f"Collection {name} was created but has wrong vector size")
                            self.logger.info(f"Created and verified collection: {name}")
                        except Exception as e:
                            self.logger.error(f"Error creating collection {name}: {e}")
                            raise
                    
                    # Create indexes if they don't exist (synchronous methods)
                    try:
                        self.qdrant._client.create_payload_index(
                            collection_name=name,
                            field_name="embedding_source",
                            field_schema="keyword"
                        )
                        self.qdrant._client.create_payload_index(
                            collection_name=name,
                            field_name="processed_at",
                            field_schema="keyword"  # Store datetime as string
                        )
                        
                        if name == "concepts":
                            self.qdrant._client.create_payload_index(
                                collection_name=name,
                                field_name="term",
                                field_schema="keyword"
                            )
                            
                        self.logger.info(f"Created indexes for collection: {name}")
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            self.logger.warning(f"Error creating indexes for {name}: {e}")
                        
                    # Final verification that collection is ready
                    try:
                        collection_info = self.qdrant._client.get_collection(name)
                        if collection_info.config.params.vectors.size != config["size"]:
                            raise ValueError(f"Collection {name} was created but has wrong vector size")
                        self.logger.info(f"Final verification: Collection {name} is ready with correct vector size")
                    except Exception as e:
                        self.logger.error(f"Error in final verification of collection {name}: {e}")
                        raise
                        
                except Exception as e:
                    self.logger.error(f"Failed to initialize collection {name}: {e}")
                    raise
                    
            self.logger.info("All collections initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TrainingCrawler: {e}")
            raise

    async def start(self):
        """Start the crawler process."""
        try:
            self.logger.info("Starting training data crawler (phases)...")
            await self.crawl_training_data_phases()
        except Exception as e:
            self.logger.error(f"Error starting crawler: {e}")
            raise

    async def stop(self):
        """Stop the crawler process."""
        try:
            self.logger.info("Stopping training data crawler...")
            self.is_processing = False
            await self.bus.close()
            await self.redis.close()
            await self.qdrant.close()
            self.logger.info("Training data crawler stopped")
        except Exception as e:
            self.logger.error(f"Error stopping crawler: {e}")
            raise

    async def _get_file_type(self, file_path: str) -> Optional[str]:
        """Get the file type based on extension and content."""
        try:
            _, ext = os.path.splitext(file_path.lower())
            self.logger.info(f"Checking file type for {file_path} (extension: {ext})")
            
            # Direct extension check first
            if ext in ['.txt', '.json', '.jsonl', '.jpg', '.jpeg', '.png', '.gif']:
                try:
                    # For text files, verify it's readable
                    if ext in ['.txt', '.json', '.jsonl']:
                        # For JSON files, trust the extension if file exists and is readable
                        if ext == '.json':
                            if os.path.exists(file_path) and os.access(file_path, os.R_OK):
                                self.logger.info(f"Confirmed JSON file: {file_path}")
                                return '.json'
                        # For text and JSONL files, do basic validation
                        elif ext in ['.txt', '.jsonl']:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read(1024)  # Read first 1KB
                                if ext == '.txt':
                                    # For text files, just verify it's readable
                                    if content:
                                        self.logger.info(f"Confirmed text file: {file_path}")
                                        return '.txt'
                                elif ext == '.jsonl':
                                    # Verify it's valid JSONL
                                    try:
                                        for line in content.split('\n'):
                                            if line.strip():
                                                json.loads(line.strip())
                                        self.logger.info(f"Confirmed JSONL file: {file_path}")
                                        return '.jsonl'
                                    except json.JSONDecodeError:
                                        self.logger.warning(f"Invalid JSONL content in {file_path}")
                    else:
                        # For image files, verify they exist and are readable
                        if os.path.exists(file_path) and os.access(file_path, os.R_OK):
                            self.logger.info(f"Confirmed image file: {file_path}")
                            return ext
                except Exception as e:
                    self.logger.error(f"Error reading file {file_path}: {e}")
            
            # If we get here, try to determine type from content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1024).strip()  # Read first 1KB
                    if content:
                        if content.startswith('{') or content.startswith('['):
                            try:
                                json.loads(content)
                                self.logger.info(f"Detected JSON content in {file_path}")
                                return '.json'
                            except json.JSONDecodeError:
                                pass
                        elif all(line.strip().startswith('{') or line.strip().startswith('[') 
                               for line in content.split('\n')[:3] if line.strip()):
                            try:
                                for line in content.split('\n')[:3]:
                                    if line.strip():
                                        json.loads(line.strip())
                                self.logger.info(f"Detected JSONL content in {file_path}")
                                return '.jsonl'
                            except json.JSONDecodeError:
                                pass
                        else:
                            # If it's readable text but not JSON/JSONL, treat as txt
                            self.logger.info(f"Detected text content in {file_path}")
                            return '.txt'
            except Exception as e:
                self.logger.error(f"Error determining content type for {file_path}: {e}")
            
            self.logger.warning(f"Could not determine file type for {file_path}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining file type for {file_path}: {e}")
            return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent cache key generation."""
        if not isinstance(text, str):
            text = str(text)
        # Remove extra whitespace, convert to lowercase, and strip
        return ' '.join(text.lower().strip().split())

    def _generate_cache_key(self, text: str, stage: str, identifier: Optional[str] = None) -> str:
        """Generate a consistent cache key for text content.
        
        Args:
            text: The text to generate a cache key for
            stage: The processing stage (e.g., 'dictionary', 'json', etc.)
            identifier: Optional identifier to make the key more specific (e.g., term)
        """
        # Normalize text using the dedicated method
        normalized_text = self._normalize_text(text)
        # Create a hash of the normalized text
        content_hash = hashlib.md5(normalized_text.encode()).hexdigest()
        # Build the cache key - only use stage and content hash
        return f"embedding:{stage}:{content_hash}"

    async def _process_dictionary_json(self, file_path: str) -> bool:
        """Process dictionary JSON files with structured format."""
        try:
            self.logger.info(f"Starting to process dictionary JSON file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.debug(f"Loaded JSON data type: {type(data)}")
            
            # Handle different dictionary formats
            items = []
            if isinstance(data, dict):
                self.logger.info("Processing dictionary format")
                items = list(data.items())
            elif isinstance(data, list):
                self.logger.info("Processing array format")
                # Array format - convert to dictionary items
                for item in data:
                    if isinstance(item, dict):
                        if "term" in item:
                            self.logger.debug("Found term-based entry")
                            items.append((item["term"], item))
                        elif "word" in item:
                            self.logger.debug("Found word-based entry")
                            items.append((item["word"], item))
                        elif "key" in item:
                            self.logger.debug("Found key-based entry")
                            items.append((item["key"], item))
                    elif isinstance(item, str):
                        self.logger.debug("Found string entry")
                        items.append((item, {"definition": item}))
            else:
                self.logger.error(f"Unsupported dictionary format in {file_path}. Expected dict or array, got {type(data)}")
                return False
            
            if not items:
                self.logger.warning(f"No valid dictionary entries found in {file_path}")
                return False
                
            self.logger.info(f"Found {len(items)} items to process")
            
            # Process in batches of 100 concepts
            batch_size = 100
            points = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1} of {(len(items) + batch_size - 1)//batch_size} ({len(batch)} items)")
                batch_points = []
                
                for term, definition in batch:
                    try:
                        # Normalize term and definition
                        term = self._normalize_text(str(term))
                        
                        # Get definition text, ensuring it's a string
                        if isinstance(definition, dict):
                            text_to_embed = definition.get("definition", "") or definition.get("text", "") or definition.get("content", "") or term
                        else:
                            text_to_embed = str(definition) or term
                            
                        # Normalize the text to embed
                        text_to_embed = self._normalize_text(text_to_embed)
                        
                        if not text_to_embed:
                            self.logger.debug(f"Skipping empty definition for term: {term}")
                            continue
                            
                        # Generate cache key using normalized text
                        cache_key = self._generate_cache_key(text_to_embed, "dictionary")
                        
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
                                    self.logger.error(f"Failed to generate embedding for term: {term}")
                                    continue
                                # Cache the new embedding with metadata
                                await self._cache_embedding(cache_key, embedding, {
                                    'source': file_path,
                                    'type': 'dictionary_entry',
                                    'term': term,
                                    'normalized_text': text_to_embed
                                })
                            except Exception as e:
                                self.logger.error(f"Error generating embedding for term {term}: {str(e)}", exc_info=True)
                                continue
                        
                        # Create point with unique ID based on normalized term and text
                        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
                        name = f"{term}:{text_to_embed}"
                        point_id = str(uuid.uuid5(namespace, name))
                        
                        # Extract additional fields if available
                        concept_data = {
                            "term": term,
                            "definition": text_to_embed,
                            "examples": [self._normalize_text(ex) for ex in (definition.get("examples", []) if isinstance(definition, dict) else [])],
                            "related": [self._normalize_text(rel) for rel in (definition.get("related", []) if isinstance(definition, dict) else [])],
                            "categories": [self._normalize_text(cat) for cat in (definition.get("categories", []) if isinstance(definition, dict) else [])],
                            "metadata": definition.get("metadata", {}) if isinstance(definition, dict) else {}
                        }
                        
                        # Add any additional fields from the definition
                        if isinstance(definition, dict):
                            for key, value in definition.items():
                                if key not in concept_data and key not in ["term", "definition", "examples", "related", "categories", "metadata"]:
                                    concept_data[key] = value
                        
                        point = {
                            "id": point_id,
                            "vector": embedding,
                            "payload": {
                                **concept_data,
                                "embedding_source": "dictionary",
                                "processed_at": datetime.utcnow().isoformat(),
                                "original_id": f"concept:{term}:{hashlib.md5(text_to_embed.encode()).hexdigest()}",
                                "normalized_text": text_to_embed,
                                "source_format": "array" if isinstance(data, list) else "dict"
                            }
                        }
                        batch_points.append(point)
                        self.logger.debug(f"Created point for term: {term[:50]}...")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing concept {term[:50]}: {str(e)}", exc_info=True)
                        continue
                
                if batch_points:
                    self.logger.info(f"Storing batch of {len(batch_points)} points")
                    # Store batch in Qdrant
                    success = await self._store_points("concepts", batch_points)
                    if not success:
                        self.logger.error(f"Failed to store batch of concepts")
                        return False
                    
                    # Store in concept dictionary
                    for point in batch_points:
                        try:
                            concept_data = {
                                "term": point["payload"]["term"],
                                "definition": point["payload"]["definition"],
                                "examples": point["payload"].get("examples", []),
                                "related": point["payload"].get("related", []),
                                "categories": point["payload"].get("categories", []),
                                "metadata": {
                                    **point["payload"].get("metadata", {}),
                                    "source": file_path,
                                    "type": "dictionary_entry",
                                    "processed_at": datetime.utcnow().isoformat(),
                                    "source_format": point["payload"].get("source_format", "unknown")
                                }
                            }
                            
                            success = await self.redis.add_concept(point["payload"]["term"], concept_data)
                            if not success:
                                self.logger.warning(f"Failed to store concept {point['payload']['term']} in concept dictionary")
                        except Exception as e:
                            self.logger.error(f"Error storing concept in dictionary: {e}")
                            continue
                    
                    points.extend(batch_points)
                    self.logger.info(f"Successfully processed batch of {len(batch_points)} points")
                
                # Add a small delay between batches
                await asyncio.sleep(0.1)
            
            if points:
                self.logger.info(f"Successfully processed {len(points)} concepts from {file_path}")
                return True
            else:
                self.logger.warning(f"No concepts extracted from {file_path}")
                return False
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {file_path}: {str(e)}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"Error processing dictionary JSON file {file_path}: {str(e)}", exc_info=True)
            return False

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

    async def _process_general_json(self, file_path: str) -> bool:
        """Process general JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            if not isinstance(content, (dict, list)):
                self.logger.error(f"Invalid JSON content in {file_path}")
                return False
            
            # Convert content to list of entries
            entries = []
            if isinstance(content, dict):
                # Handle dictionary format
                if "data" in content and isinstance(content["data"], dict):
                    # Handle structured data format with data field
                    data = content["data"]
                    if "content" in data and isinstance(data["content"], dict):
                        # Handle content object format
                        content_obj = data["content"]
                        entry = {
                            "id": content_obj.get("id", ""),
                            "text": content_obj.get("summary", ""),  # Use summary as main text
                            "metadata": {
                                "source": file_path,
                                "type": "structured_content",
                                "labels": content.get("metadata", {}).get("labels", []),
                                "domain": content.get("metadata", {}).get("domain", ""),
                                "training_config": content.get("training_config", {}),
                                "position": content_obj.get("position", {}),
                                "index": content_obj.get("index", 0)
                            }
                        }
                        entries.append(entry)
                    else:
                        # Handle other data formats
                        for key, value in data.items():
                            if isinstance(value, dict):
                                entries.append({
                                    "id": key,
                                    "text": value.get("summary", "") or str(value),
                                    "metadata": {
                                        "source": file_path,
                                        "type": "data_entry",
                                        "original_key": key
                                    }
                                })
                else:
                    # Handle simple key-value dictionary
                    for key, value in content.items():
                        entries.append({
                            "id": key,
                            "text": str(value),
                            "metadata": {
                                "source": file_path,
                                "type": "key_value",
                                "original_key": key
                            }
                        })
            else:
                # Handle list format
                for item in content:
                    if isinstance(item, dict):
                        # Try to extract meaningful content
                        text = item.get("summary", "") or item.get("text", "") or item.get("content", "") or str(item)
                        entry = {
                            "id": item.get("id", str(uuid.uuid4())),
                            "text": text,
                            "metadata": {
                                "source": file_path,
                                "type": "list_entry",
                                **{k: v for k, v in item.items() if k not in ["id", "summary", "text", "content"]}
                            }
                        }
                        entries.append(entry)
            
            if not entries:
                self.logger.warning(f"No entries extracted from {file_path}")
                return False
            
            # Process entries in batches
            points = []
            for entry in entries:
                point = await self._process_json_entry(entry, "json", file_path)
                if point:
                    points.append(point)
            
            if points:
                try:
                    await self.qdrant._client.upsert(
                        collection_name="embeddings",
                        points=points
                    )
                    self.logger.info(f"Stored {len(points)} entries from {file_path}")
                    return True
                except Exception as e:
                    self.logger.error(f"Error storing entries in Qdrant: {e}")
                    return False
            
            self.logger.warning(f"No points generated from {file_path}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing JSON file {file_path}: {e}")
            return False

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

    async def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        try:
            cached_data = await self.redis.get_cache(cache_key)
            if cached_data and isinstance(cached_data, dict) and 'embedding' in cached_data:
                self.logger.debug(f"Cache hit for key: {cache_key}")  # Changed to debug
                return cached_data['embedding']
            self.logger.debug(f"Cache miss for key: {cache_key}")  # Changed to debug
            return None
        except Exception as e:
            self.logger.error(f"Error getting cached embedding: {e}")
            return None
            
    async def _cache_embedding(self, cache_key: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Cache embedding with metadata."""
        try:
            # Ensure embedding is a list of floats
            if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                raise ValueError("Embedding must be a list of numbers")
                
            cache_data = {
                'embedding': embedding,
                'metadata': metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
            # Use a longer TTL for embeddings (7 days) since they don't change
            await self.redis.set_cache(cache_key, cache_data, 604800)  # 7 days
            self.logger.debug(f"Cached embedding for key: {cache_key}")  # Changed to debug
        except Exception as e:
            self.logger.error(f"Error caching embedding: {e}")

    async def _process_json_entry(self, entry: Dict[str, Any], stage: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single JSON entry and generate its embedding."""
        try:
            # Extract text to embed
            text_to_embed = entry.get("text", "") or entry.get("value", "") or str(entry)
            if not text_to_embed:
                return None
                
            # Generate a unique identifier for this entry (only used for logging)
            entry_id = entry.get("id", str(uuid.uuid4()))
            
            # Generate cache key - only use the content, not the UUID
            cache_key = self._generate_cache_key(text_to_embed, stage)
            
            # Try to get from cache first
            embedding = await self._get_cached_embedding(cache_key)
            if not embedding:
                self.logger.debug(f"Generating new embedding for {stage} entry {entry_id[:8]}...")
                try:
                    embedding = await asyncio.to_thread(
                        self.embedding_model.embed_query,
                        text_to_embed
                    )
                    if not embedding:
                        self.logger.error("Failed to generate embedding")
                        return None
                    # Cache the new embedding
                    await self._cache_embedding(cache_key, embedding, {
                        'source': file_path,
                        'type': f'{stage}_entry',
                        'entry_id': entry_id
                    })
                except Exception as e:
                    self.logger.error(f"Error generating embedding: {e}")
                    return None
            
            # Create point with unique ID
            point_id = str(uuid.uuid4())
            # Convert datetime to string for storage
            processed_at = datetime.utcnow().isoformat()
            return {
                "id": point_id,
                "vector": embedding,
                "payload": {
                    **entry,
                    "embedding_source": stage,
                    "processed_at": processed_at,  # Store as ISO format string
                    "original_id": entry_id
                }
            }
        except Exception as e:
            self.logger.error(f"Error processing entry: {e}")
            return None

    async def process_file(self, file_path: str) -> bool:
        """Process a file based on its directory and type."""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return False

            # Determine the stage based on directory
            rel_path = os.path.relpath(file_path, self.training_data_path)
            stage = rel_path.split(os.sep)[0]
            
            self.logger.info(f"Processing file: {rel_path}")  # Simplified logging
            
            if stage not in [s[0] for s in self.CRAWL_PHASES]:
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

            # Process based on stage and file type
            try:
                # First, try to get the content
                content = await handler(file_path)
                if content is None:
                    self.logger.warning(f"Handler returned None for {file_path}")
                    return False
                if isinstance(content, (list, tuple)) and len(content) == 0:
                    self.logger.warning(f"No content extracted from {file_path}")
                    return False
                if isinstance(content, bool) and not content:
                    self.logger.warning(f"Handler returned False for {file_path}")
                    return False

                # Process based on stage
                if stage == "dictionary":
                    if not isinstance(content, (list, bool)):
                        self.logger.error(f"Expected list of concepts or bool, got {type(content)}")
                        return False
                    
                    if isinstance(content, bool):
                        return content  # Handler already processed the content
                    
                    # Store concepts in Qdrant
                    points = []
                    for concept in content:
                        try:
                            # Process concept...
                            # Generate embedding for the concept definition
                            text_to_embed = concept.get("definition", "") or concept.get("term", "")
                            if not text_to_embed:
                                continue
                                
                            # Generate a unique cache key for this concept
                            term = concept.get("term", "")
                            cache_key = f"embedding:{term}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
                            
                            # Try to get from cache first
                            embedding = await self._get_cached_embedding(cache_key)
                            if not embedding:
                                self.logger.debug(f"Generating new embedding for concept: {term[:50]}...")  # Changed to debug
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
                            namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
                            name = f"{term}:{text_to_embed}"
                            point_id = str(uuid.uuid5(namespace, name))
                            points.append({
                                "id": point_id,
                                "vector": embedding,
                                "payload": {
                                    **concept,
                                    "embedding_source": "dictionary",
                                    "processed_at": datetime.utcnow().isoformat(),
                                    "term": term,
                                    "definition": text_to_embed,
                                    "original_id": f"concept:{term}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
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
                                    # Retrieve points one at a time
                                    result = await self.qdrant._client.retrieve(
                                        collection_name="concepts",
                                        ids=[point["id"]],
                                        with_payload=False
                                    )
                                    if result and len(result) > 0:
                                        existing_ids.append(point["id"])
                                except Exception as e:
                                    self.logger.error(f"Error checking existing point: {e}")
                                    continue
                            
                            # Filter out existing points
                            new_points = [p for p in points if p["id"] not in existing_ids]
                            
                            if new_points:
                                try:
                                    self.logger.info(f"Storing {len(new_points)} new concepts from {os.path.basename(file_path)}")  # Simplified logging
                                    await self.qdrant._client.upsert(
                                        collection_name="concepts",
                                        points=new_points
                                    )
                                    
                                    # Also store in concept dictionary
                                    for point in new_points:
                                        try:
                                            concept_data = {
                                                "term": point["payload"]["term"],
                                                "definition": point["payload"]["definition"],
                                                "examples": point["payload"].get("examples", []),
                                                "related": point["payload"].get("related", []),
                                                "categories": point["payload"].get("categories", []),
                                                "metadata": {
                                                    **point["payload"].get("metadata", {}),
                                                    "source": file_path,
                                                    "type": "dictionary_entry",
                                                    "processed_at": datetime.utcnow().isoformat()
                                                }
                                            }
                                            
                                            success = await self.redis.add_concept(point["payload"]["term"], concept_data)
                                            if not success:
                                                self.logger.warning(f"Failed to store concept {point['payload']['term']} in concept dictionary")
                                        except Exception as e:
                                            self.logger.error(f"Error storing concept in dictionary: {e}")
                                            continue
                                    
                                    self.logger.info(f"Successfully processed {os.path.basename(file_path)}")  # Simplified logging
                                except Exception as e:
                                    self.logger.error(f"Error storing concepts in Qdrant: {e}")
                                    return False
                            else:
                                self.logger.info(f"All concepts already exist for {os.path.basename(file_path)}")  # Simplified logging
                                return True
                        except Exception as e:
                            self.logger.error(f"Error storing concepts in Qdrant: {e}")
                            return False
                    
                elif stage in ["json", "jsonl"]:
                    if not isinstance(content, list):
                        self.logger.error(f"Expected list of entries, got {type(content)}")
                        return False
                    
                    # Store entries in Qdrant
                    points = []
                    for entry in content:
                        try:
                            point = await self._process_json_entry(entry, stage, file_path)
                            if point:
                                points.append(point)
                        except Exception as e:
                            self.logger.error(f"Error processing entry: {e}")
                            continue
                    
                    if points:
                        try:
                            await self.qdrant._client.upsert(
                                collection_name="embeddings",
                                points=points
                            )
                            self.logger.info(f"Successfully processed {os.path.basename(file_path)}")  # Simplified logging
                            return True
                        except Exception as e:
                            self.logger.error(f"Error storing entries in Qdrant: {e}")
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
                                cache_key = f"embedding:{node_id}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
                                
                                # Try to get from cache first
                                embedding = await self._get_cached_embedding(cache_key)
                                if not embedding:
                                    self.logger.debug(f"Generating new embedding for Wikipedia node: {node.get('title', '')[:50]}...")  # Changed to debug
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
                                
                                point_id = str(uuid.uuid4())
                                node_points.append({
                                    "id": point_id,
                                    "vector": embedding,
                                    "payload": {
                                        **node,
                                        "type": "graph_node",
                                        "embedding_source": "wikipedia",
                                        "processed_at": datetime.utcnow().isoformat(),
                                        "original_id": node.get("id", str(uuid.uuid4()))
                                    }
                                })
                            except Exception as e:
                                self.logger.error(f"Error processing Wikipedia node: {e}")
                                success = False
                                continue
                        
                        if node_points:
                            try:
                                # First check if any points already exist
                                existing_ids = []
                                for point in node_points:
                                    try:
                                        result = await self.qdrant._client.retrieve(
                                            collection_name="graph_nodes",
                                            ids=[point["id"]],
                                            with_payload=False
                                        )
                                        if result and len(result) > 0:
                                            existing_ids.append(point["id"])
                                    except Exception as e:
                                        self.logger.error(f"Error checking existing point: {e}")
                                        continue
                                
                                # Filter out existing points
                                new_points = [p for p in node_points if p["id"] not in existing_ids]
                                
                                if new_points:
                                    try:
                                        self.logger.info(f"Storing {len(new_points)} new graph nodes from {os.path.basename(file_path)}")  # Simplified logging
                                        await self.qdrant._client.upsert(
                                            collection_name="graph_nodes",
                                            points=new_points
                                        )
                                        
                                        # Also store in concept dictionary
                                        for point in new_points:
                                            try:
                                                concept_data = {
                                                    "term": point["payload"]["title"],
                                                    "definition": point["payload"]["content"],
                                                    "examples": point["payload"].get("examples", []),
                                                    "related": point["payload"].get("related", []),
                                                    "categories": point["payload"].get("categories", []),
                                                    "metadata": {
                                                        **point["payload"].get("metadata", {}),
                                                        "source": file_path,
                                                        "type": "graph_node",
                                                        "processed_at": datetime.utcnow().isoformat()
                                                    }
                                                }
                                                
                                                success = await self.redis.add_concept(point["payload"]["title"], concept_data)
                                                if not success:
                                                    self.logger.warning(f"Failed to store concept {point['payload']['title']} in concept dictionary")
                                            except Exception as e:
                                                self.logger.error(f"Error storing concept in dictionary: {e}")
                                                continue
                                        
                                        self.logger.info(f"Successfully processed {os.path.basename(file_path)}")  # Simplified logging
                                    except Exception as e:
                                        self.logger.error(f"Error storing graph nodes in Qdrant: {e}")
                                        return False
                            except Exception as e:
                                self.logger.error(f"Error storing graph nodes in Qdrant: {e}")
                                return False
                    
                    # Store edges in Redis for graph traversal
                    if edges:
                        try:
                            edge_key = f"graph:edges:{os.path.basename(file_path)}"
                            await self.redis.set_cache(edge_key, edges, 86400)  # 24 hour TTL
                            self.logger.info(f"Stored {len(edges)} graph edges from {os.path.basename(file_path)}")  # Simplified logging
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
                            cache_key = f"embedding:{chunk_id}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
                            
                            # Try to get from cache first
                            embedding = await self._get_cached_embedding(cache_key)
                            if not embedding:
                                self.logger.debug(f"Generating new embedding for text chunk {i+1}/{len(content)}...")  # Changed to debug
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
                                    "processed_at": datetime.utcnow().isoformat(),
                                    "original_id": f"{os.path.basename(file_path)}:chunk_{i}"
                                }
                            })
                        except Exception as e:
                            self.logger.error(f"Error processing text chunk: {e}")
                            continue
                    
                    if points:
                        try:
                            await self.qdrant._client.upsert(
                                collection_name="embeddings",
                                points=points
                            )
                            self.logger.info(f"Successfully processed {os.path.basename(file_path)}")  # Simplified logging
                            return True
                        except Exception as e:
                            self.logger.error(f"Error storing text chunks in Qdrant: {e}")
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
                            cache_key = f"embedding:{image_id}:{hashlib.md5(text_to_embed.encode()).hexdigest()}"
                            
                            # Try to get from cache first
                            embedding = await self._get_cached_embedding(cache_key)
                            if not embedding:
                                self.logger.debug(f"Generating new embedding for image metadata: {image_id}...")  # Changed to debug
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
                                    "processed_at": datetime.utcnow().isoformat(),
                                    "original_id": image_id
                                }
                            })
                        except Exception as e:
                            self.logger.error(f"Error processing image metadata: {e}")
                            continue
                    
                    if points:
                        try:
                            await self.qdrant._client.upsert(
                                collection_name="embeddings",
                                points=points
                            )
                            self.logger.info(f"Successfully processed {os.path.basename(file_path)}")  # Simplified logging
                            return True
                        except Exception as e:
                            self.logger.error(f"Error storing image entries in Qdrant: {e}")
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
            current_stage, _ = self.CRAWL_PHASES[self.current_phase_index]
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
            current_stage, _ = self.CRAWL_PHASES[self.current_phase_index]
            dir_path = os.path.join(self.training_data_path, current_stage, directory)
            
            if not os.path.exists(dir_path):
                return []
            
            files = []
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if not os.path.isfile(file_path):
                    continue
                    
                file_type = await self._get_file_type(file_path)
                if file_type and file_type in self.PHASE_FILE_HANDLERS[current_stage]:
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
            current_stage, _ = self.CRAWL_PHASES[self.current_phase_index]
            self.phase_progress[current_stage] = True
            
            # Move to next stage
            self.current_phase_index += 1
            if self.current_phase_index >= len(self.CRAWL_PHASES):
                # Check if all stages are completed
                if all(self.phase_progress.values()):
                    self.logger.info("All stages completed successfully")
                    return False
                else:
                    # Reset to first stage if some stages are incomplete
                    self.current_phase_index = 0
                    self.logger.info("Completed all stages, starting over")
            
            next_stage, _ = self.CRAWL_PHASES[self.current_phase_index]
            self.logger.info(f"Moving to stage: {next_stage}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error moving to next stage: {e}")
            return False

    async def crawl_next_file(self, max_retries: int = 3) -> bool:
        """Process the next file in the current stage."""
        try:
            # Get current stage
            current_stage, priority = self.CRAWL_PHASES[self.current_phase_index]
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

    async def crawl_training_data_phases(self, delay: float = 1.0):
        """Process files in phase1 > phase2 > phase3 > phase4 > phase5 order."""
        self.logger.info(f"Starting phased processing from {self.training_data_path}")
        self.is_processing = False
        self.current_phase_index = 0
        self.phase_progress = {phase: False for phase, _ in self.CRAWL_PHASES}
        while self.current_phase_index < len(self.CRAWL_PHASES):
            phase, priority = self.CRAWL_PHASES[self.current_phase_index]
            phase_path = os.path.join(self.training_data_path, phase)
            self.logger.info(f"Processing phase: {phase} (priority: {priority})")
            if not os.path.exists(phase_path):
                self.logger.warning(f"Phase directory not found: {phase_path}")
                self.phase_progress[phase] = True
                self.current_phase_index += 1
                await asyncio.sleep(delay * 2)
                continue
            files_to_process = []
            for root, _, filenames in os.walk(phase_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if file_path not in self.processed_files and file_path not in self.failed_files:
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in self.file_handlers:
                            files_to_process.append(file_path)
            if not files_to_process:
                self.logger.info(f"No more files to process in {phase}")
                self.phase_progress[phase] = True
                self.current_phase_index += 1
                await asyncio.sleep(delay * 2)
                continue
            for i in range(0, len(files_to_process), self.batch_size):
                batch = files_to_process[i:i + self.batch_size]
                self.logger.info(f"Processing batch {i//self.batch_size + 1} of {(len(files_to_process) + self.batch_size - 1)//self.batch_size}")
                for file_path in batch:
                    try:
                        self.logger.info(f"Processing file: {file_path}")
                        success = await self.process_file_phase(file_path)
                        if success:
                            self.processed_files.add(file_path)
                            self.logger.info(f"Successfully processed {file_path}")
                        else:
                            self.failed_files.add(file_path)
                            self.logger.error(f"Failed to process {file_path}")
                        await asyncio.sleep(delay)
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
                        self.failed_files.add(file_path)
                        await asyncio.sleep(delay)
                await asyncio.sleep(delay * 3)
            self.phase_progress[phase] = True
            self.current_phase_index += 1
            await asyncio.sleep(delay * 5)
    
    async def process_file_phase(self, file_path: str) -> bool:
        """Process a file based on its extension for phase-based workflow."""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return False
            ext = os.path.splitext(file_path)[1].lower()
            handler = self.file_handlers.get(ext)
            if not handler:
                self.logger.warning(f"No handler for extension {ext} in phase workflow")
                return False
            content = await handler(file_path)
            if content is None:
                self.logger.warning(f"Handler returned None for {file_path}")
                return False
            if isinstance(content, (list, tuple)) and len(content) == 0:
                self.logger.warning(f"No content extracted from {file_path}")
                return False
            if isinstance(content, bool) and not content:
                self.logger.warning(f"Handler returned False for {file_path}")
                return False
            # Use the same logic as in process_file for storing content, embeddings, etc.
            # ... (reuse the relevant logic from process_file for json/jsonl/txt/image handling) ...
            # For brevity, call the original process_file for post-processing if needed
            return await self.process_file(file_path)
        except Exception as e:
            self.logger.error(f"Error in process_file_phase: {e}")
            return False
    
    async def _process_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process text files and generate embeddings."""
        try:
            entries = []
            # Process file in chunks to avoid loading entire file into memory
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read file in chunks of 1MB
                chunk_size = 1024 * 1024
                buffer = ""
                chunk_index = 0
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        # Process remaining buffer
                        if buffer:
                            text_chunks = self.text_splitter.split_text(buffer)
                            for i, text_chunk in enumerate(text_chunks):
                                entries.append({
                                    "text": text_chunk,
                                    "source": file_path,
                                    "type": "text_chunk",
                                    "chunk_index": chunk_index + i,
                                    "total_chunks": len(text_chunks),
                                    "embedding_source": "text",
                                    "processed_at": datetime.utcnow().isoformat()
                                })
                        break
                    
                    # Add chunk to buffer
                    buffer += chunk
                    
                    # If buffer is large enough, process it
                    if len(buffer) >= chunk_size * 2:
                        text_chunks = self.text_splitter.split_text(buffer)
                        for i, text_chunk in enumerate(text_chunks):
                            entries.append({
                                "text": text_chunk,
                                "source": file_path,
                                "type": "text_chunk",
                                "chunk_index": chunk_index + i,
                                "total_chunks": len(text_chunks),
                                "embedding_source": "text",
                                "processed_at": datetime.utcnow().isoformat()
                            })
                        chunk_index += len(text_chunks)
                        buffer = ""
            
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

    async def _check_existing_points(self, collection_name: str, point_ids: List[str]) -> List[str]:
        """Check which points already exist in the collection."""
        existing_ids = []
        try:
            self.logger.info(f"Checking {len(point_ids)} points in collection {collection_name}")
            # Process points in smaller batches to avoid overwhelming the API
            batch_size = 100
            for i in range(0, len(point_ids), batch_size):
                batch_ids = point_ids[i:i + batch_size]
                try:
                    self.logger.debug(f"Checking batch {i//batch_size + 1} of {(len(point_ids) + batch_size - 1)//batch_size} ({len(batch_ids)} points)")
                    # Use retrieve instead of retrieve_points
                    result = self.qdrant._client.retrieve(
                        collection_name=collection_name,
                        ids=batch_ids,
                        with_payload=False
                    )
                    self.logger.debug(f"Retrieve result type: {type(result)}")
                    if result and hasattr(result, 'points'):
                        batch_existing = [p.id for p in result.points if p]
                        self.logger.debug(f"Found {len(batch_existing)} existing points in batch")
                        existing_ids.extend(batch_existing)
                    else:
                        self.logger.debug(f"No existing points found in batch (result: {result})")
                except Exception as e:
                    self.logger.error(f"Error checking batch of points: {str(e)}", exc_info=True)
                    continue
            self.logger.info(f"Found {len(existing_ids)} existing points out of {len(point_ids)} total")
            return existing_ids
        except Exception as e:
            self.logger.error(f"Error checking existing points: {str(e)}", exc_info=True)
            return []

    async def _store_points(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """Store points in Qdrant with proper error handling."""
        try:
            if not points:
                self.logger.info(f"No points to store in {collection_name}")
                return True
                
            self.logger.info(f"Attempting to store {len(points)} points in {collection_name}")
            # First check which points already exist
            point_ids = [p["id"] for p in points]
            self.logger.debug(f"Point IDs to check: {point_ids[:5]}... (showing first 5)")
            
            existing_ids = await self._check_existing_points(collection_name, point_ids)
            
            # Filter out existing points
            new_points = [p for p in points if p["id"] not in existing_ids]
            self.logger.info(f"Found {len(existing_ids)} existing points, {len(new_points)} new points to store")
            
            if not new_points:
                self.logger.info(f"All {len(points)} points already exist in {collection_name}")
                return True
                
            # Store new points in batches
            batch_size = 100
            for i in range(0, len(new_points), batch_size):
                batch = new_points[i:i + batch_size]
                try:
                    self.logger.debug(f"Storing batch {i//batch_size + 1} of {(len(new_points) + batch_size - 1)//batch_size} ({len(batch)} points)")
                    self.logger.debug(f"First point in batch: {batch[0] if batch else None}")
                    
                    # Use upsert instead of upsert_points
                    result = self.qdrant._client.upsert(
                        collection_name=collection_name,
                        points=batch,
                        wait=True
                    )
                    self.logger.debug(f"Upsert result type: {type(result)}")
                    self.logger.info(f"Successfully stored batch of {len(batch)} points in {collection_name}")
                except Exception as e:
                    self.logger.error(f"Error storing batch of points: {str(e)}", exc_info=True)
                    self.logger.error(f"Failed batch details - Collection: {collection_name}, Batch size: {len(batch)}, First point ID: {batch[0]['id'] if batch else None}")
                    return False
                    
            self.logger.info(f"Successfully stored all {len(new_points)} new points in {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing points in {collection_name}: {str(e)}", exc_info=True)
            return False

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

    async def _process_general_json(self, file_path: str) -> bool:
        """Process general JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            if not isinstance(content, (dict, list)):
                self.logger.error(f"Invalid JSON content in {file_path}")
                return False
            
            # Convert content to list of entries
            entries = []
            if isinstance(content, dict):
                # Handle dictionary format
                if "data" in content and isinstance(content["data"], dict):
                    # Handle structured data format with data field
                    data = content["data"]
                    if "content" in data and isinstance(data["content"], dict):
                        # Handle content object format
                        content_obj = data["content"]
                        entry = {
                            "id": content_obj.get("id", ""),
                            "text": content_obj.get("summary", ""),  # Use summary as main text
                            "metadata": {
                                "source": file_path,
                                "type": "structured_content",
                                "labels": content.get("metadata", {}).get("labels", []),
                                "domain": content.get("metadata", {}).get("domain", ""),
                                "training_config": content.get("training_config", {}),
                                "position": content_obj.get("position", {}),
                                "index": content_obj.get("index", 0)
                            }
                        }
                        entries.append(entry)
                    else:
                        # Handle other data formats
                        for key, value in data.items():
                            if isinstance(value, dict):
                                entries.append({
                                    "id": key,
                                    "text": value.get("summary", "") or str(value),
                                    "metadata": {
                                        "source": file_path,
                                        "type": "data_entry",
                                        "original_key": key
                                    }
                                })
                else:
                    # Handle simple key-value dictionary
                    for key, value in content.items():
                        entries.append({
                            "id": key,
                            "text": str(value),
                            "metadata": {
                                "source": file_path,
                                "type": "key_value",
                                "original_key": key
                            }
                        })
            else:
                # Handle list format
                for item in content:
                    if isinstance(item, dict):
                        # Try to extract meaningful content
                        text = item.get("summary", "") or item.get("text", "") or item.get("content", "") or str(item)
                        entry = {
                            "id": item.get("id", str(uuid.uuid4())),
                            "text": text,
                            "metadata": {
                                "source": file_path,
                                "type": "list_entry",
                                **{k: v for k, v in item.items() if k not in ["id", "summary", "text", "content"]}
                            }
                        }
                        entries.append(entry)
            
            if not entries:
                self.logger.warning(f"No entries extracted from {file_path}")
                return False
            
            # Process entries in batches
            points = []
            for entry in entries:
                point = await self._process_json_entry(entry, "json", file_path)
                if point:
                    points.append(point)
            
            if points:
                try:
                    await self.qdrant._client.upsert(
                        collection_name="embeddings",
                        points=points
                    )
                    self.logger.info(f"Stored {len(points)} entries from {file_path}")
                    return True
                except Exception as e:
                    self.logger.error(f"Error storing entries in Qdrant: {e}")
                    return False
            
            self.logger.warning(f"No points generated from {file_path}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing JSON file {file_path}: {e}")
            return False

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

    async def _process_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process text files and generate embeddings."""
        try:
            entries = []
            # Process file in chunks to avoid loading entire file into memory
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read file in chunks of 1MB
                chunk_size = 1024 * 1024
                buffer = ""
                chunk_index = 0
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        # Process remaining buffer
                        if buffer:
                            text_chunks = self.text_splitter.split_text(buffer)
                            for i, text_chunk in enumerate(text_chunks):
                                entries.append({
                                    "text": text_chunk,
                                    "source": file_path,
                                    "type": "text_chunk",
                                    "chunk_index": chunk_index + i,
                                    "total_chunks": len(text_chunks),
                                    "embedding_source": "text",
                                    "processed_at": datetime.utcnow().isoformat()
                                })
                        break
                    
                    # Add chunk to buffer
                    buffer += chunk
                    
                    # If buffer is large enough, process it
                    if len(buffer) >= chunk_size * 2:
                        text_chunks = self.text_splitter.split_text(buffer)
                        for i, text_chunk in enumerate(text_chunks):
                            entries.append({
                                "text": text_chunk,
                                "source": file_path,
                                "type": "text_chunk",
                                "chunk_index": chunk_index + i,
                                "total_chunks": len(text_chunks),
                                "embedding_source": "text",
                                "processed_at": datetime.utcnow().isoformat()
                            })
                        chunk_index += len(text_chunks)
                        buffer = ""
            
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