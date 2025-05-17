"""Crawler service for fetching and processing content from multiple sources."""
import asyncio
import logging
import time
import urllib.parse
import sys
import traceback
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime
import os
import json
import tempfile
import shutil
import git
from pathlib import Path
import mimetypes
import PyPDF2
import io

import backoff
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge
import aiohttp
from bs4 import BeautifulSoup

# Update imports to use relative paths
from ..common.config import (
    REDIS_URL,
    QDRANT_URL,
    GRAPH_API_URL,
    CONCEPT_DICT_URL,
    EMBEDDING_MODEL,
    MAX_RETRIES,
    RETRY_DELAY,
    CACHE_TTL,
    PRIORITY_WEIGHTS
)
from ..common.logging import setup_logging
from ..common.redis import redis_client
from ..common.qdrant import qdrant_client
from ..common.embeddings import get_embedding_model, encode_text, encode_batch

import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import scancode.api as scancode
import uuid
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import core modules from the correct location
from .core.pdf_processor import PDFProcessor
from .core.redis_client import redis_client as app_redis_client
from .core.qdrant_client import qdrant_client as app_qdrant_client
from .core.logging import setup_logging as app_setup_logging
from .core.models import CrawlItem, CrawlResult

# Setup logging with more verbose output
logger = setup_logging(__name__, level=logging.DEBUG)

# Add a top-level exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    """Log unhandled exceptions."""
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)  # Call the default handler

sys.excepthook = handle_exception

# Initialize FastAPI app with more detailed startup logging
app = FastAPI(
    title="Crawler Service",
    on_startup=[lambda: logger.info("FastAPI application starting up...")],
    on_shutdown=[lambda: logger.info("FastAPI application shutting down...")]
)

# Prometheus metrics
CRAWL_REQUESTS = Counter('crawl_requests_total', 'Total crawl requests', ['source_type'])
CRAWL_LATENCY = Histogram('crawl_latency_seconds', 'Time spent crawling', ['source_type'])
CRAWL_ERRORS = Counter('crawl_errors_total', 'Total crawl errors', ['source_type', 'error_type'])
ACTIVE_CRAWLS = Gauge('active_crawls', 'Number of active crawls', ['source_type'])
PROCESSED_ITEMS = Counter('processed_items_total', 'Total items processed', ['source_type', 'status'])

# Update priority weights to match specification
PRIORITY_WEIGHTS = {
    "dict": 1.2,    # Dictionary concepts (highest priority)
    "git": 1.0,     # Git repositories
    "pdf": 0.8,     # PDF documents
    "url": 0.6,     # Web URLs
    "graph": 0.4    # Internal system/gap topics
}

class CrawlItem(BaseModel):
    """Model for crawl items."""
    source_type: str = Field(..., description="Type of source (git, pdf, url, graph)")
    source: str = Field(..., description="URL, file path, or graph node ID")
    priority: float = Field(default=0.5, description="Priority of the crawl item")
    metadata: Optional[Union[Dict[str, Any], str]] = Field(default=None, description="Additional metadata")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of creation")
    queued_at: float = Field(default_factory=time.time, description="When the item was queued")
    attempts: int = Field(default=0, description="Number of processing attempts")

    @validator('metadata', pre=True)
    def parse_metadata(cls, v):
        """Parse metadata from JSON string if needed."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return v

class CrawlResult(BaseModel):
    """Model for crawl results."""
    item_id: str = Field(..., description="Unique identifier for the crawl result")
    source_type: str = Field(..., description="Type of source that was crawled")
    content: Any = Field(..., description="Crawled content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Content embedding")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of processing")

# Update imports to use new langchain_community packages
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http.models import VectorParams, Distance

class DictionaryCrawler:
    """Handles crawling and processing of dictionary concepts with legal compliance."""
    def __init__(self, ollama_url: str = "http://ollama:11434", model_name: str = "nomic-embed-text"):
        self.ollama = OllamaEmbeddings(
            base_url=ollama_url,
            model=model_name
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.logger = setup_logging(__name__, level=logging.DEBUG)
        self.redis = app_redis_client
        self.qdrant = app_qdrant_client
        
        # Initialize legal compliance tools
        self.nlp = spacy.load("en_core_web_lg")
        self.pii_analyzer = AnalyzerEngine()
        self.pii_anonymizer = AnonymizerEngine()
        
        # Allowed licenses (configurable)
        self.allowed_licenses = {
            "MIT", "Apache-2.0", "CC-BY-4.0", "CC0-1.0", "BSD-3-Clause"
        }
        
        # Export control keywords
        self.export_control_keywords = {
            "crypto", "encryption", "aes", "rsa", "security", "cipher"
        }

    async def check_license(self, content: str, source: str) -> Tuple[bool, str, float]:
        """Check license compliance using scancode-toolkit."""
        try:
            # Scan for licenses
            scan_results = scancode.scan_file(
                content=content.encode('utf-8'),
                filename=source,
                processes=1
            )
            
            # Extract license info
            licenses = scan_results.get('licenses', [])
            if not licenses:
                return False, "NO_LICENSE", 0.0
                
            # Get highest confidence license
            best_license = max(licenses, key=lambda x: x.get('score', 0))
            license_id = best_license.get('spdx_license_key', '')
            confidence = best_license.get('score', 0)
            
            # Check if license is allowed
            is_allowed = license_id in self.allowed_licenses
            
            return is_allowed, license_id, confidence
            
        except Exception as e:
            self.logger.error(f"License check failed: {str(e)}")
            return False, "ERROR", 0.0

    async def check_pii(self, text: str) -> Tuple[bool, str]:
        """Check for PII using Presidio."""
        try:
            # Analyze text for PII
            results = self.pii_analyzer.analyze(text=text, language='en')
            
            if not results:
                return True, text
                
            # Anonymize PII
            anonymized = self.pii_anonymizer.anonymize(
                text=text,
                analyzer_results=results
            )
            
            # Return anonymized text and whether it contained PII
            return False, anonymized.text
            
        except Exception as e:
            self.logger.error(f"PII check failed: {str(e)}")
            return False, text

    async def check_export_control(self, text: str) -> bool:
        """Check if content might be export controlled."""
        try:
            # Simple keyword-based check
            text_lower = text.lower()
            return any(keyword in text_lower for keyword in self.export_control_keywords)
            
        except Exception as e:
            self.logger.error(f"Export control check failed: {str(e)}")
            return True  # Conservative approach

    async def load_lexicon(self, lexicon_path: str) -> List[Dict[str, Any]]:
        """Load dictionary entries from a lexicon file."""
        try:
            entries = []
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
            return entries
        except Exception as e:
            self.logger.error(f"Error loading lexicon {lexicon_path}: {str(e)}")
            return []

    async def process_concept(self, concept: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single concept with legal compliance checks."""
        try:
            # Skip if processed recently
            if concept.get('processed_at', 0) > time.time() - 86400:
                self.logger.debug(f"Skipping recently processed concept: {concept.get('term')}")
                return None

            # Prepare concept text
            text = f"Term: {concept.get('term', '')}\nDefinition: {concept.get('definition', '')}"
            if concept.get('examples'):
                text += f"\nExamples: {concept.get('examples')}"
            if concept.get('related_terms'):
                text += f"\nRelated Terms: {', '.join(concept.get('related_terms', []))}"
            
            # Legal compliance checks
            license_ok, license_id, license_conf = await self.check_license(text, concept.get('source', ''))
            if not license_ok:
                self.logger.warning(f"License {license_id} not allowed for concept {concept.get('term')}")
                return None
                
            pii_ok, anonymized_text = await self.check_pii(text)
            if not pii_ok:
                self.logger.warning(f"PII found in concept {concept.get('term')}, using anonymized version")
                text = anonymized_text
                
            export_controlled = await self.check_export_control(text)
            if export_controlled:
                self.logger.warning(f"Export controlled content detected in concept {concept.get('term')}")
                # Flag for manual review instead of rejecting
                concept['needs_review'] = True
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Generate embeddings
            embeddings = []
            for chunk in chunks:
                try:
                    embedding = await self.ollama.aembed_query(chunk)
                    embeddings.append(embedding)
                except Exception as e:
                    self.logger.error(f"Error generating embedding for chunk: {str(e)}")
                    continue
            
            if not embeddings:
                return None
                
            # Prepare metadata with legal info
            metadata = {
                'term': concept.get('term'),
                'definition': concept.get('definition'),
                'chunks': chunks,
                'embeddings': [e.tolist() for e in embeddings],
                'processed_at': time.time(),
                'license': {
                    'id': license_id,
                    'confidence': license_conf
                },
                'pii_detected': not pii_ok,
                'export_controlled': export_controlled,
                'needs_review': export_controlled,
                'source': concept.get('source', ''),
                'fingerprint': str(uuid.uuid4()),
                'audit_trail': {
                    'processed_at': datetime.utcnow().isoformat(),
                    'processor': 'dictionary_crawler',
                    'version': '1.0'
                }
            }
            
            # Store in Qdrant
            try:
                await self.qdrant.upsert(
                    collection_name="dictionary_concepts",
                    points=[{
                        'id': f"dict_{concept.get('id')}",
                        'vector': embeddings[0],
                        'payload': metadata
                    }]
                )
            except Exception as e:
                self.logger.error(f"Error storing concept in Qdrant: {str(e)}")
                return None

            return metadata
            
        except Exception as e:
            self.logger.error(f"Error processing concept: {str(e)}", exc_info=True)
            return None

    async def crawl_dictionary(self):
        """Crawl and process concepts from multiple dictionary sources."""
        try:
            # Load from multiple sources
            lexicon_paths = [
                "/app/training_data/lexicons/wiktionary.jsonl",
                "/app/training_data/lexicons/wordnet.jsonl",
                "/app/training_data/lexicons/gcide.jsonl"
            ]
            
            all_concepts = []
            for path in lexicon_paths:
                if os.path.exists(path):
                    concepts = await self.load_lexicon(path)
                    all_concepts.extend(concepts)
                    self.logger.info(f"Loaded {len(concepts)} concepts from {path}")
            
            # Process concepts in batches
            batch_size = 10
            for i in range(0, len(all_concepts), batch_size):
                batch = all_concepts[i:i + batch_size]
                tasks = []
                for concept in batch:
                    tasks.append(self.process_concept(concept))
                
                # Process batch
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for concept, result in zip(batch, results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Error processing concept {concept.get('term')}: {str(result)}")
                            continue
                        if result:
                            # Update concept in dictionary if it exists
                            if concept.get('id'):
                                try:
                                    async with httpx.AsyncClient() as client:
                                        response = await client.put(
                                            f"{CONCEPT_DICT_URL}/concepts/{concept['id']}",
                                            json=result
                                        )
                                        if response.status_code != 200:
                                            self.logger.error(f"Failed to update concept {concept['term']}: {response.status_code}")
                                except Exception as e:
                                    self.logger.error(f"Error updating concept {concept.get('term')}: {str(e)}")

            self.logger.info(f"Processed {len(all_concepts)} dictionary concepts")
            
        except Exception as e:
            self.logger.error(f"Error in dictionary crawling: {str(e)}", exc_info=True)

class TrainingDataCrawler:
    """Handles crawling and processing of training data files."""
    def __init__(self):
        # Use nomic-embed-text model which produces 768-dim vectors
        self.ollama = OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            model=os.getenv("OLLAMA_MODEL", "nomic-embed-text")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.logger = setup_logging(__name__, level=logging.DEBUG)
        self.redis = app_redis_client
        self.qdrant = app_qdrant_client
        
        # Initialize training data paths
        self.training_data_path = os.getenv("TRAINING_DATA_PATH", "/app/training_data")
        self.git_training_path = os.getenv("GIT_TRAINING_PATH", os.path.join(self.training_data_path, "git"))
        self.pdf_training_path = os.getenv("PDF_TRAINING_PATH", self.training_data_path)
        self.graph_training_path = os.getenv("GRAPH_TRAINING_PATH", os.path.join(self.training_data_path, "graph (1).json"))
        
        # Initialize PDF embeddings collection with 768 dimensions for nomic-embed-text
        self.pdf_collection = "pdf_embeddings_768d"
        self._initialized = False
        self.processed_files = set()
        self.failed_files = set()
        self._current_batch = []
        self._batch_size = 10  # Process files in small batches
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = int(os.getenv("RETRY_DELAY", "5"))

    async def initialize(self):
        """Initialize the crawler asynchronously."""
        if not self._initialized:
            try:
                # Initialize Qdrant collection with 768 dimensions for nomic-embed-text
                await self.qdrant.init_collection(
                    self.pdf_collection,
                    vector_size=768
                )
                self._initialized = True
                self.logger.info("TrainingDataCrawler initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize TrainingDataCrawler: {str(e)}")
                raise

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=30
    )
    async def process_file_directly(self, file_path: str, file_type: str):
        """Process a file directly with retry logic."""
        try:
            if file_type == 'pdf':
                # Process PDF file with retry
                pdf_processor = PDFProcessor()
                chunks = await pdf_processor.process_pdf(file_path)
                
                if not chunks:
                    self.logger.warning(f"No content extracted from PDF: {file_path}")
                    return False
                
                # Process chunks in batches
                for i in range(0, len(chunks), self._batch_size):
                    batch = chunks[i:i + self._batch_size]
                    vectors = []
                    
                    for chunk in batch:
                        try:
                            # Generate embedding with retry
                            embedding = await self.ollama.aembed_query(chunk['text'])
                            vectors.append({
                                'id': str(uuid.uuid4()),
                                'vector': embedding,
                                'payload': {
                                    'text': chunk['text'],
                                    'metadata': {
                                        'source': file_path,
                                        'page': chunk.get('page', 0),
                                        'processed_at': datetime.utcnow().isoformat()
                                    }
                                }
                            })
                        except Exception as e:
                            self.logger.error(f"Error generating embedding for chunk: {str(e)}")
                            continue
                    
                    if vectors:
                        try:
                            # Upsert vectors with retry
                            await self.qdrant.upsert(
                                collection_name=self.pdf_collection,
                                points=vectors
                            )
                            
                            # Publish to ingest stream
                            for vec in vectors:
                                await self.redis.xadd(
                                    "ingest.pdf",
                                    {
                                        "file_path": file_path,
                                        "vec_id": vec['id'],
                                        "metadata": json.dumps(vec['payload']['metadata'])
                                    }
                                )
                        except Exception as e:
                            self.logger.error(f"Error upserting vectors: {str(e)}")
                            raise
                
                return True
                
            elif file_type == 'text':
                # Process text file with retry
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split text into chunks
                    chunks = self.text_splitter.split_text(content)
                    
                    # Process chunks in batches
                    for i in range(0, len(chunks), self._batch_size):
                        batch = chunks[i:i + self._batch_size]
                        vectors = []
                        
                        for chunk in batch:
                            try:
                                # Generate embedding with retry
                                embedding = await self.ollama.aembed_query(chunk)
                                vectors.append({
                                    'id': str(uuid.uuid4()),
                                    'vector': embedding,
                                    'payload': {
                                        'text': chunk,
                                        'metadata': {
                                            'source': file_path,
                                            'processed_at': datetime.utcnow().isoformat()
                                        }
                                    }
                                })
                            except Exception as e:
                                self.logger.error(f"Error generating embedding for chunk: {str(e)}")
                                continue
                        
                        if vectors:
                            try:
                                # Upsert vectors with retry
                                await self.qdrant.upsert(
                                    collection_name=self.pdf_collection,
                                    points=vectors
                                )
                                
                                # Publish to ingest stream
                                for vec in vectors:
                                    await self.redis.xadd(
                                        "ingest.crawl",
                                        {
                                            "file_path": file_path,
                                            "vec_id": vec['id'],
                                            "metadata": json.dumps(vec['payload']['metadata'])
                                        }
                                    )
                            except Exception as e:
                                self.logger.error(f"Error upserting vectors: {str(e)}")
                                raise
                    
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error processing text file {file_path}: {str(e)}")
                    raise
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    async def crawl_training_data_incremental(self, delay: float = 1.0):
        """Incrementally crawl and process files in small batches."""
        self.logger.info(f"Starting incremental training data ingestion from {self.training_data_path}")
        
        while True:
            try:
                # Process files in small batches
                batch_files = []
                for root, _, files in os.walk(self.training_data_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file_path not in self.processed_files and file_path not in self.failed_files:
                            batch_files.append(file_path)
                            if len(batch_files) >= self._batch_size:
                                break
                    if len(batch_files) >= self._batch_size:
                        break
                
                if not batch_files:
                    self.logger.info("No new files to process. Sleeping before next check...")
                    await asyncio.sleep(10)
                    continue
                
                # Process the batch
                for file_path in batch_files:
                    try:
                        mime_type, _ = mimetypes.guess_type(file_path)
                        if mime_type == 'application/pdf':
                            await self.process_file_directly(file_path, 'pdf')
                        elif mime_type and mime_type.startswith('text/'):
                            await self.process_file_directly(file_path, 'text')
                        else:
                            self.logger.warning(f"Unsupported file type for {file_path}: {mime_type}")
                            self.failed_files.add(file_path)
                            continue
                        
                        self.processed_files.add(file_path)
                        self.logger.info(f"Successfully processed {file_path}")
                        
                        # Add delay between files
                        await asyncio.sleep(delay)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {str(e)}")
                        self.failed_files.add(file_path)
                        continue
                
                # Log progress
                self.logger.info(f"Processed batch. Total processed: {len(self.processed_files)}, Failed: {len(self.failed_files)}")
                
            except Exception as e:
                self.logger.error(f"Error in incremental crawl: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

class Crawler:
    def __init__(self):
        self.redis = app_redis_client
        self.qdrant = app_qdrant_client
        # Use nomic-embed-text consistently across all crawlers
        self.embedding_model = OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            model=os.getenv("OLLAMA_MODEL", "nomic-embed-text")
        )
        self.active_crawls: Set[str] = set()
        self.processing_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        self._worker_running = False
        self.dictionary_crawler = DictionaryCrawler()
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "nomic-embed-text")  # Update default model
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = int(os.getenv("RETRY_DELAY", "5"))
        self.training_data_crawler = TrainingDataCrawler()
        self.logger = setup_logging(__name__, level=logging.DEBUG)

    async def initialize(self):
        """Initialize connections and resources."""
        if not self._initialized:
            try:
                await self.redis.connect()
                await self.qdrant.connect()
                self._initialized = True
                self.logger.info("Crawler initialized with Redis and Qdrant connections")
            except Exception as e:
                self.logger.error(f"Failed to initialize crawler: {str(e)}")
                raise

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._initialized:
            await self.redis.close()
            self._initialized = False

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, httpx.HTTPError),
        max_tries=MAX_RETRIES,
        max_time=30
    )
    async def fetch_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch content from a URL with retries."""
        session = await self.get_session()
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    return {
                        'url': url,
                        'title': soup.title.string if soup.title else url,
                        'content': soup.get_text(separator=' ', strip=True),
                        'links': [a.get('href') for a in soup.find_all('a', href=True)],
                        'timestamp': time.time()
                    }
                logger.warning(f"Failed to fetch URL {url}: status {response.status}")
                return None
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            raise

    @backoff.on_exception(
        backoff.expo,
        (git.GitCommandError, OSError),
        max_tries=MAX_RETRIES,
        max_time=30
    )
    async def fetch_git_repo(self, repo_url: str, ref: str = "HEAD") -> Optional[Dict[str, Any]]:
        """Clone and process a git repository with retries."""
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            repo = git.Repo.clone_from(repo_url, temp_dir, branch=ref)
            content = []
            
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(('.md', '.txt', '.py', '.rst')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content.append({
                                    'path': file_path,
                                    'content': f.read(),
                                    'type': 'text'
                                })
                        except UnicodeDecodeError:
                            logger.warning(f"Could not decode file {file_path}")
                            continue
                        except Exception as e:
                            logger.error(f"Error reading file {file_path}: {str(e)}")
                            continue
            
            return {
                'repo_url': repo_url,
                'ref': ref,
                'content': content,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error processing git repo {repo_url}: {str(e)}")
            raise
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    async def process_item(self, item: CrawlItem) -> Optional[CrawlResult]:
        """Process a single crawl item based on its source type."""
        try:
            ACTIVE_CRAWLS.labels(source_type=item.source_type).inc()
            start_time = time.time()
            
            # Fetch content based on source type
            content = None
            if item.source_type == 'git':
                content = await self.fetch_git_repo(item.source)
            elif item.source_type == 'url':
                content = await self.fetch_url(item.source)
            elif item.source_type == 'pdf':
                # PDF processing will be implemented separately
                logger.warning("PDF processing not implemented yet")
                content = None
            elif item.source_type == 'graph':
                # Graph node processing will be implemented separately
                logger.warning("Graph node processing not implemented yet")
                content = None
            elif item.source_type == 'dictionary':
                # Dictionary concepts are processed by the dictionary crawler
                content = await self.dictionary_crawler.process_concept(item.metadata)
            else:
                raise ValueError(f"Unknown source type: {item.source_type}")
                
            if not content:
                CRAWL_ERRORS.labels(source_type=item.source_type, error_type='fetch_failed').inc()
                return None
                
            # Generate embedding
            embedding = None
            if isinstance(content, dict) and 'content' in content:
                text_content = content['content']
                if isinstance(text_content, list):
                    text_content = ' '.join(item['content'] for item in text_content)
                try:
                    embedding = await encode_text(text_content)
                except Exception as e:
                    logger.error(f"Error generating embedding: {str(e)}")
                    CRAWL_ERRORS.labels(source_type=item.source_type, error_type='embedding_failed').inc()
                    return None

            # Create result
            result = CrawlResult(
                item_id=f"{item.source_type}:{hash(item.source)}",
                source_type=item.source_type,
                content=content,
                metadata=item.metadata or {},
                embedding=embedding.tolist() if embedding is not None else None
            )
            
            # Record metrics
            CRAWL_LATENCY.labels(source_type=item.source_type).observe(time.time() - start_time)
            PROCESSED_ITEMS.labels(source_type=item.source_type, status='success').inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {item.source_type} item {item.source}: {str(e)}")
            CRAWL_ERRORS.labels(source_type=item.source_type, error_type=type(e).__name__).inc()
            PROCESSED_ITEMS.labels(source_type=item.source_type, status='error').inc()
            return None
        finally:
            ACTIVE_CRAWLS.labels(source_type=item.source_type).dec()

    async def process_queue_item(self, msg_id: str, data: Dict[str, Any]) -> bool:
        """Process a single queue item with retry logic."""
        try:
            # Parse the crawl item
            item = CrawlItem(**{k: v for k, v in data.items() if k not in ['queued_at', 'attempts']})
            attempts = data.get('attempts', 0)
            
            if attempts >= self.max_retries:
                logger.error(f"Max retries exceeded for {item.source_type} item {item.source}")
                await self.redis.xadd('crawl_dead_letter', {
                    **data,
                    'error': 'max_retries_exceeded',
                    'last_attempt': time.time()
                })
                return True

            # Process the item
            result = await self.process_item(item)
            
            if result:
                # Store result in Qdrant if we have embeddings
                if result.embedding:
                    try:
                        await self.qdrant.upsert(
                            collection_name="crawl_results",
                            points=[{
                                'id': result.item_id,
                                'vector': result.embedding,
                                'payload': result.dict()
                            }]
                        )
                    except Exception as e:
                        logger.error(f"Error storing in Qdrant: {str(e)}")
                        raise

                # Publish to results stream
                await self.redis.xadd('crawl_results', result.dict())
                return True
            else:
                # Increment attempt count and requeue with backoff
                data['attempts'] = attempts + 1
                data['last_error'] = time.time()
                data['next_retry'] = time.time() + (self.retry_delay * (2 ** attempts))
                
                # Requeue with updated metadata
                await self.redis.xadd('crawl_queue', data)
                return True

        except Exception as e:
            logger.error(f"Error processing queue item: {str(e)}", exc_info=True)
            # Don't requeue on critical errors
            if isinstance(e, (ValueError, KeyError)):
                await self.redis.xadd('crawl_dead_letter', {
                    **data,
                    'error': str(e),
                    'last_attempt': time.time()
                })
                return True
            return False

    async def crawl_worker(self):
        """Background worker that processes items from the queue."""
        if self._worker_running:
            self.logger.warning("Crawl worker is already running")
            return

        self._worker_running = True
        self.logger.info("Starting crawl worker...")
        
        try:
            while self._worker_running:
                try:
                    # Get items from Redis stream with a timeout
                    items = await asyncio.wait_for(
                        self.redis.xread(['crawl_queue'], count=10),
                        timeout=5.0  # 5 second timeout
                    )
                    
                    if not items:
                        await asyncio.sleep(1)
                        continue

                    for stream, messages in items:
                        for msg_id, data in messages:
                            try:
                                # Check if item is ready for retry
                                next_retry = data.get('next_retry', 0)
                                if next_retry > time.time():
                                    continue

                                # Process the item
                                success = await self.process_queue_item(msg_id, data)
                                
                                if success:
                                    # Acknowledge the message
                                    await self.redis.xack('crawl_queue', 'workers', msg_id)
                                else:
                                    # Leave message in queue for retry
                                    await asyncio.sleep(self.retry_delay)
                                    
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

    async def crawl_dictionary(self):
        """Crawl and process concepts from the dictionary."""
        try:
            # Get concepts from dictionary
            asyncio.create_task(self.dictionary_crawler.crawl_dictionary())
            logger.info("Dictionary crawler started")
            
        except Exception as e:
            logger.error(f"Error in dictionary crawling: {str(e)}", exc_info=True)

    async def start(self):
        """Start the crawler service."""
        try:
            # Ensure Redis and Qdrant are connected
            await self.initialize()
            
            # Start single background worker for queue processing
            if not self._worker_running:
                self.logger.info("Starting crawler background worker...")
                asyncio.create_task(self.crawl_worker())
                self.logger.info("Crawler worker started")
            
            # Start dictionary crawler
            self.logger.info("Starting dictionary crawler...")
            asyncio.create_task(self.crawl_dictionary())
            self.logger.info("Dictionary crawler started")
            
        except Exception as e:
            self.logger.error(f"Error starting crawler: {str(e)}", exc_info=True)
            raise

    async def stop(self):
        """Stop the crawler service."""
        try:
            self._worker_running = False
            if self._session and not self._session.closed:
                await self._session.close()
            if self._initialized:
                await self.redis.close()
                self._initialized = False
            self.logger.info("Crawler service stopped")
        except Exception as e:
            self.logger.error(f"Error stopping crawler: {str(e)}", exc_info=True)
            raise

# Global crawler instance - moved to startup_event
crawler = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global crawler
    try:
        logger.info("Starting crawler service initialization...")
        
        # Initialize crawler
        logger.info("Initializing crawler...")
        crawler = Crawler()
        await crawler.initialize()
        logger.info("Crawler initialized successfully")
        
        # Initialize training data crawler
        logger.info("Initializing training data crawler...")
        await crawler.training_data_crawler.initialize()
        logger.info("Training data crawler initialized successfully")
        
        # Start crawler service
        logger.info("Starting crawler service...")
        await crawler.start()
        logger.info("Crawler service started")
        
        # Start incremental training data crawl as a background task
        async def crawl_training_data_bg():
            try:
                # Log initial stream counts
                ingest_pdf_count = await crawler.training_data_crawler.redis.xlen("ingest.pdf")
                ingest_crawl_count = await crawler.training_data_crawler.redis.xlen("ingest.crawl")
                logger.info(f"[BEFORE] ingest.pdf stream count: {ingest_pdf_count}")
                logger.info(f"[BEFORE] ingest.crawl stream count: {ingest_crawl_count}")
                
                # Start incremental crawl with a 2 second delay between files
                logger.info("Starting incremental training data processing (background)...")
                await crawler.training_data_crawler.crawl_training_data_incremental(delay=2.0)
            except Exception as e:
                logger.error(f"Error in training data background task: {str(e)}", exc_info=True)
                # Don't raise here, allow service to continue running
        
        # Create background task for incremental crawl
        asyncio.create_task(crawl_training_data_bg())
        
        logger.info("Crawler service initialization completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    try:
        logger.info("Starting crawler service shutdown...")
        if crawler is not None:
            await crawler.stop()
            logger.info("Crawler resources cleaned up successfully")
        logger.info("Crawler service shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during crawler shutdown: {str(e)}", exc_info=True)
        raise

@app.post("/crawl")
async def add_crawl_item(item: CrawlItem, background_tasks: BackgroundTasks):
    """Add a new item to the crawl queue."""
    try:
        logger.info(f"Received new crawl request: type={item.source_type}, source={item.source}")
        # Ensure Redis is connected
        await crawler.initialize()
        
        # Set priority based on source type from config
        item.priority = PRIORITY_WEIGHTS.get(item.source_type, 0.5)
        logger.debug(f"Set priority for {item.source_type} item to {item.priority}")
        
        # Add to Redis stream with metadata
        metadata = {
            **item.dict(),
            'queued_at': time.time(),
            'attempts': 0
        }
        
        logger.debug(f"Adding item to Redis crawl queue: type={item.source_type}, source={item.source}")
        await app_redis_client.xadd('crawl_queue', metadata)
        logger.info(f"Successfully queued crawl item: type={item.source_type}, source={item.source}")
        
        return {"status": "queued", "item": item}
    except Exception as e:
        logger.error(f"Error adding crawl item: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        logger.debug("Health check requested")
        # Check Redis connection
        if crawler is None or not crawler._initialized:
            logger.warning("Health check failed: crawler not initialized")
            raise HTTPException(status_code=503, detail="Crawler not initialized")
        
        # Check Redis health
        await app_redis_client.ping()
        logger.debug("Health check passed")
        return {"status": "healthy", "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Metrics endpoint."""
    return {
        "crawl_requests": {k: v._value.get() for k, v in CRAWL_REQUESTS._metrics.items()},
        "crawl_errors": {k: v._value.get() for k, v in CRAWL_ERRORS._metrics.items()},
        "active_crawls": {k: v._value.get() for k, v in ACTIVE_CRAWLS._metrics.items()},
        "processed_items": {k: v._value.get() for k, v in PROCESSED_ITEMS._metrics.items()}
    } 