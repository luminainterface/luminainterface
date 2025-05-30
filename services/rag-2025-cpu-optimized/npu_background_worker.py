"""
NPU Background Worker for Enhanced RAG 2025
==========================================

Background processing system that leverages NPU acceleration for:
- Continuous knowledge base crawling and updating
- Document preprocessing and embedding generation
- Background semantic analysis and knowledge graph building
- Quality assessment and content curation
- Training data preparation for LoRA adapters

This worker operates independently from the production CPU-optimized RAG system,
focusing on background tasks that enhance the overall knowledge base quality.
"""

import asyncio
import logging
import time
import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import redis
import numpy as np
from collections import deque

# NPU and Hardware Acceleration
import torch
import torch.nn.functional as F

# Background Processing
import aiofiles
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

# Embeddings and Processing - Fix for updated huggingface_hub API
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    # Fallback for dependency issues
    print(f"Warning: sentence_transformers import failed: {e}")
    print("Falling back to basic embedding functionality")
    SentenceTransformer = None

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

@dataclass
class BackgroundTask:
    """Represents a background processing task"""
    task_id: str
    task_type: str  # crawl, embed, analyze, quality_check, etc.
    priority: int  # 1-10, higher = more priority
    data: Dict[str, Any]
    created_at: datetime
    scheduled_for: datetime
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"  # pending, processing, completed, failed
    npu_optimized: bool = True

@dataclass 
class CrawlResult:
    """Results from background crawling"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    embeddings: List[float]
    quality_score: float
    concepts_extracted: List[str]
    processing_time: float
    npu_acceleration_used: bool

@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    relevance_score: float
    coherence_score: float
    completeness_score: float
    freshness_score: float
    source_reliability: float
    overall_quality: float

class NPUHardwareManager:
    """Manages NPU hardware detection and optimization"""
    
    def __init__(self):
        self.npu_available = False
        self.gpu_available = False
        self.device = "cpu"
        self.optimization_enabled = False
        
    async def initialize(self):
        """Initialize and detect available hardware"""
        logger.info("🔍 Detecting NPU and GPU hardware...")
        
        # Check for CUDA/GPU
        if torch.cuda.is_available():
            self.gpu_available = True
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True
            logger.info(f"✅ GPU detected: {torch.cuda.get_device_name()}")
            
            # GTX 1080 specific optimizations
            if "GTX 1080" in torch.cuda.get_device_name():
                torch.cuda.set_per_process_memory_fraction(0.8)
                logger.info("🔧 GTX 1080 optimizations applied")
        
        # Check for Intel Extension for PyTorch (NPU)
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
                self.npu_available = True
                self.device = "xpu"
                logger.info("✅ Intel NPU detected and available")
            else:
                logger.info("ℹ️ Intel Extension found but NPU not available")
        except ImportError:
            logger.info("ℹ️ Intel Extension for PyTorch not installed")
        
        # Set optimization status
        self.optimization_enabled = self.npu_available or self.gpu_available
        logger.info(f"🚀 Hardware acceleration: {'Enabled' if self.optimization_enabled else 'CPU Only'}")
        logger.info(f"   Device: {self.device}")
        
        return self.optimization_enabled

class NPUBackgroundWorker:
    """Main NPU Background Worker for RAG system enhancement"""
    
    def __init__(self, 
                 redis_url: str = "redis://redis:6379",
                 qdrant_url: str = "http://qdrant:6333",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "rag_enhanced_knowledge",
                 worker_id: str = None):
        
        self.worker_id = worker_id or f"npu_worker_{int(time.time())}"
        self.redis_url = redis_url
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        # Hardware management
        self.hardware_manager = NPUHardwareManager()
        
        # Task processing
        self.task_queue = deque()
        self.active_tasks: Dict[str, BackgroundTask] = {}
        self.completed_tasks = deque(maxlen=1000)
        self.failed_tasks = deque(maxlen=100)
        
        # Performance metrics
        self.metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "npu_accelerated_tasks": 0,
            "average_quality_score": 0.0,
            "embeddings_generated": 0,
            "documents_crawled": 0,
            "concepts_extracted": 0
        }
        
        # Configuration
        self.max_concurrent_tasks = 5
        self.batch_size = 10
        self.quality_threshold = 0.7
        self.crawl_delay = 1.0  # Delay between crawl requests
        
        # Clients (initialized in startup)
        self.redis_client = None
        self.qdrant_client = None
        self.embedding_model = None
        self.session = None
        
        # Processing state
        self.is_running = False
        self.worker_threads = []
        
    async def initialize(self):
        """Initialize worker components"""
        logger.info("🔧 Initializing NPU Background Worker...")
        
        try:
            # Initialize hardware
            await self.hardware_manager.initialize()
            
            # Initialize Redis connection with improved URL parsing and authentication
            logger.info(f"🔗 Connecting to Redis: {self.redis_url}")
            
            # Get password from environment or use default
            redis_password = os.getenv('REDIS_PASSWORD', '02211998')
            
            if self.redis_url.startswith("redis://"):
                # Parse Redis URL and add password if not present
                if '@' not in self.redis_url and redis_password:
                    # URL doesn't have password, add it
                    redis_url_with_auth = self.redis_url.replace("redis://", f"redis://:{redis_password}@")
                    logger.info(f"🔐 Adding Redis authentication to URL")
                else:
                    redis_url_with_auth = self.redis_url
                
                try:
                    # Use redis.from_url for proper URL parsing with authentication
                    self.redis_client = redis.from_url(redis_url_with_auth, decode_responses=True)
                except Exception as e:
                    logger.warning(f"URL parsing failed: {e}, trying manual connection")
                    # Fallback to manual connection
                    host = "redis"
                    port = 6379
                    self.redis_client = redis.Redis(host=host, port=port, password=redis_password, decode_responses=True)
            else:
                # Fallback for simple connections
                parts = self.redis_url.split(':')
                host = parts[0] if parts else 'redis'
                port = int(parts[1]) if len(parts) > 1 else 6379
                self.redis_client = redis.Redis(host=host, port=port, password=redis_password, decode_responses=True)
            
            # Test Redis connection
            await asyncio.to_thread(self.redis_client.ping)
            logger.info("✅ Redis connection established")
            
            # Initialize Qdrant connection
            logger.info(f"🔗 Connecting to Qdrant: {self.qdrant_url}")
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            
            # Ensure collection exists
            collections = await asyncio.to_thread(self.qdrant_client.get_collections)
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                await asyncio.to_thread(
                    self.qdrant_client.create_collection,
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.info(f"✅ Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"✅ Using existing Qdrant collection: {self.collection_name}")
            
            # Initialize embedding model with hardware acceleration
            await self._initialize_embedding_model()
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            logger.info("✅ NPU Background Worker initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    async def _initialize_embedding_model(self):
        """Initialize embedding model with NPU/GPU acceleration"""
        logger.info("🤖 Loading embedding model...")
        
        def load_model():
            if SentenceTransformer is None:
                logger.warning("SentenceTransformer not available, using mock embeddings")
                return None
                
            model = SentenceTransformer(self.embedding_model)
            
            # Apply hardware acceleration
            if self.hardware_manager.optimization_enabled:
                if self.hardware_manager.npu_available:
                    # NPU optimization
                    try:
                        import intel_extension_for_pytorch as ipex
                        model = model.to('xpu')
                        model = ipex.optimize(model)
                        logger.info("🚀 Embedding model optimized for Intel NPU")
                    except Exception as e:
                        logger.warning(f"NPU optimization failed, using GPU/CPU: {e}")
                        model = model.to(self.hardware_manager.device)
                elif self.hardware_manager.gpu_available:
                    # GPU optimization
                    model = model.to(self.hardware_manager.device)
                    if hasattr(model, 'half'):
                        model = model.half()  # FP16 for memory efficiency
                    logger.info("🚀 Embedding model optimized for GPU")
            
            return model
        
        # Load model in thread to avoid blocking
        self.embedding_model = await asyncio.to_thread(load_model)
        logger.info(f"✅ Embedding model loaded on {self.hardware_manager.device}")
    
    async def start(self):
        """Start the background worker"""
        if self.is_running:
            logger.warning("Worker already running")
            return
        
        logger.info(f"🎯 Starting NPU Background Worker: {self.worker_id}")
        self.is_running = True
        
        # Register worker in Redis
        await asyncio.to_thread(
            self.redis_client.hset,
            "npu_background_workers",
            self.worker_id,
            json.dumps({
                "started_at": datetime.now().isoformat(),
                "status": "running",
                "device": self.hardware_manager.device,
                "npu_available": self.hardware_manager.npu_available,
                "gpu_available": self.hardware_manager.gpu_available
            })
        )
        
        # Start processing loops
        tasks = [
            asyncio.create_task(self._task_consumer_loop()),
            asyncio.create_task(self._crawl_scheduler_loop()),
            asyncio.create_task(self._quality_assessment_loop()),
            asyncio.create_task(self._metrics_reporter_loop()),
            asyncio.create_task(self._health_monitor_loop())
        ]
        
        logger.info("✅ All background loops started")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"❌ Worker error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the background worker"""
        logger.info("🛑 Stopping NPU Background Worker...")
        self.is_running = False
        
        # Unregister worker
        await asyncio.to_thread(self.redis_client.hdel, "npu_background_workers", self.worker_id)
        
        # Close connections
        if self.session:
            await self.session.close()
        if self.redis_client:
            await asyncio.to_thread(self.redis_client.close)
        
        logger.info("✅ NPU Background Worker stopped")
    
    async def _task_consumer_loop(self):
        """Main task consumption loop"""
        logger.info("🔄 Starting task consumer loop")
        
        while self.is_running:
            try:
                # Get tasks from Redis stream
                tasks = await self._get_pending_tasks()
                
                if tasks:
                    for task in tasks:
                        if len(self.active_tasks) < self.max_concurrent_tasks:
                            asyncio.create_task(self._process_task(task))
                        else:
                            # Queue is full, wait
                            break
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in task consumer loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _get_pending_tasks(self) -> List[BackgroundTask]:
        """Get pending tasks from Redis streams"""
        try:
            # Read from multiple streams
            streams = [
                "npu_background_crawl",
                "npu_background_embed", 
                "npu_background_analyze",
                "npu_background_quality"
            ]
            
            results = await asyncio.to_thread(
                self.redis_client.xreadgroup,
                groupname="npu_workers",
                consumername=self.worker_id,
                streams={stream: ">" for stream in streams},
                count=self.batch_size,
                block=1000
            )
            
            tasks = []
            for stream, messages in results:
                for msg_id, fields in messages:
                    try:
                        task_data = json.loads(fields.get(b'data', b'{}').decode())
                        task = BackgroundTask(
                            task_id=msg_id.decode(),
                            task_type=fields.get(b'type', b'unknown').decode(),
                            priority=int(fields.get(b'priority', b'5')),
                            data=task_data,
                            created_at=datetime.fromisoformat(fields.get(b'created_at', datetime.now().isoformat()).decode()),
                            scheduled_for=datetime.fromisoformat(fields.get(b'scheduled_for', datetime.now().isoformat()).decode())
                        )
                        tasks.append(task)
                    except Exception as e:
                        logger.error(f"Error parsing task {msg_id}: {e}")
            
            return sorted(tasks, key=lambda t: (t.priority, t.scheduled_for), reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting pending tasks: {e}")
            return []
    
    async def _process_task(self, task: BackgroundTask):
        """Process a single background task"""
        start_time = time.time()
        task.status = "processing"
        self.active_tasks[task.task_id] = task
        
        try:
            logger.info(f"🔨 Processing task {task.task_id} ({task.task_type})")
            
            if task.task_type == "crawl":
                result = await self._process_crawl_task(task)
            elif task.task_type == "embed":
                result = await self._process_embedding_task(task)
            elif task.task_type == "analyze":
                result = await self._process_analysis_task(task)
            elif task.task_type == "quality":
                result = await self._process_quality_task(task)
            else:
                logger.warning(f"Unknown task type: {task.task_type}")
                result = False
            
            processing_time = time.time() - start_time
            
            if result:
                task.status = "completed"
                self.completed_tasks.append(task)
                self.metrics["tasks_processed"] += 1
                
                # Acknowledge task in Redis
                await self._acknowledge_task(task)
                
                logger.info(f"✅ Task {task.task_id} completed in {processing_time:.2f}s")
            else:
                await self._handle_task_failure(task)
            
            self.metrics["total_processing_time"] += processing_time
            
        except Exception as e:
            logger.error(f"❌ Error processing task {task.task_id}: {e}")
            await self._handle_task_failure(task)
        finally:
            self.active_tasks.pop(task.task_id, None)
    
    async def _process_crawl_task(self, task: BackgroundTask) -> bool:
        """Process web crawling task with NPU acceleration"""
        try:
            url = task.data.get("url")
            if not url:
                logger.error("No URL provided for crawl task")
                return False
            
            # Fetch content
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch {url}: {response.status}")
                    return False
                
                content = await response.text()
            
            # Extract text and metadata
            title = task.data.get("title", "Unknown")
            
            # Clean and process content
            processed_content = await self._process_content(content)
            
            # Generate embeddings using NPU/GPU acceleration
            embeddings = await self._generate_embeddings_batch([processed_content])
            
            # Extract concepts and assess quality
            concepts = await self._extract_concepts(processed_content)
            quality_score = await self._assess_content_quality(processed_content)
            
            # Store in vector database
            point_id = hashlib.md5(url.encode()).hexdigest()
            
            await asyncio.to_thread(
                self.qdrant_client.upsert,
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=point_id,
                    vector=embeddings[0],
                    payload={
                        "url": url,
                        "title": title,
                        "content": processed_content[:1000],  # Store excerpt
                        "concepts": concepts,
                        "quality_score": quality_score,
                        "crawled_at": datetime.now().isoformat(),
                        "worker_id": self.worker_id,
                        "npu_accelerated": self.hardware_manager.optimization_enabled
                    }
                )]
            )
            
            # Update metrics
            self.metrics["documents_crawled"] += 1
            self.metrics["concepts_extracted"] += len(concepts)
            self.metrics["embeddings_generated"] += 1
            if self.hardware_manager.optimization_enabled:
                self.metrics["npu_accelerated_tasks"] += 1
            
            logger.info(f"📄 Crawled and indexed: {title} (Quality: {quality_score:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error in crawl task: {e}")
            return False
    
    async def _process_embedding_task(self, task: BackgroundTask) -> bool:
        """Process embedding generation task"""
        try:
            texts = task.data.get("texts", [])
            if not texts:
                return False
            
            # Generate embeddings in batch
            embeddings = await self._generate_embeddings_batch(texts)
            
            # Store embeddings result
            result_key = f"embeddings:{task.task_id}"
            await asyncio.to_thread(
                self.redis_client.setex,
                result_key,
                3600,  # 1 hour TTL
                json.dumps({
                    "embeddings": embeddings,
                    "generated_at": datetime.now().isoformat(),
                    "device_used": self.hardware_manager.device
                })
            )
            
            self.metrics["embeddings_generated"] += len(embeddings)
            return True
            
        except Exception as e:
            logger.error(f"Error in embedding task: {e}")
            return False
    
    async def _process_analysis_task(self, task: BackgroundTask) -> bool:
        """Process content analysis task"""
        try:
            content = task.data.get("content", "")
            analysis_type = task.data.get("analysis_type", "general")
            
            if analysis_type == "concept_extraction":
                concepts = await self._extract_concepts(content)
                result = {"concepts": concepts}
            elif analysis_type == "quality_assessment":
                quality_score = await self._assess_content_quality(content)
                result = {"quality_score": quality_score}
            elif analysis_type == "similarity":
                query = task.data.get("query", "")
                similarity = await self._calculate_similarity(content, query)
                result = {"similarity_score": similarity}
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}"}
                return False
            
            # Store analysis result
            result_key = f"analysis:{task.task_id}"
            await asyncio.to_thread(self.redis_client.setex, result_key, 3600, json.dumps(result))
            
            return True
            
        except Exception as e:
            logger.error(f"Error in analysis task: {e}")
            return False
    
    async def _process_quality_task(self, task: BackgroundTask) -> bool:
        """Process quality assessment and improvement task"""
        try:
            document_id = task.data.get("document_id")
            
            # Retrieve document from Qdrant
            result = await asyncio.to_thread(
                self.qdrant_client.retrieve,
                collection_name=self.collection_name,
                ids=[document_id]
            )
            
            if not result:
                return False
            
            document = result[0]
            content = document.payload.get("content", "")
            
            # Assess current quality
            quality_metrics = await self._detailed_quality_assessment(content)
            
            # Update document with quality metrics
            await asyncio.to_thread(
                self.qdrant_client.set_payload,
                collection_name=self.collection_name,
                points=[document_id],
                payload={
                    "quality_metrics": asdict(quality_metrics),
                    "last_quality_check": datetime.now().isoformat()
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error in quality task: {e}")
            return False
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using NPU/GPU acceleration"""
        if not self.embedding_model:
            # Mock embeddings if model not available
            logger.warning("Using mock embeddings - model not available")
            return [[0.1] * 384 for _ in texts]  # Mock 384-dim embeddings
            
        def encode_batch():
            with torch.no_grad():
                if self.hardware_manager.optimization_enabled:
                    # Use half precision for memory efficiency on GPU
                    if self.hardware_manager.gpu_available:
                        embeddings = self.embedding_model.encode(
                            texts,
                            convert_to_tensor=True,
                            device=self.hardware_manager.device
                        )
                    else:
                        embeddings = self.embedding_model.encode(
                            texts,
                            convert_to_tensor=True,
                            device=self.hardware_manager.device
                        )
                    
                    # Convert back to CPU for storage
                    if hasattr(embeddings, 'cpu'):
                        embeddings = embeddings.cpu().numpy()
                else:
                    embeddings = self.embedding_model.encode(texts)
                
                return embeddings.tolist()
        
        return await asyncio.to_thread(encode_batch)
    
    async def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        # Simple keyword/concept extraction
        # In production, this could use more sophisticated NLP
        concepts = []
        
        # Technology concepts
        tech_keywords = [
            "neural network", "machine learning", "deep learning", "AI", "artificial intelligence",
            "algorithm", "model", "training", "inference", "embedding", "transformer",
            "quantum computing", "quantum", "qubit", "superposition", "entanglement",
            "blockchain", "cryptocurrency", "bitcoin", "ethereum", "smart contract",
            "cloud computing", "edge computing", "NPU", "GPU", "tensor", "matrix"
        ]
        
        content_lower = content.lower()
        for keyword in tech_keywords:
            if keyword in content_lower:
                concepts.append(keyword)
        
        return list(set(concepts))  # Remove duplicates
    
    async def _assess_content_quality(self, content: str) -> float:
        """Assess content quality score"""
        if not content:
            return 0.0
        
        # Simple quality heuristics
        score = 0.0
        
        # Length factor (not too short, not too long)
        length = len(content)
        if 100 <= length <= 5000:
            score += 0.3
        elif length > 50:
            score += 0.1
        
        # Sentence structure
        sentences = content.split('.')
        if len(sentences) > 2:
            score += 0.2
        
        # Technical content indicators
        tech_indicators = ["algorithm", "method", "approach", "system", "model", "analysis"]
        tech_count = sum(1 for indicator in tech_indicators if indicator.lower() in content.lower())
        score += min(tech_count * 0.1, 0.3)
        
        # Coherence (simple check for repeated words)
        words = content.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        score += unique_ratio * 0.2
        
        return min(score, 1.0)
    
    async def _detailed_quality_assessment(self, content: str) -> QualityMetrics:
        """Perform detailed quality assessment"""
        return QualityMetrics(
            relevance_score=await self._assess_content_quality(content),
            coherence_score=0.8,  # Placeholder
            completeness_score=0.7,  # Placeholder
            freshness_score=0.9,  # Placeholder
            source_reliability=0.8,  # Placeholder
            overall_quality=0.8  # Placeholder
        )
    
    async def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embeddings = await self._generate_embeddings_batch([content1, content2])
        
        # Calculate cosine similarity
        emb1, emb2 = np.array(embeddings[0]), np.array(embeddings[1])
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)
    
    async def _process_content(self, raw_content: str) -> str:
        """Clean and process raw HTML/text content"""
        # Simple text cleaning (in production, use proper HTML parsing)
        import re
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', raw_content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Limit length
        return content[:4000]
    
    async def _acknowledge_task(self, task: BackgroundTask):
        """Acknowledge completed task in Redis"""
        try:
            # This would normally acknowledge the message in the stream
            # For now, just log completion
            await asyncio.to_thread(
                self.redis_client.hset,
                f"task_results:{task.task_id}",
                mapping={
                    "status": task.status,
                    "completed_at": datetime.now().isoformat(),
                    "worker_id": self.worker_id
                }
            )
        except Exception as e:
            logger.error(f"Error acknowledging task: {e}")
    
    async def _handle_task_failure(self, task: BackgroundTask):
        """Handle failed task"""
        task.retry_count += 1
        task.status = "failed"
        
        if task.retry_count < task.max_retries:
            # Reschedule for retry
            task.scheduled_for = datetime.now() + timedelta(minutes=task.retry_count * 5)
            task.status = "pending"
            
            # Add back to queue (simplified)
            logger.info(f"🔄 Retrying task {task.task_id} ({task.retry_count}/{task.max_retries})")
        else:
            # Move to failed tasks
            self.failed_tasks.append(task)
            self.metrics["tasks_failed"] += 1
            logger.error(f"❌ Task {task.task_id} failed permanently")
    
    async def _crawl_scheduler_loop(self):
        """Schedule crawling tasks for knowledge base enhancement"""
        logger.info("📅 Starting crawl scheduler loop")
        
        # Predefined crawl targets for knowledge base enhancement
        crawl_targets = [
            {"url": "https://en.wikipedia.org/wiki/Artificial_intelligence", "priority": 8},
            {"url": "https://en.wikipedia.org/wiki/Machine_learning", "priority": 8},
            {"url": "https://en.wikipedia.org/wiki/Neural_network", "priority": 7},
            {"url": "https://en.wikipedia.org/wiki/Quantum_computing", "priority": 6},
            {"url": "https://en.wikipedia.org/wiki/Natural_language_processing", "priority": 7},
        ]
        
        while self.is_running:
            try:
                for target in crawl_targets:
                    # Check if recently crawled
                    url_hash = hashlib.md5(target["url"].encode()).hexdigest()
                    last_crawl = await asyncio.to_thread(self.redis_client.get, f"last_crawl:{url_hash}")
                    
                    if last_crawl:
                        last_time = datetime.fromisoformat(last_crawl.decode())
                        if (datetime.now() - last_time).days < 7:  # Don't crawl more than once per week
                            continue
                    
                    # Add crawl task
                    await self._add_background_task(
                        task_type="crawl",
                        data=target,
                        priority=target["priority"]
                    )
                    
                    # Mark as scheduled
                    await asyncio.to_thread(
                        self.redis_client.setex,
                        f"last_crawl:{url_hash}",
                        86400 * 7,  # 7 days
                        datetime.now().isoformat()
                    )
                    
                    await asyncio.sleep(self.crawl_delay)
                
                # Wait before next scheduling cycle (1 hour)
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in crawl scheduler: {e}")
                await asyncio.sleep(300)  # 5 minutes
    
    async def _quality_assessment_loop(self):
        """Background quality assessment and improvement"""
        logger.info("🎯 Starting quality assessment loop")
        
        while self.is_running:
            try:
                # Get documents that need quality assessment
                # For now, just log the activity
                logger.info("🔍 Running background quality assessment")
                
                # This could query Qdrant for documents with low quality scores
                # and schedule quality improvement tasks
                
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                logger.error(f"Error in quality assessment loop: {e}")
                await asyncio.sleep(600)  # 10 minutes
    
    async def _metrics_reporter_loop(self):
        """Report metrics periodically"""
        while self.is_running:
            try:
                # Update metrics in Redis
                await asyncio.to_thread(
                    self.redis_client.hset,
                    f"npu_worker_metrics:{self.worker_id}",
                    mapping={
                        **{k: str(v) for k, v in self.metrics.items()},
                        "last_updated": datetime.now().isoformat(),
                        "active_tasks": len(self.active_tasks),
                        "completed_tasks": len(self.completed_tasks),
                        "failed_tasks": len(self.failed_tasks)
                    }
                )
                
                # Log summary
                if self.metrics["tasks_processed"] > 0:
                    avg_time = self.metrics["total_processing_time"] / self.metrics["tasks_processed"]
                    logger.info(
                        f"📊 Worker metrics - Tasks: {self.metrics['tasks_processed']}, "
                        f"Avg time: {avg_time:.2f}s, NPU accelerated: {self.metrics['npu_accelerated_tasks']}"
                    )
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in metrics reporter: {e}")
                await asyncio.sleep(60)
    
    async def _health_monitor_loop(self):
        """Monitor worker health and performance"""
        while self.is_running:
            try:
                # Check memory usage
                import psutil
                memory_percent = psutil.virtual_memory().percent
                
                if memory_percent > 90:
                    logger.warning(f"⚠️ High memory usage: {memory_percent}%")
                
                # Check GPU memory if available
                if self.hardware_manager.gpu_available:
                    try:
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                        if gpu_memory > 85:
                            logger.warning(f"⚠️ High GPU memory usage: {gpu_memory:.1f}%")
                    except:
                        pass
                
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    async def _add_background_task(self, task_type: str, data: Dict[str, Any], priority: int = 5):
        """Add task to background processing queue"""
        task_id = f"{task_type}_{int(time.time())}_{hash(str(data)) % 10000}"
        
        task_data = {
            "type": task_type,
            "priority": str(priority),
            "data": json.dumps(data),
            "created_at": datetime.now().isoformat(),
            "scheduled_for": datetime.now().isoformat(),
            "worker_id": self.worker_id
        }
        
        # Add to appropriate stream
        stream_name = f"npu_background_{task_type}"
        await asyncio.to_thread(self.redis_client.xadd, stream_name, task_data)
        
        logger.debug(f"➕ Added {task_type} task to {stream_name}")

# HTTP API for monitoring and control
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="NPU Background Worker API", version="1.0.0")

# Global worker instance
worker: Optional[NPUBackgroundWorker] = None

@app.on_event("startup")
async def startup_event():
    """Initialize worker on startup"""
    global worker
    worker = NPUBackgroundWorker()
    await worker.initialize()
    # Start worker in background
    asyncio.create_task(worker.start())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if worker:
        await worker.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not worker or not worker.is_running:
        raise HTTPException(status_code=503, detail="Worker not running")
    
    return {
        "status": "healthy",
        "worker_id": worker.worker_id,
        "is_running": worker.is_running,
        "hardware": {
            "device": worker.hardware_manager.device,
            "npu_available": worker.hardware_manager.npu_available,
            "gpu_available": worker.hardware_manager.gpu_available,
            "optimization_enabled": worker.hardware_manager.optimization_enabled
        },
        "metrics": worker.metrics,
        "active_tasks": len(worker.active_tasks)
    }

@app.get("/metrics")
async def get_metrics():
    """Get detailed worker metrics"""
    if not worker:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    return {
        "worker_id": worker.worker_id,
        "metrics": worker.metrics,
        "task_counts": {
            "active": len(worker.active_tasks),
            "completed": len(worker.completed_tasks),
            "failed": len(worker.failed_tasks)
        },
        "hardware_status": {
            "device": worker.hardware_manager.device,
            "npu_available": worker.hardware_manager.npu_available,
            "gpu_available": worker.hardware_manager.gpu_available
        }
    }

@app.post("/tasks/crawl")
async def add_crawl_task(url: str, priority: int = 5):
    """Add a crawl task"""
    if not worker:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    await worker._add_background_task(
        task_type="crawl",
        data={"url": url},
        priority=priority
    )
    
    return {"status": "task_added", "type": "crawl", "url": url}

@app.post("/tasks/embed")
async def add_embedding_task(texts: List[str], priority: int = 5):
    """Add an embedding generation task"""
    if not worker:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    await worker._add_background_task(
        task_type="embed",
        data={"texts": texts},
        priority=priority
    )
    
    return {"status": "task_added", "type": "embed", "text_count": len(texts)}

@app.post("/enhanced_upsert")
async def enhanced_upsert(upsert_data: dict):
    """Enhanced upsert endpoint for integration with crawler"""
    if not worker:
        raise HTTPException(status_code=503, detail="Worker not initialized")
    
    try:
        items = upsert_data.get("items", [])
        collection = upsert_data.get("collection", worker.collection_name)
        conflict_resolution = upsert_data.get("conflict_resolution", "intelligent_merge")
        quality_filtering = upsert_data.get("quality_filtering", True)
        batch_metadata = upsert_data.get("batch_metadata", {})
        
        successful_upserts = 0
        failed_upserts = 0
        
        for item in items:
            try:
                # Quality filtering
                if quality_filtering and item.get("quality_score", 0) < worker.quality_threshold:
                    failed_upserts += 1
                    continue
                
                # Prepare point for Qdrant
                point_id = hashlib.md5(item.get("url", str(time.time())).encode()).hexdigest()
                
                # Get or generate embeddings
                embeddings = item.get("embeddings", [])
                if not embeddings and item.get("content"):
                    embeddings_list = await worker._generate_embeddings_batch([item["content"]])
                    embeddings = embeddings_list[0] if embeddings_list else []
                
                if not embeddings:
                    failed_upserts += 1
                    continue
                
                # Enhanced payload with integration metadata
                payload = {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "content": item.get("content", "")[:1000],  # Store excerpt
                    "concepts": item.get("concepts", []),
                    "quality_score": item.get("quality_score", 0.5),
                    "source_type": item.get("source_type", "enhanced_crawler"),
                    "parent_query": item.get("parent_query"),
                    "update_strategy": item.get("update_strategy", "intelligent_merge"),
                    "crawled_at": datetime.now().isoformat(),
                    "worker_id": worker.worker_id,
                    "integration_source": "enhanced_crawler_integration",
                    "batch_metadata": batch_metadata,
                    "metadata": item.get("metadata", {})
                }
                
                # Handle conflict resolution
                if conflict_resolution == "intelligent_merge":
                    # Check if point exists
                    try:
                        existing = await asyncio.to_thread(
                            worker.qdrant_client.retrieve,
                            collection_name=collection,
                            ids=[point_id]
                        )
                        
                        if existing:
                            # Merge with existing data
                            existing_payload = existing[0].payload
                            # Keep higher quality score
                            if existing_payload.get("quality_score", 0) > payload["quality_score"]:
                                # Update timestamp but keep existing content
                                existing_payload["last_updated"] = datetime.now().isoformat()
                                existing_payload["update_count"] = existing_payload.get("update_count", 0) + 1
                                payload = existing_payload
                            else:
                                # Use new content but preserve some existing metadata
                                payload["update_count"] = existing_payload.get("update_count", 0) + 1
                                payload["first_crawled"] = existing_payload.get("crawled_at", payload["crawled_at"])
                    except:
                        # Point doesn't exist, proceed with new insert
                        pass
                
                # Upsert to Qdrant
                await asyncio.to_thread(
                    worker.qdrant_client.upsert,
                    collection_name=collection,
                    points=[PointStruct(
                        id=point_id,
                        vector=embeddings,
                        payload=payload
                    )]
                )
                
                successful_upserts += 1
                worker.metrics["embeddings_generated"] += 1
                worker.metrics["documents_crawled"] += 1
                
            except Exception as e:
                logger.error(f"Failed to upsert item {item.get('url', 'unknown')}: {e}")
                failed_upserts += 1
        
        # Update worker metrics
        worker.metrics["tasks_processed"] += successful_upserts
        worker.metrics["tasks_failed"] += failed_upserts
        
        return {
            "status": "enhanced_upsert_completed",
            "successful_upserts": successful_upserts,
            "failed_upserts": failed_upserts,
            "total_items": len(items),
            "collection": collection,
            "batch_metadata": batch_metadata,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced upsert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced upsert failed: {e}")

if __name__ == "__main__":
    import uvicorn
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8905) 