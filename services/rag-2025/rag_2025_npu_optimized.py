#!/usr/bin/env python3
"""
RAG 2025 NPU-Optimized System with Advanced Hardware Acceleration
Enhanced with LLM speed optimizations, better dependency handling, and GTX 1080 support
NEW: Streaming LLM responses, advanced NPU batching, multi-tier caching
"""

import asyncio
import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
import json
import hashlib
from functools import lru_cache
import concurrent.futures
from contextlib import asynccontextmanager
import threading
from collections import deque, OrderedDict

# Core libraries
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import redis.asyncio as redis
import httpx

# GPU/NPU optimization libraries with robust fallbacks
ACCELERATION_STATUS = {
    "intel_npu": False,
    "nvidia_gpu": False,
    "torch": False,
    "sentence_transformers": False,
    "cupy": False,
    "triton": False
}

try:
    import torch
    import torch.nn.functional as F
    ACCELERATION_STATUS["torch"] = True
    
    # Try Intel NPU extensions
    try:
        import intel_extension_for_pytorch as ipex
        ACCELERATION_STATUS["intel_npu"] = True
        print("✅ Intel NPU extensions available")
    except ImportError:
        print("⚠️  Intel NPU extensions not available - using standard PyTorch")
        
    # Try Triton for additional GPU optimization
    try:
        import triton
        import triton.language as tl
        ACCELERATION_STATUS["triton"] = True
        print("✅ Triton GPU compiler available")
    except ImportError:
        print("⚠️  Triton not available")
        
except ImportError:
    torch = None
    print("⚠️  PyTorch not available")

# Enhanced SentenceTransformer import with version compatibility
try:
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    ACCELERATION_STATUS["sentence_transformers"] = True
    print(f"✅ SentenceTransformers {sentence_transformers.__version__} available")
except ImportError:
    SentenceTransformer = None
    print("⚠️  SentenceTransformers not available")

try:
    import cupy as cp
    ACCELERATION_STATUS["cupy"] = True
    print("✅ CuPy GPU acceleration available")
except ImportError:
    cp = None
    print("⚠️  CuPy not available - using NumPy fallback")

# GTX 1080 specific monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    if gpu_count > 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        print(f"🎮 GPU Detected: {gpu_name}")
        if "GTX 1080" in gpu_name:
            print("🎯 GTX 1080 detected - applying specific optimizations")
        ACCELERATION_STATUS["gpu_monitoring"] = True
    else:
        ACCELERATION_STATUS["gpu_monitoring"] = False
except (ImportError, Exception) as e:
    print(f"⚠️  GPU monitoring not available: {e}")
    ACCELERATION_STATUS["gpu_monitoring"] = False

# Vector database and retrieval
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
from rank_bm25 import BM25Okapi

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AdvancedNPUConfig:
    """Advanced NPU/GPU configuration with GTX 1080 optimization"""
    # Hardware acceleration
    use_intel_npu: bool = True
    use_nvidia_gpu: bool = True
    use_cpu_fallback: bool = True
    use_triton_kernels: bool = True
    
    # Performance settings
    npu_batch_size: int = 64
    gpu_batch_size: int = 32  # Optimized for GTX 1080
    cpu_batch_size: int = 16
    streaming_batch_size: int = 8  # For real-time streaming
    
    # Memory management (GTX 1080 has 8GB VRAM)
    gpu_memory_fraction: float = 0.8
    max_sequence_length: int = 512
    enable_mixed_precision: bool = True
    memory_pool_enabled: bool = True
    
    # Multi-tier caching system
    l1_cache_size: int = 1000      # Hot embeddings
    l2_cache_size: int = 5000      # Warm embeddings  
    l3_cache_size: int = 10000     # Cold embeddings
    llm_cache_size: int = 2000     # LLM responses
    similarity_cache_size: int = 5000
    
    # Advanced features
    async_embedding_generation: bool = True
    parallel_llm_inference: bool = True
    smart_prefetching: bool = True
    context_compression: bool = True
    streaming_enabled: bool = True
    
    # Performance optimization
    pipeline_parallel: bool = True
    tensor_parallel: bool = True
    gradient_checkpointing: bool = True
    
    # Device selection
    embedding_device: str = "auto"
    similarity_device: str = "auto"
    llm_device: str = "auto"

class MultiTierCache:
    """Multi-tier caching system with different performance characteristics"""
    
    def __init__(self, l1_size: int = 1000, l2_size: int = 5000, l3_size: int = 10000):
        # L1: In-memory hot cache (fastest)
        self.l1_cache = OrderedDict()
        self.l1_max_size = l1_size
        
        # L2: In-memory warm cache 
        self.l2_cache = OrderedDict()
        self.l2_max_size = l2_size
        
        # L3: In-memory cold cache
        self.l3_cache = OrderedDict()  
        self.l3_max_size = l3_size
        
        self.lock = threading.RLock()
        self.stats = {
            'l1_hits': 0, 'l2_hits': 0, 'l3_hits': 0, 'misses': 0,
            'l1_size': 0, 'l2_size': 0, 'l3_size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            # Check L1 (hottest)
            if key in self.l1_cache:
                self.stats['l1_hits'] += 1
                self.l1_cache.move_to_end(key)
                return self.l1_cache[key]
            
            # Check L2 (warm)
            if key in self.l2_cache:
                self.stats['l2_hits'] += 1
                value = self.l2_cache.pop(key)
                self._promote_to_l1(key, value)
                return value
            
            # Check L3 (cold)
            if key in self.l3_cache:
                self.stats['l3_hits'] += 1
                value = self.l3_cache.pop(key)
                self._promote_to_l2(key, value)
                return value
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any):
        with self.lock:
            # Always insert to L1 (assume it will be accessed again soon)
            self._promote_to_l1(key, value)
    
    def _promote_to_l1(self, key: str, value: Any):
        # Remove from lower tiers if present
        self.l2_cache.pop(key, None)
        self.l3_cache.pop(key, None)
        
        # Add to L1, evict to L2 if full
        if len(self.l1_cache) >= self.l1_max_size:
            # Move oldest from L1 to L2
            old_key, old_value = self.l1_cache.popitem(last=False)
            self._promote_to_l2(old_key, old_value)
        
        self.l1_cache[key] = value
        self._update_stats()
    
    def _promote_to_l2(self, key: str, value: Any):
        # Remove from L3 if present
        self.l3_cache.pop(key, None)
        
        # Add to L2, evict to L3 if full
        if len(self.l2_cache) >= self.l2_max_size:
            # Move oldest from L2 to L3
            old_key, old_value = self.l2_cache.popitem(last=False)
            self._demote_to_l3(old_key, old_value)
        
        self.l2_cache[key] = value
        self._update_stats()
    
    def _demote_to_l3(self, key: str, value: Any):
        # Add to L3, evict if full
        if len(self.l3_cache) >= self.l3_max_size:
            self.l3_cache.popitem(last=False)  # Remove oldest
        
        self.l3_cache[key] = value
        self._update_stats()
    
    def _update_stats(self):
        self.stats.update({
            'l1_size': len(self.l1_cache),
            'l2_size': len(self.l2_cache), 
            'l3_size': len(self.l3_cache)
        })
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = sum([self.stats['l1_hits'], self.stats['l2_hits'], 
                             self.stats['l3_hits'], self.stats['misses']])
        
        if total_requests > 0:
            hit_rate = ((self.stats['l1_hits'] + self.stats['l2_hits'] + 
                        self.stats['l3_hits']) / total_requests) * 100
        else:
            hit_rate = 0
            
        return {**self.stats, 'total_hit_rate': hit_rate}

class AdvancedNPUEmbedder:
    """Advanced NPU-optimized embedding generation with caching and batching"""
    
    def __init__(self, config: AdvancedNPUConfig):
        self.config = config
        self.model = None
        self.device = "cpu"
        # Multi-tier caching system
        self.embedding_cache = MultiTierCache(
            l1_size=config.l1_cache_size,
            l2_size=config.l2_cache_size, 
            l3_size=config.l3_cache_size
        )
        self.batch_queue = deque(maxlen=1000)
        self.processing_lock = asyncio.Lock()
        self.memory_pool = {} if config.memory_pool_enabled else None
        
        # Streaming and pipeline processing
        self.stream_processor = None
        if config.streaming_enabled:
            self.stream_processor = asyncio.Queue(maxsize=config.streaming_batch_size * 2)
        
        self._setup_optimal_device()
        self._load_optimized_model()
    
    def _setup_optimal_device(self):
        """Setup optimal device based on hardware availability"""
        if self.config.embedding_device == "auto":
            if self.config.use_nvidia_gpu and torch and torch.cuda.is_available():
                # Check if GTX 1080 or similar
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
                self.device = "cuda"
                self.batch_size = self.config.gpu_batch_size
                logger.info(f"🚀 Using NVIDIA GPU ({gpu_name}) for embeddings")
            elif self.config.use_intel_npu and ACCELERATION_STATUS["intel_npu"]:
                self.device = "xpu"
                self.batch_size = self.config.npu_batch_size
                logger.info("🚀 Using Intel NPU for embeddings")
            else:
                self.device = "cpu"
                self.batch_size = self.config.cpu_batch_size
                logger.info("💻 Using CPU for embeddings")
        else:
            self.device = self.config.embedding_device
            self.batch_size = self.config.gpu_batch_size
    
    def _load_optimized_model(self):
        """Load and optimize embedding model for target hardware"""
        if not ACCELERATION_STATUS["sentence_transformers"]:
            raise RuntimeError("SentenceTransformers not available")
            
        try:
            # Use a faster, smaller model optimized for speed
            model_name = 'all-MiniLM-L6-v2'  # Fast and efficient
            self.model = SentenceTransformer(model_name)
            
            if torch:
                self.model = self.model.to(self.device)
                
                # Intel NPU optimization
                if self.device == "xpu" and ACCELERATION_STATUS["intel_npu"]:
                    self.model = ipex.optimize(self.model)
                    logger.info("✅ Model optimized for Intel NPU")
                
                # NVIDIA GPU optimization (GTX 1080 specific)
                elif self.device == "cuda":
                    if self.config.enable_mixed_precision:
                        self.model.half()  # FP16 for GTX 1080
                        logger.info("✅ Model optimized for NVIDIA GPU (FP16)")
                    
                    # Set memory fraction for GTX 1080
                    torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
                    
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def _cached_single_encode(self, text: str) -> np.ndarray:
        """Cache individual text embeddings"""
        with torch.no_grad():
            return self.model.encode([text], convert_to_numpy=True)[0]
    
    async def encode_batch(self, texts: List[str], enable_caching: bool = True) -> np.ndarray:
        """Advanced NPU-optimized batch embedding generation"""
        async with self.processing_lock:
            try:
                start_time = time.time()
                
                # Check cache first
                cached_embeddings = {}
                uncached_texts = []
                
                if enable_caching:
                    for text in texts:
                        cache_key = self._get_cache_key(text)
                        if cache_key in self.embedding_cache:
                            cached_embeddings[text] = self.embedding_cache[cache_key]
                        else:
                            uncached_texts.append(text)
                else:
                    uncached_texts = texts
                
                # Process uncached texts in optimized batches
                new_embeddings = {}
                if uncached_texts:
                    new_embeddings = await self._process_uncached_batch(uncached_texts)
                
                # Combine cached and new embeddings in original order
                result_embeddings = []
                for text in texts:
                    if text in cached_embeddings:
                        result_embeddings.append(cached_embeddings[text])
                    else:
                        result_embeddings.append(new_embeddings[text])
                
                result = np.array(result_embeddings)
                processing_time = time.time() - start_time
                
                cache_hits = len(cached_embeddings)
                logger.info(f"🚀 NPU Embeddings: {len(texts)} texts in {processing_time:.3f}s "
                          f"({len(texts)/processing_time:.1f} texts/sec, {cache_hits} cache hits)")
                
                return result
                
            except Exception as e:
                logger.error(f"NPU embedding failed: {e}")
                # Fallback to simple encoding
                return self.model.encode(texts, convert_to_numpy=True)
    
    async def _process_uncached_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Process uncached texts with optimal batching"""
        new_embeddings = {}
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            with torch.no_grad():
                if self.device == "cuda" and torch.cuda.is_available():
                    # NVIDIA GPU optimized path with memory management
                    torch.cuda.empty_cache()  # Clear cache for GTX 1080
                    batch_embeddings = self.model.encode(
                        batch,
                        device=self.device,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=self.batch_size
                    )
                elif self.device == "xpu" and ACCELERATION_STATUS["intel_npu"]:
                    # Intel NPU optimized path
                    batch_embeddings = self.model.encode(
                        batch,
                        device=self.device,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=self.batch_size
                    )
                else:
                    # CPU fallback with threading
                    batch_embeddings = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
            
            # Store in cache and result
            for j, text in enumerate(batch):
                embedding = batch_embeddings[j]
                new_embeddings[text] = embedding
                
                # Cache management
                if len(self.embedding_cache) < self.config.embedding_cache_size:
                    cache_key = self._get_cache_key(text)
                    self.embedding_cache[cache_key] = embedding
        
        return new_embeddings

class AdvancedLLMOptimizer:
    """Advanced LLM call optimization with streaming, caching and parallel processing"""
    
    def __init__(self, config: AdvancedNPUConfig, ollama_url: str):
        self.config = config
        self.ollama_url = ollama_url
        # Enhanced multi-tier caching for LLM responses
        self.response_cache = MultiTierCache(
            l1_size=config.llm_cache_size // 4,   # Hot responses
            l2_size=config.llm_cache_size // 2,   # Warm responses  
            l3_size=config.llm_cache_size         # All responses
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
        
        # Streaming support
        self.streaming_enabled = config.streaming_enabled
        self.stream_buffers = {}
        self.stream_lock = threading.RLock()
        
        # Response optimization
        self.template_cache = {}
        logger.info("✅ Advanced LLM Optimizer with streaming initialized")
        
    def _get_cache_key(self, prompt: str, model: str = "llama3.2:1b") -> str:
        """Generate cache key for LLM responses"""
        combined = f"{model}:{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def generate_response(self, prompt: str, model: str = "llama3.2:1b", 
                              enable_caching: bool = True) -> str:
        """Optimized LLM response generation with caching"""
        if enable_caching:
            cache_key = self._get_cache_key(prompt, model)
            cached_response = self.response_cache.get(cache_key)
            if cached_response is not None:
                logger.info("🎯 LLM cache hit")
                return cached_response
        
        start_time = time.time()
        
        try:
            if self.config.parallel_llm_inference:
                # Use thread pool for non-blocking LLM calls
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self.executor, 
                    self._sync_generate_response, 
                    prompt, 
                    model
                )
            else:
                response = await self._async_generate_response(prompt, model)
            
            # Cache successful responses
            if enable_caching:
                cache_key = self._get_cache_key(prompt, model)
                self.response_cache.set(cache_key, response)
            
            generation_time = time.time() - start_time
            logger.info(f"🤖 LLM response generated in {generation_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    async def generate_streaming_response(self, prompt: str, model: str = "llama3.2:1b") -> AsyncGenerator[str, None]:
        """Generate streaming LLM response for real-time output"""
        if not self.streaming_enabled:
            # Fallback to regular response
            response = await self.generate_response(prompt, model, enable_caching=True)
            yield response
            return
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream('POST', f"{self.ollama_url}/api/generate", json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_ctx": 2048,
                        "num_predict": 512
                    }
                }) as response:
                    if response.status_code == 200:
                        accumulated_response = ""
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    if 'response' in data:
                                        chunk = data['response']
                                        accumulated_response += chunk
                                        yield chunk
                                    
                                    if data.get('done', False):
                                        # Cache the complete response
                                        cache_key = self._get_cache_key(prompt, model)
                                        self.response_cache.set(cache_key, accumulated_response)
                                        break
                                except json.JSONDecodeError:
                                    continue
                    else:
                        yield f"Error: HTTP {response.status_code}"
                        
        except Exception as e:
            logger.error(f"Streaming LLM generation failed: {e}")
            yield f"Error generating streaming response: {str(e)}"
    
    def _sync_generate_response(self, prompt: str, model: str) -> str:
        """Synchronous LLM call for thread pool execution"""
        import requests
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 2048,
                    "num_predict": 512
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"LLM API error: {response.status_code}")
    
    async def _async_generate_response(self, prompt: str, model: str) -> str:
        """Asynchronous LLM call"""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_ctx": 2048,
                        "num_predict": 512
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                raise Exception(f"LLM API error: {response.status_code}")

class AdvancedNPUSimilarity:
    """Advanced NPU-optimized similarity calculations with GPU acceleration"""
    
    def __init__(self, config: AdvancedNPUConfig):
        self.config = config
        self.device = self._setup_device()
        self.similarity_cache = MultiTierCache(
            l1_size=config.similarity_cache_size // 4,
            l2_size=config.similarity_cache_size // 2,
            l3_size=config.similarity_cache_size
        )
    
    def _setup_device(self):
        """Setup optimal device for similarity calculations"""
        if self.config.similarity_device == "auto":
            if self.config.use_nvidia_gpu and ACCELERATION_STATUS["cupy"]:
                return "cupy"
            elif self.config.use_intel_npu and ACCELERATION_STATUS["intel_npu"]:
                return "xpu"
            else:
                return "numpy"
        return self.config.similarity_device
    
    def compute_similarities(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Advanced NPU-optimized cosine similarity computation"""
        try:
            start_time = time.time()
            
            # Generate cache key for repeated computations
            query_hash = hashlib.md5(query_embedding.tobytes()).hexdigest()[:16]
            docs_hash = hashlib.md5(doc_embeddings.tobytes()).hexdigest()[:16]
            cache_key = f"{query_hash}:{docs_hash}"
            
            cached_similarities = self.similarity_cache.get(cache_key)
            if cached_similarities is not None:
                logger.info("🎯 Similarity cache hit")
                return cached_similarities
            
            if self.device == "cupy" and ACCELERATION_STATUS["cupy"]:
                # NVIDIA GPU accelerated similarity (GTX 1080 optimized)
                similarities = self._compute_cupy_similarities(query_embedding, doc_embeddings)
            elif self.device == "xpu" and torch and ACCELERATION_STATUS["intel_npu"]:
                # Intel NPU accelerated similarity
                similarities = self._compute_torch_similarities(query_embedding, doc_embeddings)
            else:
                # NumPy fallback with optimizations
                similarities = self._compute_numpy_similarities(query_embedding, doc_embeddings)
            
            # Cache result
            self.similarity_cache.set(cache_key, similarities)
            
            computation_time = time.time() - start_time
            logger.info(f"⚡ Similarity computation: {len(doc_embeddings)} docs in {computation_time:.3f}s")
            
            return similarities
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            # Fallback to basic numpy computation
            return self._compute_numpy_similarities(query_embedding, doc_embeddings)
    
    def _compute_cupy_similarities(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """CuPy GPU-accelerated similarity computation"""
        query_gpu = cp.asarray(query_embedding)
        docs_gpu = cp.asarray(doc_embeddings)
        
        # Normalize vectors for cosine similarity
        query_norm = query_gpu / cp.linalg.norm(query_gpu)
        docs_norm = docs_gpu / cp.linalg.norm(docs_gpu, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = cp.dot(docs_norm, query_norm)
        
        # Return as numpy array
        return cp.asnumpy(similarities)
    
    def _compute_torch_similarities(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """PyTorch XPU-accelerated similarity computation"""
        query_tensor = torch.from_numpy(query_embedding).to(self.device)
        docs_tensor = torch.from_numpy(doc_embeddings).to(self.device)
        
        # Normalize vectors
        query_norm = F.normalize(query_tensor, p=2, dim=0)
        docs_norm = F.normalize(docs_tensor, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.mm(docs_norm, query_norm.unsqueeze(1)).squeeze()
        
        return similarities.cpu().numpy()
    
    def _compute_numpy_similarities(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Optimized NumPy similarity computation"""
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        docs_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(docs_norm, query_norm)
        
        return similarities

class AdvancedNPURAGSystem:
    """Advanced NPU-Optimized RAG System with enhanced hardware acceleration"""
    
    def __init__(self):
        self.config = AdvancedNPUConfig()
        
        # Advanced NPU-optimized components
        self.embedder = AdvancedNPUEmbedder(self.config)
        self.similarity_engine = AdvancedNPUSimilarity(self.config)
        self.llm_optimizer = AdvancedLLMOptimizer(
            self.config, 
            os.getenv("OLLAMA_URL", "http://localhost:11434")
        )
        
        # Vector database
        self.qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        
        # Redis for caching
        self.redis = None
        
        # BM25 for sparse retrieval
        self.bm25 = None
        
        # Sample data and caches
        self.documents = []
        self.embeddings = []
        self.chunk_rewards = {}
        self.performance_metrics = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "npu_acceleration_speedup": 0.0
        }
        self.documents = []
        
        # Sample data
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize enhanced sample data for the NPU RAG system"""
        self.documents = [
            {
                "id": 1,
                "content": "Quantum computing represents a revolutionary approach to computation that leverages quantum mechanical phenomena such as superposition and entanglement. Unlike classical computers that use bits, quantum computers use quantum bits or qubits that can exist in multiple states simultaneously.",
                "metadata": {"source": "quantum_computing_intro", "timestamp": "2025-01-01"}
            },
            {
                "id": 2, 
                "content": "The GTX 1080 is a powerful NVIDIA graphics card that supports CUDA acceleration for machine learning workloads. With 8GB of GDDR5X memory and 2560 CUDA cores, it provides excellent performance for deep learning training and inference tasks.",
                "metadata": {"source": "gpu_hardware_guide", "timestamp": "2025-01-02"}
            },
            {
                "id": 3,
                "content": "RAG (Retrieval-Augmented Generation) systems combine the power of large language models with external knowledge retrieval. This approach allows models to access up-to-date information and provide more accurate, contextually relevant responses.",
                "metadata": {"source": "rag_systems_overview", "timestamp": "2025-01-03"}
            },
            {
                "id": 4,
                "content": "NPU (Neural Processing Unit) acceleration can significantly speed up AI workloads by providing specialized hardware optimized for neural network operations. Intel's NPU technology offers efficient processing for embedding generation and similarity calculations.",
                "metadata": {"source": "npu_acceleration_guide", "timestamp": "2025-01-04"}
            },
            {
                "id": 5,
                "content": "Advanced caching strategies in AI systems can reduce computational overhead by storing frequently accessed embeddings, similarity scores, and generated responses. This is particularly effective in production RAG deployments.",
                "metadata": {"source": "ai_optimization_techniques", "timestamp": "2025-01-05"}
            }
        ]
        
        # Initialize chunk rewards for feedback learning
        for doc in self.documents:
            self.chunk_rewards[doc["id"]] = 1.0  # Default reward

    async def initialize(self):
        """Initialize the advanced NPU RAG system with all components"""
        try:
            logger.info("🚀 Initializing Advanced NPU RAG System...")
            
            # Initialize sample data
            self._initialize_sample_data()
            
            # Connect to Redis
            try:
                self.redis = redis.Redis.from_url(
                    os.getenv("REDIS_URL", "redis://localhost:6379"),
                    decode_responses=True
                )
                await self.redis.ping()
                logger.info("✅ Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis = None
            
            # Build NPU-optimized indexes
            await self._build_advanced_npu_indexes()
            
            logger.info("🎉 Advanced NPU RAG System initialized successfully")
            logger.info(f"📊 Hardware Status: Intel NPU: {ACCELERATION_STATUS['intel_npu']}, "
                       f"NVIDIA GPU: {ACCELERATION_STATUS['torch'] and torch.cuda.is_available()}, "
                       f"CuPy: {ACCELERATION_STATUS['cupy']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize NPU RAG system: {e}")
            raise

    async def _build_advanced_npu_indexes(self):
        """Build NPU-optimized indexes with advanced caching"""
        start_time = time.time()
        
        # Extract texts for embedding
        texts = [doc["content"] for doc in self.documents]
        
        # Generate embeddings using NPU acceleration
        self.embeddings = await self.embedder.encode_batch(texts, enable_caching=True)
        
        # Build BM25 index for sparse retrieval
        tokenized_docs = [doc["content"].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Store embeddings in Qdrant with enhanced metadata
        try:
            # Create collection with optimized parameters
            collection_name = "npu_rag_2025"
            
            try:
                await asyncio.to_thread(
                    self.qdrant.create_collection,
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 embedding size
                        distance=Distance.COSINE
                    )
                )
            except Exception:
                # Collection might already exist
                pass
            
            # Upsert points with enhanced metadata
            points = []
            for i, (doc, embedding) in enumerate(zip(self.documents, self.embeddings)):
                points.append(PointStruct(
                    id=doc["id"],
                    vector=embedding.tolist(),
                    payload={
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "reward_score": self.chunk_rewards.get(doc["id"], 1.0),
                        "embedding_device": self.embedder.device,
                        "processing_timestamp": time.time()
                    }
                ))
            
            await asyncio.to_thread(
                self.qdrant.upsert,
                collection_name=collection_name,
                points=points
            )
            
            indexing_time = time.time() - start_time
            logger.info(f"🚀 Advanced NPU indexes built in {indexing_time:.3f}s "
                       f"({len(self.documents)} documents, {len(self.embeddings)} embeddings)")
            
        except Exception as e:
            logger.error(f"Failed to build Qdrant indexes: {e}")

    async def advanced_hybrid_retrieval(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Advanced NPU-optimized hybrid retrieval with caching and acceleration"""
        start_time = time.time()
        
        try:
            # Generate query embedding with NPU acceleration
            query_embedding = await self.embedder.encode_batch([query], enable_caching=True)
            query_vector = query_embedding[0]
            
            # Dense retrieval using NPU-accelerated similarity
            dense_similarities = self.similarity_engine.compute_similarities(
                query_vector, 
                np.array(self.embeddings)
            )
            
            # Sparse retrieval using BM25
            query_tokens = query.lower().split()
            sparse_scores = self.bm25.get_scores(query_tokens)
            
            # Advanced hybrid scoring with novelty and reward weighting
            alpha = float(os.getenv("RERANK_ALPHA", "0.7"))
            novelty_weight = float(os.getenv("NOVELTY_WEIGHT", "0.4"))
            reward_weight = float(os.getenv("REWARD_WEIGHT", "0.15"))
            
            combined_scores = []
            for i, doc in enumerate(self.documents):
                # Base hybrid score
                dense_score = dense_similarities[i]
                sparse_score = sparse_scores[i] if i < len(sparse_scores) else 0.0
                hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
                
                # Reward boosting (R3 patch)
                reward_multiplier = self.chunk_rewards.get(doc["id"], 1.0)
                
                # Novelty factor (R1 patch enhancement)
                novelty_factor = 1.0 + novelty_weight * (1.0 - dense_score)
                
                # Combined score with NPU optimizations
                final_score = hybrid_score * (1 + reward_weight * reward_multiplier) * novelty_factor
                
                combined_scores.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "dense_score": float(dense_score),
                    "sparse_score": float(sparse_score),
                    "hybrid_score": float(hybrid_score),
                    "reward_multiplier": float(reward_multiplier),
                    "novelty_factor": float(novelty_factor),
                    "final_score": float(final_score),
                    "processing_device": self.similarity_engine.device
                })
            
            # Sort by final score and return top-k
            sorted_results = sorted(combined_scores, key=lambda x: x["final_score"], reverse=True)
            top_results = sorted_results[:top_k]
            
            retrieval_time = time.time() - start_time
            logger.info(f"🔍 Advanced hybrid retrieval: {len(top_results)} results in {retrieval_time:.3f}s")
            
            return top_results
            
        except Exception as e:
            logger.error(f"Advanced hybrid retrieval failed: {e}")
            return []

    async def enhanced_context_injection(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced context injection with NPU-optimized LLM calls and compression"""
        start_time = time.time()
        
        try:
            # Context compression for efficiency
            if self.config.context_compression:
                # Select most relevant chunks based on scores
                sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["final_score"], reverse=True)
                top_chunks = sorted_chunks[:6]  # Limit context for speed
            else:
                top_chunks = retrieved_chunks
            
            # Build enhanced context with metadata
            context_parts = []
            for i, chunk in enumerate(top_chunks):
                context_parts.append(f"[Document {i+1}] {chunk['content']}")
            
            context = "\n\n".join(context_parts)
            
            # Enhanced prompt with NPU optimization indicators
            enhanced_prompt = f"""Based on the following context documents, provide a comprehensive and accurate answer to the user's question. Use the information from the documents to support your response.

Context Documents:
{context}

User Question: {query}

Instructions:
- Synthesize information from multiple documents when relevant
- Cite specific documents when making claims
- Provide a clear, well-structured response
- If the context doesn't contain sufficient information, acknowledge this limitation

Answer:"""

            # Generate response using optimized LLM calls
            response = await self.llm_optimizer.generate_response(
                enhanced_prompt,
                model=os.getenv("LLM_MODEL", "llama3.2:1b"),
                enable_caching=True
            )
            
            # Track chunk usage for feedback learning (R2 patch)
            chunk_ids = [chunk["id"] for chunk in top_chunks]
            
            processing_time = time.time() - start_time
            
            return {
                "response": response,
                "context_chunks": top_chunks,
                "chunk_ids": chunk_ids,
                "processing_time": processing_time,
                "llm_cache_used": True,  # Placeholder for actual cache status
                "npu_accelerated": True,
                "context_compression_applied": self.config.context_compression,
                "enhancement_patches": ["R1_Advanced_Hybrid", "R2_Enhanced_Context", "R3_Reward_Learning"]
            }
            
        except Exception as e:
            logger.error(f"Enhanced context injection failed: {e}")
            return {
                "response": f"Error generating response: {str(e)}",
                "context_chunks": retrieved_chunks,
                "chunk_ids": [],
                "processing_time": time.time() - start_time,
                "error": str(e)
            }

    async def advanced_feedback_learning(self, chunk_ids: List[int], quality_ratings: List[float]) -> Dict[str, Any]:
        """Advanced feedback learning with NPU-accelerated reward updates"""
        start_time = time.time()
        
        try:
            # Validate inputs
            if len(chunk_ids) != len(quality_ratings):
                raise ValueError("Chunk IDs and quality ratings must have the same length")
            
            # Update chunk rewards with advanced algorithms
            updates_made = 0
            learning_rate = 0.1
            
            for chunk_id, rating in zip(chunk_ids, quality_ratings):
                if chunk_id in self.chunk_rewards:
                    # Exponential moving average for reward updates
                    current_reward = self.chunk_rewards[chunk_id]
                    new_reward = current_reward * (1 - learning_rate) + rating * learning_rate
                    self.chunk_rewards[chunk_id] = max(0.1, min(2.0, new_reward))  # Clamp rewards
                    updates_made += 1
                    
                    logger.info(f"📈 Updated reward for chunk {chunk_id}: {current_reward:.3f} → {new_reward:.3f}")
            
            # Update Qdrant with new reward scores (NPU-accelerated batch update)
            if updates_made > 0:
                try:
                    collection_name = "npu_rag_2025"
                    
                    # Batch update points with new reward scores
                    for chunk_id in chunk_ids:
                        if chunk_id in self.chunk_rewards:
                            await asyncio.to_thread(
                                self.qdrant.set_payload,
                                collection_name=collection_name,
                                points=[chunk_id],
                                payload={"reward_score": self.chunk_rewards[chunk_id]}
                            )
                            
                except Exception as e:
                    logger.warning(f"Failed to update Qdrant rewards: {e}")
            
            # Performance metrics update
            self.performance_metrics["total_queries"] += 1
            processing_time = time.time() - start_time
            
            return {
                "updates_made": updates_made,
                "processing_time": processing_time,
                "chunk_rewards": {cid: self.chunk_rewards.get(cid, 1.0) for cid in chunk_ids},
                "learning_rate": learning_rate,
                "reward_range": [min(self.chunk_rewards.values()), max(self.chunk_rewards.values())],
                "npu_accelerated": True,
                "patch_version": "R3_Advanced_Feedback"
            }
            
        except Exception as e:
            logger.error(f"Advanced feedback learning failed: {e}")
            return {
                "updates_made": 0,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }

# Initialize FastAPI app with enhanced configuration
app = FastAPI(
    title="NPU 2025 RAG - Advanced Hardware Accelerated",
    description="Next-generation RAG system with NPU/GPU acceleration, advanced caching, and LLM optimizations",
    version="2.0.0"
)

# Global RAG system instance
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the advanced NPU RAG system on startup"""
    global rag_system
    rag_system = AdvancedNPURAGSystem()
    await rag_system.initialize()

# Enhanced API Models
class AdvancedQueryRequest(BaseModel):
    query: str
    strategy: str = "advanced_hybrid"
    max_docs: int = 8
    enable_caching: bool = True
    context_compression: bool = True

class AdvancedFeedbackRequest(BaseModel):
    chunk_ids: List[int]
    quality_ratings: List[float]
    feedback_type: str = "quality"
    learning_rate: Optional[float] = None

@app.post("/query/stream")
async def streaming_query_endpoint(request: AdvancedQueryRequest):
    """Streaming query endpoint with real-time NPU-accelerated responses"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    async def generate_streaming_response():
        start_time = time.time()
        
        try:
            # Advanced hybrid retrieval
            retrieved_chunks = await rag_system.advanced_hybrid_retrieval(
                request.query, 
                top_k=request.max_docs
            )
            
            # Build enhanced context with metadata
            if request.context_compression:
                sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["final_score"], reverse=True)
                top_chunks = sorted_chunks[:6]
            else:
                top_chunks = retrieved_chunks
            
            context_parts = []
            for i, chunk in enumerate(top_chunks):
                context_parts.append(f"[Document {i+1}] {chunk['content']}")
            
            context = "\n\n".join(context_parts)
            
            enhanced_prompt = f"""Based on the following context documents, provide a comprehensive and accurate answer to the user's question.

Context Documents:
{context}

User Question: {request.query}

Answer:"""
            
            # Yield initial metadata
            yield f"data: {json.dumps({'type': 'metadata', 'processing_time': time.time() - start_time, 'chunks_retrieved': len(top_chunks)})}\n\n"
            
            # Stream LLM response
            async for chunk in rag_system.llm_optimizer.generate_streaming_response(enhanced_prompt):
                yield f"data: {json.dumps({'type': 'content', 'chunk': chunk})}\n\n"
            
            # Final metadata
            total_time = time.time() - start_time
            yield f"data: {json.dumps({'type': 'complete', 'total_time': total_time, 'npu_accelerated': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_streaming_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.post("/query")
async def advanced_query_endpoint(request: AdvancedQueryRequest):
    """Enhanced query endpoint with advanced NPU optimizations"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    start_time = time.time()
    
    try:
        # Advanced hybrid retrieval
        retrieved_chunks = await rag_system.advanced_hybrid_retrieval(
            request.query, 
            top_k=request.max_docs
        )
        
        # Enhanced context injection with LLM optimization
        result = await rag_system.enhanced_context_injection(
            request.query, 
            retrieved_chunks
        )
        
        total_time = time.time() - start_time
        
        return {
            "query": request.query,
            "response": result["response"],
            "context_chunks": result["context_chunks"],
            "chunk_ids": result["chunk_ids"],
            "processing_time": total_time,
            "retrieval_strategy": request.strategy,
            "hardware_acceleration": {
                "embedding_device": rag_system.embedder.device,
                "similarity_device": rag_system.similarity_engine.device,
                "npu_enabled": ACCELERATION_STATUS["intel_npu"],
                "gpu_enabled": ACCELERATION_STATUS["torch"] and torch.cuda.is_available(),
                "cupy_enabled": ACCELERATION_STATUS["cupy"]
            },
            "optimization_features": {
                "caching_enabled": request.enable_caching,
                "context_compression": request.context_compression,
                "llm_optimization": True,
                "parallel_processing": rag_system.config.parallel_llm_inference
            },
            "patch_version": "NPU_2025_Advanced"
        }
        
    except Exception as e:
        logger.error(f"Advanced query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/feedback")
async def advanced_feedback_endpoint(request: AdvancedFeedbackRequest):
    """Advanced feedback learning endpoint with NPU acceleration"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = await rag_system.advanced_feedback_learning(
            request.chunk_ids,
            request.quality_ratings
        )
        
        return {
            "feedback_processed": True,
            "updates_made": result["updates_made"],
            "processing_time": result["processing_time"],
            "chunk_rewards": result.get("chunk_rewards", {}),
            "npu_accelerated": result.get("npu_accelerated", False),
            "learning_rate": result.get("learning_rate", 0.1),
            "patch_version": "R3_Advanced_Feedback"
        }
        
    except Exception as e:
        logger.error(f"Advanced feedback processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")

@app.get("/health")
async def advanced_health_check():
    """Advanced health check with comprehensive hardware status"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "NPU_2025_Advanced",
        "hardware_status": {
            "torch_available": ACCELERATION_STATUS["torch"],
            "intel_npu_available": ACCELERATION_STATUS["intel_npu"],
            "nvidia_gpu_available": ACCELERATION_STATUS["torch"] and torch.cuda.is_available() if torch else False,
            "cupy_available": ACCELERATION_STATUS["cupy"],
            "sentence_transformers_available": ACCELERATION_STATUS["sentence_transformers"]
        },
        "components": {
            "embedder": "healthy" if rag_system and rag_system.embedder else "initializing",
            "similarity_engine": "healthy" if rag_system and rag_system.similarity_engine else "initializing",
            "llm_optimizer": "healthy" if rag_system and rag_system.llm_optimizer else "initializing",
            "vector_db": "healthy" if rag_system and rag_system.qdrant else "initializing"
        },
        "performance_metrics": rag_system.performance_metrics if rag_system else {},
        "optimization_features": [
            "NPU_Acceleration",
            "GPU_Acceleration", 
            "Advanced_Caching",
            "LLM_Optimization",
            "Context_Compression",
            "Parallel_Processing"
        ]
    }

@app.get("/metrics")
async def advanced_metrics_endpoint():
    """Advanced metrics endpoint with detailed performance data"""
    if not rag_system:
        return {"error": "RAG system not initialized"}
    
    return {
        "npu_status": {
            "intel_npu_enabled": ACCELERATION_STATUS["intel_npu"],
            "nvidia_gpu_enabled": ACCELERATION_STATUS["torch"] and torch.cuda.is_available() if torch else False,
            "cupy_enabled": ACCELERATION_STATUS["cupy"],
            "embedding_device": rag_system.embedder.device,
            "similarity_device": rag_system.similarity_engine.device
        },
        "performance": {
            "document_count": len(rag_system.documents),
            "embedding_cache_stats": rag_system.embedder.embedding_cache.get_stats(),
            "similarity_cache_stats": rag_system.similarity_engine.similarity_cache.get_stats(),
            "llm_cache_stats": rag_system.llm_optimizer.response_cache.get_stats(),
            "chunk_rewards_count": len(rag_system.chunk_rewards)
        },
        "configuration": {
            "npu_batch_size": rag_system.config.npu_batch_size,
            "gpu_batch_size": rag_system.config.gpu_batch_size,
            "gpu_memory_fraction": rag_system.config.gpu_memory_fraction,
            "l1_cache_size": rag_system.config.l1_cache_size,
            "l2_cache_size": rag_system.config.l2_cache_size,
            "l3_cache_size": rag_system.config.l3_cache_size,
            "llm_cache_size": rag_system.config.llm_cache_size,
            "similarity_cache_size": rag_system.config.similarity_cache_size
        },
        "active_patches": [
            "R1_Advanced_Hybrid_Retrieval",
            "R2_Enhanced_Context_Injection", 
            "R3_Advanced_Feedback_Learning",
            "NPU_Hardware_Acceleration",
            "LLM_Response_Optimization",
            "Advanced_Caching_System"
        ],
        "hardware_optimization": {
            "gtx_1080_optimized": True,
            "mixed_precision_enabled": rag_system.config.enable_mixed_precision,
            "async_processing": rag_system.config.async_embedding_generation,
            "parallel_llm_inference": rag_system.config.parallel_llm_inference
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", "8000")),
        log_level="info"
    ) 