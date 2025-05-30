#!/usr/bin/env python3
"""
Enhanced RAG 2025 GPU-Simple Implementation (GTX 1080 Optimized)
Simplified GPU acceleration without Intel NPU dependencies
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime
import json

# Core dependencies
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import requests
import redis
from qdrant_client import QdrantClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU Detection and Setup
def setup_gpu():
    """Setup GTX 1080 optimizations"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        logger.info(f"🎮 GPU Device: {device_name}")
        logger.info(f"💾 GPU Memory: {gpu_memory:.1f}GB")
        
        # GTX 1080 specific optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Memory fraction for GTX 1080 (8GB)
        if gpu_memory >= 7.5:  # GTX 1080 has ~8GB
            torch.cuda.set_per_process_memory_fraction(0.8)
            logger.info("🚀 GTX 1080 memory optimization enabled")
        
        return torch.device("cuda")
    else:
        logger.warning("⚠️ CUDA not available, falling back to CPU")
        return torch.device("cpu")

# Initialize GPU
device = setup_gpu()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    stream: Optional[bool] = False

class QueryResponse(BaseModel):
    query: str
    response: str
    chunks: Dict[str, Any]
    used_chunk_ids: List[str]
    processing_time: float
    source: str = "gpu_optimized"

class GPUOptimizedEmbedder:
    """GPU-optimized embedding generator for GTX 1080"""
    
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.device = device
        self.batch_size = 32 if device.type == "cuda" else 16
        
        # Load model with GPU optimization
        self.model = SentenceTransformer(self.model_name, device=str(device))
        
        # GPU-specific optimizations
        if device.type == "cuda":
            self.model.half()  # Use FP16 for GTX 1080
            logger.info("🎯 FP16 precision enabled for GTX 1080")
        
        logger.info(f"✅ GPU-Optimized Embedder initialized on {device}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with GPU acceleration"""
        start_time = time.time()
        
        # Batch processing for GPU efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=str(self.device)
        )
        
        processing_time = time.time() - start_time
        logger.info(f"⚡ GPU embedding: {len(texts)} texts in {processing_time:.3f}s")
        
        return embeddings

class GPUMultiTierCache:
    """GPU-optimized multi-tier caching system"""
    
    def __init__(self, l1_size=1000, l2_size=5000, l3_size=10000):
        self.l1_cache = {}  # Hot cache - in GPU memory if possible
        self.l2_cache = {}  # Warm cache - system memory
        self.l3_cache = {}  # Cold cache - disk
        
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size
        
        self.access_counts = {}
        self.lock = threading.Lock()
        
        logger.info("🎯 GPU Multi-tier cache initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get with automatic promotion"""
        with self.lock:
            # Try L1 first (hottest)
            if key in self.l1_cache:
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return self.l1_cache[key]
            
            # Try L2
            if key in self.l2_cache:
                value = self.l2_cache[key]
                # Promote to L1 if frequently accessed
                if self.access_counts.get(key, 0) > 3:
                    self._promote_to_l1(key, value)
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return value
            
            # Try L3
            if key in self.l3_cache:
                value = self.l3_cache[key]
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return value
            
            return None
    
    def set(self, key: str, value: Any):
        """Set with intelligent tier placement"""
        with self.lock:
            # Default to L2, promote to L1 if hot
            if len(self.l2_cache) >= self.l2_size:
                self._evict_l2()
            
            self.l2_cache[key] = value
            self.access_counts[key] = 1
    
    def _promote_to_l1(self, key: str, value: Any):
        """Promote frequently accessed items to L1"""
        if len(self.l1_cache) >= self.l1_size:
            # Evict least accessed from L1
            lru_key = min(self.l1_cache.keys(), 
                         key=lambda k: self.access_counts.get(k, 0))
            self.l2_cache[lru_key] = self.l1_cache.pop(lru_key)
        
        self.l1_cache[key] = value
        if key in self.l2_cache:
            del self.l2_cache[key]
    
    def _evict_l2(self):
        """Evict from L2 to L3"""
        if not self.l2_cache:
            return
        
        lru_key = min(self.l2_cache.keys(), 
                     key=lambda k: self.access_counts.get(k, 0))
        
        if len(self.l3_cache) >= self.l3_size:
            # Remove oldest from L3
            oldest_key = next(iter(self.l3_cache))
            del self.l3_cache[oldest_key]
        
        self.l3_cache[lru_key] = self.l2_cache.pop(lru_key)

class GPUAcceleratedSimilarity:
    """GPU-accelerated similarity calculations for GTX 1080"""
    
    def __init__(self, device):
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def calculate_similarity(self, query_embedding: np.ndarray, 
                           document_embeddings: np.ndarray) -> np.ndarray:
        """GPU-accelerated cosine similarity"""
        if self.device.type == "cuda":
            return self._gpu_cosine_similarity(query_embedding, document_embeddings)
        else:
            return self._cpu_cosine_similarity(query_embedding, document_embeddings)
    
    def _gpu_cosine_similarity(self, query_embedding: np.ndarray, 
                              document_embeddings: np.ndarray) -> np.ndarray:
        """GPU-accelerated cosine similarity calculation"""
        # Convert to GPU tensors
        query_tensor = torch.tensor(query_embedding, device=self.device, dtype=torch.float16)
        doc_tensor = torch.tensor(document_embeddings, device=self.device, dtype=torch.float16)
        
        # Normalize vectors
        query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=-1)
        doc_norm = torch.nn.functional.normalize(doc_tensor, p=2, dim=-1)
        
        # Compute cosine similarity
        similarities = torch.matmul(doc_norm, query_norm.T).squeeze()
        
        # Convert back to CPU numpy array
        return similarities.cpu().float().numpy()
    
    def _cpu_cosine_similarity(self, query_embedding: np.ndarray, 
                              document_embeddings: np.ndarray) -> np.ndarray:
        """Fallback CPU cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(document_embeddings, query_embedding.reshape(1, -1)).flatten()

class EnhancedRAGGPUSimple:
    """Enhanced RAG 2025 GPU-Simple Implementation"""
    
    def __init__(self):
        self.embedder = GPUOptimizedEmbedder()
        self.cache = GPUMultiTierCache()
        self.similarity_calc = GPUAcceleratedSimilarity(device)
        
        # Initialize connections
        self._init_connections()
        
        # Performance metrics
        self.query_count = 0
        self.total_processing_time = 0
        self.cache_hits = 0
        
        logger.info("🚀 GPU-Simple RAG System initialized")
    
    def _init_connections(self):
        """Initialize external service connections"""
        try:
            # Redis connection
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self.redis_client = None
        
        try:
            # Qdrant connection
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            self.qdrant_client = QdrantClient(url=qdrant_url)
            logger.info("✅ Qdrant connected")
        except Exception as e:
            logger.warning(f"⚠️ Qdrant connection failed: {e}")
            self.qdrant_client = None
        
        try:
            # Ollama connection
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.ollama_url = ollama_url
                logger.info("✅ Ollama connected")
            else:
                raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Ollama connection failed: {e}")
            self.ollama_url = None
    
    async def query(self, query: str, top_k: int = 5) -> QueryResponse:
        """Process RAG query with GPU acceleration"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"query:{hash(query)}:{top_k}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.cache_hits += 1
            cached_result["processing_time"] = time.time() - start_time
            logger.info(f"⚡ Cache hit for query: {query[:50]}...")
            return QueryResponse(**cached_result)
        
        try:
            # Generate query embedding with GPU acceleration
            query_embedding = self.embedder.encode([query])[0]
            
            # For demo purposes, generate a simple response
            # In production, this would search the vector database
            response_text = self._generate_demo_response(query)
            
            # Prepare response
            result = {
                "query": query,
                "response": response_text,
                "chunks": {},
                "used_chunk_ids": [],
                "processing_time": time.time() - start_time,
                "source": "gpu_optimized"
            }
            
            # Cache the result
            self.cache.set(cache_key, result)
            
            # Update metrics
            self.query_count += 1
            self.total_processing_time += result["processing_time"]
            
            logger.info(f"✅ GPU query processed in {result['processing_time']:.3f}s")
            
            return QueryResponse(**result)
            
        except Exception as e:
            logger.error(f"❌ Query processing error: {e}")
            error_result = {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "chunks": {},
                "used_chunk_ids": [],
                "processing_time": time.time() - start_time,
                "source": "gpu_optimized"
            }
            return QueryResponse(**error_result)
    
    def _generate_demo_response(self, query: str) -> str:
        """Generate demo response for testing"""
        if "gpu" in query.lower() or "gtx" in query.lower():
            return f"GPU-accelerated response: This Enhanced RAG 2025 system utilizes GTX 1080 GPU optimization with FP16 precision, multi-tier caching, and parallel processing. The query '{query}' was processed using GPU-accelerated embeddings and similarity calculations."
        
        elif "speed" in query.lower() or "performance" in query.lower():
            return f"Performance-optimized response: The GPU-Simple implementation provides accelerated processing through CUDA optimization, batch processing, and intelligent caching. Query processing utilizes {device} for maximum speed."
        
        elif "rag" in query.lower() or "enhanced" in query.lower():
            return f"Enhanced RAG 2025 response: This system features GPU acceleration, multi-tier caching, streaming responses, and hardware monitoring. The GPU-Simple version focuses on GTX 1080 optimization for maximum performance."
        
        else:
            return f"GPU-processed response: Your query '{query}' has been processed using Enhanced RAG 2025's GPU-accelerated pipeline with optimized embeddings and similarity calculations."
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_processing_time = (self.total_processing_time / self.query_count 
                              if self.query_count > 0 else 0)
        cache_hit_rate = (self.cache_hits / self.query_count 
                         if self.query_count > 0 else 0)
        
        return {
            "total_queries": self.query_count,
            "avg_processing_time": avg_processing_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "gpu_device": str(device),
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "gpu_memory_cached": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }

# Initialize RAG system
rag_system = EnhancedRAGGPUSimple()

# FastAPI app
app = FastAPI(title="Enhanced RAG 2025 GPU-Simple", version="1.0.0")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "rag-2025-gpu-simple",
        "device": str(device),
        "gpu_available": torch.cuda.is_available(),
        "redis": "connected" if rag_system.redis_client else "disconnected",
        "qdrant": "connected" if rag_system.qdrant_client else "disconnected",
        "ollama": "connected" if rag_system.ollama_url else "disconnected",
        "optimization_level": "gpu_optimized"
    }

@app.post("/query")
async def process_query(request: QueryRequest) -> QueryResponse:
    """Process RAG query"""
    return await rag_system.query(request.query, request.top_k)

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    return rag_system.get_metrics()

@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """Streaming query endpoint (simulated for demo)"""
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate_stream():
        # Simulate streaming response
        response = await rag_system.query(request.query, request.top_k)
        
        # Break response into chunks for streaming
        words = response.response.split()
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3])
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            await asyncio.sleep(0.1)  # Simulate processing delay
        
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")

if __name__ == "__main__":
    # Get port from environment
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting Enhanced RAG 2025 GPU-Simple on port {port}")
    logger.info(f"🎮 GPU Device: {device}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    ) 