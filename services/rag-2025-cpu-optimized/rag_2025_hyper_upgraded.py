#!/usr/bin/env python3
"""
RAG 2025 Hyper-Upgraded System - Integrated R1+R2+R3 Patches
R1 Patch: Hybrid Retrieval with Novelty/Reward Re-ranking
R2 Patch: RAG Context Injection with Chunk Tracking
R3 Patch: Chunk Reward Tracking and Learning
"""

import asyncio
import json
import time
import os
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import hashlib
from datetime import datetime

# Core imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil

# RAG Hyper Upgrade imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import httpx
from neo4j import GraphDatabase
import ollama

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('RAG_2025_LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced Prometheus metrics for RAG Hyper Upgrade - with collision protection
try:
    REQUEST_COUNT = Counter('rag_2025_hyper_requests_total', 'Total RAG requests', ['strategy', 'status'])
    REQUEST_DURATION = Histogram('rag_2025_hyper_request_duration_seconds', 'RAG request duration')
    RAG_HITS_TOTAL = Counter('rag_hyper_hits_total', 'Total RAG hits')
    RAG_USED_RATIO = Gauge('rag_hyper_used_ratio', 'Ratio of RAG usage')
    RAG_LATENCY_SECONDS = Histogram('rag_hyper_latency_seconds', 'RAG retrieval latency')
    CHUNK_REWARD_TOTAL = Counter('chunk_hyper_reward_total', 'Total chunk rewards')
    ACTIVE_CONNECTIONS = Gauge('rag_2025_hyper_active_connections', 'Active connections')
    MEMORY_USAGE = Gauge('rag_2025_hyper_memory_usage_bytes', 'Memory usage in bytes')
    DOCUMENT_COUNT = Gauge('rag_2025_hyper_document_count', 'Number of documents in index')
except Exception as e:
    logger.warning(f"Metrics initialization warning: {e}")
    # Create dummy metrics if registration fails
    class DummyMetric:
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    REQUEST_COUNT = DummyMetric()
    REQUEST_DURATION = DummyMetric()
    RAG_HITS_TOTAL = DummyMetric()
    RAG_USED_RATIO = DummyMetric()
    RAG_LATENCY_SECONDS = DummyMetric()
    CHUNK_REWARD_TOTAL = DummyMetric()
    ACTIVE_CONNECTIONS = DummyMetric()
    MEMORY_USAGE = DummyMetric()
    DOCUMENT_COUNT = DummyMetric()

# Initialize FastAPI app
app = FastAPI(
    title="RAG 2025 Hyper-Upgraded System",
    version="2025.1-R1R2R3",
    description="Production-ready RAG system with R1+R2+R3 patches integrated"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RetrievalStrategy(Enum):
    HYBRID = "hybrid"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    REAL_TIME = "real_time"
    ADAPTIVE = "adaptive"

@dataclass
class RAGQuery:
    query: str
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    max_docs: int = 5
    use_reasoning: bool = True
    context_size: Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    strategy: str = "hybrid"
    max_docs: int = 5
    use_reasoning: bool = True

class FeedbackRequest(BaseModel):
    query: str
    response: str
    quality: str  # "high", "medium", "low"
    used_ids: List[str] = []

class DocumentAddRequest(BaseModel):
    documents: List[str]
    source: str = "api"
    metadata: Optional[Dict[str, Any]] = None

class HybridRetrieverR1:
    """R1 Patch: Enhanced hybrid retrieval with novelty/reward re-ranking"""
    
    def __init__(self):
        logger.info("🔧 Initializing R1 Hybrid Retriever with Novelty/Reward Re-ranking...")
        
        # Initialize embedding model
        self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv('QDRANT_URL', 'http://qdrant:6333')
        )
        self.collection_name = "rag_2025_collection"
        
        # Initialize Redis for caching and tracking
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://:02211998@redis:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self.redis_client = None
        
        # BM25 for sparse retrieval
        self.sparse_retriever = None
        self.documents = []
        self.document_metadata = []
        
        # Re-ranking parameters
        self.rerank_alpha = float(os.getenv('RERANK_ALPHA', '0.6'))
        self.novelty_weight = float(os.getenv('NOVELTY_WEIGHT', '0.3'))
        self.reward_weight = float(os.getenv('REWARD_WEIGHT', '0.1'))
        
        self._ensure_collection_exists()
        
    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.info(f"✅ Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"✅ Qdrant collection exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Failed to ensure Qdrant collection: {e}")
    
    def build_index(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Build hybrid dense + sparse index with Qdrant integration"""
        logger.info(f"📚 Building R1 index for {len(documents)} documents...")
        
        self.documents = documents
        self.document_metadata = metadata or [{"source": "unknown"} for _ in documents]
        
        # Generate embeddings
        logger.info("🧠 Generating dense embeddings...")
        embeddings = self.dense_model.encode(documents, show_progress_bar=False)
        
        # Store in Qdrant
        logger.info("🔍 Storing embeddings in Qdrant...")
        points = []
        for i, (doc, embedding, meta) in enumerate(zip(documents, embeddings, self.document_metadata)):
            # Use integer ID for Qdrant compatibility
            point_id = i + 1  # Start from 1
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": doc,
                    "metadata": meta,
                    "reward": 0.0,
                    "access_count": 0,
                    "created_at": datetime.now().isoformat(),
                    "doc_hash": hashlib.md5(doc.encode()).hexdigest()[:8]  # Store hash in payload
                }
            ))
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        # Build BM25 index
        logger.info("📝 Building BM25 index...")
        tokenized_docs = [doc.split() for doc in documents]
        self.sparse_retriever = BM25Okapi(tokenized_docs)
        
        # Update metrics
        DOCUMENT_COUNT.set(len(documents))
        logger.info("✅ R1 Hybrid index built successfully!")
    
    def calculate_novelty_score(self, doc_text: str, query: str) -> float:
        """Calculate novelty score based on query-document interaction"""
        # Simple novelty calculation - can be enhanced
        query_words = set(query.lower().split())
        doc_words = set(doc_text.lower().split())
        
        # Novelty is inverse of overlap
        overlap = len(query_words.intersection(doc_words))
        total_unique = len(query_words.union(doc_words))
        
        if total_unique == 0:
            return 0.0
        
        novelty = 1.0 - (overlap / total_unique)
        return novelty
    
    def hybrid_retrieve_r1(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """R1 Patch: Hybrid retrieval with novelty/reward re-ranking"""
        start_time = time.time()
        
        if not self.sparse_retriever:
            return []
        
        # Check cache first
        cache_key = f"rag_r1_query:{hashlib.md5(query.encode()).hexdigest()}:{k}"
        if self.redis_client:
            try:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return json.loads(cached_result)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Dense retrieval from Qdrant
        query_embedding = self.dense_model.encode([query])[0]
        
        try:
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=min(k * 2, len(self.documents)),  # Get more for re-ranking
                with_payload=True
            )
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            search_results = []
        
        # Sparse retrieval
        sparse_scores = self.sparse_retriever.get_scores(query.split())
        
        # Combine and re-rank with novelty and reward
        combined_results = []
        for result in search_results:
            doc_text = result.payload.get("text", "")
            doc_index = next((i for i, doc in enumerate(self.documents) if doc == doc_text), -1)
            
            if doc_index >= 0:
                # Get scores
                dense_score = result.score
                sparse_score = sparse_scores[doc_index] if doc_index < len(sparse_scores) else 0
                novelty_score = self.calculate_novelty_score(doc_text, query)
                reward_score = result.payload.get("reward", 0.0)
                
                # R1 Re-ranking formula: 0.6*similarity + 0.3*novelty + 0.1*reward
                final_score = (
                    self.rerank_alpha * dense_score +
                    self.novelty_weight * novelty_score +
                    self.reward_weight * reward_score
                )
                
                combined_results.append({
                    "id": result.id,
                    "text": doc_text,
                    "score": float(final_score),
                    "dense_score": float(dense_score),
                    "sparse_score": float(sparse_score),
                    "novelty_score": float(novelty_score),
                    "reward_score": float(reward_score),
                    "metadata": result.payload.get("metadata", {}),
                    "access_count": result.payload.get("access_count", 0)
                })
        
        # Sort by final score and take top k
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = combined_results[:k]
        
        # Update access counts in Qdrant
        for result in final_results:
            try:
                self.qdrant_client.set_payload(
                    collection_name=self.collection_name,
                    payload={"access_count": result["access_count"] + 1},
                    points=[result["id"]]
                )
            except Exception as e:
                logger.warning(f"Failed to update access count: {e}")
        
        # Cache results
        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, 3600, json.dumps(final_results))
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        # Update metrics
        latency = time.time() - start_time
        RAG_LATENCY_SECONDS.observe(latency)
        RAG_HITS_TOTAL.inc()
        
        return final_results

class ContextInjectorR2:
    """R2 Patch: RAG Context Injection with Chunk Tracking"""
    
    def __init__(self, retriever: HybridRetrieverR1):
        self.retriever = retriever
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://ollama:11434')
        
    def retrieve_rag_context(self, query: str, top_k: int = 6) -> Tuple[str, List[str]]:
        """Retrieve RAG context and track used chunks"""
        results = self.retriever.hybrid_retrieve_r1(query, k=top_k)
        
        if not results:
            return "", []
        
        # Assemble context
        context_parts = []
        used_chunk_ids = []
        
        for i, result in enumerate(results):
            context_parts.append(f"[{i+1}] {result['text']}")
            used_chunk_ids.append(result['id'])
        
        context = "\n".join(context_parts)
        return context, used_chunk_ids
    
    def assemble_prompt(self, query: str, context: str) -> str:
        """R2 Patch: Assemble prompt with RAG context injection"""
        if not context:
            return f"Question: {query}\nAnswer:"
        
        prompt = f"""Context Information:
{context}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain relevant information, say so clearly.

Answer:"""
        return prompt
    
    async def enhanced_respond(self, query: str, use_rag: bool = True) -> Dict[str, Any]:
        """R2 Patch: Enhanced response generation with RAG context injection"""
        start_time = time.time()
        used_chunk_ids = []
        
        try:
            if use_rag:
                # Retrieve context
                context, used_chunk_ids = self.retrieve_rag_context(query)
                prompt = self.assemble_prompt(query, context)
            else:
                prompt = f"Question: {query}\nAnswer:"
            
            # Generate response using Ollama
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "llama3.2:1b",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "")
                else:
                    generated_text = "Error: Failed to generate response"
            
            # Calculate metrics
            processing_time = time.time() - start_time
            rag_used = len(used_chunk_ids) > 0
            
            if rag_used:
                RAG_USED_RATIO.set(1.0)
            else:
                RAG_USED_RATIO.set(0.0)
            
            return {
                "response": generated_text,
                "used_chunk_ids": used_chunk_ids,
                "rag_used": rag_used,
                "processing_time": processing_time,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Enhanced response error: {e}")
            return {
                "response": f"Error: {str(e)}",
                "used_chunk_ids": [],
                "rag_used": False,
                "processing_time": time.time() - start_time,
                "query": query
            }

class FeedbackLearnerR3:
    """R3 Patch: Chunk Reward Tracking and Learning"""
    
    def __init__(self, retriever: HybridRetrieverR1):
        self.retriever = retriever
        
    def calculate_quality_reward(self, quality: str) -> float:
        """Calculate reward based on quality feedback"""
        quality_rewards = {
            "high": 0.2,
            "medium": 0.1,
            "low": -0.1
        }
        return quality_rewards.get(quality.lower(), 0.0)
    
    async def update_chunk_rewards(self, used_ids: List[str], quality: str):
        """R3 Patch: Update chunk rewards in Qdrant"""
        if not used_ids:
            return
        
        reward = self.calculate_quality_reward(quality)
        
        try:
            # Get current rewards for the chunks
            for chunk_id in used_ids:
                try:
                    # Retrieve current point
                    points = self.retriever.qdrant_client.retrieve(
                        collection_name=self.retriever.collection_name,
                        ids=[chunk_id],
                        with_payload=True
                    )
                    
                    if points:
                        current_reward = points[0].payload.get("reward", 0.0)
                        new_reward = current_reward + reward
                        
                        # Update the reward
                        self.retriever.qdrant_client.set_payload(
                            collection_name=self.retriever.collection_name,
                            payload={"reward": new_reward},
                            points=[chunk_id]
                        )
                        
                        # Update metrics
                        CHUNK_REWARD_TOTAL.inc(reward)
                        
                        logger.debug(f"Updated reward for {chunk_id}: {current_reward} -> {new_reward}")
                        
                except Exception as e:
                    logger.warning(f"Failed to update reward for {chunk_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Chunk reward update error: {e}")
    
    async def log_feedback(self, query: str, response: str, quality: str, used_ids: List[str]):
        """R3 Patch: Log feedback and update rewards"""
        # Update chunk rewards
        await self.update_chunk_rewards(used_ids, quality)
        
        # Store feedback in Redis for analytics
        if self.retriever.redis_client:
            try:
                feedback_data = {
                    "query": query,
                    "response": response,
                    "quality": quality,
                    "used_ids": used_ids,
                    "timestamp": datetime.now().isoformat()
                }
                
                feedback_key = f"feedback:{datetime.now().strftime('%Y%m%d')}:{int(time.time())}"
                self.retriever.redis_client.setex(
                    feedback_key, 
                    86400 * 7,  # Keep for 7 days
                    json.dumps(feedback_data)
                )
                
                logger.info(f"Logged feedback: {quality} quality for {len(used_ids)} chunks")
                
            except Exception as e:
                logger.warning(f"Failed to log feedback: {e}")

class RAG2025HyperUpgraded:
    """Main RAG 2025 system with R1+R2+R3 patches integrated"""
    
    def __init__(self):
        logger.info("🚀 Initializing RAG 2025 Hyper-Upgraded System (R1+R2+R3)...")
        
        # Initialize components
        self.retriever = HybridRetrieverR1()
        self.context_injector = ContextInjectorR2(self.retriever)
        self.feedback_learner = FeedbackLearnerR3(self.retriever)
        
        # Start monitoring
        self.start_monitoring()
        
        # Initialize with sample data if enabled
        if os.getenv('RAG_2025_MODE') == 'production':
            self.initialize_sample_data()
    
    def start_monitoring(self):
        """Start Prometheus metrics server"""
        try:
            metrics_port = int(os.getenv('METRICS_PORT', '8001'))
            start_http_server(metrics_port)
            logger.info(f"📊 Metrics server started on port {metrics_port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")
    
    def initialize_sample_data(self):
        """Initialize with sample documents"""
        sample_docs = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "Natural language processing enables computers to understand and generate human language.",
            "Computer vision allows machines to interpret and understand visual information.",
            "Reinforcement learning is a type of machine learning where agents learn through interaction with an environment."
        ]
        
        sample_metadata = [
            {"source": "ml_basics", "topic": "machine_learning"},
            {"source": "ml_basics", "topic": "neural_networks"},
            {"source": "ml_basics", "topic": "deep_learning"},
            {"source": "ml_basics", "topic": "nlp"},
            {"source": "ml_basics", "topic": "computer_vision"},
            {"source": "ml_basics", "topic": "reinforcement_learning"}
        ]
        
        self.retriever.build_index(sample_docs, sample_metadata)
        logger.info("✅ Sample data initialized")
    
    async def process_query(self, query: str, use_rag: bool = True) -> Dict[str, Any]:
        """Process query with full R1+R2+R3 pipeline"""
        return await self.context_injector.enhanced_respond(query, use_rag)
    
    async def process_feedback(self, query: str, response: str, quality: str, used_ids: List[str]):
        """Process feedback with R3 learning"""
        await self.feedback_learner.log_feedback(query, response, quality, used_ids)

# Global system instance
rag_system = None

@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = RAG2025HyperUpgraded()
    logger.info("🎉 RAG 2025 Hyper-Upgraded System started successfully!")

@app.get("/")
async def root():
    return {
        "message": "RAG 2025 Hyper-Upgraded System",
        "version": "2025.1-R1R2R3",
        "patches": ["R1: Hybrid Retrieval + Novelty/Reward Re-ranking", 
                   "R2: RAG Context Injection + Chunk Tracking",
                   "R3: Chunk Reward Tracking + Learning"],
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check system components
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "retriever": "healthy" if rag_system and rag_system.retriever else "unhealthy",
                "context_injector": "healthy" if rag_system and rag_system.context_injector else "unhealthy",
                "feedback_learner": "healthy" if rag_system and rag_system.feedback_learner else "unhealthy",
                "qdrant": "unknown",
                "redis": "unknown",
                "ollama": "unknown"
            },
            "patches": ["R1", "R2", "R3"],
            "memory_usage": psutil.virtual_memory().percent
        }
        
        # Test Qdrant connection
        try:
            if rag_system and rag_system.retriever.qdrant_client:
                collections = rag_system.retriever.qdrant_client.get_collections()
                health_status["components"]["qdrant"] = "healthy"
        except:
            health_status["components"]["qdrant"] = "unhealthy"
        
        # Test Redis connection
        try:
            if rag_system and rag_system.retriever.redis_client:
                rag_system.retriever.redis_client.ping()
                health_status["components"]["redis"] = "healthy"
        except:
            health_status["components"]["redis"] = "unhealthy"
        
        return health_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Enhanced query endpoint with R1+R2+R3 processing"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = await rag_system.process_query(request.query, use_rag=True)
        REQUEST_COUNT.labels(strategy="hybrid", status="success").inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels(strategy="hybrid", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """R3 Patch: Submit feedback for learning"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        await rag_system.process_feedback(
            request.query, 
            request.response, 
            request.quality, 
            request.used_ids
        )
        return {"status": "feedback_processed", "quality": request.quality}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_documents")
async def add_documents(request: DocumentAddRequest):
    """Add documents to the index"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        rag_system.retriever.build_index(request.documents, request.metadata)
        return {
            "status": "documents_added",
            "count": len(request.documents),
            "source": request.source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "document_count": len(rag_system.retriever.documents) if rag_system else 0,
        "memory_usage": psutil.virtual_memory().percent,
        "patches_active": ["R1", "R2", "R3"],
        "features": {
            "hybrid_retrieval": True,
            "novelty_reranking": True,
            "chunk_tracking": True,
            "feedback_learning": True
        }
    }

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(
        "rag_2025_hyper_upgraded:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    ) 