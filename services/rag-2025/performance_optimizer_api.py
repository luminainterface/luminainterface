#!/usr/bin/env python3
"""
Performance Optimizer API Service
Provides REST API endpoints for the RAG 2025 Performance Optimizer
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Import our performance optimizer
from performance_optimizer import integrate_performance_optimizer, PerformanceOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG 2025 Performance Optimizer",
    description="Advanced NPU & CPU optimization service for RAG 2025 system",
    version="1.0.0"
)

# Global optimizer instance
optimizer: Optional[PerformanceOptimizer] = None

# Pydantic models for API
class AttentionRequest(BaseModel):
    queries: List[List[float]]
    keys: List[List[float]]
    values: List[List[float]]

class AttentionResponse(BaseModel):
    result: List[List[float]]
    computation_time: float
    cache_hit: bool

class OptimizationMetricsResponse(BaseModel):
    cache_hit_ratio: float
    memory_efficiency: float
    compute_throughput: float
    energy_efficiency: float
    learning_effectiveness: float
    operations_count: int
    vectorized_ops: int
    cache_entries: int

class WeightCompressionRequest(BaseModel):
    weights: List[List[float]]
    threshold: float = 0.01

class WeightCompressionResponse(BaseModel):
    compression_ratio: float
    memory_savings: float
    original_size: int
    compressed_size: int

class SystemStatusResponse(BaseModel):
    status: str
    optimization_enabled: bool
    uptime: float
    version: str

@app.on_event("startup")
async def startup_event():
    """Initialize the performance optimizer on startup"""
    global optimizer
    logger.info("🚀 Starting RAG 2025 Performance Optimizer API...")
    
    try:
        optimizer = integrate_performance_optimizer(enable_async=True)
        logger.info("✅ Performance Optimizer initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Performance Optimizer: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    return {
        "status": "healthy",
        "service": "rag-performance-optimizer",
        "timestamp": time.time()
    }

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status and configuration"""
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    return SystemStatusResponse(
        status="operational",
        optimization_enabled=optimizer.optimization_enabled,
        uptime=time.time(),
        version="1.0.0"
    )

@app.post("/optimize/attention", response_model=AttentionResponse)
async def optimize_attention(request: AttentionRequest):
    """Optimize attention computation with all optimizations"""
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        # Convert input to numpy arrays
        queries = np.array(request.queries, dtype=np.float32)
        keys = np.array(request.keys, dtype=np.float32)
        values = np.array(request.values, dtype=np.float32)
        
        # Validate dimensions
        if queries.shape != keys.shape or queries.shape != values.shape:
            raise HTTPException(
                status_code=400, 
                detail=f"Dimension mismatch: queries={queries.shape}, keys={keys.shape}, values={values.shape}"
            )
        
        # Check cache first
        cache_key = f"attn_{hash(queries.tobytes())}_{hash(keys.tobytes())}"
        cached_result = optimizer.cache_optimizer.get(cache_key)
        cache_hit = cached_result is not None
        
        # Perform optimized attention computation
        start_time = time.time()
        if cache_hit:
            result = cached_result
        else:
            result = await optimizer.optimize_attention_computation(queries, keys, values)
        computation_time = time.time() - start_time
        
        return AttentionResponse(
            result=result.tolist(),
            computation_time=computation_time,
            cache_hit=cache_hit
        )
        
    except Exception as e:
        logger.error(f"Attention optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=OptimizationMetricsResponse)
async def get_performance_metrics():
    """Get current performance metrics"""
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        metrics = optimizer.get_performance_metrics()
        compute_stats = optimizer.compute_optimizer.computation_stats
        
        return OptimizationMetricsResponse(
            cache_hit_ratio=metrics.cache_hit_ratio,
            memory_efficiency=metrics.memory_efficiency,
            compute_throughput=metrics.compute_throughput,
            energy_efficiency=metrics.energy_efficiency,
            learning_effectiveness=metrics.learning_effectiveness,
            operations_count=compute_stats["operations_count"],
            vectorized_ops=compute_stats["vectorized_ops"],
            cache_entries=len(optimizer.cache_optimizer.cache)
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/weights", response_model=WeightCompressionResponse)
async def compress_weights(request: WeightCompressionRequest):
    """Compress model weights using sparse representation"""
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        weights = np.array(request.weights, dtype=np.float32)
        
        # Compress weights
        compressed = optimizer.compress_model_weights(weights, request.threshold)
        
        # Calculate metrics
        original_size = weights.nbytes
        compressed_size = (
            sum(idx.nbytes for idx in compressed["indices"]) + 
            compressed["values"].nbytes
        )
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        memory_savings = 1.0 - (compressed_size / original_size)
        
        return WeightCompressionResponse(
            compression_ratio=compression_ratio,
            memory_savings=memory_savings,
            original_size=original_size,
            compressed_size=compressed_size
        )
        
    except Exception as e:
        logger.error(f"Weight compression failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report")
async def get_optimization_report():
    """Get detailed optimization report"""
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        report = optimizer.generate_optimization_report()
        return {"report": report, "timestamp": time.time()}
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/comprehensive")
async def run_comprehensive_test():
    """Run comprehensive optimization test"""
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        logger.info("🧪 Running comprehensive optimization test...")
        
        # Generate test data
        test_sizes = [32, 64, 128, 256]
        results = []
        
        for size in test_sizes:
            queries = np.random.randn(size, 128).astype(np.float32)
            keys = np.random.randn(size, 128).astype(np.float32)
            values = np.random.randn(size, 128).astype(np.float32)
            
            # Test optimized attention
            start_time = time.time()
            result = await optimizer.optimize_attention_computation(queries, keys, values)
            end_time = time.time()
            
            results.append({
                "size": size,
                "computation_time": end_time - start_time,
                "output_shape": result.shape,
                "throughput": size / (end_time - start_time)
            })
        
        # Get final metrics
        metrics = optimizer.get_performance_metrics()
        
        return {
            "test_results": results,
            "final_metrics": {
                "learning_effectiveness": f"{metrics.learning_effectiveness:.1%}",
                "cache_hit_ratio": f"{metrics.cache_hit_ratio:.1%}",
                "compute_throughput": f"{metrics.compute_throughput:.1%}"
            },
            "status": "✅ TARGET ACHIEVED" if metrics.learning_effectiveness >= 1.80 else "🎯 APPROACHING TARGET",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Comprehensive test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
async def clear_cache():
    """Clear optimization cache"""
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    try:
        optimizer.cache_optimizer.cache.clear()
        optimizer.cache_optimizer.access_count.clear()
        optimizer.cache_optimizer.access_order.clear()
        
        return {
            "status": "cache_cleared",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "RAG 2025 Performance Optimizer",
        "version": "1.0.0",
        "status": "operational" if optimizer else "initializing",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "optimize_attention": "/optimize/attention",
            "compress_weights": "/optimize/weights",
            "metrics": "/metrics",
            "report": "/report",
            "test": "/test/comprehensive",
            "clear_cache": "/cache/clear"
        },
        "description": "World's first chat-triggered circular growth RAG with advanced NPU/CPU optimizations"
    }

if __name__ == "__main__":
    uvicorn.run(
        "performance_optimizer_api:app",
        host="0.0.0.0",
        port=8909,
        log_level="info",
        reload=False
    ) 