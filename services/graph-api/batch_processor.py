from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import asyncio
from loguru import logger
from prometheus_client import Counter, Histogram
from qdrant_client import QdrantClient
import redis.asyncio as redis
import json
import os

# Metrics
BATCH_PROCESSED = Counter(
    'graph_api_batches_processed_total',
    'Number of vector batches processed',
    ['status']  # 'success', 'error'
)

BATCH_LATENCY = Histogram(
    'graph_api_batch_processing_seconds',
    'Time spent processing vector batches'
)

CHUNK_SIZE = 100  # Number of vectors to process in each chunk
MAX_CONCURRENT_CHUNKS = 4  # Maximum number of chunks to process concurrently

class BatchProcessor:
    def __init__(self, model: nn.Module, device: torch.device, redis_url: str, qdrant_url: str):
        self.model = model
        self.device = device
        self.redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
        
    async def process_chunk(self, chunk: List[Dict[str, Any]], chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Process a single chunk of vectors."""
        async with self.semaphore:
            try:
                start_time = datetime.utcnow()
                
                # Extract vectors and metadata
                vectors = [item["embedding"] for item in chunk]
                metadata = [item["metadata"] for item in chunk]
                
                # Convert to tensors
                vector_tensor = torch.tensor(vectors, dtype=torch.float32, device=self.device)
                
                # Process through model
                self.model.train()
                with torch.no_grad():
                    output = self.model(vector_tensor)
                
                # Convert output to numpy and then to list
                output_vectors = output.cpu().numpy().tolist()
                
                # Prepare results
                results = []
                for i, (vector, meta) in enumerate(zip(output_vectors, metadata)):
                    node_id = meta.get("node_id")
                    if node_id:
                        # Store in Redis
                        await self.redis_client.set(
                            f"node:embedding:{node_id}",
                            json.dumps(vector)
                        )
                        
                        # Update Qdrant
                        self.qdrant_client.set_payload(
                            collection_name="graph_nodes",
                            payload={
                                "embedding": vector,
                                "last_updated": datetime.utcnow().isoformat(),
                                "chunk_index": chunk_index,
                                "total_chunks": total_chunks
                            },
                            points=[node_id]
                        )
                        
                        results.append({
                            "node_id": node_id,
                            "status": "success",
                            "vector_size": len(vector)
                        })
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                BATCH_LATENCY.observe(processing_time)
                
                return {
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "processed_vectors": len(results),
                    "processing_time": processing_time,
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
                BATCH_PROCESSED.labels(status="error").inc()
                return {
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "error": str(e),
                    "status": "error"
                }
    
    async def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of vectors in chunks."""
        try:
            vectors = batch["vectors"]
            batch_id = batch["batch_id"]
            total_vectors = len(vectors)
            
            # Calculate chunks
            chunks = [vectors[i:i + CHUNK_SIZE] for i in range(0, total_vectors, CHUNK_SIZE)]
            total_chunks = len(chunks)
            
            logger.info(f"Processing batch {batch_id} with {total_vectors} vectors in {total_chunks} chunks")
            
            # Process chunks concurrently
            chunk_tasks = [
                self.process_chunk(chunk, i, total_chunks)
                for i, chunk in enumerate(chunks)
            ]
            
            chunk_results = await asyncio.gather(*chunk_tasks)
            
            # Aggregate results
            successful_chunks = [r for r in chunk_results if r["status"] != "error"]
            failed_chunks = [r for r in chunk_results if r["status"] == "error"]
            
            total_processed = sum(r["processed_vectors"] for r in successful_chunks)
            
            result = {
                "batch_id": batch_id,
                "total_vectors": total_vectors,
                "total_chunks": total_chunks,
                "successful_chunks": len(successful_chunks),
                "failed_chunks": len(failed_chunks),
                "total_processed": total_processed,
                "chunk_results": chunk_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if failed_chunks:
                BATCH_PROCESSED.labels(status="error").inc()
                result["status"] = "partial_success"
            else:
                BATCH_PROCESSED.labels(status="success").inc()
                result["status"] = "success"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing batch {batch.get('batch_id', 'unknown')}: {str(e)}")
            BATCH_PROCESSED.labels(status="error").inc()
            return {
                "batch_id": batch.get("batch_id", "unknown"),
                "error": str(e),
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            } 