import asyncio
import logging
import os
from typing import List, Dict, Any
import httpx
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .metrics import (
    ENRICH_LATENCY,
    ENRICH_REQUESTS,
    ENRICH_CHUNKS,
    RETRIEVAL_LATENCY,
    RETRIEVAL_CHUNKS,
    CHUNK_RELEVANCE,
    MISSING_METADATA
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-coordinator")

# Initialize clients
embedder = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
qdrant = QdrantClient(os.getenv("QDRANT_URL", "http://qdrant:6333"))
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8500")

class MetaRequest(BaseModel):
    cids: List[str]

async def fetch_metadata(cids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch metadata for concept IDs from concept dictionary"""
    if not cids:
        return {}
        
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            with ENRICH_LATENCY.time():
                response = await client.post(
                    f"{CONCEPT_DICT_URL}/meta",
                    json=MetaRequest(cids=cids).dict()
                )
                response.raise_for_status()
                meta = response.json()
                
            ENRICH_REQUESTS.labels(status="success").inc()
            ENRICH_CHUNKS.inc(len(meta))
            return meta
            
    except Exception as e:
        logger.error(f"Failed to fetch metadata: {e}")
        ENRICH_REQUESTS.labels(status="error").inc()
        return {}

async def retrieve_chunks(
    query: str,
    collection: str = "concepts",
    top_k: int = 5,
    score_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Retrieve and enrich chunks from vector store
    
    Args:
        query: Search query
        collection: Qdrant collection name
        top_k: Number of chunks to retrieve
        score_threshold: Minimum relevance score
        
    Returns:
        List of chunks with metadata
    """
    try:
        # Generate query embedding
        query_vec = embedder.encode(query).tolist()
        
        # Search vector store
        with RETRIEVAL_LATENCY.time():
            results = qdrant.search(
                collection_name=collection,
                query_vector=query_vec,
                limit=top_k,
                score_threshold=score_threshold
            )
            
        RETRIEVAL_CHUNKS.inc(len(results))
        
        # Extract concept IDs
        cids = []
        for hit in results:
            if "cid" in hit.payload:
                cids.append(hit.payload["cid"])
            CHUNK_RELEVANCE.observe(hit.score)
            
        # Fetch metadata
        meta = await fetch_metadata(cids)
        
        # Enrich chunks
        enriched = []
        for hit in results:
            chunk = {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "meta": meta.get(hit.payload.get("cid", ""), {})
            }
            if not chunk["meta"] and "cid" in hit.payload:
                MISSING_METADATA.inc()
            enriched.append(chunk)
            
        return enriched
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        raise 