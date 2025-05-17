import os, asyncio, logging, time, uuid
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import aiohttp
from datetime import datetime
from lumina_core.common.bus import BusClient
from lumina_core.common.retry import with_retry

# ── Config ────────────────────────────────────────────────────────────────────
RETRIEVER_URL   = os.getenv("RETRIEVER_URL",  "http://retriever:8001/retrieve")
LLM_URL         = os.getenv("LLM_URL",        "http://ollama:11434/generate")
TOP_K           = int(os.getenv("RAG_TOP_K",              5))
RETRY_ATTEMPTS  = int(os.getenv("RAG_RETRIES",            3))
TIMEOUT_SEC     = float(os.getenv("RAG_TIMEOUT",          30))
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL,
    format="%(asctime)s | RAG | %(levelname)s | %(message)s")
log = logging.getLogger("RAG")

# ── FastAPI app & metrics ─────────────────────────────────────────────────────
app = FastAPI(title="RAG-Coordinator", version="0.3.0")
Instrumentator().instrument(app).expose(app)

REQ_LATENCY = Histogram(
    "rag_request_seconds",
    "RAG end-to-end latency",
    buckets=(.2,.5,1,2,5,10,30))
REQ_COUNTER = Counter(
    "rag_requests_total",
    "Total RAG queries",
    ["status"])
CONTEXT_TOK = Gauge(
    "rag_context_tokens",
    "Tokens passed to LLM per request")

# ── Pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    temperature: Optional[float] = 0.2

class SourceChunk(BaseModel):
    id:     str
    score:  float
    payload: Dict[str, Any]

class QueryResponse(BaseModel):
    answer:  str
    sources: List[SourceChunk]
    latency: float

# ── Helpers ───────────────────────────────────────────────────────────────────
_client = httpx.AsyncClient(timeout=TIMEOUT_SEC)

async def _call_with_retry(method, url, **kwargs):
    for attempt in range(1, RETRY_ATTEMPTS+1):
        try:
            resp = await _client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == RETRY_ATTEMPTS:
                raise
            await asyncio.sleep(1 * attempt)
            log.warning(f"{url} attempt {attempt} failed: {e}")

def _build_prompt(query:str, ctx:str) -> str:
    return (f"Answer the question truthfully. Cite sources.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:")

# ── API endpoints ─────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
@REQ_LATENCY.time()                # ⏱  histogram wrapper
async def query_rag(req: QueryRequest):
    tic = time.perf_counter()

    # 1. Retrieve top-k chunks
    r_resp = await _call_with_retry(
        "POST", RETRIEVER_URL,
        json={"query": req.query, "top_k": req.top_k})
    chunks = [SourceChunk(**c) for c in r_resp.json()]
    if not chunks:
        REQ_COUNTER.labels("no_context").inc()
        raise HTTPException(404, "No context found")

    ctx_text = "\n\n".join(c.payload.get("content", str(c.payload)) for c in chunks)
    CONTEXT_TOK.set(len(ctx_text.split()))

    # 2. Call LLM
    prompt = _build_prompt(req.query, ctx_text)
    llm_payload = {
        "prompt": prompt,
        "temperature": req.temperature,
        "stream": False
    }
    l_resp = await _call_with_retry("POST", LLM_URL, json=llm_payload)
    #  Ollama returns { "response": "...", "done":true } for non-stream
    answer = l_resp.json().get("response") or l_resp.json().get("answer") or l_resp.text

    REQ_COUNTER.labels("success").inc()
    return QueryResponse(
        answer=answer.strip(),
        sources=chunks,
        latency=round(time.perf_counter()-tic, 3))

@app.get("/health")
async def health():
    try:
        _ = await _call_with_retry("GET", RETRIEVER_URL.replace("/retrieve","/health"))
        # LLM: Ollama has /api/tags; use head request
        _ = await _call_with_retry("GET", LLM_URL.split("/generate")[0]+"/api/tags")
        return {"status":"ok"}
    except Exception as e:
        raise HTTPException(503, detail=str(e))

# ── graceful shutdown ─────────────────────────────────────────────────────────
@app.on_event("shutdown")
async def _shutdown():
    await _client.aclose()

class RAGCoordinator:
    def __init__(self, redis_url: str, qdrant_url: str, concept_dict_url: str):
        self.bus = BusClient(redis_url=redis_url)
        self.qdrant_url = qdrant_url
        self.concept_dict_url = concept_dict_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def connect(self):
        """Connect to Redis and create consumer group"""
        await self.bus.connect()
        try:
            await self.bus.create_group("rag.request", "coordinator")
        except Exception as e:
            log.info(f"Group may exist: {e}")
            
        # Create aiohttp session
        self.session = aiohttp.ClientSession()
        
    async def close(self):
        """Close connections"""
        if self.session:
            await self.session.close()
            
    async def fetch_chunks_from_qdrant(
        self, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Fetch relevant chunks from Qdrant"""
        try:
            async with self.session.get(
                f"{self.qdrant_url}/collections/concepts/points/search",
                json={
                    "vector": query,  # Assuming query is already vectorized
                    "limit": limit,
                    "with_payload": True,
                    "with_vector": False
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail="Failed to fetch chunks from Qdrant"
                    )
                    
                result = await response.json()
                chunks = []
                for hit in result["result"]:
                    chunk = {
                        "id": hit["id"],
                        "score": hit["score"],
                        "payload": hit["payload"],
                        "cid": hit["payload"].get("cid")
                    }
                    chunks.append(chunk)
                    log.info(f"Chunk fetched from Qdrant: {chunk}")
                    
                return chunks
                
        except Exception as e:
            log.error(f"Error fetching chunks from Qdrant: {e}")
            raise
            
    async def enrich_chunks_with_metadata(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich chunks with concept metadata from Concept Dictionary"""
        try:
            # Extract concept IDs
            cids = [c["cid"] for c in chunks if c.get("cid")]
            if not cids:
                return chunks
                
            # Fetch metadata
            async with self.session.post(
                f"{self.concept_dict_url}/meta",
                json={"cids": cids}
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=response.status,
                        detail="Failed to fetch concept metadata"
                    )
                    
                metadata = await response.json()
                
                # Enrich chunks
                for chunk in chunks:
                    if chunk.get("cid") in metadata:
                        chunk["meta"] = metadata[chunk["cid"]]
                        log.info(f"Concept metadata enriched: {chunk}")
                        
                return chunks
                
        except Exception as e:
            log.error(f"Error enriching chunks with metadata: {e}")
            raise
            
    @with_retry(max_attempts=3)
    async def process_request(self, msg: Dict[str, Any]):
        """Process a RAG request with retry logic and DLQ support"""
        start_time = datetime.now()
        try:
            # Extract request parameters
            query = msg.get("query")
            if not query:
                raise ValueError("Missing query")
                
            limit = msg.get("limit", 5)
            
            # Fetch chunks from Qdrant
            chunks = await self.fetch_chunks_from_qdrant(query, limit)
            
            # Enrich chunks with metadata
            enriched_chunks = await self.enrich_chunks_with_metadata(chunks)
            
            # Prepare response
            response = {
                "request_id": msg.get("request_id"),
                "chunks": enriched_chunks,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Publish response
            await self.bus.publish("rag.response", response)
            
            # Record metrics
            log.info(f"RAG request processed in {datetime.now() - start_time}")
            
        except Exception as e:
            log.error(f"Error processing RAG request: {e}")
            raise
            
    async def start(self):
        """Start consuming from rag.request stream"""
        while True:
            try:
                await self.bus.consume(
                    stream="rag.request",
                    group="coordinator",
                    consumer="worker",
                    handler=self.process_request,
                    block_ms=1000,
                    count=10
                )
            except Exception as e:
                log.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)

@app.on_event("startup")
async def startup():
    """Initialize coordinator on startup"""
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    concept_dict_url = os.getenv("CONCEPT_DICT_URL", "http://concept-dict:8828")
    
    coordinator = RAGCoordinator(redis_url, qdrant_url, concept_dict_url)
    await coordinator.connect()
    app.state.coordinator = coordinator
    # Start consumer loop
    asyncio.create_task(coordinator.start())

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    if hasattr(app.state, "coordinator"):
        await app.state.coordinator.close()

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    from prometheus_client import generate_latest
    return Response(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 