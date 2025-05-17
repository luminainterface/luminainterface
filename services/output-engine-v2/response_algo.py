import os, time, json, logging
from typing import List, Dict, Any, Optional
import httpx, redis.asyncio as aioredis
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache
from prometheus_client import Counter, Histogram, start_http_server
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# ───────── config ─────────
REDIS = aioredis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"))
DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8828")
TRAIN_URL = os.getenv("TRAINER_URL", "http://concept-trainer-growable:8710")
THOUGHT_STREAM = "thought.log"
OUT_STREAM = "output.generated"
ADAPTER_DIR = "/adapters"

token = os.getenv("HUGGINGFACE_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_8bit=True, use_auth_token=token)

adapter_cache_id = None

# 2-A · tiny LRU cache (300 lines/s)
RAG_CACHE_SIZE = 500

@lru_cache(maxsize=RAG_CACHE_SIZE)
def _cache_key(query: str, k: int) -> str:
    return f"{query[:64]}::{k}"

async def cached_rag(query: str, k: int) -> List[Dict[str, Any]]:
    key = _cache_key(query, k)
    if cached := cached_rag_cache.get(key):
        MET_RAG_HIT.inc()
        return cached
    chunks = await get_knowledge_context(query, k)
    cached_rag_cache[key] = chunks
    if len(cached_rag_cache) > RAG_CACHE_SIZE:
        cached_rag_cache.pop(next(iter(cached_rag_cache)))
    return chunks

cached_rag_cache = {}

# 2-B · Prometheus metrics (init once)
start_http_server(9105)
MET_LAT = Histogram("response_latency_seconds", "LLM latency", ["phase"])
MET_TOK = Counter("tokens_out_total", "LLM tokens out")
MET_RAG_HIT = Counter("rag_cache_hit_total", "RAG LRU hits")

# ───────── utility ─────────
async def get_intelligence_adapter() -> Optional[str]:
    """Fetches the latest LoRA adapter (intelligence) from Concept-Trainer-Growable."""
    global adapter_cache_id
    async with httpx.AsyncClient() as cli:
        r = await cli.get(f"{TRAIN_URL}/adapter/latest")
        r.raise_for_status()
        adapter_id = r.json()["adapter_id"]
    if adapter_id != adapter_cache_id:
        path = f"{ADAPTER_DIR}/{adapter_id}.bin"
        if not os.path.exists(path):
            async with httpx.AsyncClient() as cli:
                f = await cli.get(f"{TRAIN_URL}/adapters/{adapter_id}")
                f.raise_for_status()
                open(path, "wb").write(f.content)
        model.load_adapter(path)
        adapter_cache_id = adapter_id
        logging.info("Loaded adapter %s", adapter_id)
    return adapter_id

async def get_knowledge_context(query: str, k: int = 6) -> List[Dict[str, Any]]:
    """Fetches RAG chunks (knowledge) from Concept-Dictionary."""
    async with httpx.AsyncClient() as cli:
        r = await cli.get(f"{DICT_URL}/rag", params={"q": query, "k": k})
        r.raise_for_status()
        return r.json()

def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Builds a prompt combining the query and RAG chunks."""
    ctx = "\n".join([f"[{i}] {c['text']}" for i, c in enumerate(chunks)])
    return (
        "You are Lumina. Use the context to answer.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\nAnswer:"
    )

async def think_note(query: str, answer: str, novelty: float) -> None:
    """Publishes a thought note (introspection) to Redis."""
    note = f"Solved '{query[:40]}…' novelty={novelty:.2f}"
    await REDIS.xadd(THOUGHT_STREAM, {"note": note, "ts": time.time()})

# (Insert new adapter watcher (non-blocking) block.)
async def adapter_watcher():
    while True:
        try:
            await get_intelligence_adapter()
        except Exception as e:
            logging.warning("Adapter watch err %s", e)
        await asyncio.sleep(60)

# (Insert FastAPI endpoint block.)
app = FastAPI()

@app.on_event("startup")
async def _watch():
    asyncio.create_task(adapter_watcher())

@app.post("/respond", response_class=StreamingResponse)
async def respond(req: dict):
    query = req["query"]
    top = req.get("top_k", 6)
    return await generate_response(query, top)

# (Replace generate_response with the new streaming version.)
async def generate_response(query: str, top_k: int = 6):
    """Stream tokens while updating metrics, reward, thoughts."""
    t0 = time.time()
    await get_intelligence_adapter()

    # ❶ retrieve & rank
    raw_chunks = await cached_rag(query, top_k * 2)
    ranked = sorted(
        raw_chunks,
        key=lambda c: 0.6 * c.get("novelty", 0) + 0.4 * c.get("reward", 0),
        reverse=True
    )[:top_k]

    # ❷ build prompt with richer markup
    ctx = "\n".join(
        f"[{i} | nov={c.get('novelty', 0):.2f} | lic={c.get('license', '?')}] {c['text']}"
        for i, c in enumerate(ranked)
    )
    prompt = (
        "You are Lumina. Use the numbered context snippets to answer.\n\n"
        f"Context:\n{ctx}\n\nUser: {query}\nAssistant:"
    )

    # ❸ token stream
    tokens = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    gen = model.generate(
        **tokens,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.92,
        temperature=0.7,
        streamer=True
    )

    used = []

    async def token_stream():
        buffer = ""
        for tok in gen:
            word = tokenizer.decode(tok, skip_special_tokens=False)
            buffer += word
            if word in [" ", "\n"]:  # flush on whitespace
                yield buffer
                buffer = ""
        if buffer:
            yield buffer

    # ❹ after generation: reward & logs
    answer_text = tokenizer.decode(gen, skip_special_tokens=True)
    for c in ranked:
        if c["text"][:30] in answer_text:
            used.append(c["cid"])
            # reward bump
            await REDIS.xadd("concept.reward", {"cid": c["cid"], "delta": 0.2})
    await REDIS.xadd(OUT_STREAM, {
        "turn_id": str(time.time_ns()),
        "query": query,
        "answer": answer_text,
        "concepts_used": json.dumps(used),
        "latency": round(time.time() - t0, 3),
        "ts": time.time()
    })
    await think_note(query, answer_text, sum(c.get("novelty", 0) for c in ranked) / top_k)

    MET_LAT.labels(phase="full").observe(time.time() - t0)
    MET_TOK.inc(len(answer_text.split()))

    return StreamingResponse(token_stream(), media_type="text/plain") 