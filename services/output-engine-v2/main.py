import os, time, asyncio, json, logging
from typing import List
from fastapi import FastAPI, Request
import httpx, redis.asyncio as aioredis
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ───────── config ─────────
REDIS = aioredis.from_url(os.getenv("REDIS_URL","redis://redis:6379"))
DICT_URL = os.getenv("CONCEPT_DICT_URL","http://concept-dictionary:8828")
TRAIN_URL= os.getenv("TRAINER_URL","http://concept-trainer-growable:8710")
THOUGHT_STREAM = "thought.log"
OUT_STREAM     = "output.generated"
ADAPTER_DIR = "/adapters"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
token = os.getenv("HUGGINGFACE_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_auth_token=token)
model    = AutoModelForCausalLM.from_pretrained(
              "mistralai/Mistral-7B-Instruct-v0.2",
              load_in_8bit=True,
              use_auth_token=token
           )

adapter_cache_id = None

app = FastAPI(title="Output Engine v2")

# ───────── utility ─────────
async def load_latest_adapter():
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
                open(path,"wb").write(f.content)
        model.load_adapter(path)
        adapter_cache_id = adapter_id
        logging.info("Loaded adapter %s", adapter_id)

async def retrieve_chunks(query:str,k:int=6)->List[dict]:
    async with httpx.AsyncClient() as cli:
        r = await cli.get(f"{DICT_URL}/rag", params={"q":query,"k":k})
        r.raise_for_status()
        return r.json()

def build_prompt(query:str,chunks:List[dict])->str:
    ctx = "\n".join([f"[{i}] {c['text']}" for i,c in enumerate(chunks)])
    return (
        "You are Lumina. Use the context to answer.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\nAnswer:"
    )

async def think_note(query, answer, novelty):
    note = f"Solved '{query[:40]}…' novelty={novelty:.2f}"
    await REDIS.xadd(THOUGHT_STREAM, {"note": note, "ts": time.time()})

# ───────── endpoint ─────────
@app.post("/respond")
async def respond(req: Request):
    body = await req.json()
    q   = body["query"]
    top = body.get("top_k",6)

    await load_latest_adapter()
    chunks = await retrieve_chunks(q, top)
    prompt = build_prompt(q, chunks)
    tokens = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    out    = model.generate(**tokens, max_new_tokens=256)
    answer = tokenizer.decode(out[0], skip_special_tokens=True)

    used   = [c["cid"] for c in chunks if c["text"][:30] in answer]

    await REDIS.xadd(OUT_STREAM, {
        "turn_id":str(time.time_ns()),
        "query":q,
        "answer":answer,
        "concepts_used": json.dumps(used),
        "ts":time.time()
    })

    await think_note(q, answer, sum(c.get("novelty",0) for c in chunks)/top)

    return {"answer":answer, "citations":used}

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "dependencies": {
            "redis": False,
            "concept_dictionary": False,
            "concept_trainer": False,
            "model_loaded": False
        }
    }
    
    # Check Redis
    try:
        await REDIS.ping()
        health_status["dependencies"]["redis"] = True
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["redis_error"] = str(e)
    
    # Check Concept Dictionary
    try:
        async with httpx.AsyncClient() as cli:
            r = await cli.get(f"{DICT_URL}/health", timeout=5.0)
            if r.status_code == 200:
                health_status["dependencies"]["concept_dictionary"] = True
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["concept_dictionary_error"] = str(e)
    
    # Check Concept Trainer
    try:
        async with httpx.AsyncClient() as cli:
            r = await cli.get(f"{TRAIN_URL}/health", timeout=5.0)
            if r.status_code == 200:
                health_status["dependencies"]["concept_trainer"] = True
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["concept_trainer_error"] = str(e)
    
    # Check if model is loaded
    health_status["dependencies"]["model_loaded"] = model is not None and tokenizer is not None
    
    # Overall status is unhealthy if any dependency is unhealthy
    if not all(health_status["dependencies"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status

@app.get("/ready")
async def ready_check():
    """Readiness probe that checks if the service is ready to handle requests"""
    health = await health_check()
    if health["status"] != "healthy":
        return {"status": "not_ready", "reason": "health_check_failed"}
    
    # Additional readiness checks
    try:
        # Try to load the latest adapter as a readiness check
        await load_latest_adapter()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "reason": f"adapter_load_failed: {str(e)}"} 