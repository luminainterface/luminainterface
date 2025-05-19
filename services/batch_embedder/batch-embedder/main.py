import os, asyncio, time, uuid, logging, json
from typing import List
from collections import deque

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
from lumina_core.common.bus import BusClient
from lumina_core.common.stream_message import StreamMessage
from lumina_core.common.retry import with_retry
from qdrant_client import QdrantClient
from prometheus_client import Counter, Histogram, start_http_server

# ─────────────────────── config ──────────────────────────
REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379")
INGEST_STREAM = os.getenv("INGEST_CLEANED_STREAM", "ingest.cleaned")
TRAIN_URL   = os.getenv("TRAIN_URL", "http://concept-trainer-growable:8710/train_batch")
MAX_BATCH   = int(os.getenv("BATCH_SIZE", 128))
BATCH_SEC   = int(os.getenv("BATCH_SECONDS", 5))

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant   = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
bus      = BusClient(REDIS_URL)

# ─────────────────────── metrics ─────────────────────────
start_http_server(8709)
EMBED_BATCHES = Counter("embed_batches_total", "batches sent to trainer")
EMBED_VECTORS = Counter("embed_vectors_total", "vectors embedded")
TRAIN_POST    = Histogram("trainer_post_seconds", "latency posting to trainer",
                          buckets=[0.05,0.1,0.3,0.6,1,2,4])
        
# ─────────────────────── models ──────────────────────────
class CleanMsg(StreamMessage):
    fp: str
    text: str
    lang: str
    ts: float

# queue for the batch
pending: deque[CleanMsg] = deque()

# ───────────────── novelty calc ──────────────────────────
def novelty(vec: List[float]) -> float:
    hits = qdrant.search(
        collection_name="concepts",
        query_vector=vec,
        limit=3,
        with_payload=False
    )
    nearest = hits[0].score if hits else 0.0
    return max(0.0, 1 - nearest)

# ───────────────── handler ───────────────────────────────
@with_retry(bus, INGEST_STREAM)
async def on_clean(data: dict):
    msg = CleanMsg(**data)
    pending.append(msg)

async def batch_loop():
    while True:
        await asyncio.sleep(BATCH_SEC)
        if not pending: 
            continue
        batch, vecs, fps, metas = [], [], [], []
        while pending and len(batch) < MAX_BATCH:
            m = pending.popleft()
            v = embedder.encode(m.text, convert_to_numpy=True).tolist()
            vecs.append(v)
            fps.append(m.fp)
            metas.append({"lang": m.lang, "novelty": novelty(v)})
            batch.append(m)
        await post_batch(fps, vecs, metas)
        EMBED_BATCHES.inc()
        EMBED_VECTORS.inc(len(batch))

@TRAIN_POST.time()
async def post_batch(fps, vecs, metas):
    payload = {
        "batch_id": uuid.uuid4().hex,
        "embed_ids": fps,
        "vectors": vecs,
        "meta": metas
    }
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post(TRAIN_URL, json=payload)
        r.raise_for_status()

# ───────────────── main ──────────────────────────────────
async def main():
    asyncio.create_task(
        bus.consume(
            stream=INGEST_STREAM,
        group="embed",
            consumer=os.getenv("HOSTNAME","embedder"),
            handler=on_clean
        )
    )
    asyncio.create_task(batch_loop())
    while True:  # keep service alive
        await asyncio.sleep(3600)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main()) 