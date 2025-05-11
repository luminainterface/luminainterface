import os
import hashlib
import json
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
import redis.asyncio as aioredis
from lumina_core.common.bus import BusClient
from .ranker import score_url, score_pdf
from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST
import asyncio

app = FastAPI(title="Ingest Gateway")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
QUEUE_STREAM = os.getenv("QUEUE_STREAM", "ingest.queue")
FP_SET = os.getenv("FP_SET", "ingest.fp")

bus = BusClient(redis_url=REDIS_URL)
redis = aioredis.from_url(REDIS_URL)

# Prometheus metric
INGEST_RANK_SCORE = Histogram("ingest_rank_score", "Rank score for ingested items")

@app.on_event("startup")
async def startup():
    await bus.connect()
    asyncio.create_task(consume_system_topics())
    asyncio.create_task(consume_git_events())

@app.on_event("shutdown")
async def shutdown():
    await bus.close()

@app.post("/submit/url")
async def submit_url(url: str = Form(...)):
    fp = hashlib.sha256(url.encode()).hexdigest()
    # Deduplication
    if not await redis.setnx(f"fp:{fp}", 1):
        return JSONResponse({"status": "duplicate", "fp": fp}, status_code=200)
    score = score_url(url)
    INGEST_RANK_SCORE.observe(score)
    payload = {"type": "url", "payload": url, "score": score, "fp": fp}
    await bus.publish(QUEUE_STREAM, payload)
    return {"status": "queued", "fp": fp, "score": score}

@app.post("/submit/pdf")
async def submit_pdf(file: UploadFile = File(...)):
    content = await file.read()
    fp = hashlib.sha256(content).hexdigest()
    if not await redis.setnx(f"fp:{fp}", 1):
        return JSONResponse({"status": "duplicate", "fp": fp}, status_code=200)
    # Save file for downstream
    save_path = f"/tmp/{fp}.pdf"
    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)
    score = score_pdf(content)
    INGEST_RANK_SCORE.observe(score)
    payload = {"type": "pdf", "payload": save_path, "score": score, "fp": fp}
    await bus.publish(QUEUE_STREAM, payload)
    return {"status": "queued", "fp": fp, "score": score}

async def consume_system_topics():
    async def handler(msg):
        topic = msg["topic"]
        fp = hashlib.sha256(topic.encode()).hexdigest()
        if not await redis.setnx(f"fp:{fp}", 1):
            return
        score = score_url(topic)  # Use same scoring for now
        INGEST_RANK_SCORE.observe(score)
        payload = {"type": "system", "payload": topic, "score": score, "fp": fp}
        await bus.publish(QUEUE_STREAM, payload)
    await bus.consume("system.topics", group="gateway", consumer="sys", handler=handler)

async def consume_git_events():
    async def handler(msg):
        event = msg["event"]
        fp = hashlib.sha256(event.encode()).hexdigest()
        if not await redis.setnx(f"fp:{fp}", 1):
            return
        score = 0.5  # Placeholder
        INGEST_RANK_SCORE.observe(score)
        payload = {"type": "git", "payload": event, "score": score, "fp": fp}
        await bus.publish(QUEUE_STREAM, payload)
    await bus.consume("git.ingest_events", group="gateway", consumer="git", handler=handler)

@app.get("/metrics")
def metrics():
    return JSONResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST) 