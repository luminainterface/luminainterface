import os
import hashlib
import json
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Request
from fastapi.responses import JSONResponse
import aiofiles
import redis.asyncio as aioredis
from lumina_core.common.bus import BusClient
from ranker import score_url, score_pdf
from license_scanner import LicenseScanner, ScanResult
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
import asyncio
import tempfile
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

app = FastAPI(title="Ingest Gateway")

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://:02211998@redis:6379")
QUEUE_STREAM = os.getenv("QUEUE_STREAM", "ingest.queue")
FP_SET = os.getenv("FP_SET", "ingest.fp")
MIN_LICENSE_CONFIDENCE = float(os.getenv("MIN_LICENSE_CONFIDENCE", "0.8"))

# Initialize clients
bus = BusClient(redis_url=REDIS_URL)
redis = aioredis.from_url(REDIS_URL)
license_scanner = LicenseScanner(min_confidence=MIN_LICENSE_CONFIDENCE)

# Prometheus metrics
INGEST_RANK_SCORE = Histogram("ingest_rank_score", "Rank score for ingested items")
INGEST_SUCCESS = Counter("ingest_success_total", "Successful ingestions", ["type"])
INGEST_FAILURE = Counter("ingest_failure_total", "Failed ingestions", ["type", "reason"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf": ".pdf",
    "application/x-pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/json": ".json"
}

async def process_license_scan(
    content: bytes,
    file_extension: str,
    source_type: str
) -> Tuple[bool, Dict[str, Any]]:
    """Process license scan and return results."""
    try:
        scan_result = await license_scanner.scan_content(content, file_extension)
        summary = license_scanner.get_license_summary(scan_result)
        
        if scan_result.error:
            INGEST_FAILURE.labels(type=source_type, reason="license_scan_error").inc()
            raise HTTPException(
                status_code=500,
                detail=f"License scan failed: {scan_result.error}"
            )
        
        if scan_result.is_blocked:
            INGEST_FAILURE.labels(type=source_type, reason="blocked_license").inc()
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Content blocked due to license restrictions",
                    "licenses": summary["licenses"],
                    "warning": summary["warning"]
                }
            )
        
        if scan_result.warning:
            logging.warning(f"License warning for {source_type}: {scan_result.warning}")
        
        return True, summary
        
    except HTTPException:
        raise
    except Exception as e:
        INGEST_FAILURE.labels(type=source_type, reason="license_processing_error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup():
    await bus.connect()
    asyncio.create_task(consume_system_topics())
    asyncio.create_task(consume_git_events())

@app.on_event("shutdown")
async def shutdown():
    await bus.close()

@app.post("/submit/url")
async def submit_url(
    url: str = Form(...),
    content_type: Optional[str] = Form(None)
):
    """Submit a URL for ingestion."""
    try:
        fp = hashlib.sha256(url.encode()).hexdigest()
        
        # Check for duplicates
        if not await redis.setnx(f"fp:{fp}", 1):
            return JSONResponse({"status": "duplicate", "fp": fp}, status_code=200)
        
        # Score URL
        score = score_url(url)
        INGEST_RANK_SCORE.observe(score)
        
        # Prepare payload
        payload = {
            "type": "url",
            "payload": url,
            "score": score,
            "fp": fp,
            "content_type": content_type,
            "submitted_at": datetime.utcnow().isoformat()
        }
        
        # Publish to queue
        await bus.publish(QUEUE_STREAM, payload)
        INGEST_SUCCESS.labels(type="url").inc()
        
        return {
            "status": "queued",
            "fp": fp,
            "score": score,
            "submitted_at": payload["submitted_at"]
        }
        
    except Exception as e:
        INGEST_FAILURE.labels(type="url", reason="processing_error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit/file")
async def submit_file(
    file: UploadFile = File(...),
    content_type: str = Header(None)
):
    """Submit a file for ingestion."""
    if not content_type or content_type not in ALLOWED_CONTENT_TYPES:
        INGEST_FAILURE.labels(type="file", reason="invalid_content_type").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type. Must be one of: {', '.join(ALLOWED_CONTENT_TYPES.keys())}"
        )
    
    try:
        content = await file.read()
        fp = hashlib.sha256(content).hexdigest()
        file_extension = ALLOWED_CONTENT_TYPES[content_type]
        
        # Check for duplicates
        if not await redis.setnx(f"fp:{fp}", 1):
            return JSONResponse({"status": "duplicate", "fp": fp}, status_code=200)
        
        # Process license scan
        _, license_info = await process_license_scan(
            content,
            file_extension,
            "file"
        )
        
        # Save file for downstream
        save_path = f"/tmp/{fp}{file_extension}"
        async with aiofiles.open(save_path, "wb") as f:
            await f.write(content)
        
        # Score content
        score = score_pdf(content) if file_extension == ".pdf" else 0.5  # Default score for non-PDF
        
        # Prepare payload
        payload = {
            "type": "file",
            "payload": save_path,
            "score": score,
            "fp": fp,
            "content_type": content_type,
            "license_info": license_info,
            "submitted_at": datetime.utcnow().isoformat()
        }
        
        # Publish to queue
        await bus.publish(QUEUE_STREAM, payload)
        INGEST_SUCCESS.labels(type="file").inc()
        
        return {
            "status": "queued",
            "fp": fp,
            "score": score,
            "license": license_info,
            "submitted_at": payload["submitted_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        INGEST_FAILURE.labels(type="file", reason="processing_error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Redis connection
        await redis.ping()
        
        # Test license scanner
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            tmp.write(b"MIT License\nCopyright (c) 2024")
            tmp.flush()
            scan_result = await license_scanner.scan_file(tmp.name)
        
        return {
            "status": "healthy",
            "service": "ingest-gateway",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "redis": "connected",
                "license_scanner": "operational",
                "bus": "connected"
            },
            "license_scanner": {
                "min_confidence": MIN_LICENSE_CONFIDENCE,
                "blocked_licenses": len(license_scanner.BLOCKED_LICENSES),
                "allowed_licenses": len(license_scanner.ALLOWED_LICENSES)
            }
        }
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return JSONResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

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