#!/usr/bin/env python3
"""Sync-checker service for Lumina system.

This service monitors:
1. Service health and synchronization
2. Model adapter versions across services
3. Queue depths and processing rates
4. Training progress and model drift
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx
import redis.asyncio as redis
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sync-checker')

# Constants
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '300'))  # 5 minutes default
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
OUTPUT_ENGINE_URL = os.getenv('OUTPUT_ENGINE_URL', 'http://output-engine:9000')
TRAINER_URL = os.getenv('TRAINER_URL', 'http://concept-trainer-growable:8710')
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')

# Prometheus metrics
SERVICE_HEALTH = Gauge('lumina_service_health', 'Service health status', ['service'])
ADAPTER_VERSION = Gauge('lumina_adapter_version', 'Adapter version timestamp', ['service'])
QUEUE_DEPTH = Gauge('lumina_queue_depth', 'Queue depth', ['stream'])
PROCESSING_RATE = Counter('lumina_processing_rate', 'Messages processed per minute', ['stream'])
TRAINING_PROGRESS = Gauge('lumina_training_progress', 'Training progress percentage')
MODEL_DRIFT = Gauge('lumina_model_drift', 'Model drift score')
CHECK_LATENCY = Histogram('lumina_check_latency', 'Check operation latency', ['operation'])

class ServiceStatus(BaseModel):
    status: str
    adapter_id: Optional[str]
    last_update: Optional[datetime]
    queue_depth: Optional[int]
    processing_rate: Optional[float]

class SyncChecker:
    def __init__(self):
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.http = httpx.AsyncClient(timeout=30.0)
        self.app = FastAPI(title="Sync Checker")
        self.setup_routes()
        
    def setup_routes(self):
        """Set up FastAPI routes."""
        @self.app.get("/health")
        async def health():
            return {"status": "ok", "last_check": self.last_check_time}
            
        @self.app.get("/status")
        async def status():
            return {
                "services": self.service_status,
                "queues": self.queue_status,
                "training": self.training_status
            }
            
        @self.app.post("/check")
        async def manual_check():
            try:
                await self.run_check()
                return {"status": "ok", "message": "Check completed"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def check_service_health(self, service: str, url: str) -> ServiceStatus:
        """Check health of a service and get its status."""
        try:
            with CHECK_LATENCY.labels(operation=f"health_{service}").time():
                response = await self.http.get(f"{url}/health")
                response.raise_for_status()
                data = response.json()
                
                SERVICE_HEALTH.labels(service=service).set(1)
                if 'adapter_id' in data:
                    ADAPTER_VERSION.labels(service=service).set(
                        datetime.fromisoformat(data['adapter_id'].split('_')[1]).timestamp()
                    )
                
                return ServiceStatus(
                    status="ok",
                    adapter_id=data.get('adapter_id'),
                    last_update=datetime.fromisoformat(data.get('last_update', datetime.now().isoformat())),
                    queue_depth=None,
                    processing_rate=None
                )
        except Exception as e:
            logger.error(f"Error checking {service} health: {e}")
            SERVICE_HEALTH.labels(service=service).set(0)
            return ServiceStatus(status="error", adapter_id=None, last_update=None)

    async def check_queue_status(self, stream: str) -> Dict:
        """Check status of a Redis stream."""
        try:
            with CHECK_LATENCY.labels(operation=f"queue_{stream}").time():
                info = await self.redis.xinfo_stream(stream)
                groups = await self.redis.xinfo_groups(stream)
                
                total_pending = sum(int(g['pending']) for g in groups)
                QUEUE_DEPTH.labels(stream=stream).set(total_pending)
                
                # Calculate processing rate
                last_len = await self.redis.get(f"sync:last_len:{stream}")
                current_len = int(info['length'])
                if last_len:
                    rate = (current_len - int(last_len)) / (CHECK_INTERVAL / 60)
                    PROCESSING_RATE.labels(stream=stream).inc(rate)
                await self.redis.set(f"sync:last_len:{stream}", current_len)
                
                return {
                    "depth": total_pending,
                    "last_id": info['last-generated-id'],
                    "groups": len(groups)
                }
        except Exception as e:
            logger.error(f"Error checking queue {stream}: {e}")
            return {"depth": -1, "last_id": None, "groups": 0}

    async def check_training_status(self) -> Dict:
        """Check training progress and model drift."""
        try:
            with CHECK_LATENCY.labels(operation="training").time():
                # Check trainer status
                response = await self.http.get(f"{TRAINER_URL}/status")
                response.raise_for_status()
                trainer_data = response.json()
                
                TRAINING_PROGRESS.set(trainer_data.get('progress', 0))
                MODEL_DRIFT.set(trainer_data.get('drift_score', 0))
                
                return {
                    "progress": trainer_data.get('progress', 0),
                    "drift_score": trainer_data.get('drift_score', 0),
                    "last_update": trainer_data.get('last_update')
                }
        except Exception as e:
            logger.error(f"Error checking training status: {e}")
            return {"progress": -1, "drift_score": -1, "last_update": None}

    async def run_check(self):
        """Run a complete system check."""
        self.last_check_time = datetime.now()
        
        # Check services
        self.service_status = {
            "output-engine": await self.check_service_health("output-engine", OUTPUT_ENGINE_URL),
            "trainer": await self.check_service_health("trainer", TRAINER_URL)
        }
        
        # Check queues
        self.queue_status = {
            stream: await self.check_queue_status(stream)
            for stream in ['ingest.queue', 'thought.log', 'model.adapter.updated']
        }
        
        # Check training
        self.training_status = await self.check_training_status()
        
        logger.info("System check completed")

    async def start(self):
        """Start the sync checker service."""
        # Start Prometheus metrics server
        start_http_server(8000)
        
        # Initial check
        await self.run_check()
        
        # Start periodic checks
        while True:
            await asyncio.sleep(CHECK_INTERVAL)
            await self.run_check()

    async def close(self):
        """Cleanup connections."""
        await self.redis.close()
        await self.http.aclose()

async def main():
    """Main entry point."""
    checker = SyncChecker()
    try:
        await checker.start()
    finally:
        await checker.close()

if __name__ == "__main__":
    asyncio.run(main()) 