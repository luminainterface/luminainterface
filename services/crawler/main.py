#!/usr/bin/env python3
"""Crawler service for Lumina system.

This service:
1. Reads from ingest.queue
2. Validates repositories using license scanner
3. Crawls and processes content
4. Writes to ingest.raw_json
"""

import asyncio
import logging
import os
from typing import Optional
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from handlers.license_scanner import LicenseScanner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crawler')

# Constants
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))  # 1 minute default

class CrawlerService:
    def __init__(self):
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.license_scanner = LicenseScanner()
        self.app = FastAPI(title="Crawler Service")
        self.setup_routes()
        
    def setup_routes(self):
        """Set up FastAPI routes."""
        @self.app.get("/health")
        async def health():
            return {"status": "ok"}
            
        @self.app.post("/validate")
        async def validate_repo(repo: str):
            """Validate a repository URL."""
            is_allowed, reason = await self.license_scanner.check_repo(repo)
            if not is_allowed:
                raise HTTPException(status_code=403, detail=reason)
            return {"status": "ok", "message": "Repository allowed"}
            
    async def process_message(self, msg_id: str, msg_data: dict):
        """Process a message from the ingest queue."""
        try:
            # Extract repo URL
            repo_url = msg_data.get('repo_url')
            if not repo_url:
                logger.warning(f"Message {msg_id} has no repo_url")
                await self.redis.xack('ingest.queue', 'crawler', msg_id)
                return
                
            # Validate repository
            is_allowed, reason = await self.license_scanner.check_repo(repo_url)
            if not is_allowed:
                logger.warning(f"Repository {repo_url} blocked: {reason}")
                # Move to dead letter queue
                await self.redis.xadd('dead.letter', msg_data)
                await self.redis.xack('ingest.queue', 'crawler', msg_id)
                return
                
            # TODO: Implement actual crawling logic here
            # For now, just acknowledge the message
            await self.redis.xack('ingest.queue', 'crawler', msg_id)
            
        except Exception as e:
            logger.error(f"Error processing message {msg_id}: {e}")
            # Move to dead letter queue on error
            await self.redis.xadd('dead.letter', msg_data)
            await self.redis.xack('ingest.queue', 'crawler', msg_id)
            
    async def run(self):
        """Run the crawler service."""
        # Create consumer group if it doesn't exist
        try:
            await self.redis.xgroup_create('ingest.queue', 'crawler', mkstream=True)
        except redis.ResponseError as e:
            if 'BUSYGROUP' not in str(e):
                raise
                
        while True:
            try:
                # Check for pause flag
                if await self.redis.get('crawler.PAUSE'):
                    logger.info("Crawler paused due to high queue depth")
                    await asyncio.sleep(5)
                    continue
                    
                # Read messages
                messages = await self.redis.xreadgroup(
                    'crawler', 'crawler-1',
                    {'ingest.queue': '>'},
                    count=10,
                    block=1000
                )
                
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        await self.process_message(msg_id, msg_data)
                        
            except Exception as e:
                logger.error(f"Error in crawler loop: {e}")
                await asyncio.sleep(CHECK_INTERVAL)
                
    async def close(self):
        """Cleanup connections."""
        await self.redis.close()
        await self.license_scanner.close()

async def main():
    """Main entry point."""
    service = CrawlerService()
    try:
        await service.run()
    finally:
        await service.close()

if __name__ == "__main__":
    asyncio.run(main())