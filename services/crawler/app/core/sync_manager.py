"""Synchronization manager for concept storage."""
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .concept_client import ConceptClient
from .qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class SyncManager:
    def __init__(
        self,
        concept_client: ConceptClient,
        qdrant_client: QdrantClient,
        sync_interval: float = 3600.0,  # 1 hour default
        batch_size: int = 100
    ):
        self.concept_client = concept_client
        self.qdrant_client = qdrant_client
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        self._running = False
        self._current_task: Optional[asyncio.Task] = None
        self._last_sync: Optional[datetime] = None

    async def start(self):
        """Start the sync loop"""
        if self._running:
            logger.warning("Sync manager is already running")
            return

        self._running = True
        self._current_task = asyncio.create_task(self._sync_loop())
        logger.info("Sync manager started")

    async def stop(self):
        """Stop the sync loop"""
        if not self._running:
            return

        self._running = False
        if self._current_task:
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        logger.info("Sync manager stopped")

    async def _sync_loop(self):
        """Main sync loop that ensures consistency between Redis and Qdrant"""
        while self._running:
            try:
                await self._perform_sync()
                self._last_sync = datetime.utcnow()
                await asyncio.sleep(self.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying on error

    async def _perform_sync(self):
        """Perform a full sync between Redis and Qdrant"""
        logger.info("Starting sync between Redis and Qdrant")
        
        # Get all concepts from Redis
        offset = 0
        while True:
            concepts = await self.concept_client.get_concepts_by_status(
                "trained",
                limit=self.batch_size,
                offset=offset
            )
            
            if not concepts:
                break

            # Sync each concept to Qdrant
            for concept in concepts:
                try:
                    await self._sync_concept(concept)
                except Exception as e:
                    logger.error(f"Error syncing concept {concept.get('id')}: {str(e)}")

            offset += len(concepts)
            if len(concepts) < self.batch_size:
                break

        logger.info("Sync completed")

    async def _sync_concept(self, concept: Dict):
        """Sync a single concept between Redis and Qdrant"""
        concept_id = concept["id"]
        embedding = concept.get("embedding")
        
        if not embedding:
            logger.warning(f"Concept {concept_id} has no embedding, skipping")
            return

        # Check if concept exists in Qdrant
        exists = await self.qdrant_client.get_concept(concept_id)
        
        if exists:
            # Update if exists
            await self.qdrant_client.update_concept(
                concept_id,
                embedding=embedding,
                metadata=concept
            )
        else:
            # Insert if doesn't exist
            await self.qdrant_client.add_concept(
                concept_id,
                embedding=embedding,
                metadata=concept
            )

    async def force_sync(self):
        """Force an immediate sync"""
        if self._running:
            await self._perform_sync()
            self._last_sync = datetime.utcnow()
        else:
            logger.warning("Sync manager is not running")

    @property
    def last_sync_time(self) -> Optional[datetime]:
        """Get the last sync time"""
        return self._last_sync

    @property
    def time_since_last_sync(self) -> Optional[timedelta]:
        """Get the time since the last sync"""
        if self._last_sync:
            return datetime.utcnow() - self._last_sync
        return None 