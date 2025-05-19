"""Concept training pipeline implementation."""
import asyncio
import logging
from typing import Optional, Dict, List
from datetime import datetime
from .concept_client import ConceptClient, ConceptStatus
from .embeddings import CustomOllamaEmbeddings

logger = logging.getLogger(__name__)

class ConceptTrainer:
    def __init__(
        self,
        concept_client: ConceptClient,
        embeddings: CustomOllamaEmbeddings,
        poll_interval: float = 1.0,
        max_retries: int = 3
    ):
        self.concept_client = concept_client
        self.embeddings = embeddings
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self._running = False
        self._current_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the training loop"""
        if self._running:
            logger.warning("Trainer is already running")
            return

        self._running = True
        self._current_task = asyncio.create_task(self._training_loop())
        logger.info("Concept trainer started")

    async def stop(self):
        """Stop the training loop"""
        if not self._running:
            return

        self._running = False
        if self._current_task:
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        logger.info("Concept trainer stopped")

    async def _training_loop(self):
        """Main training loop that processes concepts"""
        while self._running:
            try:
                # Get next concept to train
                concept = await self.concept_client.get_next_untrained_concept()
                
                if not concept:
                    await asyncio.sleep(self.poll_interval)
                    continue

                concept_id = concept["id"]
                logger.info(f"Processing concept {concept_id}")

                # Update status to training
                await self.concept_client.update_concept_status(
                    concept_id,
                    "training"
                )

                try:
                    # Generate embedding
                    text = concept.get("text", "")
                    if not text:
                        raise ValueError("Concept has no text content")

                    embedding = await self._generate_embedding(text)
                    
                    # Update concept with embedding and mark as trained
                    success = await self.concept_client.update_concept_status(
                        concept_id,
                        "trained",
                        embedding=embedding
                    )

                    if not success:
                        raise RuntimeError("Failed to update concept status")

                    logger.info(f"Successfully trained concept {concept_id}")

                except Exception as e:
                    logger.error(f"Error training concept {concept_id}: {str(e)}")
                    await self.concept_client.update_concept_status(
                        concept_id,
                        "error",
                        error_message=str(e)
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in training loop: {str(e)}")
                await asyncio.sleep(self.poll_interval)

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text"""
        try:
            # Use the embeddings model to generate vector
            embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                text
            )
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def retry_failed_concepts(self):
        """Retry concepts that failed training"""
        success = await self.concept_client.retry_failed_concepts(
            max_retries=self.max_retries
        )
        if success:
            logger.info("Successfully reset failed concepts for retry")
        else:
            logger.error("Failed to reset failed concepts") 