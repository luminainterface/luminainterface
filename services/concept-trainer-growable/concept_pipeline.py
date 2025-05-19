"""Integration of concept training pipeline with growable model system."""
import asyncio
import logging
from typing import Optional, Dict, List
from datetime import datetime
import torch
import numpy as np

from .model import GrowableConceptNet
from services.crawler.app.core.concept_client import ConceptClient, ConceptStatus
from services.crawler.app.core.embeddings import CustomOllamaEmbeddings
from services.crawler.app.core.sync_manager import SyncManager

logger = logging.getLogger(__name__)

class GrowableConceptPipeline:
    def __init__(
        self,
        concept_client: ConceptClient,
        embeddings: CustomOllamaEmbeddings,
        model: GrowableConceptNet,
        sync_manager: SyncManager,
        poll_interval: float = 1.0,
        max_retries: int = 3,
        growth_threshold: float = 0.1
    ):
        self.concept_client = concept_client
        self.embeddings = embeddings
        self.model = model
        self.sync_manager = sync_manager
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.growth_threshold = growth_threshold
        self._running = False
        self._current_task: Optional[asyncio.Task] = None
        self._training_lock = asyncio.Lock()

    async def start(self):
        """Start the training pipeline"""
        if self._running:
            logger.warning("Pipeline is already running")
            return

        self._running = True
        self._current_task = asyncio.create_task(self._training_loop())
        logger.info("Growable concept pipeline started")

    async def stop(self):
        """Stop the training pipeline"""
        if not self._running:
            return

        self._running = False
        if self._current_task:
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        logger.info("Growable concept pipeline stopped")

    async def _training_loop(self):
        """Main training loop that processes concepts and manages model growth"""
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
                    async with self._training_lock:
                        # Generate embedding
                        text = concept.get("text", "")
                        if not text:
                            raise ValueError("Concept has no text content")

                        embedding = await self._generate_embedding(text)
                        
                        # Convert embedding to tensor and train
                        tensor = torch.tensor([embedding], dtype=torch.float32)
                        outputs = self.model(tensor)
                        
                        # Calculate loss and metrics
                        # For now, we use a dummy label (0) since we're just learning representations
                        labels = torch.zeros(1, dtype=torch.long)
                        criterion = torch.nn.NLLLoss()
                        loss = criterion(outputs, labels)
                        
                        # Backpropagate and update
                        self.model.optimizer.zero_grad()
                        loss.backward()
                        self.model.optimizer.step()
                        
                        # Record training metrics
                        accuracy = (torch.argmax(outputs, dim=1) == labels).float().mean().item()
                        drift = self._calculate_drift(concept_id, embedding)
                        self.model.record_training(concept_id, loss.item(), accuracy, drift)
                        
                        # Check if model should grow
                        should_grow, layer_idx = self.model.should_grow(concept_id)
                        if should_grow:
                            current_size = self.model.layers[layer_idx].current_capacity
                            new_size = int(current_size * 1.5)  # Grow by 50%
                            logger.info(f"Growing layer {layer_idx} from {current_size} to {new_size}")
                            self.model.grow_layer(layer_idx, new_size)
                        
                        # Update concept with embedding and mark as trained
                        success = await self.concept_client.update_concept_status(
                            concept_id,
                            "trained",
                            embedding=embedding.tolist()
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

    def _calculate_drift(self, concept_id: str, new_embedding: List[float]) -> float:
        """Calculate concept drift based on embedding changes"""
        try:
            # Get previous embedding if available
            metrics = self.model.get_concept_metrics(concept_id)
            if not metrics or 'last_embedding' not in metrics:
                return 0.0

            prev_embedding = metrics['last_embedding']
            if not prev_embedding:
                return 0.0

            # Calculate cosine distance between embeddings
            prev_tensor = torch.tensor(prev_embedding, dtype=torch.float32)
            new_tensor = torch.tensor(new_embedding, dtype=torch.float32)
            
            # Normalize vectors
            prev_norm = prev_tensor / torch.norm(prev_tensor)
            new_norm = new_tensor / torch.norm(new_tensor)
            
            # Calculate cosine similarity
            similarity = torch.dot(prev_norm, new_norm)
            
            # Convert to distance (drift)
            drift = 1.0 - similarity.item()
            
            return drift
        except Exception as e:
            logger.error(f"Error calculating drift: {str(e)}")
            return 0.0

    async def retry_failed_concepts(self):
        """Retry concepts that failed training"""
        success = await self.concept_client.retry_failed_concepts(
            max_retries=self.max_retries
        )
        if success:
            logger.info("Successfully reset failed concepts for retry")
        else:
            logger.error("Failed to reset failed concepts") 