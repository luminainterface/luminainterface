import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel

# Metrics
CONCEPTS_FETCHED = Counter(
    'concept_trainer_concepts_fetched_total',
    'Number of concepts fetched for training',
    ['status']  # 'success', 'error'
)

CONCEPTS_TRAINED = Counter(
    'concept_trainer_concepts_trained_total',
    'Number of concepts trained',
    ['status']  # 'success', 'error'
)

TRAINING_QUEUE_SIZE = Gauge(
    'concept_trainer_queue_size',
    'Current number of concepts in training queue'
)

FETCH_LATENCY = Histogram(
    'concept_trainer_fetch_latency_seconds',
    'Time spent fetching concepts from dictionary'
)

class ConceptFetcher:
    """Handles fetching and training concepts from the concept dictionary."""
    
    def __init__(
        self,
        concept_dict_url: str = "http://concept-dictionary:8000",
        batch_size: int = 10,
        fetch_interval: int = 30,
        max_retries: int = 3,
        min_quality_score: float = 0.0
    ):
        self.concept_dict_url = concept_dict_url
        self.batch_size = batch_size
        self.fetch_interval = fetch_interval
        self.max_retries = max_retries
        self.min_quality_score = min_quality_score
        self.http_client = httpx.AsyncClient()
        self.training_queue: Dict[str, Dict[str, Any]] = {}
        self._running = False
        
    async def fetch_concepts(self) -> Optional[Dict[str, Any]]:
        """Fetch a batch of untrained concepts from the dictionary."""
        try:
            with FETCH_LATENCY.time():
                response = await self.http_client.post(
                    f"{self.concept_dict_url}/concepts/fetch_queue",
                    json={
                        "batch_size": self.batch_size,
                        "max_retries": self.max_retries,
                        "min_quality_score": self.min_quality_score
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    CONCEPTS_FETCHED.labels(status="success").inc(len(data["concepts"]))
                    return data
                else:
                    logger.error(f"Failed to fetch concepts: {response.text}")
                    CONCEPTS_FETCHED.labels(status="error").inc()
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching concepts: {str(e)}")
            CONCEPTS_FETCHED.labels(status="error").inc()
            return None
            
    async def update_training_status(
        self,
        term: str,
        status: str,
        error: Optional[str] = None
    ) -> bool:
        """Update the training status of a concept."""
        try:
            training_status = {
                "term": term,
                "status": status,
                "training_started": datetime.utcnow().isoformat() if status == "training" else None,
                "training_completed": datetime.utcnow().isoformat() if status == "trained" else None,
                "error": error,
                "last_attempt": datetime.utcnow().isoformat(),
                "retry_count": self.training_queue.get(term, {}).get("retry_count", 0) + 1
            }
            
            response = await self.http_client.post(
                f"{self.concept_dict_url}/concepts/training_status/{term}",
                json=training_status
            )
            
            if response.status_code == 200:
                if status == "trained":
                    CONCEPTS_TRAINED.labels(status="success").inc()
                elif status == "failed":
                    CONCEPTS_TRAINED.labels(status="error").inc()
                return True
            else:
                logger.error(f"Failed to update training status for {term}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating training status for {term}: {str(e)}")
            return False
            
    async def process_concept(self, concept: Dict[str, Any]) -> bool:
        """Process a single concept for training."""
        term = concept["term"]
        try:
            # Mark as training
            await self.update_training_status(term, "training")
            
            # TODO: Implement actual training logic here
            # For now, we'll just simulate training
            await asyncio.sleep(1)  # Simulate training time
            
            # Update status based on simulated result
            success = True  # Simulate successful training
            if success:
                await self.update_training_status(term, "trained")
                return True
            else:
                await self.update_training_status(term, "failed", "Training failed")
                return False
                
        except Exception as e:
            logger.error(f"Error processing concept {term}: {str(e)}")
            await self.update_training_status(term, "failed", str(e))
            return False
            
    async def process_batch(self, batch_data: Dict[str, Any]):
        """Process a batch of concepts."""
        concepts = batch_data["concepts"]
        batch_id = batch_data["batch_id"]
        
        logger.info(f"Processing batch {batch_id} with {len(concepts)} concepts")
        
        # Add concepts to training queue
        for concept in concepts:
            term = concept["term"]
            self.training_queue[term] = {
                "concept": concept,
                "batch_id": batch_id,
                "retry_count": 0,
                "added_at": datetime.utcnow().isoformat()
            }
        
        TRAINING_QUEUE_SIZE.set(len(self.training_queue))
        
        # Process concepts concurrently
        tasks = [self.process_concept(concept) for concept in concepts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Remove processed concepts from queue
        for term, success in zip([c["term"] for c in concepts], results):
            if isinstance(success, bool) and success:
                self.training_queue.pop(term, None)
        
        TRAINING_QUEUE_SIZE.set(len(self.training_queue))
        
        logger.info(f"Completed batch {batch_id}. Results: {sum(1 for r in results if r is True)}/{len(results)} successful")
        
    async def run(self):
        """Run the concept fetcher continuously."""
        self._running = True
        logger.info("Starting concept fetcher")
        
        while self._running:
            try:
                # Fetch new concepts
                batch_data = await self.fetch_concepts()
                if batch_data and batch_data["concepts"]:
                    await self.process_batch(batch_data)
                    
                # Wait before next fetch
                await asyncio.sleep(self.fetch_interval)
                
            except Exception as e:
                logger.error(f"Error in concept fetcher: {str(e)}")
                await asyncio.sleep(5)  # Back off on error
                
    def stop(self):
        """Stop the concept fetcher."""
        self._running = False
        logger.info("Stopping concept fetcher")
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get current statistics about the fetcher."""
        try:
            response = await self.http_client.get(f"{self.concept_dict_url}/concepts/training_stats")
            if response.status_code == 200:
                stats = response.json()
                stats["queue_size"] = len(self.training_queue)
                stats["running"] = self._running
                return stats
            else:
                return {
                    "error": f"Failed to get stats: {response.text}",
                    "queue_size": len(self.training_queue),
                    "running": self._running
                }
        except Exception as e:
            return {
                "error": str(e),
                "queue_size": len(self.training_queue),
                "running": self._running
            } 