import redis
import json
from loguru import logger
import asyncio
from datetime import datetime
import httpx
import numpy as np
from typing import Dict, List, Optional

REDIS_HOST = "localhost"
REDIS_PORT = 6379
RETRAIN_CHANNEL = "lumina.retrain"
FEEDBACK_CHANNEL = "lumina.rag_update"
CONCEPT_TRAINER_URL = "http://concept-trainer:8906"
CONCEPT_DICT_URL = "http://concept-dictionary:8000"

class RetrainListener:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(RETRAIN_CHANNEL, FEEDBACK_CHANNEL)
        self.http_client = httpx.AsyncClient()
        
    async def get_concept_embedding(self, concept_id: str) -> Optional[List[float]]:
        """Get current concept embedding from dictionary."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{CONCEPT_DICT_URL}/concepts/{concept_id}")
                if response.status_code == 200:
                    concept_data = response.json()
                    return concept_data.get("embedding")
            return None
        except Exception as e:
            logger.error(f"Error getting concept embedding: {str(e)}")
            return None

    async def get_mistral_response(self, concept_id: str) -> Optional[List[float]]:
        """Get Mistral's response for the concept."""
        try:
            # In a real system, this would call Mistral's API
            # For now, we'll generate a random embedding
            return np.random.uniform(-1, 1, 768).tolist()
        except Exception as e:
            logger.error(f"Error getting Mistral response: {str(e)}")
            return None

    async def handle_retrain_event(self, event_data: dict):
        """Handle a retraining event."""
        try:
            concept_id = event_data.get("concept_id")
            reason = event_data.get("reason")
            metrics = event_data.get("metrics", {})
            
            logger.info(f"Processing retrain event for concept {concept_id}")
            logger.info(f"Reason: {reason}")
            logger.info(f"Metrics: {metrics}")
            
            # Get current concept embedding
            current_embedding = await self.get_concept_embedding(concept_id)
            if not current_embedding:
                logger.error(f"No embedding found for concept {concept_id}")
                return
            
            # Get Mistral's response
            mistral_response = await self.get_mistral_response(concept_id)
            if not mistral_response:
                logger.error(f"Failed to get Mistral response for concept {concept_id}")
                return
            
            # Calculate confidence delta based on metrics
            confidence_delta = metrics.get("confidence", 0.5) - 0.5
            feedback_score = metrics.get("quality", 0.7)
            
            # Prepare training request
            training_data = {
                "concept_id": concept_id,
                "nn_response": current_embedding,
                "mistral_response": mistral_response,
                "confidence_delta": confidence_delta,
                "feedback_score": feedback_score
            }
            
            # Send training request
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{CONCEPT_TRAINER_URL}/train",
                        json=training_data,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"Training completed for concept {concept_id}: {result}")
                        
                        # Update concept metrics
                        await self.update_concept_metrics(concept_id, metrics, result)
                    else:
                        logger.error(f"Training failed for concept {concept_id}: {response.text}")
            
            except Exception as e:
                logger.error(f"Error during training request: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error handling retrain event: {str(e)}")
    
    async def update_concept_metrics(self, concept_id: str, old_metrics: Dict, training_result: Dict):
        """Update concept metrics after training."""
        try:
            # Calculate new metrics
            new_metrics = {
                "drift_score": old_metrics.get("drift_score", 0.5) * 0.8,  # Reduce drift
                "confidence": min(1.0, old_metrics.get("confidence", 0.5) + 0.1),  # Increase confidence
                "quality": min(1.0, old_metrics.get("quality", 0.7) + 0.1),  # Increase quality
                "last_training": datetime.utcnow().isoformat(),
                "training_loss": training_result.get("loss", 0.0)
            }
            
            # Update concept in dictionary
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{CONCEPT_DICT_URL}/concepts/{concept_id}/update",
                    json=new_metrics
                )
                
                if response.status_code == 200:
                    logger.info(f"Updated metrics for concept {concept_id}")
                else:
                    logger.error(f"Failed to update metrics for concept {concept_id}: {response.text}")
        
        except Exception as e:
            logger.error(f"Error updating concept metrics: {str(e)}")
    
    async def run(self):
        """Run the retrain listener."""
        logger.info(f"Starting retrain listener on channels: {RETRAIN_CHANNEL}, {FEEDBACK_CHANNEL}")
        
        while True:
            try:
                message = self.pubsub.get_message()
                if message and message["type"] == "message":
                    channel = message["channel"]
                    data = json.loads(message["data"])
                    
                    if channel == RETRAIN_CHANNEL:
                        await self.handle_retrain_event(data)
                    elif channel == FEEDBACK_CHANNEL:
                        logger.info(f"Received feedback event: {data}")
                        # TODO: Implement feedback handling
                
                await asyncio.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in retrain listener: {str(e)}")
                await asyncio.sleep(1)  # Back off on error

if __name__ == "__main__":
    listener = RetrainListener()
    asyncio.run(listener.run()) 
 