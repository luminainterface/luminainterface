import asyncio
import pytest
import httpx
import json
import time
from datetime import datetime, UTC
import logging
from typing import Dict, List, Optional
import redis.asyncio as aioredis
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("concept-interaction-test")

# Test configuration
CONCEPT_DICT_URL = "http://localhost:8828"
CONCEPT_TRAINER_URL = "http://localhost:8710"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = "02211998"
REDIS_DB = 0
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TEST_TIMEOUT = 300  # 5 minutes timeout for entire test
POLL_INTERVAL = 5  # seconds between status checks
DICT_API_KEY = os.getenv("DICT_API_KEY", "")  # Get API key from environment

class ConceptInteractionTest:
    """Test suite for concept-dictionary and concept-trainer-growable interaction."""
    
    def __init__(self):
        self.redis_client = None
        self.qdrant_client = None
        self.concept_dict_client = None
        self.concept_trainer_client = None
        self.headers = {}
        if DICT_API_KEY:
            self.headers["X-API-Key"] = DICT_API_KEY
        else:
            logger.warning("No DICT_API_KEY provided - authentication may fail")
        self.test_concepts = [
            {
                "term": "test_concept_1",
                "definition": "This is a test concept for interaction testing",
                "metadata": {
                    "source": "test",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "test_id": "interaction_test_1"
                }
            },
            {
                "term": "test_concept_2",
                "definition": "Another test concept with different content",
                "metadata": {
                    "source": "test",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "test_id": "interaction_test_2"
                }
            }
        ]
        self.ingested_concepts = set()
        self.digested_concepts = set()

    async def setup(self):
        """Initialize connections and test environment."""
        try:
            # Initialize Redis client with retry logic
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to connect to Redis (attempt {attempt + 1}/{max_retries})")
                    self.redis_client = aioredis.from_url(
                        REDIS_URL,
                        encoding="utf-8",
                        decode_responses=True,
                        socket_timeout=5.0,
                        socket_connect_timeout=5.0,
                        retry_on_timeout=True
                    )
                    await self.redis_client.ping()
                    logger.info("Connected to Redis successfully")
                    break
                except Exception as redis_error:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to connect to Redis after {max_retries} attempts: {str(redis_error)}")
                        raise
                    logger.warning(f"Redis connection attempt {attempt + 1} failed: {str(redis_error)}")
                    await asyncio.sleep(retry_delay)
            
            # Initialize Qdrant client with no authentication
            try:
                self.qdrant_client = QdrantClient(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    prefer_grpc=False,  # Use HTTP instead of gRPC
                    timeout=10.0  # Increase timeout
                )
                
                # Test Qdrant connection
                collections = self.qdrant_client.get_collections()
                logger.info(f"Connected to Qdrant. Available collections: {collections}")
            except Exception as qe:
                logger.error(f"Qdrant connection failed: {str(qe)}")
                raise

            # Initialize HTTP clients for services
            self.concept_dict_client = httpx.AsyncClient(
                base_url=CONCEPT_DICT_URL,
                timeout=30.0,
                headers=self.headers
            )
            self.concept_trainer_client = httpx.AsyncClient(base_url=CONCEPT_TRAINER_URL, timeout=30.0)

            # Test service connections
            try:
                dict_response = await self.concept_dict_client.get("/health")
                trainer_response = await self.concept_trainer_client.get("/health")
                if dict_response.status_code == 200 and trainer_response.status_code == 200:
                    logger.info("Both services are healthy")
                else:
                    raise Exception("One or both services are not healthy")
            except Exception as e:
                logger.error(f"Service health check failed: {str(e)}")
                raise

            # Clean up any existing test concepts
            await self._cleanup_test_concepts()
            
            logger.info("Test environment setup complete")
            return True
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            await self.cleanup()  # Ensure cleanup on setup failure
            return False

    async def cleanup(self):
        """Clean up test resources."""
        try:
            if hasattr(self, 'redis_client') and self.redis_client is not None:
                await self.redis_client.aclose()
            if hasattr(self, 'concept_dict_client') and self.concept_dict_client is not None:
                await self.concept_dict_client.aclose()
            if hasattr(self, 'concept_trainer_client') and self.concept_trainer_client is not None:
                await self.concept_trainer_client.aclose()
            # Only attempt to clean up test concepts if the client is initialized
            if hasattr(self, 'concept_dict_client') and self.concept_dict_client is not None:
                await self._cleanup_test_concepts()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            # Do not raise further

    async def _check_services(self):
        """Verify both services are running and healthy."""
        # Check concept-dictionary
        dict_health = await self.concept_dict_client.get("/health")
        assert dict_health.status_code == 200, "Concept dictionary not healthy"
        
        # Check concept-trainer-growable
        trainer_health = await self.concept_trainer_client.get("/health")
        assert trainer_health.status_code == 200, "Concept trainer not healthy"
        
        logger.info("Both services are healthy")

    async def _cleanup_test_concepts(self):
        """Remove any test concepts from both services."""
        if not hasattr(self, 'concept_dict_client') or self.concept_dict_client is None:
            logger.warning("concept_dict_client not initialized; skipping test concept cleanup.")
            return
        try:
            # Get all concepts
            concepts = await self.concept_dict_client.get("/concepts")
            if concepts.status_code == 200:
                for concept in concepts.json():
                    if concept.get("metadata", {}).get("test_id", "").startswith("interaction_test"):
                        # Delete from concept-dictionary
                        await self.concept_dict_client.delete(f"/concepts/{concept['term']}")
                        # Delete from Qdrant
                        if self.qdrant_client is not None:
                            self.qdrant_client.delete(
                                collection_name="concepts",
                                points_selector={"ids": [concept["term"]]}
                            )
            logger.info("Test concepts cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def add_test_concepts(self):
        """Add test concepts directly to Redis."""
        for concept in self.test_concepts:
            try:
                # Generate embedding
                embedding = [0.1] * 768  # Simple test embedding
                concept["embedding"] = embedding
                # Compose the concept dict as expected by the dictionary
                concept_dict = {
                    "term": concept["term"],
                    "definition": concept["definition"],
                    "embedding": embedding,
                    "metadata": concept["metadata"],
                    "last_updated": datetime.now(UTC).isoformat(),
                    "usage_count": 1
                }
                await self.redis_client.set(f"concept:{concept['term']}", json.dumps(concept_dict))
                logger.info(f"Added test concept to Redis: {concept['term']}")
            except Exception as e:
                logger.error(f"Failed to add test concept {concept['term']} to Redis: {e}")
                raise

    async def verify_concept_ingestion(self, timeout: int = 60) -> bool:
        """Verify that concepts are ingested by the trainer."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check trainer status
                status = await self.concept_trainer_client.get("/status")
                if status.status_code == 200:
                    data = status.json()
                    # Check if concepts are in trainer's queue or model
                    for concept in self.test_concepts:
                        if concept["term"] in data.get("queued_concepts", []) or \
                           concept["term"] in data.get("trained_concepts", []):
                            self.ingested_concepts.add(concept["term"])
                
                if len(self.ingested_concepts) == len(self.test_concepts):
                    logger.info("All test concepts ingested by trainer")
                    return True
                
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"Error checking ingestion: {e}")
                await asyncio.sleep(POLL_INTERVAL)
        
        logger.error(f"Timeout waiting for concept ingestion. Ingested: {self.ingested_concepts}")
        return False

    async def verify_concept_digestion(self, timeout: int = 60) -> bool:
        """Verify that concepts are digested by the dictionary."""
        start_time = time.time()
        digestion_started = False
        
        while time.time() - start_time < timeout:
            try:
                # First check if digestion has started
                if not digestion_started:
                    status = await self.concept_dict_client.get("/concepts/digest/status")
                    if status.status_code == 200:
                        data = status.json()
                        if data["status"] == "running":
                            logger.info("Digestion process has started")
                            digestion_started = True
                        elif data["status"] == "idle":
                            # If still idle, trigger digestion again
                            trigger = await self.concept_dict_client.post("/concepts/digest/trigger")
                            if trigger.status_code != 200:
                                logger.error(f"Failed to trigger digestion: {trigger.status_code}")
                                return False
                            logger.info("Triggered digestion process")
                            digestion_started = True
                        else:
                            logger.error(f"Unexpected digestion status: {data['status']}")
                            return False
                
                # If digestion has started, check concept status
                if digestion_started:
                    for concept in self.test_concepts:
                        if concept["term"] not in self.digested_concepts:
                            try:
                                concept_data = await self.concept_dict_client.get(f"/concepts/{concept['term']}")
                                if concept_data.status_code == 200:
                                    concept_info = concept_data.json()
                                    if concept_info.get("metadata", {}).get("last_improved"):
                                        self.digested_concepts.add(concept["term"])
                                        logger.info(f"Concept {concept['term']} has been digested")
                            except Exception as e:
                                logger.error(f"Error checking concept {concept['term']}: {e}")
                                continue
                
                if len(self.digested_concepts) == len(self.test_concepts):
                    logger.info("All test concepts digested by dictionary")
                    return True
                
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"Error checking digestion: {e}")
                await asyncio.sleep(POLL_INTERVAL)
        
        logger.error(f"Timeout waiting for concept digestion. Digested: {self.digested_concepts}")
        return False

    async def run_test(self) -> bool:
        """Run the complete interaction test."""
        try:
            # Setup
            if not await self.setup():
                return False
            
            try:
                # Add test concepts
                await self.add_test_concepts()

                # Import concepts from Redis into concept-dictionary
                import_response = await self.concept_dict_client.post("/concepts/import_from_redis")
                if import_response.status_code != 200:
                    logger.error(f"Failed to import concepts from Redis: {import_response.status_code}")
                    return False
                logger.info("Successfully imported concepts from Redis into concept-dictionary")
                
                # Trigger digestion
                trigger_response = await self.concept_dict_client.post("/concepts/digest/trigger")
                if trigger_response.status_code != 200:
                    logger.error(f"Failed to trigger digestion: {trigger_response.status_code}")
                    return False
                logger.info("Successfully triggered digestion")
                
                # Verify digestion
                if not await self.verify_concept_digestion():
                    return False
                
                # Verify ingestion
                if not await self.verify_concept_ingestion():
                    return False
                
                logger.info("Test completed successfully")
                return True
            finally:
                # Clean up test concepts before closing clients
                await self._cleanup_test_concepts()
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
        finally:
            # Close clients last
            await self.cleanup()

@pytest.mark.asyncio
async def test_concept_interaction():
    """Main test function."""
    test = ConceptInteractionTest()
    success = await test.run_test()
    assert success, "Concept interaction test failed"

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_concept_interaction()) 