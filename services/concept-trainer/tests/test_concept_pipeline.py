import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Dict, Any
from loguru import logger
import sys
import os
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration
CONCEPT_DICT_URL = "http://localhost:8828"  # Using the concept-dictionary port
TRAINER_URL = "http://localhost:8710"  # Using the growable trainer port

# Configure httpx client with longer timeout
CLIENT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
CLIENT_LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=10)

TEST_CONCEPT = {
    "term": "quantum_entanglement_paradox",
    "definition": """
    A fundamental quantum mechanical phenomenon where two or more particles become correlated 
    in such a way that the quantum state of each particle cannot be described independently, 
    even when separated by large distances. This leads to apparent paradoxes such as the 
    Einstein-Podolsky-Rosen (EPR) paradox and challenges classical notions of locality and 
    causality. The phenomenon has been experimentally verified through Bell's inequality 
    violations and forms the basis for quantum computing and quantum cryptography.
    """.strip(),
    "examples": [
        "The quantum entanglement paradox was demonstrated in the famous EPR experiment.",
        "Quantum entanglement paradoxes challenge our understanding of locality in physics."
    ],
    "metadata": {
        "complexity_score": 0.95,
        "domain": "quantum_physics",
        "subdomains": ["quantum_mechanics", "quantum_information", "quantum_computing"],
        "related_concepts": [
            "bell_inequality",
            "quantum_superposition",
            "quantum_tunneling",
            "quantum_decoherence"
        ],
        "quality_score": 0.92,
        "source": "quantum_physics_textbook",
        "last_verified": datetime.utcnow().isoformat()
    },
    "force_quality_score": 1.0  # Ensure high quality score for testing
}

async def check_concept_exists(term: str) -> bool:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{CONCEPT_DICT_URL}/concepts/{term}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error checking concept: {e}")
            return False

async def inject_test_concept() -> bool:
    async with httpx.AsyncClient() as client:
        try:
            # Use the new test endpoint
            response = await client.post(
                f"{CONCEPT_DICT_URL}/test/inject_concept",
                json=TEST_CONCEPT
            )
            if response.status_code == 200:
                logger.info("Successfully injected test concept")
                return True
            else:
                logger.error(f"Failed to inject concept: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error injecting concept: {e}")
            return False

async def check_digest_status() -> dict:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{CONCEPT_DICT_URL}/concepts/digest/status")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get digest status: {response.text}")
                return {"status": "error"}
        except Exception as e:
            logger.error(f"Error checking digest status: {e}")
            return {"status": "error"}

async def check_training_status(term: str) -> dict:
    async with httpx.AsyncClient() as client:
        try:
            # First check concept-dictionary for training status
            response = await client.get(f"{CONCEPT_DICT_URL}/concepts/{term}")
            if response.status_code == 200:
                data = response.json()
                training_status = data.get("metadata", {}).get("training_status", {})
                
                # Also check trainer status
                trainer_response = await client.get(f"{TRAINER_URL}/fetcher/stats")
                if trainer_response.status_code == 200:
                    trainer_stats = trainer_response.json()
                    logger.info(f"Trainer stats: {trainer_stats}")
                
                return training_status
            else:
                logger.error(f"Failed to get training status: {response.text}")
                return {"status": "error"}
        except Exception as e:
            logger.error(f"Error checking training status: {e}")
            return {"status": "error"}

async def monitor_pipeline(max_wait: int = 300) -> None:
    start_time = time.time()
    digest_notified = False
    training_notified = False
    logger.info("Starting pipeline monitoring...")
    
    while time.time() - start_time < max_wait:
        if not await check_concept_exists(TEST_CONCEPT["term"]):
            logger.error("Test concept not found in dictionary!")
            return
            
        # Check digest status
        digest_status = await check_digest_status()
        if digest_status.get("status") == "running":
            if not digest_notified:
                logger.info("Auto-digestion is running")
                digest_notified = True
        else:
            if not digest_notified:
                logger.warning("NO DIGEST: Auto-digestion is not running")
                digest_notified = True
                
        # Check training status
        training_status = await check_training_status(TEST_CONCEPT["term"])
        if training_status.get("status") == "training":
            if not training_notified:
                logger.info("Concept is being trained")
                training_notified = True
        elif training_status.get("status") == "trained":
            logger.success("Concept has been successfully trained!")
            return
        elif training_status.get("status") == "failed":
            logger.error(f"Training failed: {training_status.get('error', 'Unknown error')}")
            return
        else:
            if not training_notified:
                logger.warning("NO TRAIN: Concept is not being trained")
                training_notified = True
                
        await asyncio.sleep(5)
    logger.error("Pipeline monitoring timed out")

async def check_service_health(client: httpx.AsyncClient, url: str, name: str) -> bool:
    """Check health of a service with detailed error logging."""
    try:
        logger.info(f"Checking {name} health at {url}")
        response = await client.get(f"{url}/health")
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"{name} health check passed: {health_data}")
            return True
        else:
            logger.error(f"{name} health check failed with status {response.status_code}: {response.text}")
            return False
    except httpx.ConnectError as e:
        logger.error(f"Connection error checking {name} health: {str(e)}")
        logger.error(f"Connection details: {e.__dict__}")
        return False
    except httpx.TimeoutException as e:
        logger.error(f"Timeout checking {name} health: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking {name} health: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def run_test():
    logger.info("Starting concept pipeline test...")
    
    # Inject test concept with detailed error handling
    logger.info("Injecting test concept...")
    try:
        if not await inject_test_concept():
            logger.error("Failed to inject test concept")
            return
    except Exception as e:
        logger.error(f"Error during concept injection: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
        
    # Verify concept exists with detailed error handling
    logger.info("Verifying concept exists...")
    try:
        if not await check_concept_exists(TEST_CONCEPT["term"]):
            logger.error("Test concept not found after injection")
            return
    except Exception as e:
        logger.error(f"Error verifying concept: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
        
    # Monitor pipeline with detailed error handling
    logger.info("Starting pipeline monitoring...")
    try:
        await monitor_pipeline()
    except Exception as e:
        logger.error(f"Error during pipeline monitoring: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return
        
    logger.info("Test completed successfully")

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    asyncio.run(run_test()) 