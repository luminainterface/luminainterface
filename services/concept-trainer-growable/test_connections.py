import os
import httpx
import redis
import logging
from qdrant_client import QdrantClient
import asyncio
import torch
from pathlib import Path
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables with localhost defaults
REDIS_URL = "redis://localhost:6379"  # Override environment variable
DICT_URL = "http://localhost:8828"    # Updated to correct port
DICT_API_KEY = os.getenv("DICT_API_KEY", "")
QDRANT_URL = "http://localhost:6333"  # Override environment variable
MODEL_PATH = Path(__file__).parent / "model_final.pth"
TRAINING_DIR = Path("training_data")  # Changed to local directory

def check_port_open(host: str, port: int) -> bool:
    """Check if a port is open on the given host"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception as e:
        logger.error(f"Error checking port {port} on {host}: {e}")
        return False

async def test_redis():
    """Test Redis connection"""
    try:
        logger.info("Testing Redis connection...")
        # First check if port is open
        host = REDIS_URL.split("://")[1].split(":")[0]
        port = int(REDIS_URL.split(":")[-1])
        if not check_port_open(host, port):
            logger.error(f"❌ Redis port {port} is not open on {host}")
            logger.info("To start Redis locally on Windows:")
            logger.info("1. Download Redis for Windows from https://github.com/microsoftarchive/redis/releases")
            logger.info("2. Install and start Redis server using: redis-server")
            return False
            
        client = redis.from_url(REDIS_URL)
        client.ping()
        logger.info("✅ Redis connection successful")
        return True
    except redis.ConnectionError as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.info("Make sure Redis server is running locally")
        return False
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False

async def test_concept_dict():
    """Test Concept Dictionary connection"""
    try:
        logger.info("Testing Concept Dictionary connection...")
        # First check if port is open
        host = DICT_URL.split("://")[1].split(":")[0]
        port = int(DICT_URL.split(":")[-1])
        if not check_port_open(host, port):
            logger.error(f"❌ Concept Dictionary port {port} is not open on {host}")
            logger.info("Make sure the Concept Dictionary service is running locally")
            return False
            
        headers = {}
        if DICT_API_KEY:
            headers["X-API-Key"] = DICT_API_KEY
        else:
            logger.warning("⚠️ No DICT_API_KEY provided - authentication may fail")
            
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{DICT_URL}/health", headers=headers, timeout=5.0)
            if resp.status_code == 200:
                logger.info("✅ Concept Dictionary connection successful")
                return True
            elif resp.status_code == 401:
                logger.error("❌ Concept Dictionary authentication failed - check DICT_API_KEY")
                return False
            else:
                logger.error(f"❌ Concept Dictionary connection failed with status {resp.status_code}")
                return False
    except httpx.ConnectError as e:
        logger.error(f"❌ Concept Dictionary connection failed: {e}")
        logger.info("Make sure the Concept Dictionary service is running locally")
        return False
    except Exception as e:
        logger.error(f"❌ Concept Dictionary connection failed: {e}")
        return False

async def test_qdrant():
    """Test Qdrant connection"""
    try:
        logger.info("Testing Qdrant connection...")
        # First check if port is open
        host = QDRANT_URL.split("://")[1].split(":")[0]
        port = int(QDRANT_URL.split(":")[-1])
        if not check_port_open(host, port):
            logger.error(f"❌ Qdrant port {port} is not open on {host}")
            logger.info("To start Qdrant locally:")
            logger.info("1. Install Docker Desktop for Windows")
            logger.info("2. Run: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
            return False
            
        client = QdrantClient(url=QDRANT_URL)
        collections = client.get_collections()
        logger.info("✅ Qdrant connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        logger.info("Make sure Qdrant is running locally (preferably via Docker)")
        return False

def check_model_file():
    """Check if model file exists"""
    try:
        logger.info("Checking for model file...")
        if MODEL_PATH.exists():
            # Try loading the model to verify it's valid
            state_dict = torch.load(MODEL_PATH)
            logger.info("✅ Model file exists and is valid")
            return True
        else:
            logger.warning("⚠️ Model file not found - system will use random initialization")
            return False
    except Exception as e:
        logger.error(f"❌ Model file exists but is invalid: {e}")
        return False

def check_training_dir():
    """Check if training directory exists"""
    try:
        logger.info("Checking training directory...")
        if TRAINING_DIR.exists():
            logger.info("✅ Training directory exists")
        else:
            logger.info("Creating training directory...")
            TRAINING_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("✅ Training directory created")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create/verify training directory: {e}")
        return False

async def run_all_checks():
    """Run all connection checks"""
    logger.info("Starting connection checks...")
    logger.info(f"Using Redis URL: {REDIS_URL}")
    logger.info(f"Using Concept Dictionary URL: {DICT_URL}")
    logger.info(f"Using Qdrant URL: {QDRANT_URL}")
    logger.info(f"Using training directory: {TRAINING_DIR.absolute()}\n")
    
    # Run all checks
    redis_ok = await test_redis()
    dict_ok = await test_concept_dict()
    qdrant_ok = await test_qdrant()
    model_ok = check_model_file()
    dir_ok = check_training_dir()
    
    # Print summary
    logger.info("\n=== Connection Check Summary ===")
    logger.info(f"Redis: {'✅' if redis_ok else '❌'}")
    logger.info(f"Concept Dictionary: {'✅' if dict_ok else '❌'}")
    logger.info(f"Qdrant: {'✅' if qdrant_ok else '❌'}")
    logger.info(f"Model File: {'✅' if model_ok else '⚠️'}")
    logger.info(f"Training Directory: {'✅' if dir_ok else '❌'}")
    logger.info("==============================\n")
    
    # Print next steps
    if not all([redis_ok, dict_ok, qdrant_ok]):
        logger.info("Next steps to get services running:")
        if not redis_ok:
            logger.info("1. Install and start Redis server locally")
        if not dict_ok:
            logger.info("2. Start the Concept Dictionary service locally")
        if not qdrant_ok:
            logger.info("3. Start Qdrant using Docker")
    
    return all([redis_ok, dict_ok, qdrant_ok, dir_ok])

if __name__ == "__main__":
    asyncio.run(run_all_checks()) 