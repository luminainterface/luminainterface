import redis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_redis_connection():
    try:
        # Try different connection methods
        urls = [
            "redis://:02211998@redis:6379",
            "redis://:02211998@localhost:6379",
            "redis://:02211998@127.0.0.1:6379"
        ]
        
        for url in urls:
            logger.info(f"Trying to connect to Redis at {url}")
            try:
                r = redis.Redis.from_url(url, decode_responses=True)
                r.ping()
                logger.info(f"Successfully connected to Redis at {url}")
                return True
            except Exception as e:
                logger.error(f"Failed to connect to Redis at {url}: {e}")
        
        return False
    except Exception as e:
        logger.error(f"Error testing Redis connection: {e}")
        return False

if __name__ == "__main__":
    test_redis_connection() 