import redis
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# Initialize Redis client
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class ConceptDB:
    """Database operations for concepts"""
    def __init__(self):
        self.redis = redis_client
        self.qdrant = qdrant_client
        self.model = model 