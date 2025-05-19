"""Configuration settings for the crawler service."""
import os
from typing import Dict, Any

# Redis settings
REDIS_URL = os.getenv("REDIS_URL", "redis://:02211998@redis:6379")

# Qdrant settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "crawl_results")

# Graph API settings
GRAPH_API_URL = os.getenv("GRAPH_API_URL", "http://graph-api:8200")

# Concept Dictionary settings
CONCEPT_DICT_URL = os.getenv("CONCEPT_DICT_URL", "http://concept-dictionary:8828")

# Ollama settings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

# Embedding model settings (legacy, kept for compatibility)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Crawler settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "86400"))  # 24 hours
MAX_CONCURRENT_CRAWLS = int(os.getenv("MAX_CONCURRENT_CRAWLS", "5"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.6"))

# File type settings
ALLOWED_EXTENSIONS = {
    '.md', '.txt', '.rst', '.py', '.js', '.ts', '.java', '.cpp', '.h', 
    '.hpp', '.c', '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt'
}

# Directory settings
SKIP_DIRS = {
    '.git', 'node_modules', 'venv', '__pycache__', 'target', 'build',
    'dist', 'vendor', 'bower_components'
}

# Priority weights for different source types
PRIORITY_WEIGHTS: Dict[str, float] = {
    'dictionary': 1.2,  # Highest priority for dictionary crawling
    'git': 1.0,
    'pdf': 0.8,
    'url': 0.6,
    'graph': 0.4
}

# Training data paths
TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "/app/training_data")
GIT_TRAINING_PATH = os.getenv("GIT_TRAINING_PATH", os.path.join(TRAINING_DATA_PATH, "git"))
PDF_TRAINING_PATH = os.getenv("PDF_TRAINING_PATH", os.path.join(TRAINING_DATA_PATH, "pdfs"))
GRAPH_TRAINING_PATH = os.getenv("GRAPH_TRAINING_PATH", os.path.join(TRAINING_DATA_PATH, "graph (1).json"))

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Metrics settings
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8400"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))

def get_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary."""
    return {
        key: value for key, value in globals().items()
        if not key.startswith('_') and isinstance(value, (str, int, float, bool, dict, set))
    } 