from pydantic import BaseSettings, HttpUrl, RedisDsn
from typing import Optional

class LuminaSettings(BaseSettings):
    # Service URLs
    GRAPH_API_URL: HttpUrl
    REDIS_URL: RedisDsn
    QDRANT_URL: HttpUrl
    OLLAMA_URL: HttpUrl
    EVENT_MUX_URL: HttpUrl
    MASTERCHAT_URL: HttpUrl
    CRAWLER_URL: HttpUrl
    PROMETHEUS_URL: HttpUrl
    
    # Optional service-specific URLs
    JAEGER_URL: Optional[HttpUrl] = None
    LOKI_URL: Optional[HttpUrl] = None
    
    # Timeouts and retries
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    REDIS_TIMEOUT: int = 5
    HEALTH_CHECK_TIMEOUT: int = 5
    
    # Feature flags
    ENABLE_TRACING: bool = True
    ENABLE_METRICS: bool = True
    ENABLE_HEALTH_CHECK: bool = True
    
    # Service limits
    MAX_BATCH_SIZE: int = 100
    MAX_QUEUE_SIZE: int = 1000
    
    # Metrics
    METRICS_PORT: int = 8000
    METRICS_PREFIX: str = "lumina"
    
    # Version
    VERSION: str = "1.0.0"
    
    # Health check thresholds
    HEALTH_CHECK_INTERVAL: int = 30  # seconds
    HEALTH_CHECK_WARNING_THRESHOLD: int = 3  # consecutive failures
    HEALTH_CHECK_CRITICAL_THRESHOLD: int = 5  # consecutive failures
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = LuminaSettings() 