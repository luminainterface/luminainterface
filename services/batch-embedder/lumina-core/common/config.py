from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class LuminaSettings(BaseSettings):
    # Service URLs
    REDIS_URL: str = "redis://redis:6379"
    QDRANT_URL: str = "http://qdrant:6333"
    OLLAMA_URL: str = "http://ollama:11434"
    GRAPH_API_URL: str = "http://graph-api:8200"
    EVENT_MUX_URL: str = "http://event-mux:8000"
    MASTERCHAT_URL: str = "http://masterchat:8000"
    CRAWLER_URL: str = "http://crawler:8400"
    PROMETHEUS_URL: str = "http://prometheus:9090"
    GRAFANA_URL: str = "http://grafana:3000"

    # Service ports
    SERVICE_PORT: int = 8000
    
    # Redis configuration
    REDIS_MAX_CONNECTIONS: int = 10
    REDIS_TIMEOUT: int = 5
    
    # Qdrant configuration
    QDRANT_TIMEOUT: int = 10
    QDRANT_GRPC_PORT: Optional[int] = None
    
    # Monitoring
    METRICS_ENABLED: bool = True
    TRACING_ENABLED: bool = True
    JAEGER_AGENT_HOST: str = "jaeger"
    JAEGER_AGENT_PORT: int = 6831
    
    # Health check
    HEALTH_CHECK_INTERVAL: int = 30
    HEALTH_CHECK_TIMEOUT: int = 5
    
    # Audit configuration
    AUDIT_INTERVAL: int = 60
    AUDIT_SCORE_THRESHOLD: float = 80.0
    
    # Event configuration
    EVENT_BATCH_SIZE: int = 100
    EVENT_PROCESSING_TIMEOUT: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> LuminaSettings:
    """Get cached settings instance."""
    return LuminaSettings()

# Example usage:
# from lumina_core.common.config import get_settings
# settings = get_settings()
# redis_url = settings.REDIS_URL 