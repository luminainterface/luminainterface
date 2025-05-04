import sys
import json
from loguru import logger
from typing import Any, Dict

def setup_logging(json_logs: bool = False) -> None:
    """Configure logging with loguru."""
    # Remove default handler
    logger.remove()
    
    if json_logs:
        # JSON format for production
        logger.add(
            sys.stdout,
            format=lambda record: json.dumps({
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "module": record["module"],
                "function": record["function"],
                "line": record["line"],
                "extra": record["extra"]
            }),
            serialize=True
        )
    else:
        # Pretty format for development
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
    
    # Add file logging
    logger.add(
        "logs/lumina.log",
        rotation="500 MB",
        retention="10 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

def log_request(request_id: str, method: str, path: str, status_code: int, duration: float) -> None:
    """Log HTTP request details."""
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2)
        }
    )

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log error with context."""
    logger.error(
        str(error),
        extra={"error_type": type(error).__name__, "context": context or {}}
    ) 