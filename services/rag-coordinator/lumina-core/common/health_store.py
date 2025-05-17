from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime, timedelta
import redis.asyncio as redis
from .metrics import service_health, service_latency, health_check_errors

logger = logging.getLogger(__name__)

class HealthStore:
    def __init__(
        self,
        redis_client: redis.Redis,
        history_ttl: int = 86400,  # 24 hours
        max_history_size: int = 1000
    ):
        self.redis = redis_client
        self.history_ttl = history_ttl
        self.max_history_size = max_history_size

    async def store_health_check(
        self,
        service_name: str,
        service_type: str,
        check_name: str,
        status: Dict[str, Any]
    ):
        """Store a health check result."""
        try:
            key = f"health:history:{service_name}:{check_name}"
            timestamp = datetime.utcnow().isoformat()
            
            # Add metadata
            status.update({
                "timestamp": timestamp,
                "service_name": service_name,
                "service_type": service_type,
                "check_name": check_name
            })
            
            # Store in Redis
            await self.redis.lpush(key, json.dumps(status))
            await self.redis.ltrim(key, 0, self.max_history_size - 1)
            await self.redis.expire(key, self.history_ttl)
            
            # Update metrics
            health_value = 1.0 if status.get("status") == "healthy" else \
                         0.5 if status.get("status") == "degraded" else 0.0
            service_health.labels(
                service=service_name,
                type=service_type,
                check=check_name
            ).set(health_value)
            
            if "latency_ms" in status:
                service_latency.labels(
                    service=service_name,
                    type=service_type,
                    check=check_name
                ).observe(status["latency_ms"] / 1000.0)
            
        except Exception as e:
            logger.error(f"Error storing health check: {str(e)}")
            health_check_errors.labels(
                service=service_name,
                type=service_type,
                check=check_name,
                error_type="store_error"
            ).inc()

    async def get_health_history(
        self,
        service_name: str,
        check_name: str,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get health check history for a service."""
        try:
            key = f"health:history:{service_name}:{check_name}"
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            
            # Get all entries
            entries = await self.redis.lrange(key, 0, -1)
            
            # Parse and filter
            history = []
            for entry in entries:
                data = json.loads(entry)
                if data["timestamp"] >= cutoff:
                    history.append(data)
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting health history: {str(e)}")
            return []

    async def get_health_trends(
        self,
        service_name: str,
        check_name: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Calculate health trends for a service."""
        try:
            history = await self.get_health_history(service_name, check_name, hours)
            if not history:
                return {
                    "success_rate": 0.0,
                    "avg_latency": 0.0,
                    "error_rate": 0.0
                }
            
            total_checks = len(history)
            successful_checks = sum(1 for check in history if check.get("status") == "healthy")
            total_latency = sum(check.get("latency_ms", 0) for check in history)
            
            return {
                "success_rate": successful_checks / total_checks if total_checks > 0 else 0.0,
                "avg_latency": total_latency / total_checks if total_checks > 0 else 0.0,
                "error_rate": (total_checks - successful_checks) / total_checks if total_checks > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating health trends: {str(e)}")
            return {
                "success_rate": 0.0,
                "avg_latency": 0.0,
                "error_rate": 0.0
            }

    async def cleanup_old_history(self):
        """Clean up old health check history."""
        try:
            keys = await self.redis.keys("health:history:*")
            for key in keys:
                await self.redis.expire(key, self.history_ttl)
        except Exception as e:
            logger.error(f"Error cleaning up health history: {str(e)}") 