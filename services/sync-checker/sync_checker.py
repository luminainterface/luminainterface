"""
Periodic model synchronization checker service.
Runs health checks and logs synchronization status to Docker logs.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
import httpx
import redis.asyncio as redis
from prometheus_api_client import PrometheusConnect
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

class SyncChecker:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        self.output_engine_url = os.getenv("OUTPUT_ENGINE_URL", "http://output-engine:9000")
        self.trainer_url = os.getenv("TRAINER_URL", "http://concept-trainer-growable:8710")
        self.prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
        self.check_interval = int(os.getenv("CHECK_INTERVAL", "300"))  # 5 minutes default
        
        self.redis = None
        self.http = None
        self.prom = None

    async def init_clients(self):
        """Initialize client connections."""
        self.redis = redis.from_url(self.redis_url)
        self.http = httpx.AsyncClient(timeout=30.0)
        self.prom = PrometheusConnect(url=self.prometheus_url)

    async def check_adapter_versions(self):
        """Check if all services report the same adapter version."""
        try:
            output_engine_health = await self.http.get(f"{self.output_engine_url}/health")
            trainer_health = await self.http.get(f"{self.trainer_url}/health")
            
            if not (output_engine_health.status_code == 200 and trainer_health.status_code == 200):
                logger.error("sync_check_failed", 
                           output_engine_status=output_engine_health.status_code,
                           trainer_status=trainer_health.status_code)
                return False
                
            output_version = output_engine_health.json()["adapter_id"]
            trainer_version = trainer_health.json()["adapter_id"]
            
            versions_match = output_version == trainer_version and output_version is not None
            
            logger.info("adapter_version_check",
                       output_engine_version=output_version,
                       trainer_version=trainer_version,
                       versions_match=versions_match)
            
            return versions_match
            
        except Exception as e:
            logger.error("version_check_error", error=str(e), exc_info=True)
            return False

    async def check_stream_lag(self):
        """Check Redis stream lag."""
        try:
            stream_info = await self.redis.xinfo_groups("model.adapter.updated")
            total_pending = sum(group["pending"] for group in stream_info)
            
            logger.info("stream_lag_check",
                       stream="model.adapter.updated",
                       total_pending=total_pending)
            
            return total_pending < 100
            
        except Exception as e:
            logger.error("stream_lag_check_error", error=str(e), exc_info=True)
            return False

    async def check_error_rate(self):
        """Check error rate from Prometheus metrics."""
        try:
            error_rate = self.prom.custom_query(
                'rate(output_errors_total[5m]) / rate(output_responses_total[5m])'
            )
            rate = float(error_rate[0]["value"][1]) if error_rate else 0
            
            logger.info("error_rate_check",
                       error_rate=rate,
                       threshold=0.05)
            
            return rate < 0.05
            
        except Exception as e:
            logger.error("error_rate_check_error", error=str(e), exc_info=True)
            return False

    async def check_alerts(self):
        """Check for active alerts."""
        try:
            alerts = self.prom.custom_query('ALERTS{alertname=~"AdapterVersionMismatch|RedisStreamLag"}')
            active_alerts = [alert["metric"]["alertname"] for alert in alerts]
            
            logger.info("alert_check",
                       active_alerts=active_alerts)
            
            return len(active_alerts) == 0
            
        except Exception as e:
            logger.error("alert_check_error", error=str(e), exc_info=True)
            return False

    async def run_check(self):
        """Run all checks and log overall status."""
        try:
            versions_ok = await self.check_adapter_versions()
            stream_ok = await self.check_stream_lag()
            error_rate_ok = await self.check_error_rate()
            alerts_ok = await self.check_alerts()
            
            all_ok = all([versions_ok, stream_ok, error_rate_ok, alerts_ok])
            
            logger.info("sync_status_check",
                       timestamp=datetime.utcnow().isoformat(),
                       versions_synced=versions_ok,
                       stream_healthy=stream_ok,
                       error_rate_acceptable=error_rate_ok,
                       no_critical_alerts=alerts_ok,
                       overall_status="HEALTHY" if all_ok else "DEGRADED")
            
            return all_ok
            
        except Exception as e:
            logger.error("sync_check_failed", error=str(e), exc_info=True)
            return False

    async def run(self):
        """Main run loop."""
        await self.init_clients()
        
        while True:
            try:
                await self.run_check()
            except Exception as e:
                logger.error("sync_checker_error", error=str(e), exc_info=True)
            
            await asyncio.sleep(self.check_interval)

async def main():
    checker = SyncChecker()
    await checker.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("sync_checker_shutdown")
        sys.exit(0) 