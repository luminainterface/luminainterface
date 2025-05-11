import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/ops')
import asyncio
import json
import logging
from typing import Dict, Any, List
import aiohttp
from prometheus_client import Counter, Histogram, make_asgi_app
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
ACTIONS_EXECUTED = Counter('auto_mitigation_actions_total', 'Total number of auto-mitigation actions executed')
ACTION_DURATION = Histogram('auto_mitigation_duration_seconds', 'Time taken to execute auto-mitigation actions')
ACTION_ERRORS = Counter('auto_mitigation_errors_total', 'Total number of auto-mitigation action failures')

# Initialize FastAPI app
app = FastAPI(title="Lumina Action Handler")

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

class Alert(BaseModel):
    """Alert model from Alertmanager"""
    status: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    startsAt: str
    endsAt: str = None

class ActionHandler:
    def __init__(self):
        self.actions = {
            'memory_rss_mb': self.handle_memory_pressure,
            'embedding_latency_ms': self.handle_embedding_latency,
            'fps_current': self.handle_fps_drop,
            'crawl_queue_ratio': self.handle_crawl_queue
        }
        
    async def handle_memory_pressure(self, alert: Alert) -> None:
        """Handle high memory pressure by restarting the crawler"""
        try:
            logger.info("Handling memory pressure alert")
            async with aiohttp.ClientSession() as session:
                # Restart crawler container
                async with session.post('http://crawler:8000/restart') as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to restart crawler: {resp.status}")
            logger.info("Successfully restarted crawler")
        except Exception as e:
            logger.error(f"Error handling memory pressure: {e}")
            raise

    async def handle_embedding_latency(self, alert: Alert) -> None:
        """Handle high embedding latency by triggering index optimization"""
        try:
            logger.info("Handling embedding latency alert")
            # Enqueue index optimization task
            await redis_client.lpush('optimization_tasks', json.dumps({
                'type': 'optimize_index',
                'priority': 'high',
                'reason': 'high_latency'
            }))
            logger.info("Successfully enqueued index optimization")
        except Exception as e:
            logger.error(f"Error handling embedding latency: {e}")
            raise

    async def handle_fps_drop(self, alert: Alert) -> None:
        """Handle FPS drop by enabling performance mode"""
        try:
            logger.info("Handling FPS drop alert")
            # Publish performance mode message to WebSocket
            await redis_client.publish('ui_commands', json.dumps({
                'command': 'performance_mode',
                'enabled': True
            }))
            logger.info("Successfully enabled performance mode")
        except Exception as e:
            logger.error(f"Error handling FPS drop: {e}")
            raise

    async def handle_crawl_queue(self, alert: Alert) -> None:
        """Handle crawl queue imbalance by triggering emergency flush"""
        try:
            logger.info("Handling crawl queue imbalance")
            # Trigger emergency queue flush
            await redis_client.publish('crawler_commands', json.dumps({
                'command': 'emergency_flush',
                'reason': 'queue_imbalance'
            }))
            logger.info("Successfully triggered emergency queue flush")
        except Exception as e:
            logger.error(f"Error handling crawl queue: {e}")
            raise

    async def process_alert(self, alert: Alert) -> None:
        """Process an alert and execute appropriate action"""
        try:
            # Extract metric name from alert labels
            metric_name = alert.labels.get('alertname', '').lower()
            
            # Find matching action handler
            handler = self.actions.get(metric_name)
            if not handler:
                logger.warning(f"No handler found for metric: {metric_name}")
                return
                
            # Execute action with timing
            with ACTION_DURATION.time():
                await handler(alert)
                
            ACTIONS_EXECUTED.inc()
            logger.info(f"Successfully processed alert: {metric_name}")
            
        except Exception as e:
            ACTION_ERRORS.inc()
            logger.error(f"Error processing alert: {e}")
            raise

# Initialize action handler
handler = ActionHandler()

@app.post("/webhook")
async def webhook(alert: Alert):
    """Handle incoming alerts from Alertmanager"""
    try:
        await handler.process_alert(alert)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8700) 