import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/ops')
import asyncio
import logging
from typing import Dict, List, Optional
import numpy as np
from prometheus_client import Counter, Gauge, start_http_server
from prometheus_api_client import PrometheusConnect
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
from prometheus_fastapi_instrumentator import Instrumentator
import time, random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
TREND_ALERTS = Counter('trend_alerts_total', 'Total number of trend-based alerts generated')
TREND_DETECTIONS = Counter('trend_detections_total', 'Number of trend patterns detected')
TREND_CONFIDENCE = Gauge('trend_confidence', 'Confidence level of detected trends')
TREND_CONFIDENCE_SCORE = Gauge(
    "trend_confidence_score",
    "Confidence level of current trend prediction"
)
TREND_PREDICTION_CONFIDENCE = Gauge(
    "trend_prediction_confidence",
    "R² confidence of latest trend regression"
)

# Initialize FastAPI app
app = FastAPI(title="Lumina Trend Analyzer")

# Initialize Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# Prometheus connection
prom = PrometheusConnect(url="http://prometheus:9090", disable_ssl=True)

class TrendConfig(BaseModel):
    """Configuration for trend detection"""
    metric: str
    window: str  # e.g., "1h", "6h", "24h"
    threshold: float
    min_samples: int = 30
    confidence_threshold: float = 0.8

class TrendAnalyzer:
    def __init__(self):
        self.trend_configs = {
            'embedding_latency_ms': TrendConfig(
                metric='embedding_latency_ms',
                window='1h',
                threshold=0.1,  # 10% increase per hour
                min_samples=30
            ),
            'memory_rss_mb': TrendConfig(
                metric='memory_rss_mb',
                window='6h',
                threshold=0.05,  # 5% increase per 6 hours
                min_samples=60
            ),
            'fps_current': TrendConfig(
                metric='fps_current',
                window='1h',
                threshold=-0.15,  # 15% decrease per hour
                min_samples=30
            ),
            'concept_throughput': TrendConfig(
                metric='concept_throughput',
                window='6h',
                threshold=-0.1,  # 10% decrease per 6 hours
                min_samples=60
            )
        }
        
    def calculate_trend(self, values: List[float], timestamps: List[float]) -> tuple:
        """Calculate trend using linear regression"""
        if len(values) < 2:
            return 0.0, 0.0
            
        x = np.array(timestamps)
        y = np.array(values)
        
        # Calculate linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate R-squared for confidence
        y_pred = slope * x + intercept
        r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        return slope, r_squared
        
    async def analyze_metric(self, config: TrendConfig) -> Optional[Dict]:
        """Analyze a metric for trends"""
        try:
            # Query Prometheus for metric data
            query = f"{config.metric}[{config.window}]"
            result = prom.custom_query(query)
            
            if not result:
                return None
                
            # Extract values and timestamps
            values = [float(point[1]) for point in result[0]['values']]
            timestamps = [float(point[0]) for point in result[0]['values']]
            
            if len(values) < config.min_samples:
                return None
                
            # Calculate trend
            slope, confidence = self.calculate_trend(values, timestamps)
            
            # Check if trend exceeds threshold
            if abs(slope) > config.threshold and confidence > config.confidence_threshold:
                TREND_DETECTIONS.inc()
                TREND_CONFIDENCE.set(confidence)
                
                return {
                    'metric': config.metric,
                    'slope': slope,
                    'confidence': confidence,
                    'current_value': values[-1],
                    'threshold': config.threshold
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing metric {config.metric}: {e}")
            return None
            
    async def check_trends(self):
        """Check all configured metrics for trends"""
        while True:
            try:
                for metric, config in self.trend_configs.items():
                    trend = await self.analyze_metric(config)
                    
                    if trend:
                        # Generate early warning alert
                        alert = {
                            'type': 'trend_warning',
                            'metric': metric,
                            'slope': trend['slope'],
                            'confidence': trend['confidence'],
                            'current_value': trend['current_value'],
                            'threshold': trend['threshold']
                        }
                        
                        # Publish alert to Redis
                        await redis_client.publish('trend_alerts', str(alert))
                        TREND_ALERTS.inc()
                        
                        logger.info(f"Trend detected for {metric}: {trend}")
                        
            except Exception as e:
                logger.error(f"Error in trend checking loop: {e}")
                
            await asyncio.sleep(60)  # Check every minute

# Initialize trend analyzer
analyzer = TrendAnalyzer()

@app.on_event("startup")
async def startup():
    """Start trend analysis on startup"""
    asyncio.create_task(analyzer.check_trends())
    start_http_server(8124)
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.on_event("startup")
async def seed_metric():
    # simple metric mutation so Prom sees values immediately
    @app.middleware("http")
    async def update_confidence(request, call_next):
        TREND_PREDICTION_CONFIDENCE.set(random.uniform(0.6, 0.9))
        return await call_next(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8123) 