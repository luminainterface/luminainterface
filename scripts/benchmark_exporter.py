from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
import json
import os
from typing import Dict, Any
import asyncio
from aiohttp import web
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
LATENCY_GAUGE = Gauge('embedding_latency_ms', 'Embedding service latency in milliseconds')
MEMORY_GAUGE = Gauge('memory_rss_mb', 'Memory usage in MB')
FPS_GAUGE = Gauge('fps_current', 'Current FPS in the UI')
CONCEPT_THROUGHPUT = Gauge('concept_throughput', 'Concepts processed per second')

# Histograms for detailed latency tracking
EMBED_LATENCY = Histogram(
    'embedding_duration_seconds',
    'Embedding request duration',
    buckets=[.010, .025, .050, .075, .100, .250, .500, .750, 1.0]
)

SEARCH_LATENCY = Histogram(
    'search_duration_seconds',
    'Vector search request duration',
    buckets=[.025, .050, .075, .100, .250, .500, .750, 1.0]
)

INFERENCE_LATENCY = Histogram(
    'inference_duration_seconds',
    'Concept inference duration',
    buckets=[.050, .100, .250, .500, .750, 1.0, 2.0]
)

# Counters for events
CRAWL_REQUESTS = Counter('crawl_requests_total', 'Total number of crawl requests')
CONCEPT_UPDATES = Counter('concept_updates_total', 'Total number of concept updates')

class BenchmarkExporter:
    def __init__(self, port: int = 9109):
        self.port = port
        self.latest_metrics: Dict[str, Any] = {}
        
    async def start(self):
        """Start the metrics server and benchmark runner."""
        # Start Prometheus metrics server
        start_http_server(self.port)
        logger.info(f"Metrics server started on port {self.port}")
        
        # Start web server for benchmark control
        app = web.Application()
        app.router.add_post('/metrics/update', self.handle_metric_update)
        app.router.add_get('/metrics/latest', self.handle_get_metrics)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port + 1)
        await site.start()
        logger.info(f"Control server started on port {self.port + 1}")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    async def handle_metric_update(self, request: web.Request) -> web.Response:
        """Handle metric updates from the benchmark runner."""
        try:
            data = await request.json()
            self.update_metrics(data)
            return web.Response(text='OK')
        except Exception as e:
            logger.error(f"Error handling metric update: {e}")
            return web.Response(status=500, text=str(e))
            
    async def handle_get_metrics(self, request: web.Request) -> web.Response:
        """Return latest metrics as JSON."""
        return web.json_response(self.latest_metrics)
        
    def update_metrics(self, data: Dict[str, Any]):
        """Update Prometheus metrics from benchmark data."""
        self.latest_metrics = data
        
        # Update gauges
        if 'embedding_latency' in data:
            LATENCY_GAUGE.set(data['embedding_latency'])
            EMBED_LATENCY.observe(data['embedding_latency'] / 1000)  # Convert to seconds
            
        if 'memory_usage' in data:
            MEMORY_GAUGE.set(data['memory_usage'])
            
        if 'fps' in data:
            FPS_GAUGE.set(data['fps'])
            
        if 'concept_throughput' in data:
            CONCEPT_THROUGHPUT.set(data['concept_throughput'])
            
        # Update histograms
        if 'search_latency' in data:
            SEARCH_LATENCY.observe(data['search_latency'] / 1000)
            
        if 'inference_latency' in data:
            INFERENCE_LATENCY.observe(data['inference_latency'] / 1000)
            
        # Update counters
        if 'crawl_requests' in data:
            CRAWL_REQUESTS.inc(data['crawl_requests'])
            
        if 'concept_updates' in data:
            CONCEPT_UPDATES.inc(data['concept_updates'])
            
        logger.info("Metrics updated successfully")

def main():
    exporter = BenchmarkExporter()
    
    try:
        asyncio.run(exporter.start())
    except KeyboardInterrupt:
        logger.info("Exporter shutting down...")
    except Exception as e:
        logger.error(f"Exporter failed: {e}")
        raise

if __name__ == '__main__':
    main() 