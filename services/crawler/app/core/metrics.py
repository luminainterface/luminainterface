from prometheus_client import Counter, Gauge, Histogram, Summary
import logging
import time
from typing import Dict, Optional
from datetime import datetime
import json
from collections import deque
import numpy as np

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("crawler")

# Prometheus metrics
CRAWLER_REQUESTS = Counter('crawler_requests_total', 'Total number of crawl requests', ['type'])
CRAWLER_FETCH_SECONDS = Histogram('crawler_fetch_seconds', 'Time spent fetching pages', 
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
CRAWLER_PROCESS_SECONDS = Histogram('crawler_process_seconds', 'Time spent processing pages',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
CRAWLER_PAGES = Counter('crawler_pages_total', 'Total number of pages crawled', ['status'])
CRAWLER_PRIORITY_AVG = Gauge('crawler_priority_avg', 'Average priority of crawled pages')
CRAWLER_ACTIVE_CRAWLS = Gauge('crawler_active_crawls', 'Number of active crawls')
CRAWLER_CACHE_HITS = Counter('crawler_cache_hits_total', 'Total number of cache hits')
CRAWLER_ERRORS = Counter('crawler_errors_total', 'Total number of crawl errors', ['type'])
CRAWLER_QUEUE_DEPTH = Gauge('crawler_queue_depth', 'Current depth of the crawl queue')
CRAWLER_CONSUMER_LAG = Gauge('crawler_consumer_lag', 'Consumer lag for crawl queue')
CRAWLER_PROCESSING_RATE = Gauge('crawler_processing_rate', 'Current processing rate (msgs/s)')
CRAWLER_CONTENT_TYPES = Counter('crawler_content_types_total', 'Content type counts', ['type'])
CRAWLER_PRIORITY_DIST = Counter('crawler_priority_distribution_total', 'Priority distribution', ['level'])

# New metrics for Phase 9
CRAWLER_RETRIES = Counter(
    "crawler_retries_total",
    "Total number of retry attempts",
    ["stream", "status"]
)

CRAWLER_DLQ_MESSAGES = Counter(
    "crawler_dlq_messages_total",
    "Total number of messages sent to dead-letter queue",
    ["stream", "reason"]
)

CRAWLER_SKIPS = Counter(
    "crawler_skips_total",
    "Total number of skipped operations",
    ["stream", "reason"]
)

CRAWLER_STREAM_DEPTH = Gauge(
    "crawler_stream_depth",
    "Current depth of input streams",
    ["stream"]
)

class MetricsCollector:
    def __init__(self, window_size: int = 60):
        """Initialize metrics collector with a sliding window for rate calculations"""
        self.window_size = window_size  # Window size in seconds
        self.processing_times = deque(maxlen=window_size)  # Store timestamps of processed items
        self.content_types = {'url': 0, 'pdf': 0, 'git': 0, 'system': 0}
        self.priority_dist = {'high': 0, 'medium': 0, 'low': 0}
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_summary_time = time.time()
        self.summary_interval = 10  # Summary every 10 seconds
        self._start_times: Dict[str, float] = {}
        
    def record_processing(self, content_type: str, priority: float, cache_hit: bool):
        """Record a processed item"""
        now = time.time()
        self.processing_times.append(now)
        
        # Update content type counts
        self.content_types[content_type] = self.content_types.get(content_type, 0) + 1
        CRAWLER_CONTENT_TYPES.labels(type=content_type).inc()
        
        # Update priority distribution
        if priority >= 0.8:
            level = 'high'
        elif priority >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        self.priority_dist[level] = self.priority_dist.get(level, 0) + 1
        CRAWLER_PRIORITY_DIST.labels(level=level).inc()
        
        # Update cache stats
        if cache_hit:
            self.cache_hits += 1
            CRAWLER_CACHE_HITS.inc()
        else:
            self.cache_misses += 1
            
        # Calculate and update processing rate
        current_window = [t for t in self.processing_times if now - t <= self.window_size]
        if current_window:
            rate = len(current_window) / self.window_size
            CRAWLER_PROCESSING_RATE.set(rate)
            
        # Log summary if interval has passed
        if now - self.last_summary_time >= self.summary_interval:
            self.log_summary()
            self.last_summary_time = now
            
    def log_summary(self):
        """Log a summary of current metrics in logfmt format"""
        now = datetime.utcnow().isoformat()
        cache_hit_ratio = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        # Calculate processing rate over the window
        current_window = [t for t in self.processing_times if time.time() - t <= self.window_size]
        processing_rate = len(current_window) / self.window_size if current_window else 0
        
        # Format metrics as logfmt
        metrics = {
            'ts': now,
            'level': 'info',
            'service': 'crawler',
            'queue_depth': CRAWLER_QUEUE_DEPTH._value.get(),
            'consumer_lag': CRAWLER_CONSUMER_LAG._value.get(),
            'processing_rate': f"{processing_rate:.1f}",
            'content_types': json.dumps(self.content_types),
            'priority_dist': json.dumps(self.priority_dist),
            'cache_hit_ratio': f"{cache_hit_ratio:.2f}",
            'fetch_latency_p95': f"{self._get_histogram_quantile(CRAWLER_FETCH_SECONDS, 0.95):.1f}s",
            'process_latency_p95': f"{self._get_histogram_quantile(CRAWLER_PROCESS_SECONDS, 0.95):.1f}s",
            'concurrent_crawls': CRAWLER_ACTIVE_CRAWLS._value.get()
        }
        
        # Convert to logfmt format
        logfmt = ' '.join(f"{k}={v}" for k, v in metrics.items())
        logger.info(logfmt)
        
    def _get_histogram_quantile(self, histogram: Histogram, quantile: float) -> float:
        """Calculate quantile from histogram buckets"""
        buckets = histogram._buckets
        if not buckets:
            return 0.0
            
        counts = [b[1] for b in buckets]
        if not any(counts):
            return 0.0
            
        total = sum(counts)
        target = total * quantile
        current = 0
        
        for bucket, count in buckets:
            current += count
            if current >= target:
                return bucket
                
        return buckets[-1][0]  # Return max bucket if quantile not found
        
    def update_queue_metrics(self, queue_depth: int, consumer_lag: int):
        """Update queue-related metrics"""
        CRAWLER_QUEUE_DEPTH.set(queue_depth)
        CRAWLER_CONSUMER_LAG.set(consumer_lag)
        
    def record_error(self, error_type: str):
        """Record an error occurrence"""
        CRAWLER_ERRORS.labels(type=error_type).inc()
        
    def record_fetch_time(self, seconds: float):
        """Record page fetch time"""
        CRAWLER_FETCH_SECONDS.observe(seconds)
        
    def record_process_time(self, seconds: float):
        """Record page processing time"""
        CRAWLER_PROCESS_SECONDS.observe(seconds)
        
    def update_active_crawls(self, count: int):
        """Update number of active crawls"""
        CRAWLER_ACTIVE_CRAWLS.set(count)
        
    def update_priority_avg(self, avg: float):
        """Update average priority"""
        CRAWLER_PRIORITY_AVG.set(avg)
        
    def record_retry(self, stream: str, attempt: int, success: bool):
        """Record a retry attempt"""
        status = "success" if success else "failure"
        CRAWLER_RETRIES.labels(stream=stream, status=status).inc()
        
    def record_dlq(self, stream: str, reason: str):
        """Record a message sent to dead-letter queue"""
        CRAWLER_DLQ_MESSAGES.labels(stream=stream, reason=reason).inc()
        
    def record_skip(self, reason: str):
        """Record a skipped operation"""
        CRAWLER_SKIPS.labels(stream="crawler", reason=reason).inc()
        
    def record_stream_depth(self, stream: str, depth: int):
        """Record current stream depth"""
        CRAWLER_STREAM_DEPTH.labels(stream=stream).set(depth)
        
    def start_operation(self, operation_id: str):
        """Start timing an operation"""
        self._start_times[operation_id] = time.time()
        
    def end_operation(self, operation_id: str):
        """End timing an operation and record duration"""
        if operation_id in self._start_times:
            duration = time.time() - self._start_times[operation_id]
            CRAWLER_PROCESS_SECONDS.observe(duration)
            del self._start_times[operation_id] 