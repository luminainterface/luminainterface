"""Prometheus metrics for the crawler service."""
from prometheus_client import Counter, Histogram

# Counters
WIKI_CRAWL_COUNTER = Counter(
    'wiki_crawl_total',
    'Number of Wikipedia crawls',
    ['status']
)

GRAPH_EDGE_CREATE = Counter(
    'graph_edge_create_total',
    'Number of graph edges created',
    ['status']
)

# Histograms
EMBEDDING_DURATION = Histogram(
    'embedding_duration_seconds',
    'Time taken to generate embeddings'
) 