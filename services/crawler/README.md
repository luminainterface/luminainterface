# Smart Crawler Service

A service for intelligently crawling Wikipedia pages and building a knowledge graph. The service integrates with:
- Qdrant for vector storage
- Redis for caching and queuing
- Graph API for relationship management
- Concept Dictionary for concept storage

## Features

- Smart crawling with adaptive depth and priority
- Intelligent content extraction and filtering
- Automatic concept linking and relationship discovery
- Vector-based similarity search
- Priority-based crawl scheduling
- Concurrent crawling with rate limiting
- Comprehensive monitoring and metrics
- Cache management with TTL

## API Endpoints

### POST /api/v1/smart-crawl
Start a smart crawl from a given Wikipedia page with adaptive depth and priority.

Request body:
```json
{
    "start_title": "string",
    "max_pages": 100,  // optional
    "min_relevance_score": 0.6,  // optional
    "max_depth": 3,  // optional
    "max_links_per_page": 15  // optional
}
```

### POST /api/v1/crawl
Legacy endpoint for basic crawling.

Request body:
```json
{
    "start_title": "string",
    "max_depth": 2,
    "max_links_per_page": 10
}
```

### GET /api/v1/search
Search for similar concepts.

Query parameters:
- `query`: Search query string
- `limit`: Maximum number of results (default: 5)

### GET /api/v1/stats
Get statistics about the current crawl process.

Response:
```json
{
    "total_pages": 100,
    "average_priority": 0.75,
    "max_priority": 0.95,
    "active_crawls": 3,
    "cache_hits": {
        "total": 50,
        "by_depth": {
            "0": 20,
            "1": 15,
            "2": 10,
            "3": 5
        }
    }
}
```

### GET /api/v1/health
Health check endpoint.

## Environment Variables

- `REDIS_URL`: Redis connection URL (default: redis://redis:6379)
- `QDRANT_URL`: Qdrant connection URL (default: http://qdrant:6333)
- `GRAPH_API_URL`: Graph API URL (default: http://graph-api:8200)
- `CONCEPT_DICT_URL`: Concept Dictionary URL (default: http://concept-dict:8000)
- `EMBEDDING_MODEL`: Sentence transformer model name (default: all-MiniLM-L6-v2)
- `WIKI_SEARCH_DEPTH`: Maximum crawl depth (default: 3)
- `WIKI_MAX_RESULTS`: Maximum links per page (default: 15)
- `MIN_RELEVANCE_SCORE`: Minimum relevance score for pages (default: 0.6)
- `MAX_CONCURRENT_CRAWLS`: Maximum number of concurrent crawls (default: 5)

## Smart Crawling Features

### Priority Calculation
The smart crawler calculates priority scores for pages based on:
- Depth in the crawl tree
- Existence in concept dictionary
- Content relevance using vector similarity
- Link relationships

### Adaptive Depth
The crawler adapts its depth based on:
- Content relevance scores
- Priority thresholds
- Available resources
- Cache hit rates

### Concurrent Processing
- Multiple pages are processed concurrently
- Rate limiting prevents overloading
- Resource usage is monitored and adjusted

### Caching Strategy
- Pages are cached with TTL
- Cache hits are tracked by depth
- Cache invalidation based on content updates

## Monitoring

### Prometheus Metrics
- `crawler_requests_total`: Total crawl requests
- `crawler_fetch_seconds`: Page fetch latency
- `crawler_process_seconds`: Page processing latency
- `crawler_pages_total`: Total pages crawled
- `crawler_priority_avg`: Average page priority
- `crawler_active_crawls`: Active crawl count
- `crawler_cache_hits_total`: Cache hit count
- `crawler_errors_total`: Error count

## Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8400
```

## Docker

Build and run with Docker:
```bash
docker build -t smart-crawler .
docker run -p 8400:8400 smart-crawler
```

## Architecture

The service is built with a modular architecture:

- `core/`: Core components
  - `smart_crawler.py`: Smart crawler implementation
  - `crawler.py`: Base crawler functionality
  - `wiki_client.py`: Wikipedia API client
  - `vector_store.py`: Qdrant vector store client
  - `graph_client.py`: Graph API client
  - `concept_client.py`: Concept Dictionary client
  - `redis_client.py`: Redis client
  - `graph_processor.py`: Graph processing utilities

- `api/`: API endpoints
  - `router.py`: FastAPI router

- `main.py`: Application entry point 