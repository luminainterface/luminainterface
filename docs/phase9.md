# Phase 9: Smart Content Processing & Enhanced Concept Linking

## Overview

Phase 9 introduces three major enhancements to the system:

1. Smart Content Extraction Service
2. Enhanced Concept Linking Service
3. Unified Retry & Dead-Letter Queue System

## Architecture

### Content Extractor Service

The Content Extractor service (`services/content-extractor`) processes raw HTML and PDF content before vectorization:

- Consumes from `ingest.raw_html` and `ingest.raw_pdf` streams
- Uses multiple libraries for robust content extraction:
  - `newspaper3k` for article extraction
  - `trafilatura` for web content cleaning
  - `langdetect` for language detection
  - `pdfminer.six` for PDF processing
- Publishes cleaned content to `ingest.cleaned` stream
- Includes Prometheus metrics for monitoring extraction success/failure

### Concept Linker Service

The Concept Linker service (`services/concept-linker`) performs advanced concept linking:

- Consumes from `concept.new` stream
- Uses spaCy for Named Entity Recognition (NER)
- Implements vector similarity search using Qdrant
- Creates relationship edges between concepts
- Publishes links to `concept.link` stream
- Includes metrics for link creation and processing

### Retry & Dead-Letter Queue System

A unified retry system implemented in `lumina-core`:

- Decorator-based retry mechanism with exponential backoff
- Dead-letter queue support for failed operations
- Configurable retry attempts and delays
- Comprehensive error tracking and logging

## Metrics & Monitoring

### Content Extractor Metrics

- `extraction_success_total`: Successful content extractions
- `extraction_skip_total`: Skipped extractions (by reason)
- `extraction_fail_total`: Failed extractions
- `extraction_seconds`: Time spent on extraction

### Concept Linker Metrics

- `link_created_total`: Links created (by type)
- `link_skip_total`: Skipped links (by reason)
- `link_fail_total`: Failed link creations
- `link_seconds`: Time spent creating links

### Retry & DLQ Metrics

- `retry_attempt_total`: Retry attempts (by operation)
- `retry_fail_total`: Failed retries
- `dlq_messages_total`: Messages in DLQ (by stream)
- `dlq_processing_seconds`: Time spent processing DLQ

## Alerting Rules

### Content Extractor Alerts

- High extraction failure rate (>10%)
- Low extraction success rate (<80%)
- High extraction latency (>5s)

### Concept Linker Alerts

- High link creation failure rate (>10%)
- Low link creation rate (<1/s)
- High processing latency (>10s)

### Retry & DLQ Alerts

- High retry rate (>20% of operations)
- DLQ depth > 1000 messages
- DLQ processing delay > 1h

## Development

### Prerequisites

- Python 3.9+
- Redis 6.0+
- Qdrant 0.7.0+
- Docker & Docker Compose

### Setup

1. Install dependencies:
```bash
# Content Extractor
cd services/content-extractor
pip install -r requirements.txt

# Concept Linker
cd services/concept-linker
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

2. Configure environment variables:
```bash
# Content Extractor
export REDIS_URL=redis://localhost:6379
export PORT=8001

# Concept Linker
export REDIS_URL=redis://localhost:6379
export QDRANT_URL=http://localhost:6333
export PORT=8002
```

3. Start services:
```bash
# Content Extractor
cd services/content-extractor
python main.py

# Concept Linker
cd services/concept-linker
python main.py
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale content-extractor=3 --scale concept-linker=2
```

## Testing

### Unit Tests

```bash
# Content Extractor
cd services/content-extractor
pytest tests/

# Concept Linker
cd services/concept-linker
pytest tests/
```

### Integration Tests

```bash
# Run integration test suite
cd tests/integration
pytest test_phase9.py
```

### Load Testing

```bash
# Run load tests
cd tests/load
locust -f test_phase9.py
```

## Monitoring & Debugging

### Grafana Dashboards

- Content Extractor Dashboard
- Concept Linker Dashboard
- Retry & DLQ Dashboard

### Logging

- Structured logging in logfmt format
- Log levels: INFO (default), DEBUG, WARNING, ERROR
- Log destination: stdout (collected by Promtail)

### Debugging Tools

- Redis CLI for stream inspection
- Qdrant UI for vector search debugging
- Prometheus for metrics analysis

## Future Enhancements

1. Content Extractor:
   - Support for more document types
   - Enhanced language detection
   - Custom extraction rules

2. Concept Linker:
   - Additional NER models
   - Relationship type classification
   - Graph-based similarity search

3. Retry System:
   - Circuit breaker pattern
   - Custom retry strategies
   - DLQ message replay 