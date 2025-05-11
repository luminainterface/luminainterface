# Lumina Event Bus Specification

This document defines the standardized event streams used across Lumina services. All events use Redis Streams for persistence, consumer groups, and at-least-once delivery.

## Stream Overview

| Stream | Producer | Consumers | Payload Schema | QoS |
|--------|----------|-----------|----------------|-----|
| `ingest.pdf` | pdf-trainer | concept-dictionary, batch-embedder | `{id: str, file_path: str, vec_id: str, ts: str}` | At-least-once, maxlen=1000 |
| `ingest.crawl` | crawler | concept-dictionary, batch-embedder | `{url: str, title: str, vec_id: str, ts: str}` | At-least-once, maxlen=1000 |
| `concept.new` | concept-dictionary | output-engine, concept-analyzer | `{cid: str, embedding_id: str, meta: dict, ts: str}` | At-least-once, maxlen=5000 |
| `output.generated` | output-engine | retrain-listener, log-embedding-worker | `{turn_id: str, concepts_used: list[str], text: str, confidence: float, ts: str}` | At-least-once, maxlen=10000 |
| `model.adapter.updated` | concept-trainer-growable | all inference services | `{adapter_id: str, ts: str}` | At-least-once, maxlen=100 |

## Consumer Groups

Each stream has one or more consumer groups for different service roles:

| Stream | Consumer Group | Purpose |
|--------|---------------|----------|
| `ingest.pdf` | `concept-dict` | Concept dictionary updates |
| `ingest.pdf` | `batch-embed` | Batch embedding generation |
| `ingest.crawl` | `concept-dict` | Concept dictionary updates |
| `ingest.crawl` | `batch-embed` | Batch embedding generation |
| `concept.new` | `output-gen` | Output generation |
| `concept.new` | `drift-monitor` | Concept drift analysis |
| `output.generated` | `retrain` | Retraining trigger |
| `output.generated` | `log-embed` | Log embedding updates |
| `model.adapter.updated` | `inference` | Model hot-swap |

## Payload Schemas

### ingest.pdf
```python
{
    "id": str,  # Unique PDF ID
    "file_path": str,  # Path to PDF file
    "vec_id": str,  # Vector store ID
    "ts": str  # ISO timestamp
}
```

### ingest.crawl
```python
{
    "url": str,  # Source URL
    "title": str,  # Page title
    "vec_id": str,  # Vector store ID
    "ts": str  # ISO timestamp
}
```

### concept.new
```python
{
    "cid": str,  # Concept ID
    "embedding_id": str,  # Vector store ID
    "meta": {
        "name": str,
        "description": str,
        "confidence": float,
        "source": str,
        "created_at": str
    },
    "ts": str  # ISO timestamp
}
```

### output.generated
```python
{
    "turn_id": str,  # Conversation turn ID
    "concepts_used": list[str],  # List of concept IDs
    "text": str,  # Generated text
    "confidence": float,  # Generation confidence
    "ts": str  # ISO timestamp
}
```

### model.adapter.updated
```python
{
    "adapter_id": str,  # Adapter model ID
    "ts": str  # ISO timestamp
}
```

## Quality of Service

### Delivery Guarantees
- All streams use Redis Streams for at-least-once delivery
- Messages are acknowledged after successful processing
- Unacknowledged messages are redelivered on consumer restart

### Retention
- Each stream has a maximum length (maxlen) to prevent unbounded growth
- Oldest messages are trimmed when maxlen is reached
- Stream lengths are monitored via Prometheus metrics

### Monitoring
- Consumer lag is exposed via `redis_stream_consumer_pending_count` metric
- Stream lengths are exposed via `redis_stream_length` metric
- Processing latency is exposed via service-specific histograms

## Error Handling

### Producer Errors
- Failed publishes are retried with exponential backoff
- After max retries, errors are logged and raised
- Producers should implement dead-letter queues for critical failures

### Consumer Errors
- Message parsing errors are logged and acknowledged
- Handler errors are logged but not acknowledged (will retry)
- Consumer crashes trigger message redelivery
- Long-running handlers should implement timeouts

## Best Practices

1. Always use the `BusClient` helper from `lumina_core.common.bus`
2. Implement proper error handling and logging
3. Monitor consumer lag and stream lengths
4. Use appropriate consumer group names
5. Set reasonable maxlen values
6. Implement proper cleanup in service shutdown 