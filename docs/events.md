# Lumina Event Bus Specification (Phase 9)

## Streams & Schemas

| Stream            | Example Payload                                                      | QoS                        |
|-------------------|---------------------------------------------------------------------|----------------------------|
| system.topics     | {"source":"gap-detector", "topic":"Graph_isomorphism"}             | PUB (fire-and-forget)      |
| ingest.queue      | {"type":"url", "payload":"https://…", "score":0.87, "fp":"hash…"}   | Persistent, ack by worker  |
| ingest.raw_html   | {"url":"…", "html":"…", "fp":"hash…", "ts":...}                   | Stream                     |
| ingest.raw_pdf    | {"file_id":"…", "vec_id":null, "fp":"hash…", "ts":...}             | Stream                     |
| concept.prune     | {"cid":"…", "reason":"stale"}                                      | Optional                   |

## Consumer Groups
- ingest.queue: group "workers" (Crawler, PDF-Trainer, etc.)
- ingest.raw_html: group "embed"
- ingest.raw_pdf: group "embed"

## Payload Schemas

### system.topics
```json
{
  "source": "gap-detector",
  "topic": "Graph_isomorphism"
}
```

### ingest.queue
```json
{
  "type": "url" | "pdf" | "system" | "git",
  "payload": "https://..." | "/tmp/abc.pdf" | ...,
  "score": 0.87,
  "fp": "hash..."
}
```

### ingest.raw_html
```json
{
  "url": "...",
  "html": "...",
  "fp": "hash...",
  "ts": 1710000000
}
```

### ingest.raw_pdf
```json
{
  "file_id": "...",
  "vec_id": null,
  "fp": "hash...",
  "ts": 1710000000
}
```

### concept.prune
```json
{
  "cid": "...",
  "reason": "stale"
}
```

## Quality of Service
- ingest.queue: At-least-once, persistent, consumer-group ack
- Deduplication: Ingest-Gateway uses Redis SETNX on fp
- Lag monitoring: redis_stream_group_pending_messages{stream="ingest.queue"}

## Metrics
- ingest_rank_score (histogram, Ingest-Gateway)
- ingest_queue_lag (Redis exporter)
- crawler_fetch_seconds (Crawler)

## Alerts
- StreamConsumerLag: redis_stream_group_pending_messages{stream="ingest.queue"} > 1000 for 5m

## Migration
- Use scripts/migrate_old_ingest.py to move old ingest.pdf/ingest.crawl to ingest.queue 