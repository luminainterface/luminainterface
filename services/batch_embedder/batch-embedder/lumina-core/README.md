# Lumina Core

Neural network-based question answering system using Wikipedia knowledge.

## Monitoring

### Alert Rules

The following alert rules are configured to monitor system health:

#### WikiQA Pipeline

- `WikiQA_High_Latency`: Triggered when p95 latency exceeds 5s for any agent
  - Threshold: 5s
  - Window: 10m
  - Severity: warning

- `WikiQA_High_Error_Rate`: Triggered when error rate exceeds 10% for any agent
  - Threshold: 10%
  - Window: 5m
  - Severity: warning

#### Neural Network Growth

- `NN_Node_Growth_Stalled`: Triggered when no new nodes are added for 24h
  - Threshold: 24h
  - Window: 15m
  - Severity: warning

### Rate Limits

Each agent has rate limits to prevent abuse:

- `CrawlAgent`: 5 requests per minute per topic
- `SummariseAgent`: 10 requests per minute per article count
- `QAAgent`: 5 requests per minute per question

Rate limit errors return:
```json
{
  "status": "error",
  "type": "rate_limit",
  "detail": "Rate limit exceeded for {key}"
}
```

### Metrics

Key metrics exposed via Prometheus:

- `crawl_requests_total`: Total crawl requests by status
- `summarise_requests_total`: Total summarization requests by status
- `qa_requests_total`: Total QA requests by status
- `synapse_nodes_total`: Total number of neural network nodes
- `synapse_nodes_total_last_update`: Timestamp of last node addition

## Development

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Redis:
```bash
docker run -d -p 6379:6379 redis
```

3. Run tests:
```bash
pytest
```

### Architecture

The system consists of three main agents:

1. `CrawlAgent`: Crawls Wikipedia articles for a given topic
2. `SummariseAgent`: Summarizes articles using Mistral
3. `QAAgent`: Answers questions using article knowledge

Each agent is rate-limited and monitored via Prometheus metrics. 