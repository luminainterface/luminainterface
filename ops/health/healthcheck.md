# Lumina Health Check System

## Overview

The Lumina health check system provides advanced service health monitoring across all microservices. It goes beyond simple "200 OK" responses by validating dependencies, data freshness, and critical-path metrics.

## Health Check Components

### 1. Core Health Checks

Each service implements the following health checks:

| Check | Description | Threshold |
|-------|-------------|-----------|
| Redis | Connectivity and ping | 1s timeout |
| Qdrant | Vector store health | 2s timeout |
| Latency | P95 response time | < 0.25s |
| Semantic | Concept drift | < 0.30 |
| Dependencies | Downstream services | 1s timeout |

### 2. Service Dependencies

The system monitors dependencies between services:

```yaml
EXPECTED_SVC:
  graph-api: http://graph-api:8200/health
  event-mux: http://event-mux:8000/health
  masterchat: http://masterchat:8000/health
  crawler: http://crawler:8400/health
```

### 3. Health Check Response

Example health check response:

```json
{
  "healthy": true,
  "version": "git-sha",
  "duration_s": "0.123",
  "checks": {
    "redis": {"ok": true, "detail": ""},
    "qdrant": {"ok": true, "detail": "Status 200"},
    "latency": {"ok": true, "detail": "p95=0.123s"},
    "semantic": {"ok": true, "detail": "drift=0.15"},
    "graph-api": {"ok": true, "detail": "Status 200"},
    "event-mux": {"ok": true, "detail": "Status 200"},
    "masterchat": {"ok": true, "detail": "Status 200"},
    "crawler": {"ok": true, "detail": "Status 200"}
  }
}
```

## Monitoring & Alerting

### 1. Prometheus Metrics

The system exposes the following metrics:

- `service_health`: Gauge (0=bad, 1=good) with `check` label
- `service_health_duration_seconds`: Histogram of check durations

### 2. Alert Rules

```yaml
# Critical Alert (30s)
- alert: ServiceHealthCritical
  expr: service_health == 0
  for: 30s
  labels:
    severity: critical

# Warning Alert (5m)
- alert: ServiceHealthWarning
  expr: service_health == 0
  for: 5m
  labels:
    severity: warning
```

### 3. Grafana Dashboard

The health check dashboard includes:

- Service Health Status Panel
  - Shows current health state of all checks
  - Color-coded (green=healthy, red=unhealthy)
  - Updates every 10s

- Health Check Duration Panel
  - Tracks check execution times
  - Helps identify slow checks
  - Shows trends over time

## Implementation

### 1. Adding to a Service

```python
from ops.health.health import router as health_router

app = FastAPI()
app.include_router(health_router)
```

### 2. Environment Variables

Required environment variables:

```bash
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://qdrant:6333/health
PROM_URL=http://prometheus:9090
GIT_SHA=your-git-sha
```

### 3. Endpoints

- `/health`: Main health check endpoint
- `/health/metrics`: Prometheus metrics endpoint

## Best Practices

1. **Timeout Configuration**
   - Redis: 1s
   - Qdrant: 2s
   - Dependencies: 1s
   - Prometheus queries: 2s

2. **Error Handling**
   - All checks are wrapped in try/except
   - Detailed error messages in response
   - No single check failure blocks others

3. **Performance**
   - Checks run concurrently
   - Timeouts prevent hanging
   - Metrics track check durations

4. **Maintenance**
   - Version tracking via GIT_SHA
   - Configurable thresholds
   - Easy to add new checks

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis service status
   - Verify REDIS_URL format
   - Check network connectivity

2. **Qdrant Health Check Failed**
   - Verify Qdrant service is running
   - Check QDRANT_URL configuration
   - Review Qdrant logs

3. **High Latency**
   - Check service resource usage
   - Review downstream dependencies
   - Monitor network latency

4. **Semantic Drift**
   - Review concept analysis logs
   - Check embedding quality
   - Verify model performance

### Debugging

1. Check service logs:
```bash
docker logs <service-name>
```

2. Test health endpoint:
```bash
curl http://<service>:<port>/health
```

3. View Prometheus metrics:
```bash
curl http://<service>:<port>/health/metrics
```

## Contributing

To add new health checks:

1. Add check function to `ops/health/health.py`
2. Add check to `advanced_health()` function
3. Update documentation
4. Add appropriate metrics
5. Update Grafana dashboard if needed

## Security

- Health endpoints are internal only
- No sensitive data in responses
- Timeouts prevent DoS attacks
- Metrics are rate-limited 