# Lumina Project Roadmap

## Risk Assessment & Mitigation Plan

### High-Priority Risks

#### 1. Unbounded Data Growth
**Why it matters:** Crawls + vectors will hit GBs → TBs in weeks; DuckDB & Qdrant on a single volume will eventually choke.

**Mitigations:**
- Set size cap on Qdrant collections (`max_vectors_per_shard`)
- Implement nightly prune job for stale clusters
- S3/MinIO off-load for > 30-day snapshots

#### 2. Cold-start Latency
**Why it matters:** First LLM request after container restart can take 10–30s → planner appears "hung".

**Mitigations:**
- Warm-up task in MasterChat startup
- Add `/ping` endpoint for health checks
- Implement readiness probe in Kubernetes

#### 3. Rate-limit Evasion & Abuse
**Why it matters:** IP-based buckets break under NAT or VPN exit nodes; attackers can rotate IPs.

**Mitigations:**
- Switch limiter to API key + IP pair
- Add Cloudflare or Traefik rate-limit at edge
- Implement request fingerprinting

#### 4. Token / Cost Explosions
**Why it matters:** Large files (e.g., PDFs > 5MB) can trigger expensive embedding calls.

**Mitigations:**
- Middleware checks Content-Length header
- Drop requests > 1MB by default
- Extend budget guard to crawler

#### 5. Vector Duplication / Drift
**Why it matters:** Same page fetched by different URLs → duplicate nodes; embeddings model upgrades create incompatible vectors.

**Mitigations:**
- Implement canonical URL normalizer
- Add vector version field
- Create re-embed job for version mismatches

### Immediate Mitigations (≤ 1 hour each)

1. **Disk Sentinel**
```yaml
# Prometheus Rule
- alert: LowDiskSpace
  expr: node_filesystem_avail_bytes{mountpoint="/var/lib/docker"} < 10GB
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Low disk space on {{ $labels.mountpoint }}"
```

2. **Canonical URL Utility**
```python
def normalize_url(url: str) -> str:
    """Normalize URL to canonical form."""
    parsed = urllib.parse.urlparse(url)
    # Sort query parameters
    query = urllib.parse.parse_qs(parsed.query)
    sorted_query = urllib.parse.urlencode(sorted(query.items()))
    return urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        sorted_query,
        parsed.fragment
    ))
```

3. **Warm-up Endpoint**
```python
@app.get("/ping")
async def ping():
    """Warm-up endpoint for cold start mitigation."""
    try:
        # Warm up LLM
        await llm.generate("ping")
        return {"status": "ok", "latency_ms": latency}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

4. **Force Layout Node Limit**
```typescript
const MAX_NODES = 3000;

function renderGraph(nodes: Node[]) {
  if (nodes.length > MAX_NODES) {
    showMessage("Zoom in to load more nodes");
    return renderSubset(nodes.slice(0, MAX_NODES));
  }
  return renderFullGraph(nodes);
}
```

### Sprint Planning

#### Sprint 1
- [ ] Migrate Qdrant to external volume
- [ ] Implement S3 snapshot plugin
- [ ] Add disk space monitoring
- [ ] Deploy URL normalizer

#### Sprint 2
- [ ] WebGL renderer for SubgraphView
- [ ] Accessibility color palette pass
- [ ] Implement node limit controls
- [ ] Add warm-up endpoints

#### Sprint 3
- [ ] Blue/Green deployment setup
- [ ] Scripted restore procedures
- [ ] Chaos testing framework
- [ ] Rate limit improvements

### Testing & Validation

#### Load Testing
```bash
# Locust configuration
locust -f load_test.py --users 100 --spawn-rate 10
```

#### Chaos Testing
```bash
# Chaos test scenarios
- Kill graph-api container
- Revoke API key
- Simulate Redis failure
- Trigger disk space alert
```

#### Game Day Scenarios
1. API key revocation
2. Container crash recovery
3. Rate limit breach
4. Disk space emergency

### Monitoring & Alerts

#### Critical Alerts
- Disk space < 10GB
- Redis lag > 5s
- API error rate > 1%
- Cold start latency > 30s

#### Warning Alerts
- Rate limit usage > 80%
- Vector count approaching limit
- Backup age > 24h
- Memory usage > 80%

### Documentation

#### Required Documentation
- [ ] Disaster recovery runbook
- [ ] Rate limit configuration guide
- [ ] Monitoring dashboard setup
- [ ] Chaos testing procedures

### Security

#### Key Management
- Monthly key rotation
- Docker Swarm/K8s Secrets
- No secrets in env files
- API key audit trail

#### Access Control
- Role-based access
- API key tiering
- IP allowlisting
- Request signing

### Performance

#### Optimization Targets
- Cold start < 5s
- Graph render < 1s for 3k nodes
- API latency p95 < 200ms
- Backup restore < 15min

### Compliance

#### Requirements
- Respect robots.txt
- Rate limit compliance
- Data retention policies
- Privacy considerations

## Next Steps

1. Review and prioritize immediate mitigations
2. Schedule sprint planning
3. Set up monitoring dashboards
4. Begin load testing
5. Document runbooks

## Updates

- 2024-03-XX: Initial roadmap created
- 2024-03-XX: Added immediate mitigations
- 2024-03-XX: Defined sprint structure 