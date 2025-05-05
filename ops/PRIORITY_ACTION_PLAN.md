# Operations Priority Action Plan (48h)

## P0: Critical Path (Next 4-8 hours)

### System Health Verification
```bash
# Run these commands in order
docker compose pull && docker compose up -d
./scripts/align_ports.sh
./scripts/smoke.sh
```

**Success Criteria:**
- [ ] All services show healthy in `docker compose ps`
- [ ] Smoke tests pass without errors
- [ ] Frontend loads without red boundary
- [ ] Grafana "Graph-Ops" dashboard shows:
  - Non-zero events/sec
  - Redis lag ≈ 0

### Port & Environment Alignment
- Implement `scripts/align_ports.sh` in CI pipeline
- Add post-compose-up verification step

## P1: Security & Monitoring (Next 12-24 hours)

### Monitoring Setup
1. Prometheus Scrape Verification:
   - [ ] masterchat service
   - [ ] event-mux service
   - [ ] crawler service

2. Grafana Panel Implementation:
   - [ ] Redis lag monitoring
   - [ ] Planner p95 tracking

3. Alert Configuration:
   ```yaml
   alerts:
   - RedisLagHigh: >1k for 2m
   - PlannerStuck: p99 > 15s
   - CrawlerErrors: >5% 5xx
   ```

### Secrets Management
1. Remove from Git:
   - [ ] Audit .env files for exposed keys
   - [ ] Move OPENAI_API_KEY to secure storage
2. Implementation:
   - [ ] Configure Docker Secrets
   - [ ] Set up GitHub Encrypted Secrets
   - [ ] Update CI pipeline for secret injection

### Backup Implementation
- [ ] Configure GitHub Action for `freeze_and_push.sh`
- [ ] Schedule for 02:00 UTC daily
- [ ] Set 30-day S3 artifact expiration

## P2: Infrastructure Stability (24-36 hours)

### CI Pipeline Enhancement
1. Stage 1: Fast Checks
   - Linting
   - Unit tests
2. Stage 2: Integration
   - Compose deployment
   - Smoke test execution
3. Stage 3: Frontend
   - UI build verification
   - Cypress happy-path testing
4. Stage 4: Deployment
   - Auto-deploy to staging VPS on main

### Resource Monitoring
- [ ] Set up container resource tracking in Grafana
- [ ] Configure 60% utilization alerts
- [ ] Implement CRAWL_BATCH_SIZE limits

## P3: Security & Scalability (36-48 hours)

### TLS & CORS Implementation
- [ ] Deploy Traefik/Caddy reverse proxy
- [ ] Configure Let's Encrypt automation
- [ ] Lock CORS to *.luminainterface.dev

### Kafka Integration Preparation
- [ ] Create separate compose profile for Kafka
- [ ] Implement basic consumer setup
- [ ] Configure offset monitoring

### Logging Infrastructure
- [ ] Select between Loki/Elasticsearch
- [ ] Implement container log shipping
- [ ] Set up service-based log categorization

## Immediate Action Script

```bash
#!/bin/bash
set -euo pipefail

echo "🚀 Starting system verification..."

# 1. Pull & boot
echo "📥 Pulling latest images and starting services..."
docker compose pull && docker compose up -d

# 2. Ports → UI
echo "🔄 Aligning ports..."
./scripts/align_ports.sh

# 3. Smoke
echo "🔍 Running smoke tests..."
./scripts/smoke.sh || { echo "❌ Smoke tests failed"; exit 1; }

# 4. Dashboards
echo "📊 Opening monitoring dashboard..."
open http://localhost:3000/d/graph-ops

echo "✅ System verification complete!"
```

## Additional Tasks (If Time Permits)
- [ ] Add CI verification badge to backcurrentstate.md
- [ ] Implement weekly DuckDB vacuum job
- [ ] Set up nightly container cleanup cron

## Success Metrics
- All P0 items completed within 8 hours
- P1 items completed within 24 hours
- No exposed secrets in version control
- Automated backup system operational
- Monitoring system providing actionable alerts 