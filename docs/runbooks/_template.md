# Runbook Template

## Service: [Service Name]
**Owner:** [Primary Owner] (@slack-handle)
**Last Updated:** [YYYY-MM-DD]

## Overview
Brief description of the service and its critical dependencies.

## Detection

### Symptoms
- [ ] Primary symptom (e.g., "High error rate in API")
- [ ] Secondary symptoms
- [ ] Related metrics to check

### Monitoring
```yaml
metrics:
  - name: service_errors_total
    type: counter
    threshold: > 1% of requests
    dashboard: /grafana/dashboards/service-health

  - name: service_latency_seconds
    type: histogram
    threshold: p95 > 2s
    dashboard: /grafana/dashboards/service-performance
```

### Alerts
```yaml
alerts:
  - name: HighErrorRate
    condition: rate(service_errors_total[5m]) > 0.01
    severity: critical
    runbook: /runbooks/high-error-rate.md

  - name: HighLatency
    condition: histogram_quantile(0.95, rate(service_latency_seconds_bucket[5m])) > 2
    severity: warning
    runbook: /runbooks/high-latency.md
```

## Impact

### Business Impact
- [ ] User-facing impact
- [ ] Revenue impact
- [ ] Compliance impact

### Technical Impact
- [ ] System dependencies affected
- [ ] Data integrity concerns
- [ ] Performance degradation

## Diagnosis

### Quick Checks
1. [ ] Check service logs
   ```bash
   kubectl logs -n <namespace> deployment/<service> --tail=100
   ```

2. [ ] Verify metrics
   ```bash
   curl -s localhost:9090/metrics | grep service_
   ```

3. [ ] Check dependencies
   ```bash
   curl -s localhost:8080/health
   ```

### Common Issues
1. [ ] Issue 1
   - Symptoms
   - Root cause
   - Verification steps

2. [ ] Issue 2
   - Symptoms
   - Root cause
   - Verification steps

## Mitigation

### Immediate Actions
1. [ ] Action 1
   ```bash
   # Command or steps
   ```

2. [ ] Action 2
   ```bash
   # Command or steps
   ```

### Rollback Plan
1. [ ] Step 1
   ```bash
   # Rollback command
   ```

2. [ ] Step 2
   ```bash
   # Rollback command
   ```

## Prevention

### Long-term Fixes
- [ ] Fix 1
- [ ] Fix 2
- [ ] Fix 3

### Monitoring Improvements
- [ ] Add metric 1
- [ ] Add alert 2
- [ ] Update dashboard 3

### Documentation Updates
- [ ] Update runbook
- [ ] Add troubleshooting guide
- [ ] Update architecture docs

## Communication

### Internal
- [ ] Slack channel: #incidents
- [ ] Email: incidents@lumina.ai
- [ ] PagerDuty: Service Team

### External
- [ ] Status page: status.lumina.ai
- [ ] Customer support: support@lumina.ai
- [ ] Public postmortem: blog.lumina.ai

## References

### Related Documentation
- [Architecture Overview](/docs/architecture.md)
- [Deployment Guide](/docs/deployment.md)
- [Troubleshooting Guide](/docs/troubleshooting.md)

### Tools
- [Grafana Dashboard](https://grafana.lumina.ai/d/service-health)
- [Logs](https://logs.lumina.ai)
- [Metrics](https://metrics.lumina.ai)

## Updates

- YYYY-MM-DD: Initial runbook created
- YYYY-MM-DD: Added new mitigation steps
- YYYY-MM-DD: Updated monitoring section 