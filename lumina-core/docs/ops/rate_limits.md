# Rate Limit Monitoring

## Dashboard Setup

1. Import the dashboard:
   ```bash
   # In Grafana UI:
   Dashboards -> Import -> Upload JSON file
   # Select: lumina-core/grafana_dashboards/rate_limit.json
   ```

2. Configure Prometheus datasource:
   - Name: `prometheus`
   - URL: `http://prometheus:9090` (adjust if different)

## Dashboard Panels

1. **Requests vs Blocks (per key)**
   - Shows rate of successful requests vs rate-limited blocks
   - Helps identify bursty clients
   - Legend shows mean and max rates

2. **Token Bucket Level (Chat)**
   - Gauge showing available tokens per key
   - Red: < 3 tokens (near limit)
   - Yellow: 3-7 tokens
   - Green: > 7 tokens

3. **Block Ratio (per key)**
   - Percentage of requests that hit rate limits
   - Alert threshold: > 25% for 5 minutes
   - Helps tune limits for each key

4. **Chat Requests Rate (per key)**
   - Raw request rate per API key
   - Useful for capacity planning
   - Compare against tier limits

## Alert Rules

Add these to your Prometheus rules:

```yaml
groups:
  - name: lumina-rate-limits
    rules:
      - alert: HighBlockRatio
        expr: |
          sum(rate(rate_limit_hits_total{result="block"}[5m])) by (api_key) /
          sum(rate(rate_limit_hits_total[5m])) by (api_key) > 0.25
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate limit block ratio for {{ $labels.api_key }}"
          description: "{{ $value | humanizePercentage }} of requests blocked in last 5m"

      - alert: BurstyClient
        expr: |
          sum(rate(rate_limit_hits_total{result="block"}[1m])) by (api_key) > 5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Bursty client detected: {{ $labels.api_key }}"
          description: "{{ $value }} requests blocked in last minute"
```

## Tuning Limits

1. Monitor block ratios:
   - < 5%: Consider increasing limits
   - > 25%: Consider decreasing limits
   - Sudden spikes: Check for client issues

2. Watch token bucket levels:
   - Consistently low: Increase bucket size
   - Always full: Decrease bucket size

3. Review request patterns:
   - Bursty clients: Consider burst allowances
   - Steady high usage: Consider tier upgrades 