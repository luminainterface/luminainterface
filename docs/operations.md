## Snapshot and Restore

The Lumina stack supports full snapshot and restore capabilities, allowing you to:
- Capture the entire stack state (containers, volumes, configs)
- Restore on a new host with zero data loss
- Automatically verify stream and adapter state

### Taking a Snapshot

To create a snapshot of the current stack state:

```bash
./scripts/freeze.sh
```

This will:
1. Save all container images
2. Archive all volumes (Redis, Qdrant, adapters, Grafana, Prometheus)
3. Copy configuration files
4. Capture stream state
5. Create a timestamped archive (e.g., `lumina-snapshot-20240315_143301.tar.gz`)

### Restoring from a Snapshot

To restore the stack on a new host:

```bash
./scripts/restore.sh lumina-snapshot-20240315_143301.tar.gz
```

The restore process:
1. Loads all container images
2. Restores all volumes
3. Applies configuration files
4. Starts services
5. Verifies stream and adapter state

### Monitoring and Alerts

The stack includes reliability monitoring:

1. Redis AOF monitoring:
   - Alerts on high fsync latency (>25ms)
   - Tracks AOF rewrite status
   - Monitors memory usage

2. Stream monitoring:
   - Consumer lag alerts (>500 messages)
   - Stream memory usage
   - Consumer group health

View alerts in Grafana or configure notifications in your preferred channel.

### Best Practices

1. Take regular snapshots:
   ```bash
   # Daily snapshot
   0 0 * * * /path/to/lumina/scripts/freeze.sh
   ```

2. Store snapshots securely:
   - Use encrypted storage
   - Keep multiple generations
   - Test restores periodically

3. Monitor reliability metrics:
   - Check Grafana dashboards
   - Review Prometheus alerts
   - Monitor stream lag

4. Before major updates:
   - Take a snapshot
   - Verify restore works
   - Keep snapshot until update is verified

### Troubleshooting

Common issues and solutions:

1. Restore fails with "volume exists":
   ```bash
   docker volume rm redis_data qdrant_data adapters_data grafana_data prometheus_data
   ./scripts/restore.sh snapshot.tar.gz
   ```

2. Stream verification fails:
   - Check Redis logs: `docker compose logs redis`
   - Verify AOF status: `docker compose exec redis redis-cli info persistence`
   - Check consumer groups: `docker compose exec redis redis-cli xinfo groups ingest.crawl`

3. Adapter restore issues:
   - Check trainer logs: `docker compose logs concept-trainer-growable`
   - Verify adapter volume: `docker volume inspect adapters_data`
   - Test adapter endpoint: `curl http://localhost:8710/adapters/test` 