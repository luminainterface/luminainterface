# Backup Verification System

## Overview
The backup verification system ensures the integrity and availability of system backups through automated checks and monitoring.

## Components

### 1. Verification Script (`scripts/verify_backup.sh`)
- Verifies tar archive integrity
- Checks MD5 checksums
- Exports Prometheus metrics
- Handles retention policies

### 2. Docker Service (`docker-compose.backup.yml`)
- Runs verification script in container
- Exposes metrics endpoint
- Manages resource limits
- Handles automatic restarts

### 3. Prometheus Alerts (`prometheus/rules/backup_alerts.yml`)
- Monitors verification status
- Tracks backup age
- Alerts on slow verifications

## Setup

1. Build and start the service:
```bash
docker compose -f docker-compose.backup.yml up -d
```

2. Verify metrics endpoint:
```bash
curl http://localhost:9090/metrics
```

3. Check Prometheus targets:
```bash
curl http://localhost:9090/-/healthy
```

## Monitoring

### Grafana Dashboard
- Backup verification status
- Backup size trends
- Verification duration
- Age of backups

### Alerts
- Critical: Failed verifications
- Warning: Old backups (>2 days)
- Warning: Slow verifications (>5 minutes)

## Maintenance

### Adding New Backups
1. Place backup file in `/backups` directory
2. Generate MD5 checksum:
```bash
md5sum backup.tar > backup.tar.md5
```

### Retention Policy
- Backups older than 31 days are automatically excluded
- Adjust `RETENTION_DAYS` in docker-compose file if needed

## Troubleshooting

### Common Issues
1. **Missing MD5 file**
   - Generate checksum file for backup
   - Restart verification service

2. **Invalid tar archive**
   - Verify backup creation process
   - Check for corruption

3. **Slow verification**
   - Check system resources
   - Consider increasing memory limit

### Logs
```bash
docker compose -f docker-compose.backup.yml logs -f
``` 