# Backup Verification System

## Overview
The backup verification system ensures the integrity and availability of system backups through automated checks and monitoring.

## Components

### 1. Verification Script
- Located at `scripts/verify_backup.sh`
- Performs integrity checks on backup files
- Generates Prometheus metrics
- Exposes metrics on port 9090

### 2. Docker Service
- Defined in `docker-compose.backup.yml`
- Runs verification script in container
- Mounts backup directory
- Connects to monitoring network

### 3. Prometheus Alerts
- Defined in `prometheus/rules/backup_alerts.yml`
- Monitors verification status
- Alerts on failures and issues
- Tracks backup age and verification duration

## Setup

1. Build and start the service:
```bash
docker compose -f docker-compose.backup.yml up -d
```

2. Verify metrics are being collected:
```bash
curl localhost:9090/metrics
```

3. Check Prometheus targets:
```bash
curl localhost:9090/-/healthy
```

## Monitoring

### Grafana Dashboard
- Access at `http://localhost:3000/d/crawler-logs`
- Shows backup verification status
- Displays backup sizes and ages
- Tracks verification duration

### Alerts
- Backup verification failures
- Old backups (>2 days)
- Slow verifications (>5 minutes)

## Maintenance

### Adding New Backups
1. Place backup files in `/backups` directory
2. Include MD5 checksum file (`.md5`)
3. Verification will run automatically

### Updating Retention
1. Modify `RETENTION_DAYS` in `docker-compose.backup.yml`
2. Restart service:
```bash
docker compose -f docker-compose.backup.yml restart
```

## Troubleshooting

### Common Issues

1. **Verification Fails**
   - Check backup file exists
   - Verify MD5 checksum file present
   - Check file permissions

2. **Metrics Not Available**
   - Verify service is running
   - Check port 9090 is accessible
   - Review container logs

3. **Alerts Not Firing**
   - Check Prometheus configuration
   - Verify alert rules are loaded
   - Review alert manager setup

### Logs
```bash
# View service logs
docker compose -f docker-compose.backup.yml logs -f

# Check metrics endpoint
curl localhost:9090/metrics
``` 