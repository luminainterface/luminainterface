# Lumina Operations

## Backup System

### Quick Start
```bash
# Create and push backup to S3
./freeze_and_push.sh

# Verify a backup locally
./verify_freeze.sh lumina_freeze_*.tar.gz

# Restore from S3 (replace YYYY/MM/DD with actual path)
aws s3 cp s3://lumina-backups/YYYY/MM/DD/lumina_freeze_*.tar.gz . && \
verify_freeze.sh lumina_freeze_*.tar.gz && \
docker compose up
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `SNAP_WAIT` | 5 | Seconds to wait for Qdrant snapshot |
| `MAX_RETRIES` | 3 | Retry attempts for snapshot creation |
| `ARCHIVE_COMPRESSION` | 9 | Gzip compression level (1-9) |
| `AWS_BUCKET` | lumina-backups | S3 bucket name |
| `AWS_REGION` | us-east-1 | AWS region |
| `AWS_PREFIX` | YYYY/MM/DD | S3 path prefix |
| `SLACK_WEBHOOK` | - | Optional webhook for notifications |
| `DEBUG` | - | Enable verbose logging |

### Monitoring Thresholds
- Alert if backup size exceeds 2GB
- Alert if no backup exists for 14 days
- Monitor S3 storage growth monthly

### Tuning Examples

#### Development (Speed)
```bash
# .env
SNAP_WAIT=0
ARCHIVE_COMPRESSION=1
```

#### Production (Reliability)
```bash
# .env
SNAP_WAIT=5
ARCHIVE_COMPRESSION=9
MAX_RETRIES=3
```

#### CI/CD Pipeline
```bash
# .env
SNAP_WAIT=0
ARCHIVE_COMPRESSION=1
DEBUG=true
```

### GitHub Actions

The backup system includes a manual workflow that can be triggered from the GitHub UI:

1. Go to Actions → Run Backup
2. Select environment (staging/production)
3. Monitor workflow progress
4. Download `backup-url` artifact for S3 location

### Verification

Each backup includes:
- SHA-256 checksum
- Snapshot manifest
- Component inventory

Run `verify_freeze.sh` to validate backup integrity.

### Monitoring

- Backup size and S3 URL are logged
- Optional Slack notifications
- GitHub Actions artifacts (30-day retention)
- S3 path: `s3://lumina-backups/YYYY/MM/DD/lumina_freeze_*.tar.gz`