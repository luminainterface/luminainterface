#!/bin/bash

set -euo pipefail

# Load environment defaults
set -a
[ -f .env ] && . .env
set +a

# Configurable parameters with production-safe defaults
: ${SNAP_WAIT:=5}  # Default 5s wait for snapshot
: ${MAX_RETRIES:=3}  # Default 3 retries for snapshot creation
: ${QDRANT_URL:="http://localhost:6333"}  # Qdrant API endpoint
: ${ARCHIVE_COMPRESSION:=9}  # Default to maximum compression for production
: ${AWS_BUCKET:="lumina-backups"}  # S3 bucket name
: ${AWS_REGION:="us-east-1"}  # AWS region
: ${AWS_PREFIX:=$(date +'%Y/%m/%d')}  # Date-based prefix

# Enable verbose logging if DEBUG is set
[ "${DEBUG:-}" ] && set -x

# Create timestamp for the backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="lumina_freeze_${TIMESTAMP}"
LAST_BACKUP_FILE="last_backup_url.txt"

# Cleanup function
cleanup() {
    if [ -d "$BACKUP_DIR" ]; then
        rm -rf "$BACKUP_DIR"
    fi
}
trap cleanup EXIT

echo "Starting Lumina backup process..."

# Run freeze script
echo "Creating system snapshot..."
if ! ./freeze.sh; then
    echo "Freeze process failed"
    exit 1
fi

# Get the freeze archive
ARCHIVE=$(ls lumina_freeze_*.tar.gz)
if [ ! -f "$ARCHIVE" ]; then
    echo "Freeze archive not found"
    exit 1
fi

# Verify the freeze
echo "Verifying backup integrity..."
if ! ./verify_freeze.sh "$ARCHIVE"; then
    echo "Verification failed"
    exit 1
fi

# Upload to S3
echo "Uploading to S3..."
S3_PATH="s3://${AWS_BUCKET}/${AWS_PREFIX}/${ARCHIVE}"
if ! aws s3 cp "$ARCHIVE" "$S3_PATH" --only-show-errors; then
    echo "S3 upload failed"
    exit 1
fi

# Get backup size
BACKUP_SIZE=$(stat -f %z "$ARCHIVE" 2>/dev/null || stat -c %s "$ARCHIVE")

# Write backup URL to file
echo "https://${AWS_BUCKET}.s3.${AWS_REGION}.amazonaws.com/${AWS_PREFIX}/${ARCHIVE}" > "$LAST_BACKUP_FILE"

# Optional Slack notification
if [ -n "${SLACK_WEBHOOK:-}" ]; then
    curl -s -X POST "$SLACK_WEBHOOK" \
        -H 'Content-Type: application/json' \
        -d "{
            \"text\": \"✅ Lumina backup complete\",
            \"attachments\": [{
                \"fields\": [
                    {
                        \"title\": \"Size\",
                        \"value\": \"$(numfmt --to=iec-i --suffix=B $BACKUP_SIZE)\",
                        \"short\": true
                    },
                    {
                        \"title\": \"Location\",
                        \"value\": \"<https://${AWS_BUCKET}.s3.${AWS_REGION}.amazonaws.com/${AWS_PREFIX}/${ARCHIVE}|View in S3>\",
                        \"short\": true
                    }
                ]
            }]
        }"
fi

echo "Backup complete: ${S3_PATH}"
echo "URL saved to: ${LAST_BACKUP_FILE}"
exit 0 