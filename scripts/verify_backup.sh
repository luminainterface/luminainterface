#!/bin/bash

# Configuration
BACKUP_DIR="/backups"
RETENTION_DAYS=31
PROMETHEUS_PORT=9090
METRICS_FILE="/tmp/backup_metrics.prom"

# Create metrics file
cat > $METRICS_FILE << EOF
# HELP backup_verification_status Status of backup verification (1=success, 0=failure)
# TYPE backup_verification_status gauge
# HELP backup_size_bytes Size of backup in bytes
# TYPE backup_size_bytes gauge
# HELP backup_age_seconds Age of backup in seconds
# TYPE backup_age_seconds gauge
# HELP backup_verification_duration_seconds Time taken to verify backup
# TYPE backup_verification_duration_seconds gauge
EOF

# Function to verify backup
verify_backup() {
    local backup_file=$1
    local start_time=$(date +%s)
    local status=0
    local error_msg=""

    # Check if file exists
    if [ ! -f "$backup_file" ]; then
        error_msg="Backup file not found"
        status=0
    else
        # Verify tar archive
        if ! tar -tf "$backup_file" > /dev/null 2>&1; then
            error_msg="Invalid tar archive"
            status=0
        else
            # Verify MD5 checksum
            local md5_file="${backup_file}.md5"
            if [ -f "$md5_file" ]; then
                if ! md5sum -c "$md5_file" > /dev/null 2>&1; then
                    error_msg="MD5 checksum mismatch"
                    status=0
                else
                    status=1
                fi
            else
                error_msg="MD5 file not found"
                status=0
            fi
        fi
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local size=$(stat -c %s "$backup_file" 2>/dev/null || echo 0)
    local age=$((end_time - $(stat -c %Y "$backup_file" 2>/dev/null || echo $end_time)))

    # Append metrics
    cat >> $METRICS_FILE << EOF
backup_verification_status{file="$backup_file"} $status
backup_size_bytes{file="$backup_file"} $size
backup_age_seconds{file="$backup_file"} $age
backup_verification_duration_seconds{file="$backup_file"} $duration
EOF

    # Log result
    if [ $status -eq 1 ]; then
        echo "✅ Backup verification successful: $backup_file"
    else
        echo "❌ Backup verification failed: $backup_file - $error_msg"
    fi

    return $status
}

# Main execution
echo "Starting backup verification..."

# Find all backup files
find "$BACKUP_DIR" -type f -name "*.tar" -mtime -$RETENTION_DAYS | while read -r backup_file; do
    verify_backup "$backup_file"
done

# Start Prometheus metrics server
echo "Starting metrics server on port $PROMETHEUS_PORT..."
while true; do
    (echo -e "HTTP/1.1 200 OK\nContent-Type: text/plain\n"; cat $METRICS_FILE) | nc -l -p $PROMETHEUS_PORT
done 