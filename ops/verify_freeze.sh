#!/bin/bash

set -euo pipefail

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <freeze-archive.tar.gz>"
    exit 1
fi

ARCHIVE="$1"
CHECKSUM="${ARCHIVE}.sha256"

# Verify checksum file exists
if [ ! -f "$CHECKSUM" ]; then
    echo "Error: Checksum file not found: $CHECKSUM"
    exit 1
fi

# Verify checksum
echo "Verifying checksum..."
if ! sha256sum -c "$CHECKSUM"; then
    echo "Error: Checksum verification failed"
    exit 1
fi

# Create temporary directory
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Extract archive
echo "Extracting archive..."
if ! tar xzf "$ARCHIVE" -C "$TEMP_DIR"; then
    echo "Error: Archive extraction failed"
    exit 1
fi

# Find the backup directory
BACKUP_DIR=$(find "$TEMP_DIR" -maxdepth 1 -type d -name "lumina_freeze_*" | head -n 1)
if [ -z "$BACKUP_DIR" ]; then
    echo "Error: Could not find backup directory in archive"
    exit 1
fi

# Verify snapshot JSON
if [ ! -f "$BACKUP_DIR/snapshot.json" ]; then
    echo "Error: Snapshot JSON not found"
    exit 1
fi

# Verify snapshot name in JSON
if ! jq -e '.name' "$BACKUP_DIR/snapshot.json" > /dev/null; then
    echo "Error: Invalid snapshot JSON format"
    exit 1
fi

# Verify Qdrant data
if [ ! -f "$BACKUP_DIR/qdrant_data.tar.gz" ]; then
    echo "Error: Qdrant data not found"
    exit 1
fi

# Verify UI build
if [ ! -d "$BACKUP_DIR/ui_dist" ]; then
    echo "Error: UI build not found"
    exit 1
fi

echo "Verification successful: $ARCHIVE"
exit 0 