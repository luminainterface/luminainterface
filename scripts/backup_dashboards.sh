#!/bin/bash

# Configuration
GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASSWORD="lumina"
BACKUP_DIR="./backups/grafana"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Get all dashboards
echo "Fetching dashboard list..."
DASHBOARDS=$(curl -s -u "$GRAFANA_USER:$GRAFANA_PASSWORD" "$GRAFANA_URL/api/search?type=dash-db" | jq -r '.[].uid')

# Backup each dashboard
for uid in $DASHBOARDS; do
    echo "Backing up dashboard $uid..."
    curl -s -u "$GRAFANA_USER:$GRAFANA_PASSWORD" "$GRAFANA_URL/api/dashboards/uid/$uid" > "$BACKUP_DIR/dashboard_${uid}_${DATE}.json"
done

# Create a tar archive
echo "Creating backup archive..."
tar -czf "$BACKUP_DIR/grafana_dashboards_${DATE}.tar.gz" -C "$BACKUP_DIR" .

# Clean up individual files
rm "$BACKUP_DIR"/dashboard_*_"$DATE".json

echo "Backup completed: $BACKUP_DIR/grafana_dashboards_${DATE}.tar.gz" 