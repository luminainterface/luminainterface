FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy maintenance script
COPY scripts/maintenance.py /app/
RUN chmod +x /app/maintenance.py

# Create cron job
RUN echo "0 2 * * 1 root /usr/local/bin/python /app/maintenance.py >> /var/log/cron.log 2>&1" > /etc/cron.d/maintenance-cron \
    && chmod 0644 /etc/cron.d/maintenance-cron \
    && touch /var/log/cron.log

# Create entrypoint script
RUN echo '#!/bin/sh\nservice cron start\ntail -f /var/log/cron.log' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"] 