# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    build-essential \
    bash \
    git \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy lumina-core from the project root
COPY ../../lumina-core ./lumina-core
RUN pip install -e ./lumina-core

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH="/app:/app/lumina-core" \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_MODEL=all-MiniLM-L6-v2

# Create cache directory for sentence-transformers
RUN mkdir -p /root/.cache/torch/sentence_transformers

# Create startup script
COPY scripts/wait-for-it.sh /app/scripts/wait-for-it.sh
RUN chmod +x /app/scripts/wait-for-it.sh

# Create health check script
RUN echo '#!/bin/bash\n\
curl -f http://localhost:8000/health || exit 1' > /app/healthcheck.sh && \
chmod +x /app/healthcheck.sh

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD /app/healthcheck.sh

# Create startup script
RUN echo '#!/bin/bash\n\
\n\
# Wait for dependencies\n\
/app/scripts/wait-for-it.sh redis:6379 -- echo "Redis is up" || exit 1\n\
/app/scripts/wait-for-it.sh concept-dict:8000 -- echo "Concept Dictionary is up" || exit 1\n\
/app/scripts/wait-for-it.sh graph-api:8829 -- echo "Graph API is up" || exit 1\n\
\n\
# Start the application\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8000\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Run the startup script
CMD ["/app/start.sh"] 