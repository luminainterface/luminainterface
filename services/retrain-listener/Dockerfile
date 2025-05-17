FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    build-essential \
    bash \
    git \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY services/retrain-listener/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy lumina-core (the whole package) and install it in editable mode
COPY lumina-core /app/lumina-core
RUN pip install -e /app/lumina-core

# Copy retrain-listener service code (from services/retrain-listener) into /app
COPY services/retrain-listener /app

# Set environment variables
ENV PYTHONPATH="/app:/app/lumina-core" \
    PYTHONUNBUFFERED=1 \
    EMBED_MODEL=nomic-embed-text \
    BATCH_SIZE=64

# Create health check script
RUN echo '#!/bin/bash\n\
curl -f http://localhost:8680/health || exit 1' > /app/healthcheck.sh && \
chmod +x /app/healthcheck.sh

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD /app/healthcheck.sh

# Create startup script
RUN echo '#!/bin/bash\n\
\n\
# Wait for dependencies\n\
/app/scripts/wait-for-it.sh redis:6379 -- echo "Redis is up" || exit 1\n\
/app/scripts/wait-for-it.sh batch-embedder:8709 -- echo "Batch Embedder is up" || exit 1\n\
\n\
# Start the application\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8680\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 8680

# Run the startup script
CMD ["/app/start.sh"] 