FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create scripts directory and add wait-for-it.sh
RUN mkdir -p /app/scripts && \
    curl -o /app/scripts/wait-for-it.sh https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh && \
    chmod +x /app/scripts/wait-for-it.sh

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/training_data

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PDF_DIR=/app/training_data \
    EMBEDDING_MODEL=nomic-embed-text \
    QDRANT_URL=http://qdrant:6333 \
    REDIS_URL=redis://redis:6379 \
    SLEEP_INTERVAL=60

# Create health check script
RUN echo '#!/bin/bash\n\
curl -f http://localhost:8841/health || exit 1' > /app/healthcheck.sh && \
chmod +x /app/healthcheck.sh

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD /app/healthcheck.sh

# Create startup script
RUN echo '#!/bin/bash\n\
\n\
# Wait for dependencies\n\
/app/scripts/wait-for-it.sh redis:6379 -- echo "Redis is up" || exit 1\n\
/app/scripts/wait-for-it.sh qdrant:6333 -- echo "Qdrant is up" || exit 1\n\
\n\
# Start both the FastAPI health server and the PDF trainer\n\
python /app/main.py & \n\
python /app/pdf_trainer.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port for health checks
EXPOSE 8841

# Run the startup script
CMD ["/app/start.sh"] 