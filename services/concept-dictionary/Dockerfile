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
    EMBEDDING_DIM=384 \
    EMBEDDING_MULTI_MODULE=false

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
/app/scripts/wait-for-it.sh neo4j:7687 -- echo "Neo4j is up" || exit 1\n\
\n\
# Start the application\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8000\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Run the startup script
CMD ["/app/start.sh"] 