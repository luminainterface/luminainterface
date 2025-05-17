FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    redis-tools \
    build-essential \
    bash \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create scripts directory
RUN mkdir -p /app/scripts

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy lumina-core (as lumina_core) into /app/lumina_core (for consistency with import lumina_core.common)
COPY lumina-core /app/lumina_core

# Install lumina-core in development mode
RUN pip install -e /app/lumina_core

# Copy concept-dictionary service code
COPY . /app/

# Set environment variables
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_DIM=384 \
    EMBEDDING_MULTI_MODULE=false \
    PORT=8828

# Create health check script
RUN echo '#!/bin/bash\n\
curl -f http://localhost:8828/health || exit 1' > /app/healthcheck.sh && \
chmod +x /app/healthcheck.sh

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD /app/healthcheck.sh

# Expose port
EXPOSE 8828

# Run the application directly with uvicorn
CMD ["uvicorn", "concept_dictionary.main:app", "--host", "0.0.0.0", "--port", "8828"] 