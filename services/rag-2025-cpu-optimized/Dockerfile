FROM python:3.11-slim

# Build argument for main file
ARG MAIN_FILE=rag_2025_cpu_optimized.py
ARG PORT=8902

# Set environment variables for better Python performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV MAIN_FILE=${MAIN_FILE}
ENV PORT=${PORT}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache /app/learning_cycles /app/crawled_content /app/optimization_cache

# Expose port (default 8902, but configurable)
EXPOSE ${PORT}

# Health check with dynamic port
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with the specified main file
CMD python ${MAIN_FILE} 