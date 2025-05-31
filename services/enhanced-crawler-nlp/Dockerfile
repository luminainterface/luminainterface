FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Additional dependencies for enhanced crawler
RUN pip install --no-cache-dir \
    aiohttp \
    asyncio \
    uvicorn \
    fastapi \
    pydantic \
    redis \
    requests \
    feedparser \
    beautifulsoup4 \
    lxml \
    python-multipart

# Copy application code
COPY enhanced_crawler_nlp_v2_intelligent.py .

# Create directories for data and logs
RUN mkdir -p /app/logs /app/arxiv_data /app/filtered_content

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8850

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8850/health || exit 1

# Expose port
EXPOSE 8850

# Run the application
CMD ["python", "enhanced_crawler_nlp_v2_intelligent.py"]
