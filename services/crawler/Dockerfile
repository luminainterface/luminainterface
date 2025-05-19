# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set up working directory
WORKDIR /app

# Install system dependencies including libgomp1 for scancode-toolkit
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgomp1 \
    libarchive-dev \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# Install scancode-toolkit and verify installation
RUN pip install --no-cache-dir scancode-toolkit[full] && \
    scancode --version

# Copy requirements first to leverage Docker cache
COPY services/crawler/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy setup.py and install the package in development mode
COPY lumina_core/setup.py .
COPY lumina_core/ /app/lumina_core/
RUN pip install -e .

# Copy the crawler service code
COPY services/crawler/app /app/services/crawler/app/
COPY services/crawler/handlers /app/services/crawler/handlers/
COPY services/crawler/main.py /app/services/crawler/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8400/health || exit 1

# Run the application
WORKDIR /app/services/crawler
CMD ["python", "main.py"] 