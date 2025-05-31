FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY enhanced_fact_checking_layer.py .
COPY enhanced_fact_checking_service.py .
COPY fact_checking.db ./fact_checking.db

# Expose port
EXPOSE 8885

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8885/health || exit 1

# Run the service
CMD ["python", "enhanced_fact_checking_service.py"] 