FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY enhanced_fact_checker_with_web_search.py .

# Create directories
RUN mkdir -p logs cache patterns

# Expose port
EXPOSE 8886

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8886/health || exit 1

# Run the application
CMD ["python", "enhanced_fact_checker_with_web_search.py"] 