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
COPY meta_orchestration_controller.py .
COPY enhanced_real_world_benchmark.py .

# Expose port
EXPOSE 8999

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=5 \
  CMD curl -f http://localhost:8999/health || exit 1

# Run the application
CMD ["python", "meta_orchestration_controller.py"] 