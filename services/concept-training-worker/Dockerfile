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

# Additional dependencies for concept training
RUN pip install --no-cache-dir \
    aiohttp \
    asyncio \
    uvicorn \
    fastapi \
    pydantic \
    redis \
    requests \
    torch \
    transformers \
    datasets \
    accelerate \
    peft \
    python-multipart

# Copy application code
COPY concept_training_worker_v2_enhanced.py .

# Create directories for data and logs
RUN mkdir -p /app/logs /app/training_data /app/lora_models /app/assessments

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8851

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8851/health || exit 1

# Expose port
EXPOSE 8851

# Run the application
CMD ["python", "concept_training_worker_v2_enhanced.py"]
