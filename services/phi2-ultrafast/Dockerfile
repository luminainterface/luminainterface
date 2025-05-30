FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
RUN pip install transformers torch accelerate aiohttp uvicorn fastapi

# Copy service files
COPY app.py /app/
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8892

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8892/health || exit 1

# Start service
CMD ["python", "app.py"]
