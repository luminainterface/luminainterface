FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including scancode-toolkit
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install scancode-toolkit
RUN pip install --no-cache-dir scancode-toolkit

# Copy requirements first and install them
COPY services/ingest_gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy lumina-core and install it
COPY lumina_core /app/lumina_core
RUN pip install -e /app/lumina_core

# Copy ingest-gateway service code
COPY services/ingest_gateway /app

# Expose port
EXPOSE 8601

ENV PYTHONPATH=/app

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8601"] 