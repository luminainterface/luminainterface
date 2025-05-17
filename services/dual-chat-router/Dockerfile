FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY services/dual-chat-router/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy lumina-core
COPY lumina-core /app/lumina-core

# Install lumina-core in development mode
RUN pip install -e /app/lumina-core

# Copy only the dual-chat-router service code
COPY services/dual-chat-router /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8300

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8300"] 