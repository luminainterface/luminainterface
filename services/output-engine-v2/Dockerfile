FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY services/output-engine-v2/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy lumina-core from root
COPY lumina-core /app/lumina-core

# Install lumina-core in development mode
RUN pip install -e /app/lumina-core

# Copy service code
COPY services/output-engine-v2/. /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 9000

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"] 