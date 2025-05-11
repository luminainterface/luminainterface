FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for model persistence
RUN mkdir -p /app/data && chmod 777 /app/data

# Expose port
EXPOSE 8905

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8905"] 