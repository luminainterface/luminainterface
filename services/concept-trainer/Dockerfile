FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch first
RUN pip install torch>=2.1.0

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy lumina_core (as lumina_core) into /app/lumina_core
COPY lumina_core /app/lumina_core
RUN pip install -e /app/lumina_core

# Copy output-engine package
COPY output-engine /app/output-engine
RUN pip install -e /app/output-engine

# Copy application code
COPY services/concept-trainer /app/services/concept-trainer
WORKDIR /app/services/concept-trainer

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:${PYTHONPATH}"

# Run the service
EXPOSE 8813
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8813"] 