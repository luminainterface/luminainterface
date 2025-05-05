FROM python:3.11-slim
WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt prometheus_client

COPY lumina_core/ /app/lumina_core
ENV PYTHONPATH="/app:${PYTHONPATH}"
RUN pip install --no-cache-dir -e /app/lumina_core

# Copy application code
COPY crawler/ /app/crawler/
WORKDIR /app/crawler

EXPOSE 8400
CMD ["uvicorn", "metrics_server:app", "--host", "0.0.0.0", "--port", "8400"] 