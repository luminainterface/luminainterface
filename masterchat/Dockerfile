FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Set PYTHONPATH early
ENV PYTHONPATH="/app"

# Install dependencies in the correct order with specific versions
RUN pip install --no-cache-dir \
    "fastapi==0.110.0" \
    "uvicorn[standard]==0.29.0" \
    "redis==5.0.4" \
    "duckdb==0.10.3" \
    "sse-starlette==2.0.0" \
    "apscheduler==3.10.4" \
    "mistralai==0.1.8" \
    "jinja2>=3.1.0" \
    "prometheus-fastapi-instrumentator==6.1.0"

# Copy application code
COPY . .
COPY ../ops /app/ops
COPY ./lumina_core /app/lumina_core
RUN ls -l /app/lumina_core && ls -l /app/lumina_core/common

RUN rm -rf /app/lumina-core || true
ENV PYTHONPATH=/app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["env", "PYTHONPATH=/app", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 