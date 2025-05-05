FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all necessary files first
COPY requirements.txt ./
COPY lumina_core/ /app/lumina_core
COPY masterchat/ /app/masterchat/

# Set PYTHONPATH early
ENV PYTHONPATH="/app:/app/masterchat:${PYTHONPATH}"

# Install dependencies in the correct order with specific versions
RUN pip install --no-cache-dir \
    "fastapi==0.110.0" \
    "uvicorn[standard]==0.29.0" \
    "redis==5.0.4" \
    "duckdb==0.10.3" \
    "sse-starlette==2.0.0" \
    "apscheduler==3.10.4" \
    "mistralai==0.1.8" \
    "jinja2>=3.1.0" && \
    pip install --no-cache-dir -e /app/lumina_core

WORKDIR /app/masterchat

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 