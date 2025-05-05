FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY lumina_core/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY lumina_core/ /app/lumina_core/
COPY graph_api/main.py /app/main.py

# Set Python path
ENV PYTHONPATH=/app

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8200"] 