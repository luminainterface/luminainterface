FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install CLI script
COPY scripts/masterchat_cli.py /usr/local/bin/masterchat
RUN chmod +x /usr/local/bin/masterchat

# Environment variables
ENV PYTHONPATH=/app
ENV LLM_TEMP=0.3
ENV GOAL_FILE=/app/goal_lattice.yml

# Expose port
EXPOSE 8000

# Set up entrypoint
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 