FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies  
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ai_detection_mitigation.py .

# Create directories
RUN mkdir -p patterns logs

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application
CMD ["python", "-c", "from ai_detection_mitigation import *; import flask; app = flask.Flask(__name__); app.run(host='0.0.0.0', port=5001)"] 