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
COPY enhanced_research_backend.py .
COPY ai_failsafe_trap_system.py .
COPY publication_excellence_demo.py .
COPY publication_excellence_v2.py .
COPY ai_detection_mitigation.py .

# Create directories
RUN mkdir -p templates output logs

# Copy frontend template
COPY research_paper_frontend.html templates/

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "enhanced_research_backend.py"] 