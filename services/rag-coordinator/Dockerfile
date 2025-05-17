FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with verbose output
RUN pip install --no-cache-dir -v -r requirements.txt && \
    pip list | grep telemetry

# Copy the service code and lumina_core
COPY . .

# Install lumina_core in development mode
WORKDIR /app/lumina_core
RUN pip install -e . && \
    pip list | grep telemetry

WORKDIR /app

# Verify Python path and installed packages
RUN python -c "import sys; print(sys.path)" && \
    pip list

# Expose the API port
EXPOSE 8828

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8828"] 