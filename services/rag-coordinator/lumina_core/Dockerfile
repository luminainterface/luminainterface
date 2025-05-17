FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package code and setup files
COPY setup.py .
COPY lumina_core/ /app/lumina_core/

# Install the package in development mode
RUN pip install -e .

# Expose the API port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "lumina_core.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 