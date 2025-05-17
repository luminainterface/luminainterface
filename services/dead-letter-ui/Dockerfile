FROM python:3.11-slim

WORKDIR /app

# Install git, curl and then install dependencies
RUN apt-get update && apt-get install -y git curl

# Copy requirements first and install them
COPY services/dead-letter-ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy lumina-core and install it
COPY lumina_core /app/lumina_core
RUN pip install -e /app/lumina_core

# Copy dead-letter-ui service code
COPY services/dead-letter-ui /app

# Expose port
EXPOSE 8602

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8602"] 