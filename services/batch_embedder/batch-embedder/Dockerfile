FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/* \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements first for better caching
COPY services/batch_embedder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy lumina_core (the whole package) and install it
COPY lumina_core /app/lumina_core
WORKDIR /app/lumina_core
RUN pip install -e .
WORKDIR /app

# Copy batch-embedder service code (from services/batch_embedder) into /app
COPY services/batch_embedder /app

# Copy wait-for-it.sh script
COPY scripts/wait-for-it.sh /app/scripts/wait-for-it.sh
RUN chmod +x /app/scripts/wait-for-it.sh

# Create startup script
RUN echo '#!/bin/bash\n\
\n\
# Wait for dependencies\n\
/app/scripts/wait-for-it.sh redis:6379 -- echo "Redis is up" || exit 1\n\
/app/scripts/wait-for-it.sh concept-trainer-growable:8710 -- echo "Concept Trainer Growable is up" || exit 1\n\
\n\
# Start the application\n\
exec python main.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose metrics port
EXPOSE 8709

# Run the startup script
CMD ["/app/start.sh"] 