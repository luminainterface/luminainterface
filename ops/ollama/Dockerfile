FROM ollama/ollama:latest

USER root

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy init script
COPY init.sh /init.sh
RUN chmod +x /init.sh

# Start Ollama and run init script
ENTRYPOINT ["/bin/bash", "-c", "/bin/ollama serve & sleep 5 && /init.sh && wait"] 