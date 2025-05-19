FROM ollama/ollama:latest
USER root
RUN apt-get update && apt-get install -y wget
# Removed USER 1000 so that ollama runs as root (and can write to /.ollama) 