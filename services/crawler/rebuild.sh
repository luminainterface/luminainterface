#!/bin/bash

# Stop and remove existing containers
docker-compose down

# Remove all Python cache files
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type d -name "*.egg-info" -exec rm -r {} +
find . -type d -name "*.egg" -exec rm -r {} +
find . -type d -name ".pytest_cache" -exec rm -r {} +
find . -type d -name ".coverage" -delete
find . -type d -name "htmlcov" -exec rm -r {} +

# Remove Docker images
docker rmi $(docker images -q 'crawler_crawler' 'crawler_concept-dictionary') 2>/dev/null || true

# Rebuild and start services
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 10

# Check service health
docker-compose ps 