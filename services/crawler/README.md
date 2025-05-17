# Lumina Crawler Service

A robust service for crawling and processing training data files, generating embeddings, and storing them in a vector database. The service supports multiple file types (JSON, JSONL, PDF, text) and provides a REST API for file processing and monitoring.

## Features

- **Multi-format Support**: Process JSON, JSONL, PDF, and text files
- **Embedding Generation**: Generate embeddings using Ollama models
- **Vector Storage**: Store embeddings in Qdrant vector database
- **Queue Management**: Process files asynchronously using Redis streams
- **Health Monitoring**: Comprehensive health checks and metrics
- **REST API**: Easy-to-use HTTP endpoints for file processing and monitoring
- **Incremental Processing**: Automatically process new files in the training data directory
- **Error Handling**: Robust error handling with dead letter queue
- **Metrics**: Prometheus metrics for monitoring

## Architecture

The service consists of several key components:

1. **File Processor**: Handles different file types and generates embeddings
2. **Crawler**: Manages worker coordination and file processing queue
3. **Vector Store**: Stores and retrieves embeddings
4. **Redis Client**: Manages message queues and caching
5. **FastAPI Application**: Provides REST API endpoints

## Prerequisites

- Docker and Docker Compose
- Redis (for message queues)
- Qdrant (for vector storage)
- Ollama (for embedding generation)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| REDIS_URL | Redis connection URL | redis://:02211998@redis:6379 |
| QDRANT_URL | Qdrant connection URL | http://qdrant:6333 |
| OLLAMA_URL | Ollama service URL | http://ollama:11434 |
| OLLAMA_MODEL | Ollama model name | all-MiniLM-L6-v2 |
| TRAINING_DATA_PATH | Path to training data | /app/training_data |
| CHUNK_SIZE | Text chunk size | 1000 |
| CHUNK_OVERLAP | Chunk overlap size | 200 |
| BATCH_SIZE | Processing batch size | 32 |
| PROCESS_INTERVAL | Background processing interval | 3600 |
| INQUIRY_CHECK_INTERVAL | Queue check interval | 300 |
| RETRAIN_INTERVAL | MLBridge retrain interval | 100 |

## API Endpoints

### Health and Status

- `GET /health`: Health check endpoint
- `GET /status`: Get detailed crawler status
- `GET /training_crawler/health`: Training crawler health status
- `GET /training_crawler/metrics`: Training crawler metrics
- `GET /mlbridge/health`: MLBridge health status
- `GET /mlbridge/metrics`: MLBridge metrics

### File Processing

- `POST /process?file_path=<path>`: Process a single file
- `POST /restart`: Restart the crawler service

## Usage

1. Build and start the service:

```bash
docker-compose up -d
```

2. Process a file:

```bash
curl -X POST "http://localhost:8000/process?file_path=/path/to/file.json"
```

3. Check service status:

```bash
curl "http://localhost:8000/health"
```

4. Monitor metrics:

```bash
curl "http://localhost:8000/training_crawler/metrics"
```

## File Processing

### Supported File Types

1. **JSON Files**
   - Handles both dictionary and list structures
   - Converts dictionary entries to list format
   - Generates embeddings for each entry

2. **JSONL Files**
   - Processes each line as a separate JSON object
   - Validates JSON format
   - Generates embeddings for each line

3. **PDF Files**
   - Extracts text content
   - Splits into chunks
   - Generates embeddings for each chunk

4. **Text Files**
   - Splits into chunks
   - Generates embeddings for each chunk

### Processing Flow

1. File is added to the crawl queue
2. Worker picks up the file
3. File processor determines type and processes accordingly
4. Embeddings are generated and stored in Qdrant
5. Results are published to the results stream

## Error Handling

- Failed files are moved to a dead letter queue
- Processing errors are logged and reported
- Retry mechanism for transient failures
- Comprehensive error reporting in API responses

## Monitoring

The service provides several monitoring endpoints:

- Health checks
- Processing statistics
- Queue lengths
- Error rates
- MLBridge metrics
- Training crawler metrics

## Development

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
pytest
```

4. Start the service:
```bash
uvicorn app.main:app --reload
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 