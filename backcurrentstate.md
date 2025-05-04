# Lumina Backend Current State

## Core Components

### 1. Ollama Bridge (`lumina_core/llm/ollama_bridge.py`)
- Environment-driven configuration (`OLLAMA_URL`, `OLLAMA_MODEL`)
- Streaming response support
- Token counting
- Error handling
- Async HTTP client with timeout

### 2. Memory System (`lumina_core/memory/qdrant_store.py`)
- Qdrant vector store integration
- Nomic embeddings (`nomic-ai/nomic-embed-text-v1`)
- Conversation tracking with UUIDs
- Similar message retrieval (top-3)
- Metrics collection

### 3. Embedding Cache (`lumina_core/utils/cache.py`)
- Two-level caching:
  - In-memory LRU cache
  - Optional Redis backend
- Configurable cache size
- 24-hour Redis expiry
- Error handling and logging

### 4. OpenAI Compatibility (`lumina_core/api/openai_compat.py`)
- `/v1/chat/completions` endpoint
- Streaming and non-streaming support
- Token counting
- Memory integration
- `/v1/models` endpoint

## API Endpoints

### Chat API
- `POST /chat`
  - Streaming responses
  - Memory integration
  - Token counting
  - Error handling

### Metrics API
- `GET /metrics/summary`
  - Conversation count
  - Token usage
  - Vector count

### Health API
- `GET /health`
  - Deep service checks
  - Per-service status
  - Degraded state handling

### OpenAI-Compatible API
- `POST /v1/chat/completions`
  - Streaming support
  - Token counting
  - Memory integration
- `GET /v1/models`
  - Model listing

## Infrastructure

### Logging
- Structured logging with loguru
- JSON format for production
- Colorized format for development
- File rotation and compression
- Request/response logging
- Error context logging

### Rate Limiting
- Redis-backed rate limiter
- Configurable limits
- Protection against DoS

### CORS
- Configurable origins
- Method and header support

## Testing

### Unit Tests
- Ollama bridge mocking
- Qdrant store mocking
- OpenAI compatibility
- Streaming responses
- Error handling

### Test Coverage
- Chat endpoints
- Memory operations
- OpenAI compatibility
- Health checks
- Metrics

## Environment Variables
```
OLLAMA_URL=http://llm-engine:11434
OLLAMA_MODEL=phi2
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://localhost:6379
EMBEDDING_CACHE_SIZE=1000
```

## Dependencies
```
fastapi>=0.104.0
uvicorn>=0.24.0
redis>=5.0.1
qdrant-client>=1.6.0
sentence-transformers>=2.2.2
pydantic>=2.4.2
python-dotenv>=1.0.0
httpx>=0.25.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
loguru>=0.7.0
fastapi-limiter>=0.1.5
```

## Usage Examples

### OpenAI Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# Streaming
stream = client.chat.completions.create(
    model="phi2",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Non-streaming
response = client.chat.completions.create(
    model="phi2",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### LangChain
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    model_name="phi2"
)

messages = [HumanMessage(content="Hello!")]
response = chat(messages)
print(response.content)
```

## Running the Service

### Development
```bash
uvicorn lumina_core.api.main:app --reload
```

### Production
```bash
REDIS_URL=redis://localhost:6379 \
EMBEDDING_CACHE_SIZE=1000 \
JSON_LOGS=true \
uvicorn lumina_core.api.main:app --host 0.0.0.0 --port 8000
```

## Next Steps
1. Vector pruning job for Qdrant maintenance
2. Enhanced error handling and retries
3. Metrics aggregation and visualization
4. Load testing and performance optimization
5. Documentation and API examples 