# Lumina Core

## Using Lumina as an OpenAI Drop-in

Lumina provides an OpenAI-compatible API surface that works with most OpenAI SDKs and libraries. This makes it easy to integrate with existing tools and frameworks.

### Environment Setup

1. Set required environment variables:
```bash
# Core services
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi2
QDRANT_URL=http://localhost:6333

# Optional: Redis for caching and rate limiting
REDIS_URL=redis://localhost:6379
EMBEDDING_CACHE_SIZE=1000
```

2. Start the services:
```bash
# Start Redis (if using caching)
docker run -d -p 6379:6379 redis

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Start Ollama with your model
ollama run phi2

# Start Lumina
uvicorn lumina_core.api.main:app --host 0.0.0.0 --port 8000
```

### Integration Examples

#### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Any value works
)

# Non-streaming
response = client.chat.completions.create(
    model="phi2",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="phi2",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

#### Using LangChain

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

#### Using cURL

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Features

- **OpenAI Compatibility**: Works with most OpenAI SDKs and libraries
- **Streaming Support**: Real-time token streaming
- **Memory Integration**: Automatic conversation history
- **Embedding Cache**: Faster responses for repeated queries
- **Rate Limiting**: Protection against abuse
- **Health Checks**: Monitor service status

### Limitations

- Currently supports only the chat completions endpoint
- Token counting is approximate
- No fine-tuning support
- Limited model selection (currently only phi2)

### Troubleshooting

1. **Connection Issues**
   - Verify all services are running
   - Check environment variables
   - Ensure ports are accessible

2. **Rate Limiting**
   - Default: 10 requests per minute
   - Adjust via `RATE_LIMIT_PER_MINUTE` env var

3. **Cache Issues**
   - Clear Redis cache: `redis-cli FLUSHALL`
   - Check Redis connection: `redis-cli PING`

4. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```
   This will show the status of all services.

## API Key Configuration

The backend requires an API key for all endpoints except `/health`. The key should be:

1. Set in the backend environment:
   ```bash
   export LUMINA_API_KEY="your-secure-key-here"
   ```

2. Included in frontend requests:
   ```typescript
   // Add to all API calls
   headers: {
     'X-API-Key': process.env.NEXT_PUBLIC_LUMINA_API_KEY
   }
   ```

3. Stored in frontend environment:
   ```env
   # .env.local
   NEXT_PUBLIC_LUMINA_API_KEY=your-secure-key-here
   ```

## Rate Limits

- Chat endpoint: 10 requests per minute per API key
- Admin endpoints: 5 requests per minute per API key
- Metrics endpoint: 30 requests per minute per API key

Rate limit responses:
- Status: 429 Too Many Requests
- Body: `{"detail": "Rate limit exceeded"}`

## Monitoring

The backend exposes Prometheus metrics at `/metrics` and a summary at `/metrics/summary`. Key metrics include:

- Request latency (p95)
- Cache hit/miss rates
- Rate limit hits/blocks
- Service health status

See `docs/ops/rate_limits.md` for Grafana dashboard setup and alert configuration. 