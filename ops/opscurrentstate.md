# Lumina Operations Infrastructure

## Development Environment

### Docker Compose Services
- **hub-api**: FastAPI backend service
  - Hot-reload enabled via bind-mount
  - Excludes `__pycache__` and `.pytest_cache`
  - Environment: development mode
  - Port: 8000

- **ui**: Next.js frontend service
  - Hot-reload enabled via bind-mount
  - Excludes `node_modules` and `.next`
  - Environment: development mode
  - Port: 3000

- **vector-db**: Qdrant vector database
  - Latest stable image
  - Persistent storage via volume
  - Port: 6333

- **llm-engine**: Ollama LLM service
  - Latest stable image
  - Persistent model storage
  - Port: 11434

- **redis**: Redis cache
  - Alpine-based image
  - Persistent storage
  - Port: 6379

- **scheduler**: Background task processor
  - Hot-reload enabled
  - Environment: development mode

### Development Workflow
1. Start services: `docker compose up --build`
2. Code changes trigger auto-rebuild
3. Hot-reload enabled for both frontend and backend

## CI/CD Pipeline

### GitHub Actions Workflow
- Triggers on push to `main`/`develop` and PRs
- Steps:
  1. Python setup (3.11)
  2. Node.js setup (20.x)
  3. Python dependencies (ruff, pytest)
  4. Node.js dependencies
  5. Python linting (ruff)
  6. JavaScript linting (eslint)
  7. Python tests (pytest -q)
  8. Docker build verification

## Backup System

### Freeze Script Features
- Configurable parameters (via .env or environment):
  - `SNAP_WAIT`: Snapshot wait time (default: 5s)
  - `MAX_RETRIES`: Retry attempts (default: 3)
  - `QDRANT_URL`: Vector DB endpoint
  - `ARCHIVE_COMPRESSION`: gzip level (default: 6)
  - `DEBUG`: Verbose logging toggle

- Backup components:
  1. UI build output
  2. Docker images
  3. Qdrant snapshots (with polling)
  4. Ollama models
  5. Redis data
  6. Configuration files
  7. Environment files

- Integrity features:
  - Retry logic with exponential backoff
  - Snapshot polling with jq validation
  - Archive integrity checks
  - SHA256 checksums
  - Cleanup on failure
  - Parallel compression (pigz) with gzip fallback

### Tuning Freeze Performance

#### Environment Variables Reference
| Variable | Default | Description | Dev Value | Prod Value |
|----------|---------|-------------|-----------|------------|
| `SNAP_WAIT` | 5s | Wait between snapshot polls | 0 | 5 |
| `MAX_RETRIES` | 3 | Retry attempts for operations | 3 | 3 |
| `QDRANT_URL` | localhost:6333 | Vector DB endpoint | - | - |
| `ARCHIVE_COMPRESSION` | 6 | gzip level (1-9) | 1 | 9 |
| `DEBUG` | unset | Verbose logging | 1 | unset |

#### Common Tuning Scenarios

1. **Development Speed**
```bash
# .env
SNAP_WAIT=0
ARCHIVE_COMPRESSION=1
DEBUG=1
```

2. **Production Reliability**
```bash
# .env
SNAP_WAIT=5
ARCHIVE_COMPRESSION=9
MAX_RETRIES=5
```

3. **CI/CD Pipeline**
```bash
# .env
SNAP_WAIT=0
ARCHIVE_COMPRESSION=1
MAX_RETRIES=3
```

4. **Debug Mode**
```bash
# .env
DEBUG=1
SNAP_WAIT=0
ARCHIVE_COMPRESSION=1
```

### Usage
```bash
# Development (fast compression)
ARCHIVE_COMPRESSION=1 ./freeze.sh

# Production (maximum compression)
ARCHIVE_COMPRESSION=9 ./freeze.sh

# Debug mode
DEBUG=1 ./freeze.sh

# CI/CD (minimal wait)
SNAP_WAIT=0 ./freeze.sh

# Using .env file
cp .env.sample .env  # Copy and edit defaults
./freeze.sh         # Script will use .env values
```

## Security Notes
- No sensitive data in version control
- Environment files excluded from git
- Docker images built from verified sources
- Checksums for archive verification

## Monitoring
- Container health checks via restart policy
- Error logging to stderr
- Debug mode for verbose output
- CI pipeline status checks

## Future Improvements
1. Add coverage reporting to CI
2. Implement Storybook for UI components
3. Add Docker registry authentication support
4. Consider adding monitoring metrics 