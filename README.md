# Lumina Interface 🌌

[![CI](https://github.com/jtran/luminainterface/actions/workflows/ci.yml/badge.svg)](https://github.com/jtran/luminainterface/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Latest Release](https://img.shields.io/github/v/release/jtran/luminainterface?include_prereleases)](https://github.com/jtran/luminainterface/releases)

Lumina is a neural-network interface that bridges human conversation with AI systems via a self-growing language+memory core. This repo contains **all services**—backend, UI, and ops scripts—needed to run Lumina locally or on a server.

## Quick Start (Local)

```bash
git clone https://github.com/jtran/luminainterface.git
cd luminainterface
docker compose up --build
# UI → http://localhost:3000
# API → http://localhost:8000  (requires X-API-Key header set to $LUMINA_API_KEY)
```

## Project Structure

| Path | Purpose |
|------|---------|
| `lumina_core/` | FastAPI backend, Ollama proxy, vector memory, rate-limits |
| `ui/` | Vanilla JS front-end (chat, metrics, shortcuts) |
| `ops/` | Docker Compose, freeze/verify scripts, backups |
| `docs/` | Grafana dashboards, Prometheus alert rules, backlog |

## CI / CD

Lint ➔ tests ➔ build ➔ freeze verify ➔ optional S3 backup

## Alpha Release Notes

Version: v0.1-alpha

Known limits:
- Single-user API key
- 10 RPM chat limit
- No auth UI
- Search feature coming soon

All feedback welcome! File an issue or discussion.

## Development

See individual component READMEs for detailed setup:
- [Backend Setup](lumina_core/README.md)
- [Frontend Setup](ui/README.md)
- [Ops Guide](ops/README.md)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

At its core, Lumina is a neural-network interface that bridges human conversation with AI systems via a self-growing language+memory core. This repo contains **all services**—backend, UI, and ops scripts—needed to run Lumina locally or on a server.

[![CI](https://github.com/jtran/luminainterface/actions/workflows/ci.yml/badge.svg)](https://github.com/jtran/luminainterface/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/jtran/luminainterface?include_prereleases)](https://github.com/jtran/luminainterface/releases)

## Quick Start (Local)

```bash
# Clone and enter
git clone https://github.com/jtran/luminainterface.git
cd luminainterface

# Set API key
export LUMINA_API_KEY="your-secure-key-here"

# Start all services
docker compose up --build

# Access the interface
UI → http://localhost:3000
API → http://localhost:8000
```

## Project Structure

```
lumina_core/     # FastAPI backend, Ollama proxy, vector memory
├── api/         # API routes and middleware
├── memory/      # Vector store and caching
├── models/      # Data models and schemas
└── tests/       # Backend test suite

ui/              # Vanilla JS front-end
├── src/         # Source files
├── public/      # Static assets
└── tests/       # Frontend tests

ops/             # Operations and deployment
├── docker/      # Dockerfile and compose
├── scripts/     # Backup and verification
└── staging/     # Staging configuration

docs/            # Documentation
├── api/         # API documentation
├── ops/         # Operations guides
└── alerts/      # Alert rules and dashboards
```

## Features

- 🤖 **OpenAI Compatibility**: Works with most OpenAI SDKs and libraries
- 🌊 **Streaming Support**: Real-time token streaming
- 🧠 **Memory Integration**: Automatic conversation history
- ⚡ **Embedding Cache**: Faster responses for repeated queries
- 🛡️ **Rate Limiting**: Protection against abuse
- 📊 **Health Checks**: Monitor service status

## Alpha Release Notes

Version: v0.1-alpha

### Known Limitations
- Single-user API key authentication
- 10 requests per minute chat limit
- No authentication UI
- Limited model selection (currently only phi2)
- No fine-tuning support
- Token counting is approximate

### Getting Started
1. Set your API key in the environment
2. Include the key in all requests: `X-API-Key: your-key`
3. See [API Documentation](docs/api/README.md) for endpoints

## Development

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Node.js 18+

### Running Tests
```bash
# Backend tests
cd lumina_core
pytest

# Frontend tests
cd ui
npm test
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Run tests and linting
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📚 [Documentation](docs/README.md)
- 🐛 [Issue Tracker](https://github.com/jtran/luminainterface/issues)
- 💬 [Discussions](https://github.com/jtran/luminainterface/discussions)

# Lumina Project - Backup & Logging Infrastructure

## Overview
This branch implements backup verification and logging infrastructure for the Lumina project, focusing on data integrity, monitoring, and observability.

## Key Components

### 1. Backup Verification System
- Automated backup integrity checks
- Prometheus metrics export
- Configurable retention policies
- Docker-based deployment
- Alert rules for monitoring

### 2. Logging Infrastructure
- Centralized logging with Loki
- Grafana dashboards for visualization
- Log aggregation and analysis
- Performance monitoring

## Implementation Details

### Files for Review

#### Backup Verification
1. `scripts/verify_backup.sh`
   - Core verification logic
   - MD5 checksum validation
   - Metrics generation

2. `docker-compose.backup.yml`
   - Service configuration
   - Resource limits
   - Network setup

3. `scripts/Dockerfile.backup-verifier`
   - Container definition
   - Dependencies
   - Runtime configuration

#### Monitoring
1. `prometheus/rules/backup_alerts.yml`
   - Alert definitions
   - Thresholds
   - Team assignments

2. `grafana/dashboards/crawler_logs.json`
   - Log visualization
   - Error tracking
   - Performance metrics

#### Documentation
1. `docs/backup-verification.md`
   - System overview
   - Setup instructions
   - Troubleshooting guide

## Review Focus Areas

### 1. Security
- Backup integrity verification
- Access controls
- Secret management

### 2. Scalability
- Resource allocation
- Performance optimization
- Data retention

### 3. Maintainability
- Documentation completeness
- Error handling
- Alert configuration

### 4. Monitoring
- Metric selection
- Dashboard usability
- Alert thresholds

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/luminainterface/luminainterface.git
cd luminainterface
```

2. Switch to the review branch:
```bash
git checkout polish-pass
```

3. Start the services:
```bash
docker compose -f docker-compose.backup.yml up -d
```

4. Access the dashboards:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## Review Process
1. Review the code changes in the `polish-pass` branch
2. Focus on the files listed above
3. Leave comments and suggestions using GitHub's review tools
4. Pay special attention to:
   - Security considerations
   - Error handling
   - Documentation completeness
   - Monitoring effectiveness

## Version
Current tag: v0.2.0 