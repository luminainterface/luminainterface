# Output Engine v2

## Overview
Output Engine v2 is a dual-input, dual-brain microservice that generates user-facing answers and internal "thoughts" by combining:
- **Vocabulary/Knowledge** from Concept-Dictionary (RAG chunks)
- **Intelligence/Adapter** from Concept-Trainer-Growable (LoRA weights)

## Architecture
- Receives a query via `/respond`
- Fetches RAG chunks from Concept-Dictionary
- Loads latest LoRA adapter from Concept-Trainer-Growable
- Builds a prompt and generates an answer using the LLM
- Publishes output events to Redis streams (`output.generated`, `thought.log`)
- Returns the answer and citations

## Endpoints
- `POST /respond` — Main endpoint for user queries
  - Input: `{ "query": "...", "top_k": 6 }`
  - Output: `{ "answer": "...", "citations": [cid, ...] }`

## Setup
1. Build the Docker image:
   ```sh
   docker build -t output-engine-v2 .
   ```
2. Run with Docker Compose (see main compose file for service definition)
3. Ensure environment variables are set:
   - `REDIS_URL`
   - `CONCEPT_DICT_URL`
   - `TRAINER_URL`

## Streams
- `output.generated` — All user-facing outputs
- `thought.log` — Internal thought notes for introspection/learning

## Requirements
- Python 3.11+
- CUDA GPU for LLM inference (recommended)
- Redis, Concept-Dictionary, Concept-Trainer-Growable services running

## Extending
- Add metrics via Prometheus FastAPI Instrumentator
- Add more endpoints for health, metrics, or debugging 