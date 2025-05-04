# Lumina: Self-Expanding Knowledge Circle

## Dev Mode Quick Start (Docker Compose)

**This system is currently running in development mode.**

- Uses Docker Compose with --profile dev for live code reload
- Python services (indexer, ws-bridge, rag-worker) use uvicorn --reload and bind-mount source code
- Frontend is run outside Docker with pnpm start for instant hot reload
- .env-dev sets DEV_RELOAD=true for all dev containers
- To start dev stack:
  
```bash
docker compose --profile dev up -d
# or use: make dev
```

- To stop dev stack:
  
```bash
docker compose --profile dev down
# or use: make stop
```

- To switch to production:
  
```bash
docker compose down
docker compose up -d
```

- All code edits are instantly reflected in running containers (no rebuild needed)

---

## System Overview

Lumina is an advanced self-expanding knowledge system that combines a Wiki-based RAG (Retrieval-Augmented Generation) system with Phi-2 language model integration, deployed using Podman containers. The system implements a "living knowledge circle" that continuously grows and improves through use.

## System Flow

### 1. Initialization and Training
```
Start
  │
  ↓
Load Base Phi-2 Model
  │
  ↓
Initial Training ←── /graphs/ Directory
  │                  (Knowledge Base)
  ↓
Boot Sequence
  │
  ↓
System Ready
```

### 2. Operational Loop
```
User Query ─────→ Chat Interface ─────→ Topic Extraction
    ↑               │                        │
    │               ↓                        ↓
Response ←─── Knowledge Retrieval      Wiki Crawler
    │               │                        │
    │               ↓                        ↓
    └────── Gap Analysis ←──── Save to /graphs/ Directory
                │                        │
                ↓                        ↓
         Trigger Retraining ←─── Chat History Processing
                │                        │
                ↓                        ↓
         Update Knowledge Base     Update Chat Memory
```

### 3. Continuous Improvement Cycle
```
Chat Interaction ──→ Identify Knowledge Gaps
        ↑                     │
        │                     ↓
Update Model ←─── Auto-retrain on New Data
        ↑                     │
        │                     ↓
        └───── Autosave to /graphs/
```

## Core Components

### 1. Training Pipeline
- **Initial Training**
  - Loads base Phi-2 model (GGUF format)
  - Processes existing /graphs/ knowledge
  - Generates initial embeddings
  - Creates base LoRA adapters

- **Retraining Triggers**
  - Knowledge gap detection
  - Topic coverage analysis
  - Performance metrics
  - User interaction patterns

- **Autosave System**
  - Periodic graph snapshots
  - Incremental updates
  - Version control
  - State persistence

### 2. Web Interface
- **Chat System**
  - Real-time interaction
  - Context awareness
  - Source attribution
  - Confidence scoring

- **Knowledge Gap Detection**
  - Missing topic identification
  - Uncertainty tracking
  - Query pattern analysis
  - Performance monitoring

- **Wiki Crawler Integration**
  - Dynamic crawl triggering
  - Priority queue management
  - Content relevance scoring
  - Automatic expansion

### 3. Knowledge Management
- **Graph Storage (/graphs/)**
  - Version-controlled knowledge base
  - Incremental updates
  - Metadata tracking
  - Recovery points

- **Vector Indices**
  - Dense embeddings
  - Sparse vectors (BM25)
  - Graph relationships
  - Hybrid retrieval

- **State Management**
  - Redis for short-term memory
  - Qdrant for long-term storage
  - Event system
  - Recovery mechanisms

### 4. Chat Memory System

#### A. Turn Data Model
```json
{
  "id": "chat_20250501T171455Z_8af394",   // ULID or snowflake
  "session": "alpha",
  "ts": 1714587295.55,                    // epoch timestamp
  "user": "What is Lumina?",
  "assistant": "Lumina is ...",
  "confidence": 0.78,                      // rag-worker log-prob
  "cite_ids": ["Q42", "art_62e3f5"],      // source chunks
  "embedding": [/* 384 floats */]          // MiniLM vector
}
```

#### B. Storage Architecture
1. **Short-term Memory (Redis)**
   - Recent conversation context
   - Crawler state persistence
   - Query history for topic extraction
   - Event channels for real-time updates
   - System state recovery data

2. **ScratchPad Memory (Redis TTL)**
   - Private working memory for request processing
   - Short-lived (5 min TTL) hash storage
   - Chain-of-thought reasoning space
   - Internal notes and retrieval tracking
   - Automatic cleanup via TTL
   - Optional compression to long-term storage

3. **Long-term Memory (Qdrant)**
   
```python
# Qdrant collection
qdrant.upsert(
    collection_name="chat_long",
    points=[{
        "id": turn["id"],
        "vector": turn["embedding"],
        "payload": {
            "ts": turn["ts"],
            "session": turn["session"],
            "confidence": turn["confidence"],
            "text": turn["assistant"]
        }
    }]
)
```

3. **Persistent Journal**
   
```python
# /graphs/chat/YYYY/MM/DD/turns.ndjson
day = datetime.utcfromtimestamp(turn["ts"]).strftime("%Y/%m/%d")
path = GRAPH_DIR / "chat" / day
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(turn) + "\n")
```

#### C. Memory Configuration
```yaml
# docker-compose.yml environment variables
rag-worker:
  environment:
    # Recall Parameters
    - RECALL_TOP_K=3              # similar past turns to fetch
    - RECALL_MAX_DAYS=14          # memory time window
    - HISTORY_TURNS=4             # in-session context
    
    # Training Parameters
    - TRAIN_MIN_CONF=0.2          # minimum confidence threshold
    - TRAIN_BATCH=128             # batch size for training
    - TRAIN_MAX_AGE=7            # training data age limit
    - LR=0.0001                  # learning rate
    - EPOCHS=3                   # training epochs
```

#### D. Inference Pipeline
```python
class ChatMemory:
    def get_context(self, message):
        # 1. Short-term context
        prompt_blocks = self.get_recent_history(HISTORY_TURNS)
        
        # 2. Long-term semantic recall
        vec = self.embed(message)
        since = time.time() - RECALL_MAX_DAYS * 86400
        hits = self.qdrant.search(
            collection_name="chat_long",
            query_vector=vec,
            limit=RECALL_TOP_K,
            filter={"must": [{"key": "ts", "range": {"gte": since}}]}
        )
        mem_blocks = [h.payload["text"] for h in hits]
        
        # 3. Assemble prompt
        return self.build_prompt(prompt_blocks, mem_blocks, message)
```

#### E. Training Integration
```python
class LoRATrainer:
    async def listener(self):
        async for msg in self.sub.listen():
            turn = json.loads(msg["data"])
            
            # Check eligibility
            if turn["confidence"] < TRAIN_MIN_CONF:
                self.stash_sample(turn)
            
            # Train when batch ready
            if self.ready_for_train():
                path = self.create_jsonl(self.train_pool())
                adapter_id = self.run_peft(path)
                self.publish_adapter(adapter_id)
    
    @redis_subscriber("adapter.updated")
    def on_adapter(self, path):
        self.llm.load_adapter(path)  # Hot-swap adapter
```

#### F. Monitoring Metrics
```python
# Prometheus metrics
MEMORY_HITS = Counter(
    "chat_memory_hits_total",
    "Number of successful memory recalls"
)

LORA_SAMPLES = Counter(
    "chat_lora_samples_total",
    "Accumulated training samples"
)

LORA_TRAINS = Counter(
    "chat_lora_train_events_total",
    "Number of training events"
)

ADAPTER_VERSION = Gauge(
    "adapter_version",
    "Current LoRA adapter version",
    ["service"]
)
```

#### G. Implementation Flow
1. **Turn Processing**
   
```python
def process_turn(turn):
    # Log and index
    log_and_index(turn)
    
    # Trigger training if needed
    if turn["confidence"] < TRAIN_MIN_CONF:
        redis.publish("qa.turn", json.dumps(turn))
```

2. **Memory Retrieval**
   
```python
def get_context(query):
    # Get short-term history
    recent = get_recent_turns()
    
    # Get semantic matches
    similar = search_semantic_memory(query)
    
    return combine_context(recent, similar)
```

3. **Training Loop**
   
```python
async def training_loop():
    while True:
        # Collect samples
        if samples_ready():
            # Train new adapter
            train_adapter()
            
            # Hot-swap in rag-worker
            notify_adapter_update()
        
        await asyncio.sleep(CHECK_INTERVAL)
```

3. **Memory Pipeline**
   
```
User Query → Topic Extraction
        ↓           ↑
   Vector Search ← Crawler Update
        ↓           ↑
   Context Build ← State Persistence
        ↓           ↑
   Response Gen → Event Broadcast
```

   **ScratchPad Pipeline**
   
```
Request → Create ScratchPad (TTL)
      ↓
   Process Query → Write Internal Notes
      ↓
   Generate Response → Compress & Store
      ↓
   Return Answer → Auto-cleanup
```

### 5. Autonomous Learning Loop

#### A. Event Stream Architecture
```
┌────────────┐   GRAPH_ADD   ┌──────────────┐   PAD_COMPRESSED   ┌─────────┐
│  Crawler   │──────────────▶│   Indexer    │────────────────────▶│ Trainer │
└────────────┘               └──────────────┘                    └─────────┘
     ▲  │                       │   ▲                                │   │
     │  └───CRAWL_REQ───────────┘   │                                │   └──ADAPTER_UPDATED
     │                              │                                ▼
┌────┴──────┐   GAP_REQ     ┌───────┴────────┐   THINK_NOTE    ┌────────────┐
│   Planner │──────────────▶│ Scratch-Pad VM │────────────────▶│ Compressor │
└───────────┘               └────────────────┘                 └────────────┘
```

#### B. Core Services

1. **Crawler Service**
   - Listens for CRAWL_REQ events
   - Processes Wikipedia articles
   - Emits GRAPH_ADD events
   - Rate-limited by CRAWL_BUDGET_PER_HR

2. **Indexer Service**
   - Processes GRAPH_ADD events
   - Updates vector store
   - Detects topic sparsity
   - Emits GAP_REQ when needed

3. **Planner Service**
   - Runs every 10 minutes
   - Analyzes retrieval metrics
   - Prioritizes crawl targets
   - Emits CRAWL_REQ events
   
```python
async def planner_loop():
    while True:
        gaps = await redis.zpopmin("gaps:topics", 5)
        if not gaps:
            await asyncio.sleep(600)
            continue
        for topic,_ in gaps:
            await redis.xadd("crawl:req", {
                "url": topic,
                "prio": 1.0
            })
```

4. **Scratch-Pad VM**
   - Headless LLM processing
   - Generates reflections on new content
   - Uses TTL-based ScratchPad storage
   - Emits THINK_NOTE events
   
```python
pad = ScratchPad(ttl=300)
for _ in range(N):
    q = f"What new synergy exists in {title}?"
    answer = llm(q, max_tokens=80)["choices"][0]["text"]
    pad.write({"note": answer})
    await redis.xadd("think:note", {
        "pad": pad.id,
        "text": answer
    })
pad.destroy()
```

5. **Compressor Service**
   - Processes expired ScratchPads
   - Generates embeddings
   - Stores in vector database
   - Cleans up raw data
   
```python
async def on_pad_expire(pad_id):
    txt = " ".join(pad.read_all())
    vec = embed(txt)
    qdrant.upsert("scratch_mem", [{
        "id": pad_id,
        "vector": vec
    }])
    pad.destroy()
```

#### C. Configuration
```yaml
# Environment Variables
PAD_TTL: 300              # ScratchPad lifetime in seconds
THINK_ROUNDS: 3           # Reflection notes per article
GAP_THRESHOLD: 0.40       # Topic sparsity threshold
CRAWL_BUDGET_PER_HR: 50   # Rate limit for crawler
```

#### D. Monitoring
```python
# Prometheus Metrics
SCRATCH_NOTES_TOTAL = Counter(
    "scratch_notes_total",
    "Number of reflection notes generated"
)

PADS_COMPRESSED_TOTAL = Counter(
    "pads_compressed_total",
    "Number of ScratchPads processed"
)

AUTO_CRAWL_REQ_TOTAL = Counter(
    "auto_crawl_req_total",
    "Number of autonomous crawl requests"
)

GAP_HIT_RATE = Histogram(
    "gap_hit_rate",
    "Distribution of retrieval hit rates",
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)
```

#### E. Alerts
- gap_hit_rate < 0.6 for 1 hour: Potential embedding drift
- scratch_notes_total < 10 per hour: VM may be stalled
- pads_compressed_total < pads_created_total: Compression backlog

#### F. Module Maturity Tracking
| Component | Status | Metric to Watch | "Done" Definition |
|-----------|--------|----------------|-------------------|
| Crawler | ✅ | articles_ingested_total ↑ | 100 articles/day within CPU budget |
| Indexer | ✅ | faiss_query_seconds p95 < 50ms | 100K vectors handled |
| Retrieval | ✅ | retrieval_hit_ratio > 0.7 | Answers always cite IDs |
| Scratch VM | 🔄 | scratch_notes_total | Generates ≥3 notes/article |
| Planner | 🔄 | auto_crawl_req_total | Keeps queue depth 20-40 |
| LoRA Trainer | 📝 | adapter_version_total | New adapter every N days |
| Monitoring | ✅ | Dashboard & latency alert | All services exported & scrape UP |

### 6. Monitoring Architecture

#### A. Metrics Registry
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class MetricsRegistryManager:
    """Central metrics registry for all Lumina services."""
    
    # Crawler Metrics
    articles_ingested_total = Counter(
        "articles_ingested_total",
        "Total number of articles processed",
        ["source"]
    )
    crawl_queue_depth = Gauge(
        "crawl_queue_depth",
        "Current number of URLs in crawl queue"
    )
    
    # Indexer Metrics
    faiss_query_seconds = Histogram(
        "faiss_query_seconds",
        "Time spent on vector search",
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
    )
    vectors_indexed_total = Counter(
        "vectors_indexed_total",
        "Total number of vectors stored"
    )
    
    # Retrieval Metrics
    retrieval_hit_ratio = Gauge(
        "retrieval_hit_ratio",
        "Ratio of successful retrievals"
    )
    retrieval_latency = Histogram(
        "retrieval_latency_seconds",
        "Time spent on retrieval operations",
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    # Scratch VM Metrics
    scratch_notes_total = Counter(
        "scratch_notes_total",
        "Number of reflection notes generated"
    )
    pad_compression_seconds = Histogram(
        "pad_compression_seconds",
        "Time spent compressing ScratchPads",
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    # LoRA Training Metrics
    adapter_version = Gauge(
        "adapter_version",
        "Current LoRA adapter version",
        ["service"]
    )
    training_samples_total = Counter(
        "training_samples_total",
        "Total number of samples used for training"
    )
    
    @classmethod
    def start_server(cls, port: int = 9300):
        """Start Prometheus metrics server."""
        start_http_server(port)
```

#### B. Service Integration
```python
# Example service integration
from metrics.registry import MetricsRegistryManager as M

class IndexerService:
    def process_article(self, article):
        with M.faiss_query_seconds.time():
            hits = self.index.search(article.vector, k=5)
        M.retrieval_hits.inc(len(hits))
        M.vectors_indexed_total.inc()
```

#### C. Prometheus Configuration
```yaml
# docker/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'lumina'
    static_configs:
      - targets:
        - 'crawler:9300'
        - 'indexer:9301'
        - 'rag_worker:9302'
        - 'scratch_vm:9303'
        - 'planner:9304'
    metrics_path: '/metrics'
    relabel_configs:
      - source_labels: [__address__]
        target_label: service
        regex: '([^:]+):.*'
```

#### D. Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Lumina System Health",
    "panels": [
      {
        "title": "Generation Latency",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(retrieval_latency_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Crawl Queue Depth",
        "type": "gauge",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "crawl_queue_depth"
        }]
      },
      {
        "title": "Indexing QPS",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [{
          "expr": "rate(vectors_indexed_total[5m])"
        }]
      }
    ]
  }
}
```

### 7. Development Roadmap

#### A. Soft-Crawler/Chat Milestone
The "Soft-Crawler/Chat" milestone enables real-time knowledge ingestion and querying:
1. Seed a new URL → crawler ingests it
2. See the node pop into the graph in < 10s
3. Ask chat about that topic → get answer citing fresh content

**Component Status:**
| Component | Status | Blocking? | Notes |
|-----------|--------|-----------|-------|
| Ingest | ✅ Done | No | Crawler core, queue, metrics |
| Index | ✅ Done | No | FAISS upsert, hybrid search |
| Reflection | ✅ Done | Optional | Scratch VM + compression |
| Prompt Build | ✅ Done | No | Retrieval chunks + history |
| LLM | ✅ Done | No | Phi-2 via llama.cpp |
| Chat REST | ✅ Done | No | /chat returns answer + cite_ids |
| Web UI | ✅ Done | No | ChatPane input / history |
| Graph UI | 🟡 MVP | Yes | Needs WS hook from backend |
| Streamer | 🟡 TODO | Yes | /stream WS that forwards GRAPH_ADD |
| WS Bridge | 🟡 TODO | Yes | Redis → WS forwarder |
| Security | – | No | Not needed for soft demo |

**Implementation Plan:**
1. **WS Forwarder (40 lines, 30 min)**
   
```python
# services/ws_bridge/main.py
from fastapi import FastAPI, WebSocket
import asyncio, json, redis.asyncio as aioredis

app, conns, r = FastAPI(), set(), aioredis.from_url("redis://redis")

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept(); conns.add(ws)
    try:
        while True: await asyncio.sleep(3600)
    finally: conns.remove(ws)

async def pump():
    sub = r.pubsub(); await sub.subscribe("article.ingested")
    async for msg in sub.listen():
        if msg['type']!='message': continue
        meta = json.loads(msg['data'])
        evt = json.dumps({
            "type": "GRAPH_ADD",
            "source": meta['id'],
            "target": "wiki",
            "label": "ingested"
        })
        dead = []
        for c in conns:
            try: await c.send_text(evt)
            except: dead.append(c)
        for d in dead: conns.discard(d)

@app.on_event("startup")
async def _(): asyncio.create_task(pump())
```

2. **Frontend WS Hook (15 lines, 15 min)**
   
```javascript
const ws = new WebSocket("ws://localhost:7210/stream");
ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if(m.type === "GRAPH_ADD") {
    links.push({
      source: m.source,
      target: m.target,
      label: m.label
    });
    if(!nodes.find(n => n.id === m.source)) {
      nodes.push({
        id: m.source,
        label: m.source
      });
    }
    update();
  }
};
```

3. **Graph Auto-layout (10 lines, 10 min)**
   - Call update() after new node
   - Add size limit check
   - Implement smooth transitions

**Testing Flow:**
```bash
# 1. Start crawl
redis-cli ZADD crawl:queue 0 "Artificial_intelligence"

# 2. Watch node appear in graph

# 3. Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize Artificial intelligence"}'
```

#### B. Future Milestones
1. **C– (Current)**
   - Basic chat functionality
   - Crawler ↔ Indexer ↔ Vector DB integration
   - Initial graph visualization

2. **C (Next)**
   - Scratch VM + Planner feedback
   - Self-seeding topics
   - Real-time graph updates

3. **B**
   - LoRA Trainer online
   - Confidence > 0.85
   - Nightly adapter updates
   - Fine-grained monitoring

4. **A**
   - Distributed crawlers
   - Multi-model ensemble
   - CI/CD pipeline
   - Security hardening
   - Polished UI

## Core Architecture

### 1. Knowledge Flow
```
User Interaction ─→ Query Processing ─→ Topic Extraction
      ↑                                       ↓
Response Generation                     Crawler Queue
      ↑                                       ↓
Knowledge Retrieval ←─ Vector Storage ←─ Wiki Crawling
      ↑                                       ↓
Memory System     ←───── Gap Analysis ←─── Learning Loop
```

### 2. Main Components

#### A. Plugin System
```yaml
# Module Manifest (modules/my-module/manifest.yml)
name: my-module
version: 0.1.0
description: "Module description"
api_version: "1.0"
dependencies:
  - lumina-core >= 0.1.0
  - other-module >= 0.2.0
endpoints:
  - path: /api/v1/my-feature
    method: POST
    description: "Feature description"
permissions:
  - read:vector_store
  - write:chat_history
```

#### B. Freeze System
```bash
#!/bin/bash
# ops/freeze.sh

# Capture container states
docker-compose down
docker save $(docker-compose config --services | xargs -I{} echo lumina-{}) -o lumina-images.tar

# Backup volumes
tar czf lumina-volumes.tar.gz \
    ./vector_store \
    ./redis_dump \
    ./graphs \
    ./adapters

# Bundle everything
tar czf lumina-snapshot-$(date +%Y%m%d).tar.gz \
    lumina-images.tar \
    lumina-volumes.tar.gz \
    docker-compose.yml \
    .env
```

#### C. Wiki Crawler System
- **Async Job Queue** (BullMQ/RQ)
  - Non-blocking crawl operations
  - WebSocket real-time updates
  - IndexedDB storage integration
  - Concurrent URL processing

- **Intelligent Seed Selection**
  
```python
priority_score = PageRank × novelty × recency
```

  - PageRank from Wiki dumps
  - Novelty scoring for new content
  - Recency-based prioritization

#### D. Knowledge Graph
- **Enhanced Structure**
  
```json
{
  "nodes": [{
    "id": "article_url",
    "title": "Article Title",
    "summary": "Article summary text...",
    "fingerprint": "hash(title + first_sentence)",
    "wikidata": "Q12345"
  }],
  "links": [{
    "source": "article_url_1",
    "target": "article_url_2",
    "relations": ["page_link", "semantic"]
  }]
}
```

- Content fingerprinting
- Wikidata integration
- Relation-typed edges

#### E. RAG Implementation
- **Tri-Vector Index**
  - Dense embeddings (BGE-small-en-v1.5)
  - BM25 sparse vectors
  - Graph-based PageRank
  
- **Self-Improvement Loop**
  - QA interaction logging
  - Confidence scoring
  - Unknown term detection
  - LoRA adaptation

#### F. Memory Systems
- **Short-term (Redis)**
  - Recent context
  - Crawler state
  - Query history
  - Event channels

- **ScratchPad (Redis TTL)**
  
```python
class ScratchPad:
    def __init__(self, ttl=300):
        self.id = f"pad:{uuid.uuid4().hex}"
        self.ttl = ttl
    
    def write(self, note:str):
        timestamp = time.time()
        r.hset(self.id, timestamp, note)
        r.expire(self.id, self.ttl)
    
    def read_all(self):
        return [json.loads(n) for n in r.hvals(self.id)]
    
    def destroy(self):
        r.delete(self.id)
```

- **Long-term (Qdrant)**
  - Semantic vectors
  - Smart reindexing
  - Knowledge tracking
  - Gap analysis

### 3. Container Architecture

#### Services
1. **Hub API (FastAPI)**
   - Plugin registry
   - Authentication
   - Request routing
   - Service discovery

2. **LLM Engine (llama.cpp/Ollama)**
   - Local model inference
   - Hot-swap model support
   - Quantization management
   - Batch processing

3. **Scheduler (Prefect/Airflow-lite)**
   - Periodic task management
   - "Dream" cycle orchestration
   - Resource allocation
   - Failure recovery

4. **UI (Next.js/PySide)**
   - Chat interface
   - Module dashboard
   - System monitoring
   - Configuration management

## Development Roadmap

### Sprint 0: Minimal Viable Skeleton

#### Repository Structure
```
lumina/
├── lumina-core/          # Core model + memory
├── modules/              # Plugin modules
│   ├── home-security/    # Example: motion-cam interface
│   └── game-dev-agent/   # Example: Unity/Unreal helper
├── ops/                  # Operations
│   ├── docker-compose.yml
│   └── freeze.sh
└── README.md
```

#### Essential Services
| Service | Technology | Purpose |
|---------|------------|----------|
| hub-api | FastAPI | Gateway, auth, plugin registry |
| vector-db | Qdrant | Long-term memory / RAG |
| llm-engine | llama.cpp/Ollama | Local inference, hot-swap models |
| scheduler | Prefect/Airflow-lite | Nightly "dream" runs, periodic tasks |
| ui | Next.js/PySide | Chat + module dashboard |

### Sprint 1: Core Functionality (2 weeks)

#### Backlog
1. **"Hello World" Chat**
   - Hub-API proxy to local phi-2 GGUF
   - Functional chat window
   - Vector DB storage integration

2. **Memory System**
   - Message embedding pipeline
   - Similarity search implementation
   - Retrieval optimization

3. **Plugin Scaffold**
   - Module generator (`lumina plugin init`)
   - Manifest validation
   - Test framework

4. **Freeze System**
   - Container state capture
   - Volume backup
   - Configuration bundling

## System Features

### 1. Self-Improvement
- Continuous learning from interactions
- Automatic knowledge expansion
- LoRA domain adaptation
- Performance analytics

### 2. Memory Efficiency
- Lazy index merging
- Chunked processing
- Smart GPU management
- Async operations

### 3. Scalability
- Priority scheduling
- Background tasks
- Queue-based processing
- State persistence

### 4. Quality Assurance
- Structure preservation
- Relation awareness
- Two-stage retrieval
- Graph-based ranking

## Maintenance

### 1. Regular Tasks
- Redis persistence (60s)
- Knowledge recrawling (7 days)
- Topic extraction
- Gap analysis
- State backup

### 2. Monitoring
```bash
# Check knowledge loop
podman logs -f api | grep "knowledge_loop"

# Monitor crawler
redis-cli -n 0 keys "crawler:*"

# View topic extraction
podman logs -f api | grep "topic_extraction"
```

### 3. Backup Strategy
```bash
# Backup
tar czf backup.tar.gz graphs/ vector_store/ redis_dump/

# Restore
tar xzf backup.tar.gz
podman-compose restart
```

## Future Improvements

1. Enhanced Model Support
   - Additional language models
   - Multi-model ensembles
   - Custom model training

2. System Optimization
   - Graph pruning
   - Incremental updates
   - Multi-hop reasoning

3. Interface Enhancements
   - User authentication
   - Custom visualizations
   - Advanced analytics

4. Infrastructure
   - Distributed crawling
   - Cloud deployment
   - Automatic scaling

## Best Practices

1. Knowledge Building
   - Start with focused topics
   - Regular graph exports
   - Monitor growth patterns
   - Adjust parameters

2. Resource Management
   - Monitor GPU usage
   - Optimize batch sizes
   - Regular cleanup
   - State persistence

3. Security
   - Regular backups
   - Access control
   - Data encryption
   - Update management 