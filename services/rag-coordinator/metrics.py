from prometheus_client import Histogram, Counter

# RAG enrichment metrics
ENRICH_LATENCY = Histogram(
    "rag_enrich_latency_seconds",
    "Time spent enriching RAG chunks with metadata",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

ENRICH_REQUESTS = Counter(
    "rag_enrich_requests_total",
    "Number of metadata enrichment requests",
    ["status"]  # success, error
)

ENRICH_CHUNKS = Counter(
    "rag_enrich_chunks_total",
    "Number of chunks enriched with metadata"
)

# RAG retrieval metrics
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent retrieving chunks from vector store",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

RETRIEVAL_CHUNKS = Counter(
    "rag_retrieval_chunks_total",
    "Number of chunks retrieved from vector store"
)

# RAG quality metrics
CHUNK_RELEVANCE = Histogram(
    "rag_chunk_relevance",
    "Relevance score of retrieved chunks",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

MISSING_METADATA = Counter(
    "rag_missing_metadata_total",
    "Number of chunks missing metadata after enrichment"
) 