from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class FileMetadata(BaseModel):
    """Metadata for processed files."""
    source: str
    file_type: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    size: Optional[int] = None
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    error: Optional[str] = None

class FileChunk(BaseModel):
    """A chunk of text from a file."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: FileMetadata
    chunk_index: int
    total_chunks: int

class FileEntry(BaseModel):
    """A single entry from a JSON/JSONL file."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key: Optional[str] = None
    value: Any
    metadata: FileMetadata
    type: str = "file_entry"

class FileVector(BaseModel):
    """A vector embedding with metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: List[float]
    payload: Dict[str, Any]
    collection: str
    metadata: FileMetadata

class ProcessedFile(BaseModel):
    """Result of file processing."""
    file_path: str
    status: str
    vectors: List[FileVector]
    metadata: FileMetadata
    error: Optional[str] = None

class FileProcessingConfig(BaseModel):
    """Configuration for file processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 5.0
    cache_ttl: int = 86400  # Cache TTL in seconds (24 hours)
    min_chunk_size: int = 500  # Minimum chunk size for memory-aware chunking
    max_chunk_size: int = 2000  # Maximum chunk size for memory-aware chunking
    min_batch_size: int = 16  # Minimum batch size for vector storage
    max_batch_size: int = 64  # Maximum batch size for vector storage
    memory_chunk_ratio: float = 0.1  # Ratio of available memory to use for chunking (10%)
    memory_batch_ratio: float = 0.05  # Ratio of available memory to use for batching (5%)
    supported_types: List[str] = ["json", "jsonl", "text", "pdf"]
    collection_mapping: Dict[str, str] = {
        "json": "json_embeddings",
        "jsonl": "jsonl_embeddings",
        "text": "text_embeddings",
        "pdf": "pdf_vectors_768"
    } 