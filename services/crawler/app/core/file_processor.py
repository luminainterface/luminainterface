import os
import json
import logging
import mimetypes
import hashlib
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
import asyncio
import psutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..models.file_item import (
    FileMetadata, FileChunk, FileEntry, FileVector,
    ProcessedFile, FileProcessingConfig
)

class FileProcessor:
    """Handles processing of different file types and generates embeddings."""
    
    def __init__(
        self,
        embedding_model,
        vector_store,
        redis_client,
        config: Optional[FileProcessingConfig] = None
    ):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.redis = redis_client
        self.config = config or FileProcessingConfig()
        
        # Initialize mimetypes
        mimetypes.init()
        mimetypes.add_type('application/json', '.json')
        mimetypes.add_type('application/jsonl', '.jsonl')
        mimetypes.add_type('application/x-jsonlines', '.jsonl')
        
        # Initialize text splitter with memory-aware chunking after all other initializations
        chunk_size = self._get_optimal_chunk_size()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
    def _get_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory."""
        try:
            # Get available memory in MB
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            
            # Calculate chunk size based on available memory and config ratios
            chunk_size = max(
                self.config.min_chunk_size,
                min(
                    self.config.max_chunk_size,
                    int(available_memory * self.config.memory_chunk_ratio)
                )
            )
            
            self.logger.info(
                f"Memory-aware chunking: available={available_memory:.2f}MB, "
                f"ratio={self.config.memory_chunk_ratio}, "
                f"chunk_size={chunk_size}"
            )
            return chunk_size
        except Exception as e:
            self.logger.warning(
                f"Error calculating optimal chunk size: {e}, "
                f"using default: {self.config.chunk_size}"
            )
            return self.config.chunk_size
            
    def _generate_cache_key(self, text: str, file_type: str) -> str:
        """Generate a cache key for text content."""
        # Create a hash of the text content
        content_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{file_type}:{content_hash}"
        
    async def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        try:
            cached_data = await self.redis.get_cache(cache_key)
            if cached_data and 'embedding' in cached_data:
                self.logger.info(f"Cache hit for key: {cache_key}")
                return cached_data['embedding']
            self.logger.debug(f"Cache miss for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting cached embedding: {e}")
            return None
            
    async def _cache_embedding(self, cache_key: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Cache embedding with metadata."""
        try:
            cache_data = {
                'embedding': embedding,
                'metadata': metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
            await self.redis.set_cache(cache_key, cache_data, self.config.cache_ttl)
            self.logger.info(f"Cached embedding for key: {cache_key}")
        except Exception as e:
            self.logger.error(f"Error caching embedding: {e}")

    async def process_file(self, file_path: str) -> ProcessedFile:
        """Process a file and generate embeddings."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Get file metadata
            metadata = await self._get_file_metadata(file_path)
            
            # Determine file type
            file_type = await self._get_file_type(file_path)
            if not file_type:
                raise ValueError(f"Unsupported file type: {metadata.mime_type}")
            
            # Process based on file type
            vectors = []
            if file_type == 'json':
                vectors = await self._process_json(file_path, metadata)
            elif file_type == 'jsonl':
                vectors = await self._process_jsonl(file_path, metadata)
            elif file_type == 'pdf':
                vectors = await self._process_pdf(file_path, metadata)
            elif file_type == 'text':
                vectors = await self._process_text(file_path, metadata)
            
            # Store vectors in batches with memory management
            if vectors:
                collection = self.config.collection_mapping[file_type]
                batch_size = self._get_optimal_batch_size()
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    await self.vector_store.upsert_vectors(
                        vectors=[v.vector for v in batch],
                        metadata=[v.metadata.dict() for v in batch],
                        ids=[v.id for v in batch]
                    )
                    self.logger.info(f"Stored batch {i//batch_size + 1} of {(len(vectors) + batch_size - 1)//batch_size}")
                    # Allow other tasks to run
                    await asyncio.sleep(0.1)
            
            return ProcessedFile(
                file_path=file_path,
                status="success",
                vectors=vectors,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return ProcessedFile(
                file_path=file_path,
                status="error",
                vectors=[],
                metadata=metadata,
                error=str(e)
            )
            
    def _get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        try:
            # Get available memory in MB
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            
            # Calculate batch size based on available memory and config ratios
            batch_size = max(
                self.config.min_batch_size,
                min(
                    self.config.max_batch_size,
                    int(available_memory * self.config.memory_batch_ratio)
                )
            )
            
            self.logger.info(
                f"Memory-aware batching: available={available_memory:.2f}MB, "
                f"ratio={self.config.memory_batch_ratio}, "
                f"batch_size={batch_size}"
            )
            return batch_size
        except Exception as e:
            self.logger.warning(
                f"Error calculating optimal batch size: {e}, "
                f"using default: {self.config.batch_size}"
            )
            return self.config.batch_size

    async def _process_json(self, file_path: str, metadata: FileMetadata) -> List[FileVector]:
        """Process JSON file and generate embeddings with caching."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entries = []
            if isinstance(data, dict):
                for key, value in data.items():
                    entries.append(FileEntry(
                        key=key,
                        value=value,
                        metadata=metadata
                    ))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        entries.append(FileEntry(
                            value=item,
                            metadata=metadata
                        ))
            
            vectors = []
            for entry in entries:
                try:
                    text_to_embed = str(entry.value) if isinstance(entry.value, (str, int, float)) else json.dumps(entry.value)
                    cache_key = self._generate_cache_key(text_to_embed, 'json')
                    
                    # Try to get from cache first
                    embedding = await self._get_cached_embedding(cache_key)
                    if not embedding:
                        # Generate new embedding if not in cache
                        embedding = self.embedding_model.embed_query(text_to_embed)
                        # Cache the new embedding
                        await self._cache_embedding(cache_key, embedding, {
                            'source': file_path,
                            'type': 'json_entry',
                            'key': entry.key if hasattr(entry, 'key') else None
                        })
                    
                    vectors.append(FileVector(
                        vector=embedding,
                        payload=entry.dict(),
                        collection=self.config.collection_mapping['json'],
                        metadata=metadata
                    ))
                except Exception as e:
                    self.logger.error(f"Error generating embedding for JSON entry: {str(e)}")
                    continue
            
            return vectors
            
        except Exception as e:
            self.logger.error(f"Error processing JSON file: {str(e)}")
            raise

    async def _process_jsonl(self, file_path: str, metadata: FileMetadata) -> List[FileVector]:
        """Process JSONL file and generate embeddings with caching."""
        try:
            vectors = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if isinstance(entry, dict):
                            text_to_embed = str(entry.get('text', '')) or json.dumps(entry)
                            cache_key = self._generate_cache_key(text_to_embed, 'jsonl')
                            
                            # Try to get from cache first
                            embedding = await self._get_cached_embedding(cache_key)
                            if not embedding:
                                # Generate new embedding if not in cache
                                embedding = self.embedding_model.embed_query(text_to_embed)
                                # Cache the new embedding
                                await self._cache_embedding(cache_key, embedding, {
                                    'source': file_path,
                                    'type': 'jsonl_entry'
                                })
                            
                            vectors.append(FileVector(
                                vector=embedding,
                                payload={
                                    **entry,
                                    'source': file_path,
                                    'type': 'jsonl_entry'
                                },
                                collection=self.config.collection_mapping['jsonl'],
                                metadata=metadata
                            ))
                    except json.JSONDecodeError:
                        self.logger.warning(f"Invalid JSON line in {file_path}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing JSONL line: {str(e)}")
                        continue
            
            return vectors
            
        except Exception as e:
            self.logger.error(f"Error processing JSONL file: {str(e)}")
            raise

    async def _process_pdf(self, file_path: str, metadata: FileMetadata) -> List[FileVector]:
        """Process PDF file and generate embeddings with caching."""
        try:
            from ..core.pdf_processor import PDFProcessor
            pdf_processor = PDFProcessor()
            
            with open(file_path, 'rb') as f:
                pdf_content = f.read()
            
            text = await pdf_processor.extract_text(pdf_content)
            if not text:
                self.logger.warning(f"No text extracted from PDF: {file_path}")
                return []
            
            # Use memory-aware chunking
            chunks = self.text_splitter.split_text(text)
            vectors = []
            
            for i, chunk in enumerate(chunks):
                try:
                    cache_key = self._generate_cache_key(chunk, 'pdf')
                    
                    # Try to get from cache first
                    embedding = await self._get_cached_embedding(cache_key)
                    if not embedding:
                        # Generate new embedding if not in cache
                        embedding = self.embedding_model.embed_query(chunk)
                        # Cache the new embedding
                        await self._cache_embedding(cache_key, embedding, {
                            'source': file_path,
                            'type': 'pdf_chunk',
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        })
                    
                    vectors.append(FileVector(
                        vector=embedding,
                        payload={
                            'text': chunk,
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        },
                        collection=self.config.collection_mapping['pdf'],
                        metadata=metadata
                    ))
                except Exception as e:
                    self.logger.error(f"Error generating embedding for PDF chunk: {str(e)}")
                    continue
            
            return vectors
            
        except Exception as e:
            self.logger.error(f"Error processing PDF file: {str(e)}")
            raise

    async def _process_text(self, file_path: str, metadata: FileMetadata) -> List[FileVector]:
        """Process text file and generate embeddings with caching."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use memory-aware chunking
            chunks = self.text_splitter.split_text(content)
            vectors = []
            
            for i, chunk in enumerate(chunks):
                try:
                    cache_key = self._generate_cache_key(chunk, 'text')
                    
                    # Try to get from cache first
                    embedding = await self._get_cached_embedding(cache_key)
                    if not embedding:
                        # Generate new embedding if not in cache
                        embedding = self.embedding_model.embed_query(chunk)
                        # Cache the new embedding
                        await self._cache_embedding(cache_key, embedding, {
                            'source': file_path,
                            'type': 'text_chunk',
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        })
                    
                    vectors.append(FileVector(
                        vector=embedding,
                        payload={
                            'text': chunk,
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        },
                        collection=self.config.collection_mapping['text'],
                        metadata=metadata
                    ))
                except Exception as e:
                    self.logger.error(f"Error generating embedding for text chunk: {str(e)}")
                    continue
            
            return vectors
            
        except Exception as e:
            self.logger.error(f"Error processing text file: {str(e)}")
            raise

    async def _get_file_metadata(self, file_path: str) -> FileMetadata:
        """Get metadata for a file."""
        try:
            stat = os.stat(file_path)
            mime_type, encoding = mimetypes.guess_type(file_path)
            return FileMetadata(
                source=file_path,
                file_type=os.path.splitext(file_path)[1][1:],
                size=stat.st_size,
                mime_type=mime_type,
                encoding=encoding
            )
        except Exception as e:
            self.logger.error(f"Error getting file metadata: {str(e)}")
            return FileMetadata(source=file_path, file_type="unknown")

    async def _get_file_type(self, file_path: str) -> Optional[str]:
        """Determine file type based on extension and content."""
        try:
            _, ext = os.path.splitext(file_path.lower())
            
            # Direct extension check
            if ext == '.json':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    return 'json'
                except:
                    pass
            elif ext == '.jsonl':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            json.loads(first_line)
                            return 'jsonl'
                except:
                    pass
            
            # Content-based check
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1024).strip()
                    if content.startswith('{') or content.startswith('['):
                        try:
                            json.loads(content)
                            return 'json'
                        except:
                            pass
                    elif content and all(line.strip().startswith('{') or line.strip().startswith('[') 
                                       for line in content.split('\n')[:3] if line.strip()):
                        try:
                            for line in content.split('\n')[:3]:
                                if line.strip():
                                    json.loads(line.strip())
                            return 'jsonl'
                        except:
                            pass
            except:
                pass
            
            # MIME type check
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type == 'application/pdf':
                return 'pdf'
            elif mime_type and mime_type.startswith('text/'):
                return 'text'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining file type: {str(e)}")
            return None