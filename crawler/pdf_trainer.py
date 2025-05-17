import os
import time
import json
from typing import List, Dict, Optional, Tuple
from qdrant_client import QdrantClient
from redis.asyncio import Redis
from pathlib import Path
import fitz  # PyMuPDF
import asyncio
import requests
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import logging
from datetime import datetime
import aiohttp
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-trainer")

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
EMBEDDING_CHANNEL = os.getenv("EMBEDDING_CHANNEL", "emb.raw")
EMBEDDING_READY_CHANNEL = os.getenv("EMBEDDING_READY_CHANNEL", "embeddings.ready")
CONCEPT_TRAINER_URL = os.getenv("CONCEPT_TRAINER_URL", "http://concept-trainer-growable:8681/api/v1/vectors/ingest")
CONFIDENCE_METRIC_URL = os.getenv("CONFIDENCE_METRIC_URL", "http://concept-trainer-growable:8905/metrics")
CONFIDENCE_THRESHOLD = 0.9
SLEEP_INTERVAL = 60  # seconds

class PDFSection:
    def __init__(self, title: str, content: str, page_num: int, confidence: float = 1.0):
        self.title = title
        self.content = content
        self.page_num = page_num
        self.confidence = confidence

class PDFTrainer:
    def __init__(self, revectorize=False):
        self.revectorize = revectorize
        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        self.redis_client = Redis.from_url(REDIS_URL)
        self.collection = os.getenv("QDRANT_COLLECTION", "pdf_embeddings")
        self.pdf_dir = Path(os.getenv("PDF_DIR", "training_data/"))

        # Initialize TF-IDF vectorizer for section detection
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize DBSCAN for section clustering
        self.dbscan = DBSCAN(
            eps=0.3,
            min_samples=2,
            metric='cosine'
        )

    def extract_text_with_structure(self, pdf_path: Path) -> List[PDFSection]:
        """Extract text from PDF with preserved structure and section detection."""
        doc = fitz.open(pdf_path)
        sections = []
        current_section = None
        current_content = []
        
        for page_num, page in enumerate(doc):
            # Get text blocks with their positions
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                            
                        # Check if this is a potential section header
                        is_header = self._is_potential_header(text)
                        
                        if is_header and current_section:
                            # Save previous section
                            sections.append(PDFSection(
                                title=current_section,
                                content="\n".join(current_content),
                                page_num=page_num
                            ))
                            current_section = text
                            current_content = []
                        elif is_header:
                            current_section = text
                        elif current_section:
                            current_content.append(text)
                        else:
                            # No section header found yet, treat as introduction
                            current_section = "Introduction"
                            current_content.append(text)
        
        # Add the last section
        if current_section and current_content:
            sections.append(PDFSection(
                title=current_section,
                content="\n".join(current_content),
                page_num=len(doc) - 1
            ))
        
        # Perform ML-based section refinement
        refined_sections = self._refine_sections(sections)
        return refined_sections

    def _is_potential_header(self, text: str) -> bool:
        """Determine if text is likely a section header using heuristics."""
        # Check common header patterns
        header_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\d+\.\s+[A-Z][a-z]+',  # Numbered sections
            r'^[A-Z][a-z]+\s+\d+',  # Section names with numbers
            r'^Chapter\s+\d+',  # Chapter headers
            r'^Section\s+\d+',  # Section headers
        ]
        
        return any(re.match(pattern, text) for pattern in header_patterns)

    def _refine_sections(self, sections: List[PDFSection]) -> List[PDFSection]:
        """Refine sections using ML-based clustering."""
        if not sections:
            return sections
            
        # Extract features using TF-IDF
        texts = [f"{s.title} {s.content[:200]}" for s in sections]  # Use title and content preview
        try:
            features = self.tfidf.fit_transform(texts)
            
            # Cluster sections
            clusters = self.dbscan.fit_predict(features.toarray())
            
            # Merge sections in the same cluster
            merged_sections = []
            current_cluster = -1
            current_section = None
            
            for section, cluster in zip(sections, clusters):
                if cluster == -1:  # Noise point
                    merged_sections.append(section)
                elif cluster != current_cluster:
                    if current_section:
                        merged_sections.append(current_section)
                    current_cluster = cluster
                    current_section = section
                else:
                    # Merge with current section
                    current_section.content += f"\n\n{section.content}"
                    current_section.confidence = min(current_section.confidence, section.confidence)
            
            if current_section:
                merged_sections.append(current_section)
                
            return merged_sections
        except Exception as e:
            logger.error(f"Error in section refinement: {str(e)}")
            return sections

    def _chunk_text(self, text: str, max_length: int = 1000) -> List[str]:
        """Split text into semantic chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    async def process_pdf(self, pdf_path: Path) -> None:
        """Process a single PDF file and queue chunks for embedding."""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text with structure
            sections = self.extract_text_with_structure(pdf_path)
            logger.info(f"Extracted {len(sections)} sections from {pdf_path}")
            
            # Process each section
            for section in sections:
                # Chunk the section content
                chunks = self._chunk_text(section.content)
                
                for chunk_idx, chunk in enumerate(chunks):
                    # Generate unique ID for this chunk
                    chunk_id = f"{pdf_path.stem}_{section.title}_{chunk_idx}"
                    
                    # Prepare metadata
                    metadata = {
                        "filename": pdf_path.name,
                        "section": section.title,
                        "page": section.page_num,
                        "chunk": chunk_idx,
                        "confidence": section.confidence,
                        "processed_at": datetime.utcnow().isoformat(),
                        "embedded": False,  # Mark as not yet embedded
                        "source": "pdf_trainer"
                    }
                    
                    # Store raw text in Qdrant with embedded=false
        self.qdrant_client.upsert(
            collection_name=self.collection,
            points=[{
                            "id": chunk_id,
                            "vector": [],  # Empty vector - will be filled by batch embedder
                "payload": {
                                "text": chunk,
                                "metadata": metadata
                }
            }]
        )

                    # Queue for embedding via Redis stream
                    await self.redis_client.xadd(
                        EMBEDDING_CHANNEL,
                        {
                            "doc_id": chunk_id,
                            "text": chunk,
                            "metadata": json.dumps(metadata)
                        }
                    )
                    
                    logger.info(f"Queued chunk {chunk_id} for embedding")
            
            # Notify about completion
            await self.notify_concept_trainer(pdf_path)
            logger.info(f"Successfully processed {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}", exc_info=True)
            raise

    async def wait_for_embeddings(self, chunk_ids: List[str], timeout: int = 300) -> bool:
        """Wait for chunks to be embedded by the batch embedder."""
        start_time = time.time()
        remaining_ids = set(chunk_ids)
        
        while remaining_ids and (time.time() - start_time) < timeout:
            # Check which chunks have been embedded
            results = self.qdrant_client.retrieve(
                collection_name=self.collection,
                ids=list(remaining_ids)
            )
            
            for point in results:
                if point.payload.get("metadata", {}).get("embedded", False):
                    remaining_ids.remove(point.id)
            
            if remaining_ids:
                await asyncio.sleep(1)
        
        return len(remaining_ids) == 0

    async def notify_concept_trainer(self, pdf_path: Path) -> None:
        """Notify concept trainer about new PDF processing."""
        message = json.dumps({
            "event": "pdf_processed",
            "pdf": pdf_path.stem,
            "timestamp": datetime.utcnow().isoformat()
        })
        await self.redis_client.publish("lumina.growth", message)

    async def process_all_pdfs(self) -> None:
        """Process all PDFs in the directory."""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf in pdf_files:
            if self.revectorize or not self.is_already_embedded(pdf):
                await self.process_pdf(pdf)

    def is_already_embedded(self, pdf: Path) -> bool:
        """Check if PDF has already been processed and embedded."""
        result = self.qdrant_client.retrieve(
            collection_name=self.collection,
            ids=[f"{pdf.stem}_Introduction_0"]  # Check first chunk of first section
        )
        if not result:
            return False
        return result[0].payload.get("metadata", {}).get("embedded", False)

async def get_model_confidence(url: str = CONFIDENCE_METRIC_URL) -> float:
    """Get the current model confidence from the concept trainer metrics endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                if response.status != 200:
                    logger.error(f"Failed to get confidence metrics: {response.status}")
                    return 1.0
                
                # Try to parse as JSON first
                try:
                    data = await response.json()
                    return float(data.get("avg_confidence", 1.0))
                except Exception:
                    # Fallback: parse from Prometheus text format
                    text = await response.text()
                    for line in text.splitlines():
                        if "avg_confidence" in line:
                            try:
                                return float(line.split()[-1])
                            except Exception:
                                continue
                return 1.0
    except Exception as e:
        logger.error(f"Error getting model confidence: {str(e)}")
        return 1.0

async def continuous_pdf_training():
    """Main training loop with confidence-based retraining."""
    trainer = PDFTrainer(revectorize=True)
    
    while True:
        try:
            logger.info("Starting PDF processing cycle")
            await trainer.process_all_pdfs()
            
            # Check model confidence
            confidence = await get_model_confidence()
            logger.info(f"Current model confidence: {confidence}")
            
            if confidence < CONFIDENCE_THRESHOLD:
                logger.info(f"Confidence {confidence} below threshold {CONFIDENCE_THRESHOLD}, retraining immediately")
                # Trigger immediate reprocessing of all PDFs
                trainer.revectorize = True
                continue
            
            trainer.revectorize = False
            logger.info(f"Sleeping for {SLEEP_INTERVAL} seconds")
            await asyncio.sleep(SLEEP_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in training loop: {str(e)}", exc_info=True)
            await asyncio.sleep(60)  # Wait before retry

if __name__ == "__main__":
    asyncio.run(continuous_pdf_training())
