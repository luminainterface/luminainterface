import os
import time
import json
from typing import List, Dict, Optional, Tuple
from qdrant_client import QdrantClient
from pathlib import Path
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import asyncio
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import logging
from datetime import datetime
import uuid
import random
import requests
from lumina_core.common.bus import BusClient, StreamMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-trainer")

# Remove concept trainer URLs since we're not using direct connection
SLEEP_INTERVAL = 60  # seconds

# Initialize bus client
bus = BusClient(redis_url=os.getenv("REDIS_URL", "redis://redis:6379"))

class PDFSection:
    def __init__(self, title: str, content: str, page_num: int, confidence: float = 1.0):
        self.title = title
        self.content = content
        self.page_num = page_num
        self.confidence = confidence

class PDFTrainer:
    def __init__(self, revectorize=False):
        # Always force reprocessing
        self.revectorize = True
        self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
        self.model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
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
        """Process a single PDF file and write to JSON file only."""
        try:
            logger.info(f"[process_pdf] Starting processing for: {pdf_path}")
            # Ensure training_data directory exists
            training_data_dir = Path("training_data")
            if not training_data_dir.exists():
                logger.warning("training_data directory missing, creating it.")
                training_data_dir.mkdir(parents=True, exist_ok=True)
            # Ensure training_concepts.json exists
            json_path = training_data_dir / "training_concepts.json"
            if not json_path.exists():
                logger.info("training_concepts.json missing, creating it.")
                with open(json_path, "w", encoding="utf-8") as f:
                    pass

            # Extract raw text from PDF
            try:
                doc = fitz.open(pdf_path)
                raw_text = "\n".join(page.get_text() for page in doc)
                logger.info(f"[process_pdf] Extracted raw text from {pdf_path}, length: {len(raw_text)}")
                # Write raw text as a JSON object
                raw_text_obj = {
                    "filename": pdf_path.name,
                    "raw_text": raw_text,
                    "extracted_at": datetime.utcnow().isoformat()
                }
                with open(json_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(raw_text_obj) + "\n")
                logger.info(f"[process_pdf] Wrote raw text JSON for {pdf_path}")
            except Exception as raw_exc:
                logger.error(f"[process_pdf] Failed to extract/write raw text for {pdf_path}: {raw_exc}")
            
            # Extract text with structure
            try:
                sections = self.extract_text_with_structure(pdf_path)
                logger.info(f"[process_pdf] Extracted {len(sections)} sections from {pdf_path}")
            except Exception as extract_exc:
                logger.error(f"[process_pdf] Extraction failed for {pdf_path}: {extract_exc}")
                return
            
            wrote_any = False
            seen_chunks = set()  # Track unique chunks to avoid duplication
            
            # Process each section
            for section_idx, section in enumerate(sections):
                logger.info(f"[process_pdf] Section {section_idx}: '{section.title}' (page {section.page_num})")
                # Chunk section text
                chunks = self._chunk_text(section.content)
                logger.info(f"[process_pdf] Section {section_idx} chunked into {len(chunks)} chunks.")
                
                for chunk_idx, chunk in enumerate(chunks):
                    # Skip if duplicate
                    chunk_hash = hash(chunk.strip())
                    if chunk_hash in seen_chunks:
                        logger.info(f"[process_pdf] Skipping duplicate chunk {chunk_idx} of section {section_idx}")
                        continue
                        
                    # Generate embedding
                    try:
                        embedding = self.model.encode(chunk)
                        vec_id = str(uuid.uuid4())
                        logger.info(f"[process_pdf] Embedded chunk {chunk_idx} of section {section_idx}")
                        
                        # Publish to ingest.pdf stream
                        await bus.publish(
                            stream="ingest.pdf",
                            data={
                                "id": vec_id,
                                "file_path": str(pdf_path),
                                "vec_id": vec_id,
                                "ts": time.time(),
                                "metadata": {
                                    "filename": pdf_path.name,
                                    "section": section.title,
                                    "page": section.page_num,
                                    "chunk": chunk_idx,
                                    "processed_at": datetime.utcnow().isoformat()
                                }
                            }
                        )
                        logger.info(f"[process_pdf] Published chunk {chunk_idx} to ingest.pdf stream")
                        
                    except Exception as embed_exc:
                        logger.error(f"[process_pdf] Embedding failed for {pdf_path} section {section.title} chunk {chunk_idx}: {embed_exc}")
                        continue
                        
                    # Prepare concept dict
                    concept = {
                        "term": chunk,
                        "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                        "label": section.title,
                        "truth": section.confidence,
                        "growth": 0.5,  # Placeholder, can be improved
                        "metadata": {
                            "filename": pdf_path.name,
                            "section": section.title,
                            "page": section.page_num,
                            "chunk": chunk_idx,
                            "processed_at": datetime.utcnow().isoformat()
                        }
                    }
                    
                    # Write as JSON line
                    try:
                        logger.info(f"[process_pdf] Writing concept to JSON: {json_path} (section {section_idx} chunk {chunk_idx})")
                        with open(json_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(concept) + "\n")
                        logger.info(f"[process_pdf] Successfully wrote concept for {pdf_path} section {section.title} chunk {chunk_idx}")
                        wrote_any = True
                        seen_chunks.add(chunk_hash)
                    except Exception as file_exc:
                        logger.error(f"[process_pdf] Failed to write concept to JSON: {file_exc}")
                    await asyncio.sleep(0.1)
                    
            # If nothing was written, write a placeholder
            if not wrote_any:
                logger.warning(f"[process_pdf] No data was written for {pdf_path}, writing placeholder.")
                placeholder = {"term": None, "error": "No data processed", "filename": str(pdf_path)}
                with open(json_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(placeholder) + "\n")
            logger.info(f"[process_pdf] Finished processing {pdf_path}")
        except Exception as e:
            logger.error(f"[process_pdf] Error processing PDF {pdf_path}: {e}")

    async def process_all_pdfs(self) -> None:
        """Process all PDFs in the directory."""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf in pdf_files:
            if self.revectorize or not self.is_already_embedded(pdf):
                await self.process_pdf(pdf)
                logger.info(f"Successfully processed {pdf}")

    def is_already_embedded(self, pdf: Path) -> bool:
        """Check if PDF has already been processed."""
        try:
            result = self.qdrant_client.scroll(
                collection_name=self.collection,
                scroll_filter={
                    "must": [
                        {"key": "metadata.filename", "match": {"value": pdf.name}}
                    ]
                },
                limit=1
            )
            return bool(result[0])
        except Exception as e:
            logger.error(f"Error checking if already embedded: {e}")
            return False

async def continuous_pdf_training():
    """Main training loop."""
    trainer = PDFTrainer(revectorize=True)
    
    # Connect to Redis
    await bus.connect()
    
    try:
        while True:
            try:
                logger.info("Starting PDF processing cycle")
                await trainer.process_all_pdfs()
                logger.info(f"Sleeping for {SLEEP_INTERVAL} seconds")
                await asyncio.sleep(SLEEP_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in training loop: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry
    finally:
        await bus.close()

if __name__ == "__main__":
    asyncio.run(continuous_pdf_training())
