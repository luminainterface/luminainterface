import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
import fitz  # PyMuPDF
from pydantic import BaseModel
import aiohttp
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import os

from shared.log_config import setup_logging
from crawler.concept_extractor import ConceptExtractor

logger = setup_logging('pdf-trainer')

class TrainingDocument(BaseModel):
    """Represents a training document with its extracted content and metadata."""
    filename: str
    content: str
    sections: List[Dict[str, str]]  # List of {title: str, content: str}
    metadata: Dict[str, str]
    source: str = "training_manual"
    timestamp: datetime

class PDFTrainer:
    """Handles ingestion and processing of training PDFs."""
    
    def __init__(self, config: Dict, revectorize: bool = False):
        self.config = config
        self.revectorize = revectorize
        self.training_dir = Path(config['training_dir'])
        self.graph_api_url = config['graph_api_url']
        self.vector_api_url = config['vector_api_url']
        self.processed_files = set()
        self.concept_extractor = ConceptExtractor(config)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Unified collection name from config/env
        self.collection_name = os.getenv('QDRANT_COLLECTION', config.get('collection_name', 'unified_knowledge'))
        
        # Initialize Qdrant collection
        # asyncio.run(self._init_qdrant_collection())
        
    async def _init_qdrant_collection(self):
        """Initialize the Qdrant collection for storing document embeddings."""
        try:
            async with aiohttp.ClientSession() as session:
                # Create collection if it doesn't exist
                collection_config = {
                    "name": self.collection_name,
                    "vectors": {
                        "size": 384,  # Size of all-MiniLM-L6-v2 embeddings
                        "distance": "Cosine"
                    }
                }
                
                async with session.put(
                    f"{self.vector_api_url}/collections/{self.collection_name}",
                    json=collection_config
                ) as response:
                    if response.status == 200:
                        logger.info(f"Created Qdrant collection '{self.collection_name}'")
                    elif response.status == 400:
                        logger.info(f"Qdrant collection '{self.collection_name}' already exists")
                    else:
                        logger.error(f"Error creating Qdrant collection: {await response.text()}")
                        
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {str(e)}", exc_info=True)
        
    async def process_training_directory(self):
        """Process all PDF files in the training directory."""
        try:
            pdf_files = list(self.training_dir.glob('**/*.pdf'))
            logger.info(f"Found {len(pdf_files)} PDF files in training directory")
            
            for pdf_file in pdf_files:
                if pdf_file.name not in self.processed_files:
                    await self.process_pdf(pdf_file)
                    self.processed_files.add(pdf_file.name)
                    
        except Exception as e:
            logger.error(f"Error processing training directory: {str(e)}", exc_info=True)
    
    async def process_pdf(self, pdf_path: Path) -> Optional[TrainingDocument]:
        """Process a single PDF file and extract its content."""
        try:
            doc = fitz.open(pdf_path)
            content = ""
            sections = []
            current_section = {"title": "Introduction", "content": ""}
            
            for page in doc:
                text = page.get_text()
                content += text + "\n"
                
                # Simple section detection based on headers
                lines = text.split('\n')
                for line in lines:
                    if line.strip().isupper() and len(line.strip()) < 100:
                        # Save previous section
                        if current_section["content"].strip():
                            sections.append(current_section)
                        # Start new section
                        current_section = {"title": line.strip(), "content": ""}
                    else:
                        current_section["content"] += line + "\n"
            
            # Add last section
            if current_section["content"].strip():
                sections.append(current_section)
            
            training_doc = TrainingDocument(
                filename=pdf_path.name,
                content=content,
                sections=sections,
                metadata={
                    "title": pdf_path.stem,
                    "pages": str(len(doc)),
                    "processed_at": datetime.utcnow().isoformat()
                },
                timestamp=datetime.utcnow()
            )
            
            await self.embed_document(training_doc)
            logger.info(f"Processed training document: {pdf_path.name}")
            return training_doc
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
            return None
    
    async def embed_document(self, doc: TrainingDocument):
        """Embed document content and store in vector database."""
        try:
            async with aiohttp.ClientSession() as session:
                # Embed each section separately for better retrieval
                for section in doc.sections:
                    # Create chunk metadata
                    chunk_metadata = {
                        **doc.metadata,
                        "section": section["title"],
                        "source": doc.source,
                        "filename": doc.filename,
                        "type": "pdf_section"
                    }
                    # Extract concepts from section
                    concepts = await self.concept_extractor.extract_concepts(
                        section["content"],
                        chunk_metadata
                    )
                    chunk_metadata["concepts"] = [c.label for c in concepts]
                    # Generate embedding using sentence-transformers
                    embedding = self.embedding_model.encode(section["content"])
                    # Validate embedding shape
                    expected_dim = self.embedding_model.get_sentence_embedding_dimension()
                    if len(embedding) != expected_dim:
                        logger.error(f"Embedding size mismatch: {len(embedding)} != {expected_dim} for section '{section['title']}' in '{doc.filename}'")
                        continue
                    # Log embedding statistics
                    logger.info(f"Embedding stats for '{doc.filename}' section '{section['title']}': min={embedding.min()}, max={embedding.max()}, mean={embedding.mean()}, std={embedding.std()}")
                    # Store in Qdrant
                    payload = {
                        "points": [{
                            "id": f"{doc.filename}_{section['title']}",
                            "vector": embedding.tolist(),
                            "payload": chunk_metadata
                        }]
                    }
                    async with session.put(
                        f"{self.vector_api_url}/collections/{self.collection_name}/points",
                        json=payload
                    ) as response:
                        if response.status != 200:
                            logger.error(f"Error storing section {section['title']}: {await response.text()}")
                            continue
                        chunk_id = f"{doc.filename}_{section['title']}"
                        # Create relationships between chunk and concepts
                        for concept in concepts:
                            rel_payload = {
                                "id": f"{chunk_id}_{concept.label}",
                                "source": chunk_id,
                                "target": concept.label,
                                "type": "contains",
                                "properties": {
                                    "confidence": concept.confidence
                                }
                            }
                            async with session.post(
                                f"{self.graph_api_url}/edges",
                                json=rel_payload
                            ) as rel_response:
                                if rel_response.status != 200:
                                    logger.error(f"Error creating concept relationship: {await rel_response.text()}")
                # Add document to knowledge graph
                graph_payload = {
                    "id": f"doc_{doc.filename.lower().replace(' ', '_')}",
                    "type": "training_document",
                    "properties": {
                        "filename": doc.filename,
                        "title": doc.metadata["title"],
                        "source": doc.source,
                        "sections": [s["title"] for s in doc.sections],
                        "processed_at": doc.timestamp.isoformat()
                    }
                }
                async with session.post(
                    f"{self.graph_api_url}/nodes",
                    json=graph_payload
                ) as response:
                    if response.status != 200:
                        logger.error(f"Error adding document to graph: {await response.text()}")
            logger.info(f"Successfully embedded document: {doc.filename}")
        except Exception as e:
            logger.error(f"Error embedding document {doc.filename}: {str(e)}", exc_info=True)
    
    async def run(self):
        """Main run loop for the PDF trainer."""
        while True:
            try:
                await self.process_training_directory()
                await asyncio.sleep(self.config.get('scan_interval', 3600))  # Default 1 hour
            except Exception as e:
                logger.error(f"Error in PDF trainer: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry

# Example usage:
# config = {
#     'training_dir': 'training_data',
#     'graph_api_url': 'http://graph-api:8000',
#     'vector_api_url': 'http://vector-api:8000',
#     'scan_interval': 3600
# }
# trainer = PDFTrainer(config)
# asyncio.run(trainer.run()) 