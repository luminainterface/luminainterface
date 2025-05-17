"""PDF processing module for the crawler service."""
import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import PyPDF2
from collections import defaultdict
import re
import asyncio
from functools import partial

logger = logging.getLogger(__name__)

@dataclass
class PDFMetadata:
    """Metadata extracted from a PDF file."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[List[str]] = None
    pages: int = 0
    sections: Optional[Dict[str, List[int]]] = None
    processed_at: Optional[str] = None

class PDFProcessor:
    """Handles PDF file processing and text extraction."""
    
    # Common section headers to detect
    SECTION_HEADERS = {
        'introduction', 'abstract', 'summary', 'background',
        'methods', 'methodology', 'approach', 'implementation',
        'results', 'findings', 'discussion', 'conclusion',
        'references', 'bibliography', 'appendix', 'acknowledgments',
        'related work', 'future work', 'limitations'
    }
    
    # Technical terms to keep during concept extraction
    KEEP_TERMS = {
        'api', 'algorithm', 'architecture', 'backend', 'cache',
        'database', 'deployment', 'design', 'development', 'docker',
        'endpoint', 'framework', 'frontend', 'function', 'graph',
        'implementation', 'interface', 'kubernetes', 'library',
        'microservice', 'model', 'network', 'optimization', 'pattern',
        'performance', 'protocol', 'queue', 'redis', 'rest', 'security',
        'service', 'system', 'test', 'thread', 'vector', 'web'
    }

    def __init__(self):
        """Initialize the PDF processor."""
        self.logger = logging.getLogger(__name__)

    async def _detect_sections(self, pdf_reader: PyPDF2.PdfReader, max_pages: int = 5) -> Dict[str, List[int]]:
        """Detect section headers from the first few pages of the PDF."""
        # Run CPU-bound operation in thread pool
        loop = asyncio.get_event_loop()
        sections = await loop.run_in_executor(
            None,
            partial(self._detect_sections_sync, pdf_reader, max_pages)
        )
        return sections

    def _detect_sections_sync(self, pdf_reader: PyPDF2.PdfReader, max_pages: int = 5) -> Dict[str, List[int]]:
        """Synchronous version of section detection."""
        sections = defaultdict(list)
        pattern = re.compile(r'^(?:\d+\.)?\s*([A-Z][A-Za-z\s]+)(?::|\.|\s*$)', re.MULTILINE)
        
        # Only check first few pages for section headers
        for page_num in range(min(max_pages, len(pdf_reader.pages))):
            text = pdf_reader.pages[page_num].extract_text()
            matches = pattern.finditer(text)
            
            for match in matches:
                header = match.group(1).strip().lower()
                if header in self.SECTION_HEADERS:
                    sections[header].append(page_num + 1)
        
        return dict(sections)

    async def extract_metadata(self, pdf_path: str) -> PDFMetadata:
        """Extract metadata from a PDF file."""
        try:
            # Run file I/O in thread pool
            loop = asyncio.get_event_loop()
            with await loop.run_in_executor(None, open, pdf_path, 'rb') as file:
                pdf_reader = await loop.run_in_executor(None, PyPDF2.PdfReader, file)
                info = pdf_reader.metadata
                
                # Extract basic metadata
                metadata = PDFMetadata(
                    title=info.get('/Title'),
                    author=info.get('/Author'),
                    subject=info.get('/Subject'),
                    keywords=info.get('/Keywords', '').split(',') if info.get('/Keywords') else None,
                    pages=len(pdf_reader.pages)
                )
                
                # Detect sections
                metadata.sections = await self._detect_sections(pdf_reader)
                
                return metadata
                
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
            return PDFMetadata()

    async def extract_text(self, pdf_path: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract text, concepts, and sections from a PDF file."""
        try:
            # Run file I/O in thread pool
            loop = asyncio.get_event_loop()
            with await loop.run_in_executor(None, open, pdf_path, 'rb') as file:
                pdf_reader = await loop.run_in_executor(None, PyPDF2.PdfReader, file)
                text = ""
                concepts = set()
                sections = defaultdict(str)
                current_section = "main"
                
                # Extract text and detect sections
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    # Run CPU-bound text extraction in thread pool
                    page_text = await loop.run_in_executor(None, page.extract_text)
                    if not page_text:
                        continue
                        
                    # Check for section headers
                    lines = page_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Check if line is a section header
                        header_match = re.match(r'^(?:\d+\.)?\s*([A-Z][A-Za-z\s]+)(?::|\.|\s*$)', line)
                        if header_match:
                            header = header_match.group(1).strip().lower()
                            if header in self.SECTION_HEADERS:
                                current_section = header
                                continue
                        
                        # Add text to current section
                        sections[current_section] += line + '\n'
                        
                        # Extract potential concepts
                        words = re.findall(r'\b[A-Za-z][A-Za-z-]+\b', line.lower())
                        concepts.update(word for word in words if word in self.KEEP_TERMS)
                
                # Combine all text
                text = '\n\n'.join(f"Section: {section}\n{content.strip()}"
                                 for section, content in sections.items()
                                 if content.strip())
                
                return text, list(concepts), dict(sections)
                
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return "", [], {}

    async def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF file and return extracted data."""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            # Extract metadata
            metadata = await self.extract_metadata(pdf_path)
            
            # Extract text and concepts
            text, concepts, sections = await self.extract_text(pdf_path)
            
            # Create chunks based on sections
            chunks = []
            for section, content in sections.items():
                if content.strip():
                    chunks.append({
                        'text': content.strip(),
                        'section': section,
                        'metadata': {
                            'source': pdf_path,
                            'section': section,
                            'concepts': [c for c in concepts if c in content.lower()]
                        }
                    })
            
            return {
                'text': text,
                'chunks': chunks,
                'metadata': {
                    'title': metadata.title,
                    'author': metadata.author,
                    'subject': metadata.subject,
                    'keywords': metadata.keywords,
                    'pages': metadata.pages,
                    'sections': metadata.sections,
                    'concepts': list(concepts),
                    'processed_at': metadata.processed_at
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise 