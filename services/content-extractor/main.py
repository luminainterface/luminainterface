from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
import asyncio
import logging
from typing import Dict, Any, Optional
import newspaper
import trafilatura
from langdetect import detect
from bs4 import BeautifulSoup
import pdfminer.high_level
import io
from datetime import datetime
import json
from lumina_core.common.bus import BusClient
from lumina_core.common.retry import with_retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("content-extractor")

# Prometheus metrics
EXTRACT_OK = Counter("extract_ok_total", "Successful extractions", ["type"])
EXTRACT_SKIP = Counter("extract_skip_total", "Skipped extractions", ["reason"])
EXTRACT_FAIL = Counter("extract_fail_total", "Failed extractions")
EXTRACT_SECONDS = Histogram("extract_seconds", "Time spent extracting content",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
# New metrics
EXTRACT_QUALITY = Histogram("extract_quality", "Content quality metrics",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
EXTRACT_LENGTH = Histogram("extract_length", "Content length in characters",
    buckets=[100, 500, 1000, 2000, 5000, 10000, 20000])
EXTRACT_LANGUAGE = Counter("extract_language_total", "Content language distribution", ["lang"])

class Skip(Exception):
    """Raised when content should be skipped"""
    pass

class ContentExtractor:
    def __init__(self, redis_url: str):
        self.bus = BusClient(redis_url=redis_url)
        self.min_content_length = 250
        self.supported_languages = {"en"}
        self.supported_types = {"html", "pdf", "docx", "txt"}
        
    async def connect(self):
        """Connect to Redis and create consumer groups"""
        await self.bus.connect()
        # Create consumer groups for input streams
        for stream in ["ingest.raw_html", "ingest.raw_pdf", "ingest.raw_docx", "ingest.raw_txt"]:
            try:
                await self.bus.create_group(stream, "extractor")
            except Exception as e:
                logger.info(f"Group may exist: {e}")
                
    def calculate_quality_score(self, text: str, title: str = "") -> float:
        """Calculate content quality score based on various factors"""
        score = 0.0
        
        # Length score (0-0.3)
        length_score = min(len(text) / 5000, 1.0) * 0.3
        score += length_score
        
        # Title presence (0-0.2)
        if title and len(title) > 0:
            score += 0.2
            
        # Paragraph structure (0-0.2)
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 3:
            score += 0.2
            
        # Sentence structure (0-0.3)
        sentences = text.split(". ")
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if 5 <= avg_sentence_length <= 25:
            score += 0.3
            
        return min(score, 1.0)
        
    async def extract_text(self, text: str, content_type: str = "txt") -> Dict[str, Any]:
        """Extract and clean content from plain text"""
        try:
            if not text:
                raise Skip("no_content")
                
            # Basic cleaning
            text = " ".join(text.split())  # Normalize whitespace
            
            # Language detection
            lang = detect(text)
            if lang not in self.supported_languages:
                raise Skip(f"unsupported_language:{lang}")
                
            # Length check
            if len(text) < self.min_content_length:
                raise Skip("too_short")
                
            # Calculate quality metrics
            quality_score = self.calculate_quality_score(text)
            EXTRACT_QUALITY.observe(quality_score)
            EXTRACT_LENGTH.observe(len(text))
            EXTRACT_LANGUAGE.labels(lang=lang).inc()
            
            return {
                "text": text,
                "title": "",  # Plain text doesn't have titles
                "lang": lang,
                "type": content_type,
                "quality_score": quality_score
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
            
    async def extract_html(self, html: str) -> Dict[str, Any]:
        """Extract and clean content from HTML"""
        try:
            # Use newspaper3k for initial extraction
            article = newspaper.Article(url="")
            article.set_html(html)
            article.parse()
            
            # Further cleaning with trafilatura
            cleaned = trafilatura.extract(article.html, include_comments=False, 
                                        include_tables=True, deduplicate=True)
            
            if not cleaned:
                raise Skip("no_content")
                
            # Language detection
            lang = detect(cleaned)
            if lang not in self.supported_languages:
                raise Skip(f"unsupported_language:{lang}")
                
            # Length check
            if len(cleaned) < self.min_content_length:
                raise Skip("too_short")
                
            return {
                "text": cleaned,
                "title": article.title,
                "lang": lang,
                "type": "html"
            }
            
        except Exception as e:
            logger.error(f"HTML extraction failed: {e}")
            raise
            
    async def extract_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract and clean content from PDF"""
        try:
            # Extract text using pdfminer
            text = pdfminer.high_level.extract_text(io.BytesIO(pdf_bytes))
            
            if not text:
                raise Skip("no_content")
                
            # Basic cleaning
            text = " ".join(text.split())  # Normalize whitespace
            
            # Language detection
            lang = detect(text)
            if lang not in self.supported_languages:
                raise Skip(f"unsupported_language:{lang}")
                
            # Length check
            if len(text) < self.min_content_length:
                raise Skip("too_short")
                
            return {
                "text": text,
                "title": "",  # PDFs often don't have reliable titles
                "lang": lang,
                "type": "pdf"
            }
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
            
    @with_retry("ingest.raw_html", max_attempts=3, dead_letter_stream="ingest.dlq.html")
    async def process_html(self, msg: Dict[str, Any]):
        """Process HTML content with retry logic and DLQ support"""
        start_time = datetime.now()
        try:
            result = await self.extract_html(msg["html"])
            result["source_id"] = msg.get("source_id")
            result["timestamp"] = datetime.utcnow().isoformat()
            
            # Calculate quality metrics
            quality_score = self.calculate_quality_score(result["text"], result["title"])
            result["quality_score"] = quality_score
            EXTRACT_QUALITY.observe(quality_score)
            EXTRACT_LENGTH.observe(len(result["text"]))
            EXTRACT_LANGUAGE.labels(lang=result["lang"]).inc()
            
            # Publish to cleaned stream
            await self.bus.publish("ingest.cleaned", result)
            
            # Record metrics
            EXTRACT_OK.labels(type="html").inc()
            EXTRACT_SECONDS.observe((datetime.now() - start_time).total_seconds())
            
        except Skip as s:
            EXTRACT_SKIP.labels(reason=str(s)).inc()
            raise
        except Exception as e:
            EXTRACT_FAIL.inc()
            raise
            
    @with_retry("ingest.raw_pdf", max_attempts=3, dead_letter_stream="ingest.dlq.pdf")
    async def process_pdf(self, msg: Dict[str, Any]):
        """Process PDF content with retry logic and DLQ support"""
        start_time = datetime.now()
        try:
            result = await self.extract_pdf(msg["pdf_bytes"])
            result["source_id"] = msg.get("source_id")
            result["timestamp"] = datetime.utcnow().isoformat()
            
            # Calculate quality metrics
            quality_score = self.calculate_quality_score(result["text"])
            result["quality_score"] = quality_score
            EXTRACT_QUALITY.observe(quality_score)
            EXTRACT_LENGTH.observe(len(result["text"]))
            EXTRACT_LANGUAGE.labels(lang=result["lang"]).inc()
            
            # Publish to cleaned stream
            await self.bus.publish("ingest.cleaned", result)
            
            # Record metrics
            EXTRACT_OK.labels(type="pdf").inc()
            EXTRACT_SECONDS.observe((datetime.now() - start_time).total_seconds())
            
        except Skip as s:
            EXTRACT_SKIP.labels(reason=str(s)).inc()
            raise
        except Exception as e:
            EXTRACT_FAIL.inc()
            raise
            
    @with_retry("ingest.raw_docx", max_attempts=3, dead_letter_stream="ingest.dlq.docx")
    async def process_docx(self, msg: Dict[str, Any]):
        """Process DOCX content with retry logic and DLQ support"""
        start_time = datetime.now()
        try:
            # TODO: Implement DOCX extraction using python-docx
            # For now, just extract text
            result = await self.extract_text(msg["text"], "docx")
            result["source_id"] = msg.get("source_id")
            result["timestamp"] = datetime.utcnow().isoformat()
            
            # Publish to cleaned stream
            await self.bus.publish("ingest.cleaned", result)
            
            # Record metrics
            EXTRACT_OK.labels(type="docx").inc()
            EXTRACT_SECONDS.observe((datetime.now() - start_time).total_seconds())
            
        except Skip as s:
            EXTRACT_SKIP.labels(reason=str(s)).inc()
            raise
        except Exception as e:
            EXTRACT_FAIL.inc()
            raise
            
    @with_retry("ingest.raw_txt", max_attempts=3, dead_letter_stream="ingest.dlq.txt")
    async def process_txt(self, msg: Dict[str, Any]):
        """Process TXT content with retry logic and DLQ support"""
        start_time = datetime.now()
        try:
            result = await self.extract_text(msg["text"], "txt")
            result["source_id"] = msg.get("source_id")
            result["timestamp"] = datetime.utcnow().isoformat()
            
            # Publish to cleaned stream
            await self.bus.publish("ingest.cleaned", result)
            
            # Record metrics
            EXTRACT_OK.labels(type="txt").inc()
            EXTRACT_SECONDS.observe((datetime.now() - start_time).total_seconds())
            
        except Skip as s:
            EXTRACT_SKIP.labels(reason=str(s)).inc()
            raise
        except Exception as e:
            EXTRACT_FAIL.inc()
            raise
            
    async def start(self):
        """Start consuming from input streams"""
        while True:
            try:
                # Process all supported content types
                for stream, handler in [
                    ("ingest.raw_html", self.process_html),
                    ("ingest.raw_pdf", self.process_pdf),
                    ("ingest.raw_docx", self.process_docx),
                    ("ingest.raw_txt", self.process_txt)
                ]:
                    await self.bus.consume(
                        stream=stream,
                        group="extractor",
                        consumer="worker",
                        handler=handler,
                        block_ms=1000,
                        count=10
                    )
                    
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)

# FastAPI app
app = FastAPI(title="Content Extractor Service")

@app.on_event("startup")
async def startup():
    """Initialize extractor on startup"""
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    extractor = ContentExtractor(redis_url)
    await extractor.connect()
    app.state.extractor = extractor
    # Start consumer loop
    asyncio.create_task(extractor.start())

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    from prometheus_client import generate_latest
    return Response(generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 