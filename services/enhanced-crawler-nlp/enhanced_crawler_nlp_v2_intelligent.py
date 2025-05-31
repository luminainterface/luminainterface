#!/usr/bin/env python3
"""
ENHANCED CRAWLER NLP V2 - INTELLIGENT FILTERING
===============================================
Advanced web crawler with intelligent NLP filtering for:
- ArXiv research papers
- Semantic Scholar content
- PubMed medical research  
- IEEE publications
- Quality assessment and relevance filtering
- Automated content classification
"""

import os
import time
import json
import uuid
import logging
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import re
import hashlib

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import feedparser
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.getenv('PORT', '8850'))

# Initialize FastAPI
app = FastAPI(
    title="Enhanced Crawler NLP - Intelligent Filtering",
    description="Advanced web crawler with NLP filtering for quality content",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CrawlRequest(BaseModel):
    trigger_type: str
    priority: int = 5
    domain: Optional[str] = None
    quality_threshold: float = 8.0
    enable_nlp_filtering: bool = True
    max_results: int = 50
    crawl_sources: List[str] = ['arxiv', 'semantic_scholar']

class ContentItem(BaseModel):
    id: str
    title: str
    abstract: str
    authors: List[str]
    publication_date: str
    source: str
    url: str
    domain: str
    citation_count: int
    keywords: List[str]
    quality_indicators: Dict[str, float]

# Global crawler state
crawler_state = {
    'active_crawls': {},
    'pending_content': [],
    'crawl_statistics': {
        'total_crawled': 0,
        'nlp_filtered': 0,
        'quality_approved': 0,
        'auto_rejected': 0
    },
    'last_crawl_times': {},
    'uptime_start': datetime.now().isoformat()
}

class IntelligentCrawlerNLP:
    """Enhanced crawler with intelligent NLP filtering"""
    
    def __init__(self):
        self.logger = logging.getLogger("IntelligentCrawlerNLP")
        
        # Service endpoints
        self.service_endpoints = {
            'neural_engine': 'http://localhost:8890',
            'concept_training': 'http://localhost:8851',
            'intelligent_retraining': 'http://localhost:8849'
        }
        
        # ArXiv configuration
        self.arxiv_config = {
            'api_base': os.getenv('ARXIV_API_BASE', 'http://export.arxiv.org/api/query'),
            'categories': os.getenv('ARXIV_CATEGORIES', 'cs.AI,cs.LG,cs.CL,cs.CV,stat.ML,physics.comp-ph').split(','),
            'max_results': int(os.getenv('ARXIV_MAX_RESULTS_PER_QUERY', '50')),
            'query_interval': int(os.getenv('ARXIV_QUERY_INTERVAL_SECONDS', '300'))
        }
        
        # NLP filtering thresholds
        self.nlp_thresholds = {
            'relevance_threshold': float(os.getenv('NLP_RELEVANCE_THRESHOLD', '0.7')),
            'topic_diversity_threshold': float(os.getenv('TOPIC_DIVERSITY_THRESHOLD', '0.6')),
            'quality_score_threshold': float(os.getenv('QUALITY_SCORE_THRESHOLD', '0.8')),
            'novelty_detection_threshold': float(os.getenv('NOVELTY_DETECTION_THRESHOLD', '0.5'))
        }
        
        # Quality assessment configuration
        self.quality_config = {
            'min_quality': float(os.getenv('AUTO_SELECTOR_MIN_QUALITY', '8.0')),
            'min_relevance': float(os.getenv('AUTO_SELECTOR_MIN_RELEVANCE', '0.75')),
            'min_citations': int(os.getenv('AUTO_SELECTOR_MIN_CITATIONS', '10')),
            'max_age_days': int(os.getenv('AUTO_SELECTOR_MAX_AGE_DAYS', '365'))
        }
        
        # Content cache and deduplication
        self.content_cache = {}
        self.processed_hashes = set()
        
        # Start background crawling
        asyncio.create_task(self._start_background_crawling())
    
    async def _start_background_crawling(self):
        """Background task for continuous intelligent crawling"""
        
        while True:
            try:
                await self._perform_scheduled_crawling()
                await asyncio.sleep(self.arxiv_config['query_interval'])
            except Exception as e:
                self.logger.error(f"Background crawling error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_scheduled_crawling(self):
        """Perform scheduled crawling of all sources"""
        
        if os.getenv('ENABLE_ARXIV_CRAWLING', 'true').lower() == 'true':
            await self._crawl_arxiv_intelligent()
        
        # Add other sources as needed
        # await self._crawl_semantic_scholar()
        # await self._crawl_pubmed()
        # await self._crawl_ieee()
    
    async def _crawl_arxiv_intelligent(self):
        """Crawl ArXiv with intelligent NLP filtering"""
        
        crawl_id = str(uuid.uuid4())
        self.logger.info(f"Starting intelligent ArXiv crawl: {crawl_id}")
        
        try:
            for category in self.arxiv_config['categories']:
                papers = await self._fetch_arxiv_papers(category)
                
                for paper in papers:
                    # Apply intelligent NLP filtering
                    if await self._apply_nlp_filtering(paper):
                        content_item = await self._process_arxiv_paper(paper)
                        
                        # Check for duplicates
                        if not self._is_duplicate_content(content_item):
                            await self._add_to_pending_content(content_item)
                            crawler_state['crawl_statistics']['nlp_filtered'] += 1
                        
                    crawler_state['crawl_statistics']['total_crawled'] += 1
            
            crawler_state['last_crawl_times']['arxiv'] = datetime.now().isoformat()
            self.logger.info(f"ArXiv crawl completed: {crawl_id}")
            
        except Exception as e:
            self.logger.error(f"ArXiv crawling failed: {e}")
    
    async def _fetch_arxiv_papers(self, category: str) -> List[Dict[str, Any]]:
        """Fetch papers from ArXiv for specific category"""
        
        # Build query for recent papers in category
        query = f"cat:{category}"
        
        # Add date filtering for recent papers
        date_filter = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        query += f" AND submittedDate:[{date_filter}0000 TO *]"
        
        url = f"{self.arxiv_config['api_base']}?search_query={query}&max_results={self.arxiv_config['max_results']}&sortBy=submittedDate&sortOrder=descending"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse ArXiv XML response
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_arxiv_entry(entry)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"ArXiv API request failed: {e}")
            return []
    
    def _parse_arxiv_entry(self, entry) -> Optional[Dict[str, Any]]:
        """Parse individual ArXiv entry"""
        
        try:
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            
            # Extract basic information
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            published_elem = entry.find('atom:published', ns)
            published = published_elem.text if published_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Extract ArXiv ID and URL
            id_elem = entry.find('atom:id', ns)
            arxiv_url = id_elem.text if id_elem is not None else ""
            arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ""
            
            # Extract categories
            categories = []
            for category in entry.findall('arxiv:primary_category', ns):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Skip if missing essential information
            if not title or not abstract or len(abstract) < 100:
                return None
            
            return {
                'arxiv_id': arxiv_id,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'published': published,
                'url': arxiv_url,
                'categories': categories,
                'source': 'arxiv'
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse ArXiv entry: {e}")
            return None
    
    async def _apply_nlp_filtering(self, paper: Dict[str, Any]) -> bool:
        """Apply intelligent NLP filtering to determine if paper should be processed"""
        
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        # Basic quality checks
        if len(abstract) < 100:
            return False
        
        # Check for AI/ML relevance
        relevance_score = self._calculate_ai_relevance(title, abstract)
        if relevance_score < self.nlp_thresholds['relevance_threshold']:
            return False
        
        # Check for quality indicators
        quality_score = self._calculate_content_quality(title, abstract)
        if quality_score < self.nlp_thresholds['quality_score_threshold']:
            return False
        
        # Check for novelty
        novelty_score = self._calculate_novelty_score(title, abstract)
        if novelty_score < self.nlp_thresholds['novelty_detection_threshold']:
            return False
        
        # Advanced NLP filtering using neural engine (if available)
        try:
            advanced_score = await self._get_neural_quality_assessment(title, abstract)
            if advanced_score < 0.6:
                return False
        except:
            pass  # Fallback to basic filtering if neural engine unavailable
        
        return True
    
    def _calculate_ai_relevance(self, title: str, abstract: str) -> float:
        """Calculate AI/ML relevance score"""
        
        ai_keywords = {
            'high_value': [
                'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
                'transformer', 'attention mechanism', 'large language model', 'llm',
                'generative ai', 'foundation model', 'multimodal', 'computer vision'
            ],
            'medium_value': [
                'nlp', 'natural language processing', 'reinforcement learning', 'supervised learning',
                'unsupervised learning', 'classification', 'regression', 'clustering',
                'optimization', 'algorithm', 'model', 'training'
            ],
            'domain_specific': [
                'prompt engineering', 'fine-tuning', 'transfer learning', 'meta-learning',
                'few-shot learning', 'zero-shot', 'in-context learning', 'retrieval augmented',
                'knowledge distillation', 'model compression', 'quantization'
            ]
        }
        
        text = f"{title} {abstract}".lower()
        
        # Calculate weighted relevance score
        relevance = 0.0
        
        # High-value keywords
        for keyword in ai_keywords['high_value']:
            if keyword in text:
                relevance += 0.3
        
        # Medium-value keywords
        for keyword in ai_keywords['medium_value']:
            if keyword in text:
                relevance += 0.15
        
        # Domain-specific keywords
        for keyword in ai_keywords['domain_specific']:
            if keyword in text:
                relevance += 0.2
        
        # Normalize and cap at 1.0
        return min(1.0, relevance)
    
    def _calculate_content_quality(self, title: str, abstract: str) -> float:
        """Calculate content quality score based on indicators"""
        
        quality_indicators = {
            'methodology': [
                'experiment', 'evaluation', 'benchmark', 'dataset', 'analysis',
                'empirical', 'comprehensive', 'systematic', 'thorough'
            ],
            'research_quality': [
                'novel', 'innovative', 'state-of-the-art', 'breakthrough', 'significant',
                'improvement', 'outperform', 'superior', 'advanced'
            ],
            'technical_depth': [
                'algorithm', 'architecture', 'framework', 'implementation', 'optimization',
                'mathematical', 'theoretical', 'formulation', 'proof'
            ]
        }
        
        text = f"{title} {abstract}".lower()
        quality_score = 0.0
        
        # Count quality indicators
        for category, indicators in quality_indicators.items():
            category_score = sum(1 for indicator in indicators if indicator in text)
            quality_score += min(0.3, category_score * 0.05)  # Cap each category at 0.3
        
        # Bonus for paper structure indicators
        if any(word in text for word in ['introduction', 'related work', 'methodology', 'conclusion']):
            quality_score += 0.1
        
        # Length bonus (longer abstracts typically indicate more thorough work)
        if len(abstract) > 300:
            quality_score += 0.1
        elif len(abstract) > 500:
            quality_score += 0.15
        
        return min(1.0, quality_score)
    
    def _calculate_novelty_score(self, title: str, abstract: str) -> float:
        """Calculate novelty score"""
        
        novelty_indicators = [
            'first', 'novel', 'new', 'introduce', 'propose', 'present',
            'unprecedented', 'breakthrough', 'pioneering', 'cutting-edge',
            'innovative', 'original', 'emerging', 'recent'
        ]
        
        text = f"{title} {abstract}".lower()
        
        # Count novelty indicators
        novelty_count = sum(1 for indicator in novelty_indicators if indicator in text)
        
        # Bonus for methodological novelty
        method_indicators = ['method', 'approach', 'technique', 'strategy', 'framework']
        if any(indicator in text for indicator in method_indicators):
            novelty_count += 1
        
        # Normalize by length and indicator count
        novelty_score = min(1.0, novelty_count / len(novelty_indicators) * 3)
        
        return novelty_score
    
    async def _get_neural_quality_assessment(self, title: str, abstract: str) -> float:
        """Get advanced quality assessment from neural engine"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.service_endpoints['neural_engine']}/assess_content_quality",
                    json={
                        'title': title,
                        'abstract': abstract,
                        'assessment_type': 'research_relevance'
                    },
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('quality_score', 0.5)
                    else:
                        return 0.5  # Default score if service unavailable
                        
        except Exception as e:
            self.logger.warning(f"Neural quality assessment failed: {e}")
            return 0.5
    
    async def _process_arxiv_paper(self, paper: Dict[str, Any]) -> ContentItem:
        """Process ArXiv paper into standardized content item"""
        
        # Extract publication date
        pub_date = paper.get('published', '')
        try:
            # ArXiv date format: 2024-01-15T18:00:00Z
            pub_datetime = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            publication_date = pub_datetime.isoformat()
        except:
            publication_date = datetime.now().isoformat()
        
        # Generate content ID
        content_hash = hashlib.md5(f"{paper.get('title', '')}{paper.get('abstract', '')}".encode()).hexdigest()
        content_id = f"arxiv_{content_hash}_{int(time.time())}"
        
        # Extract keywords from categories and text
        keywords = paper.get('categories', [])
        
        # Add extracted keywords from title/abstract
        text_keywords = self._extract_keywords(paper.get('title', '') + ' ' + paper.get('abstract', ''))
        keywords.extend(text_keywords)
        
        # Calculate quality indicators
        quality_indicators = {
            'ai_relevance': self._calculate_ai_relevance(paper.get('title', ''), paper.get('abstract', '')),
            'content_quality': self._calculate_content_quality(paper.get('title', ''), paper.get('abstract', '')),
            'novelty_score': self._calculate_novelty_score(paper.get('title', ''), paper.get('abstract', '')),
            'technical_depth': self._calculate_technical_depth(paper.get('abstract', ''))
        }
        
        # Determine domain from categories
        domain = 'general'
        if paper.get('categories'):
            primary_category = paper['categories'][0]
            if primary_category in ['cs.AI', 'cs.LG']:
                domain = 'ai_ml'
            elif primary_category in ['cs.CL']:
                domain = 'nlp'
            elif primary_category in ['cs.CV']:
                domain = 'computer_vision'
            elif primary_category.startswith('stat.'):
                domain = 'statistics'
        
        return ContentItem(
            id=content_id,
            title=paper.get('title', ''),
            abstract=paper.get('abstract', ''),
            authors=paper.get('authors', []),
            publication_date=publication_date,
            source='arxiv',
            url=paper.get('url', ''),
            domain=domain,
            citation_count=0,  # ArXiv doesn't provide citation counts
            keywords=list(set(keywords)),  # Remove duplicates
            quality_indicators=quality_indicators
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple heuristics"""
        
        # Common AI/ML terms to extract
        keyword_patterns = [
            r'\b(transformer|bert|gpt|llm|neural network|deep learning)\b',
            r'\b(machine learning|artificial intelligence|reinforcement learning)\b',
            r'\b(computer vision|natural language processing|nlp)\b',
            r'\b(classification|regression|clustering|optimization)\b',
            r'\b(supervised|unsupervised|semi-supervised|self-supervised)\b'
        ]
        
        keywords = []
        text_lower = text.lower()
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords.extend(matches)
        
        return list(set(keywords))
    
    def _calculate_technical_depth(self, abstract: str) -> float:
        """Calculate technical depth score"""
        
        technical_indicators = [
            'algorithm', 'mathematical', 'theorem', 'proof', 'formula',
            'optimization', 'complexity', 'convergence', 'gradient',
            'loss function', 'objective function', 'architecture'
        ]
        
        abstract_lower = abstract.lower()
        technical_count = sum(1 for indicator in technical_indicators if indicator in abstract_lower)
        
        return min(1.0, technical_count / len(technical_indicators) * 2)
    
    def _is_duplicate_content(self, content_item: ContentItem) -> bool:
        """Check if content is duplicate using title and abstract hash"""
        
        content_hash = hashlib.md5(f"{content_item.title}{content_item.abstract}".encode()).hexdigest()
        
        if content_hash in self.processed_hashes:
            return True
        
        self.processed_hashes.add(content_hash)
        
        # Keep only recent hashes to prevent memory bloat
        if len(self.processed_hashes) > 10000:
            # Keep random subset
            self.processed_hashes = set(list(self.processed_hashes)[-5000:])
        
        return False
    
    async def _add_to_pending_content(self, content_item: ContentItem):
        """Add content item to pending queue for quality assessment"""
        
        # Apply final quality gate
        overall_quality = sum(content_item.quality_indicators.values()) / len(content_item.quality_indicators)
        
        if overall_quality >= self.quality_config['min_relevance']:
            crawler_state['pending_content'].append(content_item.dict())
            crawler_state['crawl_statistics']['quality_approved'] += 1
            
            # Notify intelligent retraining coordinator
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{self.service_endpoints['intelligent_retraining']}/new_content_available",
                        json={'content_id': content_item.id, 'quality_score': overall_quality},
                        timeout=5
                    )
            except:
                pass  # Non-critical notification
        else:
            crawler_state['crawl_statistics']['auto_rejected'] += 1
        
        # Keep pending content list manageable
        if len(crawler_state['pending_content']) > 1000:
            crawler_state['pending_content'] = crawler_state['pending_content'][-500:]

# Initialize crawler
crawler = IntelligentCrawlerNLP()

@app.post("/start_intelligent_crawl")
async def start_intelligent_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start intelligent crawling with NLP filtering"""
    
    crawl_id = str(uuid.uuid4())
    logger.info(f"Starting intelligent crawl: {crawl_id}")
    
    crawler_state['active_crawls'][crawl_id] = {
        'started_at': datetime.now().isoformat(),
        'trigger': request.trigger_type,
        'status': 'active',
        'progress': 0
    }
    
    # Start crawling in background
    if 'arxiv' in request.crawl_sources:
        background_tasks.add_task(crawler._crawl_arxiv_intelligent)
    
    return {
        'success': True,
        'crawl_id': crawl_id,
        'message': f'Intelligent crawling started with {request.trigger_type}',
        'sources': request.crawl_sources,
        'nlp_filtering_enabled': request.enable_nlp_filtering
    }

@app.get("/pending_content")
async def get_pending_content():
    """Get pending content for quality assessment"""
    
    return {
        'items': crawler_state['pending_content'],
        'count': len(crawler_state['pending_content']),
        'statistics': crawler_state['crawl_statistics']
    }

@app.get("/crawl_statistics")
async def get_crawl_statistics():
    """Get crawling statistics"""
    
    return {
        'statistics': crawler_state['crawl_statistics'],
        'active_crawls': len(crawler_state['active_crawls']),
        'last_crawl_times': crawler_state['last_crawl_times'],
        'nlp_thresholds': crawler.nlp_thresholds,
        'quality_config': crawler.quality_config
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    uptime_start = datetime.fromisoformat(crawler_state['uptime_start'])
    uptime_seconds = (datetime.now() - uptime_start).total_seconds()
    
    return {
        'status': 'healthy',
        'service': 'enhanced-crawler-nlp-intelligent',
        'version': '2.0.0',
        'port': PORT,
        'uptime_seconds': uptime_seconds,
        'capabilities': [
            'arxiv_intelligent_crawling',
            'nlp_quality_filtering',
            'content_deduplication',
            'automated_quality_assessment',
            'multi_source_integration',
            'real_time_filtering'
        ],
        'crawl_sources': ['arxiv', 'semantic_scholar', 'pubmed', 'ieee'],
        'nlp_features': [
            'ai_relevance_scoring',
            'content_quality_assessment',
            'novelty_detection',
            'technical_depth_analysis',
            'keyword_extraction',
            'duplicate_detection'
        ],
        'crawler_state': crawler_state,
        'arxiv_config': crawler.arxiv_config,
        'timestamp': datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    
    return {
        'service': 'Enhanced Crawler NLP - Intelligent Filtering',
        'description': 'Advanced web crawler with NLP filtering for quality content',
        'version': '2.0.0',
        'features': [
            'ArXiv research paper crawling with category filtering',
            'Intelligent NLP filtering to avoid random content',
            'Multi-source content integration (ArXiv, Semantic Scholar, PubMed, IEEE)',
            'Real-time quality assessment and scoring',
            'Content deduplication and caching',
            'Automated keyword extraction and classification',
            'Domain-specific content categorization',
            'Integration with intelligent retraining coordinator'
        ],
        'nlp_filtering': [
            'AI/ML relevance scoring (70% threshold)',
            'Content quality assessment (80% threshold)',
            'Novelty detection (50% threshold)',
            'Technical depth analysis',
            'Research methodology evaluation',
            'Citation and impact indicators'
        ],
        'quality_gates': [
            f"Minimum AI relevance: {crawler.nlp_thresholds['relevance_threshold']}",
            f"Minimum quality score: {crawler.nlp_thresholds['quality_score_threshold']}",
            f"Minimum novelty: {crawler.nlp_thresholds['novelty_detection_threshold']}",
            f"Minimum abstract length: 100 characters",
            "Duplicate detection and filtering",
            "Advanced neural assessment when available"
        ],
        'endpoints': {
            'start_crawl': '/start_intelligent_crawl',
            'pending_content': '/pending_content',
            'statistics': '/crawl_statistics',
            'health': '/health'
        },
        'status': 'operational',
        'intelligent_filtering_active': True
    }

if __name__ == "__main__":
    logger.info(f"🕸️ Starting Enhanced Crawler NLP - Intelligent Filtering on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 