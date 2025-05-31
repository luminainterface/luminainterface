#!/usr/bin/env python3
"""
RESEARCH PAPER ORCHESTRATOR WITH LORA LEARNING
==============================================

Specialized orchestrator for research paper generation that adapts the
revolutionary LoRA learning concepts from the math-focused orchestrator
to text generation and academic content quality improvement.

This system:
1. Tracks content quality patterns
2. Identifies weak areas in academic writing
3. Creates LoRA-style training examples for improvement
4. Uses adaptive routing for different content types
5. Implements recursive learning for content enhancement

Goal: Achieve A+ (97+/100) research paper quality
"""

import asyncio
import aiohttp
import json
import time
import logging
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
import statistics

class ContentType(Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"

class QualityRoute(Enum):
    BASIC_GENERATION = "basic_generation"
    ENHANCED_STRUCTURE = "enhanced_structure"
    CITATION_FOCUS = "citation_focus"
    FULL_ORCHESTRATION = "full_orchestration"

@dataclass
class ContentPerformanceRecord:
    timestamp: str
    content_query: str
    content_hash: str
    content_type: str
    complexity_score: float
    quality_route: str
    
    basic_content: str
    basic_quality_score: float
    basic_word_count: int
    basic_time: float
    
    enhanced_content: str
    enhanced_quality_score: float
    enhanced_word_count: int
    enhanced_time: float
    
    improvement_gained: bool
    quality_delta: float
    content_target: str
    weakness_pattern: Optional[str] = None
    needs_lora_training: bool = False

@dataclass
class ContentLoRAExample:
    input_query: str
    weak_content: str
    strong_content: str
    improvement_techniques: List[str]
    content_type: str
    quality_improvement: str

class ResearchContentOrchestrator:
    """Specialized orchestrator for research content with LoRA learning"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_database()
        
        # Performance tracking
        self.performance_history: List[ContentPerformanceRecord] = []
        self.lora_training_queue: List[ContentLoRAExample] = []
        
        # Quality routing thresholds
        self.quality_thresholds = {
            QualityRoute.BASIC_GENERATION: 0.80,
            QualityRoute.ENHANCED_STRUCTURE: 0.85,
            QualityRoute.CITATION_FOCUS: 0.90,
            QualityRoute.FULL_ORCHESTRATION: 0.95
        }
        
        # Content weakness patterns (for LoRA training)
        self.weakness_patterns = {
            'insufficient_citations': {'count': 0, 'lora_needed': True},
            'weak_technical_depth': {'count': 0, 'lora_needed': True},
            'poor_structure': {'count': 0, 'lora_needed': True},
            'informal_tone': {'count': 0, 'lora_needed': True},
            'inadequate_length': {'count': 0, 'lora_needed': False}
        }
        
        # Service endpoints
        self.services = {
            'enhanced_execution': 'http://localhost:8998',
            'ollama': 'http://localhost:11434'
        }
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ResearchContentOrchestrator")
        self.logger.info("📝 Research Content Orchestrator with LoRA Learning initialized")
    
    def setup_database(self):
        """Setup SQLite database for content performance tracking"""
        self.db_path = "research_content_performance.db"
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS content_performance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                content_hash TEXT,
                content_query TEXT,
                content_type TEXT,
                complexity_score REAL,
                quality_route TEXT,
                basic_quality_score REAL,
                basic_word_count INTEGER,
                basic_time REAL,
                enhanced_quality_score REAL,
                enhanced_word_count INTEGER,
                enhanced_time REAL,
                improvement_gained INTEGER,
                quality_delta REAL,
                weakness_pattern TEXT,
                needs_lora_training INTEGER
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS content_lora_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_query TEXT,
                weak_content TEXT,
                strong_content TEXT,
                improvement_techniques TEXT,
                content_type TEXT,
                quality_improvement TEXT,
                used_for_training INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info("📊 Content performance tracking database initialized")
    
    def assess_content_complexity(self, query: str, target_quality: float) -> Tuple[float, ContentType]:
        """Assess content complexity and type for routing decisions"""
        query_lower = query.lower()
        
        # Determine content type
        content_type = ContentType.INTRODUCTION  # Default
        
        if 'abstract' in query_lower:
            content_type = ContentType.ABSTRACT
        elif 'introduction' in query_lower:
            content_type = ContentType.INTRODUCTION
        elif 'literature' in query_lower or 'review' in query_lower:
            content_type = ContentType.LITERATURE_REVIEW
        elif 'methodology' in query_lower or 'method' in query_lower:
            content_type = ContentType.METHODOLOGY
        elif 'result' in query_lower:
            content_type = ContentType.RESULTS
        elif 'discussion' in query_lower:
            content_type = ContentType.DISCUSSION
        elif 'conclusion' in query_lower:
            content_type = ContentType.CONCLUSION
        
        # Base complexity calculation
        complexity = 1.0
        
        # Complexity factors
        word_target_indicators = re.findall(r'\d+\s*words?', query_lower)
        if word_target_indicators:
            try:
                target_words = int(re.findall(r'\d+', word_target_indicators[0])[0])
                if target_words > 800:
                    complexity += 1.0
                elif target_words > 500:
                    complexity += 0.5
            except:
                pass
        
        # Academic requirements boost complexity
        academic_requirements = [
            'citation', 'reference', 'publication', 'academic', 'research',
            'methodology', 'analysis', 'comprehensive', 'technical'
        ]
        
        academic_score = sum(1 for req in academic_requirements if req in query_lower)
        complexity += academic_score * 0.2
        
        # Target quality affects complexity
        complexity += (target_quality - 0.8) * 2  # Quality above 0.8 increases complexity
        
        return min(complexity, 5.0), content_type
    
    def decide_quality_routing(self, complexity: float, content_type: ContentType, target_quality: float) -> QualityRoute:
        """Decide routing based on complexity and target quality"""
        
        # Route based on target quality primarily
        if target_quality >= 0.95:
            return QualityRoute.FULL_ORCHESTRATION
        elif target_quality >= 0.90:
            return QualityRoute.CITATION_FOCUS
        elif target_quality >= 0.85:
            return QualityRoute.ENHANCED_STRUCTURE
        else:
            return QualityRoute.BASIC_GENERATION
    
    async def basic_content_generation(self, query: str) -> Dict[str, Any]:
        """Basic content generation using Ollama"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "llama3.2:1b",
                    "prompt": query,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }
                
                async with session.post(f"{self.services['ollama']}/api/generate", 
                                      json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result.get('response', '').strip()
                        
                        return {
                            'success': True,
                            'content': content,
                            'quality_score': self.calculate_basic_quality(content),
                            'processing_time': time.time() - start_time,
                            'method': 'basic_generation',
                            'word_count': len(content.split())
                        }
        except Exception as e:
            self.logger.error(f"Basic content generation failed: {e}")
        
        return {
            'success': False,
            'content': '',
            'quality_score': 0.0,
            'processing_time': time.time() - start_time,
            'method': 'basic_generation',
            'word_count': 0
        }
    
    async def enhanced_structure_generation(self, query: str, content_type: ContentType) -> Dict[str, Any]:
        """Enhanced generation with improved structure"""
        start_time = time.time()
        
        # Add structure guidance to query
        structure_guidance = self.get_structure_guidance(content_type)
        enhanced_query = f"""
        {query}
        
        Structure guidance:
        {structure_guidance}
        
        Ensure proper academic structure, logical flow, and appropriate paragraph breaks.
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "llama3.2:1b",
                    "prompt": enhanced_query,
                    "stream": False,
                    "options": {
                        "temperature": 0.6,
                        "top_p": 0.8
                    }
                }
                
                async with session.post(f"{self.services['ollama']}/api/generate", 
                                      json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result.get('response', '').strip()
                        
                        return {
                            'success': True,
                            'content': content,
                            'quality_score': self.calculate_enhanced_quality(content, content_type),
                            'processing_time': time.time() - start_time,
                            'method': 'enhanced_structure',
                            'word_count': len(content.split())
                        }
        except Exception as e:
            self.logger.error(f"Enhanced structure generation failed: {e}")
        
        # Fallback to basic generation
        return await self.basic_content_generation(query)
    
    async def citation_focused_generation(self, query: str, content_type: ContentType) -> Dict[str, Any]:
        """Generation with focus on citations and academic rigor"""
        start_time = time.time()
        
        citation_guidance = """
        Include proper academic citations in your response. Use formats like:
        - (Author et al., 2023)
        - Recent research demonstrates... (Smith, 2024)
        - According to Johnson et al. (2023)...
        
        Ensure technical depth and academic tone throughout.
        """
        
        enhanced_query = f"""
        {query}
        
        {citation_guidance}
        
        Structure guidance:
        {self.get_structure_guidance(content_type)}
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                # Try with better model settings for academic content
                payload = {
                    "model": "llama3.2:1b",
                    "prompt": enhanced_query,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "top_p": 0.7,
                        "top_k": 40
                    }
                }
                
                async with session.post(f"{self.services['ollama']}/api/generate", 
                                      json=payload, timeout=90) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result.get('response', '').strip()
                        
                        return {
                            'success': True,
                            'content': content,
                            'quality_score': self.calculate_citation_quality(content, content_type),
                            'processing_time': time.time() - start_time,
                            'method': 'citation_focused',
                            'word_count': len(content.split())
                        }
        except Exception as e:
            self.logger.error(f"Citation focused generation failed: {e}")
        
        # Fallback to enhanced structure
        return await self.enhanced_structure_generation(query, content_type)
    
    async def full_orchestration_generation(self, query: str, content_type: ContentType, context: Dict[str, Any]) -> Dict[str, Any]:
        """Full orchestration using the enhanced execution suite"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query": query,
                    "context": {
                        **context,
                        "content_type": content_type.value,
                        "orchestration_mode": "research_content",
                        "quality_focus": "academic_excellence",
                        "reasoning_requirement": "comprehensive"
                    }
                }
                
                async with session.post(f"{self.services['enhanced_execution']}/execute",
                                      json=payload, timeout=120) as response:
                    if response.status == 200:
                        result = await response.json()
                        final_result = result.get('final_result', {})
                        
                        content = final_result.get('response', '')
                        quality_score = self.calculate_orchestration_quality(content, content_type)
                        
                        return {
                            'success': True,
                            'content': content,
                            'quality_score': quality_score,
                            'processing_time': time.time() - start_time,
                            'method': 'full_orchestration',
                            'word_count': len(content.split()),
                            'orchestration_metadata': result
                        }
        except Exception as e:
            self.logger.error(f"Full orchestration failed: {e}")
        
        # Fallback to citation focused
        return await self.citation_focused_generation(query, content_type)
    
    def get_structure_guidance(self, content_type: ContentType) -> str:
        """Get structure guidance for specific content types"""
        
        guidance_map = {
            ContentType.ABSTRACT: """
            - Opening statement (background/context)
            - Problem statement and research gap
            - Methodology overview
            - Key findings or contributions
            - Implications and significance
            """,
            ContentType.INTRODUCTION: """
            - Contextual background
            - Problem evolution and current state
            - Research gap identification
            - Research objectives and questions
            - Methodological approach overview
            - Paper structure and contributions
            """,
            ContentType.LITERATURE_REVIEW: """
            - Theoretical foundations
            - Historical development
            - Current state-of-the-art analysis
            - Comparative analysis of approaches
            - Research gaps and opportunities
            - Positioning of current work
            """,
            ContentType.METHODOLOGY: """
            - Research design and approach
            - Data collection procedures
            - Analysis methods and tools
            - Validation and verification
            - Limitations and assumptions
            """
        }
        
        return guidance_map.get(content_type, "Ensure logical flow and proper academic structure.")
    
    def calculate_basic_quality(self, content: str) -> float:
        """Calculate basic quality score"""
        if not content:
            return 0.0
        
        quality_score = 0.0
        word_count = len(content.split())
        
        # Word count factor (0-0.3)
        if word_count >= 100:
            quality_score += 0.3
        elif word_count >= 50:
            quality_score += 0.2
        else:
            quality_score += 0.1
        
        # Academic vocabulary (0-0.2)
        academic_words = ['research', 'analysis', 'methodology', 'framework', 'approach', 'significant']
        academic_count = sum(1 for word in academic_words if word.lower() in content.lower())
        quality_score += min(0.2, academic_count * 0.05)
        
        # Structure indicators (0-0.2)
        structure_words = ['furthermore', 'however', 'therefore', 'additionally', 'moreover']
        structure_count = sum(1 for word in structure_words if word.lower() in content.lower())
        quality_score += min(0.2, structure_count * 0.05)
        
        # Paragraph structure (0-0.15)
        paragraphs = content.split('\n\n')
        if len(paragraphs) >= 3:
            quality_score += 0.15
        elif len(paragraphs) >= 2:
            quality_score += 0.10
        
        # Technical depth (0-0.15)
        technical_words = ['implementation', 'evaluation', 'optimization', 'algorithm', 'performance']
        technical_count = sum(1 for word in technical_words if word.lower() in content.lower())
        quality_score += min(0.15, technical_count * 0.04)
        
        return min(quality_score, 1.0)
    
    def calculate_enhanced_quality(self, content: str, content_type: ContentType) -> float:
        """Calculate enhanced quality score with structure consideration"""
        base_quality = self.calculate_basic_quality(content)
        
        # Enhanced structure bonus (0-0.1)
        structure_bonus = 0.0
        if content_type == ContentType.ABSTRACT and len(content.split()) >= 200:
            structure_bonus += 0.05
        elif content_type == ContentType.INTRODUCTION and len(content.split()) >= 400:
            structure_bonus += 0.05
        elif content_type == ContentType.LITERATURE_REVIEW and len(content.split()) >= 600:
            structure_bonus += 0.05
        
        # Logical flow indicators
        flow_indicators = ['first', 'second', 'finally', 'in conclusion', 'consequently']
        flow_count = sum(1 for indicator in flow_indicators if indicator.lower() in content.lower())
        structure_bonus += min(0.05, flow_count * 0.02)
        
        return min(base_quality + structure_bonus, 1.0)
    
    def calculate_citation_quality(self, content: str, content_type: ContentType) -> float:
        """Calculate quality with citation and academic rigor focus"""
        enhanced_quality = self.calculate_enhanced_quality(content, content_type)
        
        # Citation bonus (0-0.15)
        citation_patterns = ['et al.', '(20', '(19', 'doi:', 'Retrieved from']
        citation_count = sum(1 for pattern in citation_patterns if pattern in content)
        citation_bonus = min(0.15, citation_count * 0.03)
        
        # Academic tone bonus (0-0.1)
        academic_tone_words = ['demonstrates', 'indicates', 'suggests', 'reveals', 'establishes']
        tone_count = sum(1 for word in academic_tone_words if word.lower() in content.lower())
        tone_bonus = min(0.1, tone_count * 0.025)
        
        return min(enhanced_quality + citation_bonus + tone_bonus, 1.0)
    
    def calculate_orchestration_quality(self, content: str, content_type: ContentType) -> float:
        """Calculate highest quality score for orchestrated content"""
        citation_quality = self.calculate_citation_quality(content, content_type)
        
        # Orchestration excellence bonus (0-0.1)
        excellence_indicators = [
            'comprehensive', 'substantial', 'innovative', 'systematic',
            'rigorous', 'thorough', 'extensive', 'detailed'
        ]
        excellence_count = sum(1 for word in excellence_indicators if word.lower() in content.lower())
        excellence_bonus = min(0.1, excellence_count * 0.02)
        
        # Complexity and depth bonus (0-0.05)
        complexity_indicators = ['framework', 'architecture', 'paradigm', 'taxonomy']
        complexity_count = sum(1 for word in complexity_indicators if word.lower() in content.lower())
        complexity_bonus = min(0.05, complexity_count * 0.015)
        
        return min(citation_quality + excellence_bonus + complexity_bonus, 1.0)
    
    def check_content_quality_target(self, content: str, target_description: str) -> bool:
        """Check if content meets the target quality description"""
        # Simple heuristic checking
        word_count = len(content.split())
        
        # Extract word target if mentioned
        word_targets = re.findall(r'(\d+)\s*words?', target_description.lower())
        if word_targets:
            target_words = int(word_targets[0])
            # Content should be within 70-130% of target
            if word_count < target_words * 0.7 or word_count > target_words * 1.3:
                return False
        
        # Check for key requirements
        if 'comprehensive' in target_description.lower() and word_count < 200:
            return False
        
        if 'citation' in target_description.lower() and 'et al.' not in content and '(20' not in content:
            return False
        
        return len(content.strip()) > 50  # Minimum content requirement
    
    def identify_weakness_pattern(self, content: str, content_type: ContentType, target_quality: float, actual_quality: float) -> Optional[str]:
        """Identify weakness patterns for LoRA training"""
        
        # Only identify weaknesses if there's a significant quality gap
        if actual_quality >= target_quality * 0.9:
            return None
        
        # Check for specific weaknesses
        if 'et al.' not in content and '(20' not in content and content_type in [ContentType.LITERATURE_REVIEW, ContentType.INTRODUCTION]:
            return 'insufficient_citations'
        
        technical_words = ['framework', 'methodology', 'analysis', 'implementation']
        if not any(word in content.lower() for word in technical_words):
            return 'weak_technical_depth'
        
        if len(content.split('\n\n')) < 2:
            return 'poor_structure'
        
        informal_indicators = ['we think', 'i believe', 'personally', 'in my opinion']
        if any(phrase in content.lower() for phrase in informal_indicators):
            return 'informal_tone'
        
        word_count = len(content.split())
        if content_type == ContentType.ABSTRACT and word_count < 150:
            return 'inadequate_length'
        elif content_type == ContentType.INTRODUCTION and word_count < 300:
            return 'inadequate_length'
        elif content_type == ContentType.LITERATURE_REVIEW and word_count < 400:
            return 'inadequate_length'
        
        return None
    
    async def smart_content_orchestrate(self, query: str, target_description: str, target_quality: float = 0.90, context: Dict[str, Any] = None) -> ContentPerformanceRecord:
        """Smart content orchestration with adaptive routing and performance tracking"""
        
        if context is None:
            context = {}
        
        content_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        timestamp = datetime.now().isoformat()
        
        # Step 1: Assess complexity and content type
        complexity, content_type = self.assess_content_complexity(query, target_quality)
        quality_route = self.decide_quality_routing(complexity, content_type, target_quality)
        
        self.logger.info(f"📝 Content query: {query[:60]}...")
        self.logger.info(f"📊 Complexity: {complexity:.1f}, Type: {content_type.value}")
        self.logger.info(f"🎯 Quality route: {quality_route.value}")
        
        # Step 2: Generate basic content for comparison
        basic_start = time.time()
        basic_result = await self.basic_content_generation(query)
        basic_time = time.time() - basic_start
        basic_meets_target = self.check_content_quality_target(basic_result.get('content', ''), target_description)
        
        # Step 3: Generate enhanced content based on routing
        enhanced_result = basic_result  # Default to basic
        
        if quality_route == QualityRoute.BASIC_GENERATION:
            enhanced_result = basic_result
        elif quality_route == QualityRoute.ENHANCED_STRUCTURE:
            enhanced_result = await self.enhanced_structure_generation(query, content_type)
        elif quality_route == QualityRoute.CITATION_FOCUS:
            enhanced_result = await self.citation_focused_generation(query, content_type)
        elif quality_route == QualityRoute.FULL_ORCHESTRATION:
            enhanced_result = await self.full_orchestration_generation(query, content_type, context)
        
        enhanced_meets_target = self.check_content_quality_target(enhanced_result.get('content', ''), target_description)
        
        # Step 4: Calculate improvements
        improvement_gained = enhanced_meets_target and not basic_meets_target
        quality_delta = enhanced_result.get('quality_score', 0.0) - basic_result.get('quality_score', 0.0)
        
        # Step 5: Identify weakness patterns
        weakness_pattern = self.identify_weakness_pattern(
            enhanced_result.get('content', ''),
            content_type,
            target_quality,
            enhanced_result.get('quality_score', 0.0)
        )
        
        if weakness_pattern:
            self.weakness_patterns[weakness_pattern]['count'] += 1
        
        # Step 6: Create performance record
        record = ContentPerformanceRecord(
            timestamp=timestamp,
            content_query=query,
            content_hash=content_hash,
            content_type=content_type.value,
            complexity_score=complexity,
            quality_route=quality_route.value,
            
            basic_content=basic_result.get('content', ''),
            basic_quality_score=basic_result.get('quality_score', 0.0),
            basic_word_count=basic_result.get('word_count', 0),
            basic_time=basic_time,
            
            enhanced_content=enhanced_result.get('content', ''),
            enhanced_quality_score=enhanced_result.get('quality_score', 0.0),
            enhanced_word_count=enhanced_result.get('word_count', 0),
            enhanced_time=enhanced_result.get('processing_time', 0.0),
            
            improvement_gained=improvement_gained,
            quality_delta=quality_delta,
            content_target=target_description,
            weakness_pattern=weakness_pattern,
            needs_lora_training=improvement_gained or (weakness_pattern is not None)
        )
        
        # Step 7: Save performance record and create LoRA example if beneficial
        self.save_content_performance_record(record)
        
        if record.needs_lora_training and enhanced_meets_target and not basic_meets_target:
            improvement_techniques = self.extract_improvement_techniques(
                basic_result.get('content', ''),
                enhanced_result.get('content', ''),
                quality_route
            )
            
            lora_example = ContentLoRAExample(
                input_query=query,
                weak_content=basic_result.get('content', ''),
                strong_content=enhanced_result.get('content', ''),
                improvement_techniques=improvement_techniques,
                content_type=content_type.value,
                quality_improvement=f"Quality improved from {basic_result.get('quality_score', 0.0):.3f} to {enhanced_result.get('quality_score', 0.0):.3f}"
            )
            
            self.lora_training_queue.append(lora_example)
            self.save_content_lora_example(lora_example)
        
        return record
    
    def extract_improvement_techniques(self, weak_content: str, strong_content: str, quality_route: QualityRoute) -> List[str]:
        """Extract improvement techniques used"""
        techniques = []
        
        # Analyze differences between weak and strong content
        weak_words = set(weak_content.lower().split())
        strong_words = set(strong_content.lower().split())
        
        new_words = strong_words - weak_words
        
        # Check for specific improvements
        if 'et al.' in strong_content and 'et al.' not in weak_content:
            techniques.append('Added academic citations')
        
        if len(strong_content.split()) > len(weak_content.split()) * 1.2:
            techniques.append('Expanded content depth')
        
        if quality_route in [QualityRoute.ENHANCED_STRUCTURE, QualityRoute.FULL_ORCHESTRATION]:
            techniques.append('Improved structural organization')
        
        if quality_route in [QualityRoute.CITATION_FOCUS, QualityRoute.FULL_ORCHESTRATION]:
            techniques.append('Enhanced academic rigor')
        
        academic_improvements = new_words.intersection({'framework', 'methodology', 'analysis', 'comprehensive', 'systematic'})
        if academic_improvements:
            techniques.append('Increased technical vocabulary')
        
        return techniques
    
    def save_content_performance_record(self, record: ContentPerformanceRecord):
        """Save content performance record to database"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO content_performance_records 
            (timestamp, content_hash, content_query, content_type, complexity_score, quality_route,
             basic_quality_score, basic_word_count, basic_time,
             enhanced_quality_score, enhanced_word_count, enhanced_time,
             improvement_gained, quality_delta, weakness_pattern, needs_lora_training)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.timestamp, record.content_hash, record.content_query, record.content_type,
            record.complexity_score, record.quality_route,
            record.basic_quality_score, record.basic_word_count, record.basic_time,
            record.enhanced_quality_score, record.enhanced_word_count, record.enhanced_time,
            int(record.improvement_gained), record.quality_delta, record.weakness_pattern,
            int(record.needs_lora_training)
        ))
        
        conn.commit()
        conn.close()
    
    def save_content_lora_example(self, example: ContentLoRAExample):
        """Save content LoRA example to database"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO content_lora_examples
            (timestamp, input_query, weak_content, strong_content, improvement_techniques,
             content_type, quality_improvement)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            example.input_query,
            example.weak_content,
            example.strong_content,
            json.dumps(example.improvement_techniques),
            example.content_type,
            example.quality_improvement
        ))
        
        conn.commit()
        conn.close()
    
    def get_content_improvement_statistics(self) -> Dict[str, Any]:
        """Get content improvement statistics from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Overall statistics
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_content_queries,
                AVG(CAST(improvement_gained as FLOAT)) as improvement_rate,
                AVG(quality_delta) as avg_quality_delta,
                AVG(enhanced_time) as avg_processing_time
            FROM content_performance_records
        """)
        
        overall_stats = cursor.fetchone()
        
        # By content type
        cursor = conn.execute("""
            SELECT 
                content_type,
                COUNT(*) as count,
                AVG(CAST(improvement_gained as FLOAT)) as improvement_rate,
                AVG(enhanced_quality_score) as avg_quality
            FROM content_performance_records
            GROUP BY content_type
        """)
        
        by_type_stats = cursor.fetchall()
        
        # By quality route
        cursor = conn.execute("""
            SELECT 
                quality_route,
                COUNT(*) as count,
                AVG(CAST(improvement_gained as FLOAT)) as improvement_rate,
                AVG(enhanced_time) as avg_time
            FROM content_performance_records
            GROUP BY quality_route
        """)
        
        by_route_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            'overall': {
                'total_content_queries': overall_stats[0] if overall_stats[0] else 0,
                'improvement_rate': overall_stats[1] if overall_stats[1] else 0.0,
                'avg_quality_delta': overall_stats[2] if overall_stats[2] else 0.0,
                'avg_processing_time': overall_stats[3] if overall_stats[3] else 0.0
            },
            'by_content_type': {row[0]: {'count': row[1], 'improvement_rate': row[2], 'avg_quality': row[3]} 
                               for row in by_type_stats},
            'by_quality_route': {row[0]: {'count': row[1], 'improvement_rate': row[2], 'avg_time': row[3]} 
                                for row in by_route_stats},
            'weakness_patterns': dict(self.weakness_patterns),
            'lora_training_examples': len(self.lora_training_queue)
        }

# Test the system
async def test_research_content_orchestrator():
    """Test the research content orchestrator"""
    
    print("📝 RESEARCH CONTENT ORCHESTRATOR WITH LORA LEARNING")
    print("=" * 60)
    
    orchestrator = ResearchContentOrchestrator()
    
    # Test queries for different content types
    test_queries = [
        {
            'query': 'Write a comprehensive Abstract section for a research paper on "Distributed AI Orchestration: A Comprehensive Analysis of Multi-Service Architectures". Target length: 300 words. Include background, problem statement, methodology, findings, and implications.',
            'target': 'comprehensive abstract with 300 words, academic structure',
            'quality': 0.95
        },
        {
            'query': 'Write a comprehensive Introduction section for a research paper on "Distributed AI Orchestration: A Comprehensive Analysis of Multi-Service Architectures". Target length: 800 words. Include contextual background, research gap, objectives, and paper structure.',
            'target': 'comprehensive introduction with 800 words, proper citations',
            'quality': 0.90
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}: {test['query'][:50]}...")
        
        # Run smart orchestration
        record = await orchestrator.smart_content_orchestrate(
            query=test['query'],
            target_description=test['target'],
            target_quality=test['quality']
        )
        
        print(f"✅ Content generated!")
        print(f"   Quality: {record.enhanced_quality_score:.3f}")
        print(f"   Words: {record.enhanced_word_count}")
        print(f"   Route: {record.quality_route}")
        print(f"   Improvement: {record.improvement_gained}")
        
        if record.weakness_pattern:
            print(f"   Weakness: {record.weakness_pattern}")
    
    # Show statistics
    stats = orchestrator.get_content_improvement_statistics()
    print(f"\n📊 ORCHESTRATOR STATISTICS:")
    print(f"   Total queries: {stats['overall']['total_content_queries']}")
    print(f"   Improvement rate: {stats['overall']['improvement_rate']:.1%}")
    print(f"   LoRA examples: {stats['lora_training_examples']}")

if __name__ == "__main__":
    asyncio.run(test_research_content_orchestrator()) 