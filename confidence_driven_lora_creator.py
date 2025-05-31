#!/usr/bin/env python3
"""
CONFIDENCE-DRIVEN LORA CREATOR
==============================
Intelligent system that:
- Monitors AI confidence levels in real-time
- Detects knowledge gaps and uncertainty patterns
- Automatically creates targeted LoRAs when confidence drops
- Learns from "I don't know" responses to improve system knowledge
- Prioritizes gap-filling based on query frequency and importance
"""

import os
import time
import json
import uuid
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import statistics
import re
from collections import defaultdict, Counter

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = int(os.getenv('PORT', '8848'))

# Initialize FastAPI
app = FastAPI(
    title="Confidence-Driven LoRA Creator",
    description="Creates LoRAs automatically when AI confidence drops below thresholds",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConfidenceLevel(str, Enum):
    HIGH = "high"          # 0.8+
    MEDIUM = "medium"      # 0.5-0.8
    LOW = "low"           # 0.2-0.5
    VERY_LOW = "very_low"  # <0.2
    UNKNOWN = "unknown"    # explicit "I don't know"

class GapSeverity(str, Enum):
    CRITICAL = "critical"    # Core functionality affected
    HIGH = "high"           # Important feature missing
    MEDIUM = "medium"       # Nice-to-have knowledge
    LOW = "low"            # Minor enhancement

class ConfidenceRequest(BaseModel):
    query: str
    response: str
    confidence_score: Optional[float] = None
    response_time: float
    model_used: str
    context: Optional[Dict[str, Any]] = None

class KnowledgeGap(BaseModel):
    gap_id: str
    query: str
    domain: str
    gap_type: str  # factual, procedural, conceptual, technical
    severity: GapSeverity
    confidence_score: float
    frequency: int
    first_detected: str
    last_detected: str
    failed_attempts: int
    keywords: List[str]
    suggested_sources: List[str]

# Global state for confidence tracking
confidence_state = {
    'knowledge_gaps': {},           # gap_id -> KnowledgeGap
    'confidence_history': [],       # Recent confidence scores
    'query_patterns': defaultdict(int),  # query pattern -> frequency
    'domain_confidence': {},        # domain -> avg confidence
    'active_lora_requests': {},     # lora_id -> creation status
    'gap_priorities': [],          # Prioritized list of gaps to address
    'learning_statistics': {
        'total_queries_analyzed': 0,
        'low_confidence_detected': 0,
        'loras_created_from_gaps': 0,
        'knowledge_gaps_filled': 0,
        'avg_confidence_improvement': 0.0
    }
}

class ConfidenceDrivenLoRACreator:
    """Main class for confidence-driven LoRA creation"""
    
    def __init__(self):
        self.logger = logging.getLogger("ConfidenceDrivenLoRACreator")
        
        # Service endpoints
        self.service_endpoints = {
            'enhanced_crawler': 'http://localhost:8850',
            'concept_training': 'http://localhost:8851',
            'lora_coordination': 'http://localhost:8995',
            'ultimate_chat': 'http://localhost:8950',
            'neural_engine': 'http://localhost:8890',
            'retraining_coordinator': 'http://localhost:8849',
            'llm_gap_detector': 'http://localhost:8997',
            'rag_coordination': 'http://localhost:8952'
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'lora_creation_threshold': 0.3,      # Create LoRA if confidence below this
            'urgent_lora_threshold': 0.1,       # High priority if below this
            'gap_detection_threshold': 0.5,     # Start tracking as potential gap
            'frequency_multiplier': 3            # Frequency before considering urgent
        }
        
        # Domain classification patterns
        self.domain_patterns = {
            'ai_ml': ['machine learning', 'artificial intelligence', 'neural network', 'deep learning', 'algorithm'],
            'quantum_computing': ['quantum', 'qubit', 'superposition', 'entanglement', 'quantum computing'],
            'medicine': ['medical', 'health', 'disease', 'treatment', 'diagnosis', 'drug', 'clinical'],
            'technology': ['software', 'programming', 'computer', 'technology', 'system', 'development'],
            'science': ['research', 'study', 'experiment', 'scientific', 'physics', 'chemistry', 'biology'],
            'finance': ['financial', 'money', 'investment', 'banking', 'economic', 'market'],
            'general': []  # fallback
        }
        
        # Uncertainty indicators
        self.uncertainty_patterns = [
            r"i don't know",
            r"i'm not sure",
            r"i don't have enough information",
            r"i cannot provide",
            r"i'm not familiar with",
            r"i don't have access to",
            r"i'm unable to",
            r"i lack information",
            r"i'm not certain",
            r"i cannot determine",
            r"unclear",
            r"uncertain",
            r"possibly",
            r"might be",
            r"could be",
            r"perhaps",
            r"it's difficult to say"
        ]
        
        # Start background monitoring
        asyncio.create_task(self._start_confidence_monitoring())
        asyncio.create_task(self._start_gap_analysis())
    
    async def _start_confidence_monitoring(self):
        """Background task for monitoring confidence patterns"""
        
        while True:
            try:
                await self._analyze_confidence_trends()
                await self._update_domain_confidence()
                await self._prioritize_knowledge_gaps()
                await asyncio.sleep(120)  # Check every 2 minutes
            except Exception as e:
                self.logger.error(f"Confidence monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _start_gap_analysis(self):
        """Background task for analyzing and addressing knowledge gaps"""
        
        while True:
            try:
                await self._process_high_priority_gaps()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Gap analysis error: {e}")
                await asyncio.sleep(600)
    
    async def process_confidence_assessment(self, request: ConfidenceRequest) -> Dict[str, Any]:
        """Process a confidence assessment and detect potential gaps"""
        
        query = request.query.lower()
        response = request.response.lower()
        
        # Update statistics
        confidence_state['learning_statistics']['total_queries_analyzed'] += 1
        
        # Determine confidence level
        confidence_level, detected_confidence = self._analyze_response_confidence(
            request.response, request.confidence_score
        )
        
        # Add to confidence history
        confidence_state['confidence_history'].append({
            'timestamp': datetime.now().isoformat(),
            'query': request.query,
            'confidence_score': detected_confidence,
            'confidence_level': confidence_level.value,
            'model_used': request.model_used,
            'response_time': request.response_time
        })
        
        # Keep only recent history
        if len(confidence_state['confidence_history']) > 1000:
            confidence_state['confidence_history'] = confidence_state['confidence_history'][-500:]
        
        # Detect knowledge gap if confidence is low
        gap_detected = False
        gap_info = None
        
        if confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW, ConfidenceLevel.UNKNOWN]:
            gap_info = await self._detect_and_record_knowledge_gap(request, confidence_level, detected_confidence)
            gap_detected = True
            confidence_state['learning_statistics']['low_confidence_detected'] += 1
        
        # Check if LoRA creation should be triggered
        lora_triggered = False
        if (detected_confidence < self.confidence_thresholds['lora_creation_threshold'] or 
            confidence_level == ConfidenceLevel.UNKNOWN):
            
            lora_triggered = await self._trigger_confidence_driven_lora(gap_info or {
                'query': request.query,
                'domain': self._classify_domain(request.query),
                'confidence_score': detected_confidence
            })
        
        return {
            'confidence_assessment': {
                'confidence_level': confidence_level.value,
                'confidence_score': detected_confidence,
                'gap_detected': gap_detected,
                'lora_triggered': lora_triggered
            },
            'gap_info': gap_info,
            'recommendations': self._generate_learning_recommendations(request, detected_confidence)
        }
    
    def _analyze_response_confidence(self, response: str, provided_confidence: Optional[float]) -> Tuple[ConfidenceLevel, float]:
        """Analyze response to determine confidence level"""
        
        response_lower = response.lower()
        
        # Check for explicit uncertainty indicators
        uncertainty_score = 0
        for pattern in self.uncertainty_patterns:
            if re.search(pattern, response_lower):
                uncertainty_score += 1
        
        # If many uncertainty indicators, it's unknown/very low confidence
        if uncertainty_score >= 3:
            return ConfidenceLevel.UNKNOWN, 0.0
        elif uncertainty_score >= 2:
            return ConfidenceLevel.VERY_LOW, 0.1
        elif uncertainty_score >= 1:
            return ConfidenceLevel.LOW, 0.3
        
        # Use provided confidence if available
        if provided_confidence is not None:
            if provided_confidence >= 0.8:
                return ConfidenceLevel.HIGH, provided_confidence
            elif provided_confidence >= 0.5:
                return ConfidenceLevel.MEDIUM, provided_confidence
            elif provided_confidence >= 0.2:
                return ConfidenceLevel.LOW, provided_confidence
            else:
                return ConfidenceLevel.VERY_LOW, provided_confidence
        
        # Analyze response characteristics
        response_length = len(response.split())
        
        # Very short responses often indicate uncertainty
        if response_length < 10:
            return ConfidenceLevel.LOW, 0.3
        
        # Look for hedging language
        hedging_patterns = ['probably', 'likely', 'possibly', 'might', 'could', 'seems', 'appears']
        hedging_count = sum(1 for pattern in hedging_patterns if pattern in response_lower)
        
        if hedging_count >= 3:
            return ConfidenceLevel.MEDIUM, 0.6
        elif hedging_count >= 1:
            return ConfidenceLevel.MEDIUM, 0.7
        
        # Default to medium confidence if no clear indicators
        return ConfidenceLevel.MEDIUM, 0.6
    
    async def _detect_and_record_knowledge_gap(self, request: ConfidenceRequest, 
                                              confidence_level: ConfidenceLevel, 
                                              confidence_score: float) -> Dict[str, Any]:
        """Detect and record a knowledge gap"""
        
        query = request.query
        domain = self._classify_domain(query)
        keywords = self._extract_keywords(query)
        
        # Generate gap ID based on content similarity
        gap_id = self._generate_gap_id(query, domain, keywords)
        
        current_time = datetime.now().isoformat()
        
        # Check if gap already exists
        if gap_id in confidence_state['knowledge_gaps']:
            # Update existing gap
            existing_gap = confidence_state['knowledge_gaps'][gap_id]
            existing_gap['frequency'] += 1
            existing_gap['last_detected'] = current_time
            existing_gap['failed_attempts'] += 1
            
            # Update confidence score (moving average)
            existing_gap['confidence_score'] = (
                existing_gap['confidence_score'] * 0.7 + confidence_score * 0.3
            )
            
            gap_info = existing_gap
        else:
            # Create new gap
            severity = self._determine_gap_severity(query, domain, confidence_level)
            
            gap_info = {
                'gap_id': gap_id,
                'query': query,
                'domain': domain,
                'gap_type': self._classify_gap_type(query),
                'severity': severity.value,
                'confidence_score': confidence_score,
                'frequency': 1,
                'first_detected': current_time,
                'last_detected': current_time,
                'failed_attempts': 1,
                'keywords': keywords,
                'suggested_sources': self._suggest_learning_sources(domain, keywords)
            }
            
            confidence_state['knowledge_gaps'][gap_id] = gap_info
        
        # Update query patterns
        pattern = self._normalize_query_pattern(query)
        confidence_state['query_patterns'][pattern] += 1
        
        return gap_info
    
    def _classify_domain(self, query: str) -> str:
        """Classify the domain of a query"""
        
        query_lower = query.lower()
        
        for domain, patterns in self.domain_patterns.items():
            if domain == 'general':
                continue
            
            matches = sum(1 for pattern in patterns if pattern in query_lower)
            if matches > 0:
                return domain
        
        return 'general'
    
    def _classify_gap_type(self, query: str) -> str:
        """Classify the type of knowledge gap"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'process', 'procedure', 'steps']):
            return 'procedural'
        elif any(word in query_lower for word in ['what', 'define', 'explain', 'concept']):
            return 'conceptual'
        elif any(word in query_lower for word in ['why', 'reason', 'cause', 'because']):
            return 'causal'
        elif any(word in query_lower for word in ['technical', 'implementation', 'code', 'algorithm']):
            return 'technical'
        else:
            return 'factual'
    
    def _determine_gap_severity(self, query: str, domain: str, confidence_level: ConfidenceLevel) -> GapSeverity:
        """Determine the severity of a knowledge gap"""
        
        # Core AI/ML queries are critical
        if domain == 'ai_ml' and confidence_level == ConfidenceLevel.UNKNOWN:
            return GapSeverity.CRITICAL
        
        # Medical information is high priority
        if domain == 'medicine':
            return GapSeverity.HIGH
        
        # Technical implementation gaps are medium to high
        if 'technical' in self._classify_gap_type(query):
            return GapSeverity.MEDIUM if confidence_level == ConfidenceLevel.LOW else GapSeverity.HIGH
        
        # Default based on confidence level
        if confidence_level == ConfidenceLevel.UNKNOWN:
            return GapSeverity.HIGH
        elif confidence_level == ConfidenceLevel.VERY_LOW:
            return GapSeverity.MEDIUM
        else:
            return GapSeverity.LOW
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        
        # Simple keyword extraction
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'why', 'when', 'where', 'which', 'who'}
        
        words = re.findall(r'\w+', text.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Get most frequent/important words
        return list(set(keywords))[:10]
    
    def _generate_gap_id(self, query: str, domain: str, keywords: List[str]) -> str:
        """Generate a unique ID for a knowledge gap"""
        
        # Create ID based on domain and key concepts
        key_concepts = sorted(keywords[:3])  # Top 3 keywords
        id_string = f"{domain}_{'-'.join(key_concepts)}_{hash(query) % 10000}"
        return id_string
    
    def _normalize_query_pattern(self, query: str) -> str:
        """Normalize query to identify patterns"""
        
        # Simple pattern normalization
        normalized = re.sub(r'\b\d+\b', 'NUMBER', query.lower())
        normalized = re.sub(r'\b[a-z]+ing\b', 'VERB', normalized)
        return normalized[:100]  # Limit length
    
    def _suggest_learning_sources(self, domain: str, keywords: List[str]) -> List[str]:
        """Suggest sources for learning about the gap"""
        
        sources = []
        
        # Domain-specific sources
        if domain == 'ai_ml':
            sources.extend(['arxiv.org', 'papers.nips.cc', 'openai.com/research'])
        elif domain == 'quantum_computing':
            sources.extend(['quantum-computing.ibm.com', 'arxiv.org/list/quant-ph'])
        elif domain == 'medicine':
            sources.extend(['pubmed.ncbi.nlm.nih.gov', 'nejm.org'])
        elif domain == 'technology':
            sources.extend(['github.com', 'stackoverflow.com', 'ieee.org'])
        
        # General academic sources
        sources.extend(['scholar.google.com', 'researchgate.net', 'semanticscholar.org'])
        
        return sources[:5]
    
    async def _trigger_confidence_driven_lora(self, gap_info: Dict[str, Any]) -> bool:
        """Trigger LoRA creation for a knowledge gap"""
        
        try:
            lora_request_id = f"confidence_driven_{gap_info.get('gap_id', str(uuid.uuid4()))}"
            
            # Prepare LoRA creation request
            lora_request = {
                'trigger_type': 'confidence_driven',
                'gap_info': gap_info,
                'priority': 8 if gap_info.get('severity') in ['critical', 'high'] else 5,
                'domain_focus': gap_info.get('domain', 'general'),
                'target_keywords': gap_info.get('keywords', []),
                'learning_sources': gap_info.get('suggested_sources', [])
            }
            
            # Send to enhanced crawler for content gathering
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.service_endpoints['enhanced_crawler']}/start_intelligent_crawl",
                    json={
                        'trigger_type': f'confidence_gap_{lora_request_id}',
                        'priority': lora_request['priority'],
                        'domain': gap_info.get('domain'),
                        'quality_threshold': 8.5,
                        'enable_nlp_filtering': True,
                        'max_results': 20,
                        'crawl_sources': ['arxiv'],
                        'target_keywords': gap_info.get('keywords', [])
                    },
                    timeout=15
                ) as response:
                    if response.status == 200:
                        # Track the LoRA request
                        confidence_state['active_lora_requests'][lora_request_id] = {
                            'request_id': lora_request_id,
                            'gap_id': gap_info.get('gap_id'),
                            'status': 'crawling',
                            'started_at': datetime.now().isoformat(),
                            'domain': gap_info.get('domain'),
                            'keywords': gap_info.get('keywords', [])
                        }
                        
                        confidence_state['learning_statistics']['loras_created_from_gaps'] += 1
                        
                        self.logger.info(f"Triggered confidence-driven LoRA creation: {lora_request_id}")
                        return True
                    else:
                        self.logger.error(f"Failed to trigger LoRA crawling: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Failed to trigger confidence-driven LoRA: {e}")
            return False
    
    def _generate_learning_recommendations(self, request: ConfidenceRequest, confidence_score: float) -> List[str]:
        """Generate recommendations for improving knowledge"""
        
        recommendations = []
        
        if confidence_score < 0.3:
            recommendations.append("Consider specialized training on this topic")
            recommendations.append("Gather more comprehensive training data")
        
        if confidence_score < 0.5:
            recommendations.append("Supplement with domain-specific knowledge sources")
            recommendations.append("Cross-reference with multiple authoritative sources")
        
        domain = self._classify_domain(request.query)
        if domain != 'general':
            recommendations.append(f"Focus on {domain} domain expertise enhancement")
        
        return recommendations
    
    async def _analyze_confidence_trends(self):
        """Analyze confidence trends to identify patterns"""
        
        if len(confidence_state['confidence_history']) < 10:
            return
        
        recent_scores = [
            entry['confidence_score'] 
            for entry in confidence_state['confidence_history'][-50:]
        ]
        
        avg_confidence = statistics.mean(recent_scores)
        confidence_trend = 'stable'
        
        if len(recent_scores) >= 20:
            first_half = statistics.mean(recent_scores[:len(recent_scores)//2])
            second_half = statistics.mean(recent_scores[len(recent_scores)//2:])
            
            if second_half > first_half * 1.1:
                confidence_trend = 'improving'
            elif second_half < first_half * 0.9:
                confidence_trend = 'declining'
        
        # Store trend analysis
        confidence_state['confidence_trends'] = {
            'avg_confidence': avg_confidence,
            'trend': confidence_trend,
            'last_updated': datetime.now().isoformat()
        }
    
    async def _update_domain_confidence(self):
        """Update confidence levels by domain"""
        
        domain_scores = defaultdict(list)
        
        for entry in confidence_state['confidence_history'][-100:]:
            domain = self._classify_domain(entry['query'])
            domain_scores[domain].append(entry['confidence_score'])
        
        for domain, scores in domain_scores.items():
            if scores:
                confidence_state['domain_confidence'][domain] = {
                    'avg_confidence': statistics.mean(scores),
                    'sample_size': len(scores),
                    'last_updated': datetime.now().isoformat()
                }
    
    async def _prioritize_knowledge_gaps(self):
        """Prioritize knowledge gaps for LoRA creation"""
        
        gaps = list(confidence_state['knowledge_gaps'].values())
        
        # Score gaps based on frequency, severity, and recency
        scored_gaps = []
        current_time = datetime.now()
        
        for gap in gaps:
            last_detected = datetime.fromisoformat(gap['last_detected'])
            days_since = (current_time - last_detected).days
            
            # Recency factor (more recent = higher score)
            recency_factor = max(0.1, 1.0 - (days_since / 30))
            
            # Severity multiplier
            severity_multiplier = {
                'critical': 4.0,
                'high': 3.0,
                'medium': 2.0,
                'low': 1.0
            }.get(gap['severity'], 1.0)
            
            # Frequency factor
            frequency_factor = min(gap['frequency'] / 10, 2.0)
            
            priority_score = (
                (1.0 - gap['confidence_score']) * 
                severity_multiplier * 
                frequency_factor * 
                recency_factor
            )
            
            scored_gaps.append({
                'gap': gap,
                'priority_score': priority_score
            })
        
        # Sort by priority score
        scored_gaps.sort(key=lambda x: x['priority_score'], reverse=True)
        confidence_state['gap_priorities'] = scored_gaps
    
    async def _process_high_priority_gaps(self):
        """Process high-priority gaps for LoRA creation"""
        
        if not confidence_state['gap_priorities']:
            return
        
        # Process top gaps that haven't been addressed recently
        for scored_gap in confidence_state['gap_priorities'][:5]:
            gap = scored_gap['gap']
            gap_id = gap['gap_id']
            
            # Check if already being processed
            if any(req['gap_id'] == gap_id for req in confidence_state['active_lora_requests'].values()):
                continue
            
            # Check priority threshold
            if scored_gap['priority_score'] > 2.0:  # High priority threshold
                self.logger.info(f"Processing high-priority gap: {gap_id}")
                await self._trigger_confidence_driven_lora(gap)

# Initialize the creator
creator = ConfidenceDrivenLoRACreator()

@app.post("/assess_confidence")
async def assess_confidence(request: ConfidenceRequest):
    """Assess confidence and potentially trigger LoRA creation"""
    
    logger.info(f"Assessing confidence for query: {request.query[:100]}...")
    
    result = await creator.process_confidence_assessment(request)
    
    return {
        'success': True,
        'assessment_result': result,
        'timestamp': datetime.now().isoformat()
    }

@app.post("/report_uncertainty")
async def report_uncertainty(query: str, response: str, context: Dict[str, Any] = None):
    """Report when AI explicitly says it doesn't know something"""
    
    logger.info(f"Uncertainty reported for: {query[:100]}...")
    
    # Create confidence request with unknown confidence
    confidence_request = ConfidenceRequest(
        query=query,
        response=response,
        confidence_score=0.0,  # Explicit uncertainty
        response_time=0.0,
        model_used="unknown",
        context=context or {}
    )
    
    result = await creator.process_confidence_assessment(confidence_request)
    
    return {
        'success': True,
        'message': 'Uncertainty noted and LoRA creation triggered if needed',
        'assessment_result': result,
        'automatic_lora_triggered': result['confidence_assessment']['lora_triggered']
    }

@app.get("/knowledge_gaps")
async def get_knowledge_gaps():
    """Get current knowledge gaps"""
    
    return {
        'total_gaps': len(confidence_state['knowledge_gaps']),
        'knowledge_gaps': list(confidence_state['knowledge_gaps'].values()),
        'gap_priorities': confidence_state['gap_priorities'][:10],  # Top 10
        'domain_distribution': Counter([gap['domain'] for gap in confidence_state['knowledge_gaps'].values()])
    }

@app.get("/confidence_analytics")
async def get_confidence_analytics():
    """Get confidence analytics and trends"""
    
    return {
        'confidence_trends': confidence_state.get('confidence_trends', {}),
        'domain_confidence': confidence_state['domain_confidence'],
        'learning_statistics': confidence_state['learning_statistics'],
        'recent_assessments': confidence_state['confidence_history'][-20:],
        'active_lora_requests': confidence_state['active_lora_requests'],
        'top_query_patterns': dict(confidence_state['query_patterns'].most_common(10))
    }

@app.post("/gap_filled")
async def gap_filled(gap_id: str, improvement_score: float):
    """Notify that a knowledge gap has been filled"""
    
    if gap_id in confidence_state['knowledge_gaps']:
        # Archive the gap
        gap = confidence_state['knowledge_gaps'][gap_id]
        gap['filled_at'] = datetime.now().isoformat()
        gap['improvement_score'] = improvement_score
        
        # Update statistics
        confidence_state['learning_statistics']['knowledge_gaps_filled'] += 1
        
        # Update average improvement
        current_avg = confidence_state['learning_statistics']['avg_confidence_improvement']
        total_filled = confidence_state['learning_statistics']['knowledge_gaps_filled']
        new_avg = ((current_avg * (total_filled - 1)) + improvement_score) / total_filled
        confidence_state['learning_statistics']['avg_confidence_improvement'] = new_avg
        
        # Remove from active gaps
        del confidence_state['knowledge_gaps'][gap_id]
        
        logger.info(f"Knowledge gap filled: {gap_id} (improvement: {improvement_score})")
        
        return {
            'success': True,
            'message': f'Gap {gap_id} marked as filled',
            'improvement_score': improvement_score
        }
    
    return {
        'success': False,
        'message': f'Gap {gap_id} not found'
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    return {
        'status': 'healthy',
        'service': 'confidence-driven-lora-creator',
        'version': '1.0.0',
        'port': PORT,
        'capabilities': [
            'real_time_confidence_assessment',
            'knowledge_gap_detection',
            'automatic_lora_triggering',
            'uncertainty_pattern_recognition',
            'domain_specific_gap_analysis',
            'priority_based_learning',
            'confidence_trend_analysis'
        ],
        'gap_detection_features': [
            'explicit_uncertainty_detection',
            'confidence_threshold_monitoring',
            'query_pattern_analysis',
            'domain_classification',
            'gap_severity_assessment',
            'learning_source_suggestions'
        ],
        'statistics': confidence_state['learning_statistics'],
        'active_monitoring': {
            'total_gaps_tracked': len(confidence_state['knowledge_gaps']),
            'active_lora_requests': len(confidence_state['active_lora_requests']),
            'confidence_history_size': len(confidence_state['confidence_history'])
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    
    return {
        'service': 'Confidence-Driven LoRA Creator',
        'description': 'Creates LoRAs automatically when AI confidence drops below thresholds',
        'version': '1.0.0',
        'key_features': [
            'Real-time confidence monitoring and gap detection',
            'Automatic LoRA creation when AI says "I don\'t know"',
            'Domain-specific knowledge gap analysis',
            'Priority-based learning queue management',
            'Uncertainty pattern recognition and learning',
            'Confidence trend analysis and improvement tracking'
        ],
        'confidence_thresholds': creator.confidence_thresholds,
        'supported_domains': list(creator.domain_patterns.keys()),
        'gap_severity_levels': [level.value for level in GapSeverity],
        'confidence_levels': [level.value for level in ConfidenceLevel],
        'uncertainty_indicators': len(creator.uncertainty_patterns),
        'workflow': [
            '1. Monitor AI responses for confidence indicators',
            '2. Detect explicit uncertainty ("I don\'t know" phrases)',
            '3. Classify knowledge gap by domain and severity',
            '4. Prioritize gaps based on frequency and importance',
            '5. Trigger targeted content crawling for gap topics',
            '6. Create specialized LoRAs to fill knowledge gaps',
            '7. Validate improvement and track learning progress'
        ],
        'endpoints': {
            'assess_confidence': '/assess_confidence',
            'report_uncertainty': '/report_uncertainty',
            'knowledge_gaps': '/knowledge_gaps',
            'confidence_analytics': '/confidence_analytics',
            'gap_filled': '/gap_filled',
            'health': '/health'
        },
        'integration_ready': True,
        'intelligent_gap_detection_active': True
    }

if __name__ == "__main__":
    logger.info(f"🧠💡 Starting Confidence-Driven LoRA Creator on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 