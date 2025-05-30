#!/usr/bin/env python3
"""
🧠 NEURAL THOUGHT ENGINE - ENHANCED CONCEPT DETECTION & REASONING
Core AI processing service for the Ultimate AI Orchestration Architecture v10

This service provides:
- Advanced concept detection & pattern recognition
- Multi-modal reasoning & thought generation  
- Strategic thought coordination with enhanced systems
- Integration with phi-2 and enhanced reasoning pipelines
"""

import os
import sys
import json
import time
import redis
import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThoughtRequest(BaseModel):
    """Thought generation request"""
    prompt: str
    context: Optional[str] = None
    max_tokens: int = 200
    temperature: float = 0.7
    concept_detection: bool = True
    reasoning_mode: str = "enhanced"  # enhanced, basic, creative, analytical

class ConceptDetectionRequest(BaseModel):
    """Concept detection request"""
    text: str
    detection_depth: str = "deep"  # surface, deep, comprehensive
    context_awareness: bool = True

@dataclass
class ConceptMatch:
    """Concept detection result"""
    concept: str
    confidence: float
    context: str
    reasoning: str
    timestamp: datetime

class NeuralThoughtEngine:
    """🧠 Neural Thought Engine - Enhanced Concept Detection & Reasoning"""
    
    def __init__(self):
        self.redis_client = self._setup_redis()
        self.phi2_host = os.getenv('PHI2_HOST', 'localhost')
        self.phi2_port = int(os.getenv('PHI2_PORT', 8892))
        self.rag_coordination_host = os.getenv('RAG_COORDINATION_HOST', 'localhost')
        self.rag_coordination_port = int(os.getenv('RAG_COORDINATION_PORT', 8952))
        
        # Reasoning parameters
        self.concept_threshold = float(os.getenv('CONCEPT_THRESHOLD', 0.7))
        self.reasoning_depth = float(os.getenv('REASONING_DEPTH', 0.8))
        self.creativity_factor = float(os.getenv('CREATIVITY_FACTOR', 0.6))
        self.analytical_weight = float(os.getenv('ANALYTICAL_WEIGHT', 0.75))
        
        # Concept detection patterns
        self.concept_patterns = {
            "technical": ["algorithm", "implementation", "optimization", "architecture", "framework"],
            "problem_solving": ["issue", "challenge", "solution", "approach", "strategy"],
            "creative": ["design", "innovation", "artistic", "creative", "imagination"],
            "analytical": ["analysis", "evaluation", "assessment", "comparison", "metrics"],
            "philosophical": ["ethics", "meaning", "purpose", "existence", "consciousness"],
            "scientific": ["hypothesis", "experiment", "theory", "research", "discovery"]
        }
        
        # Thought cache
        self.thought_cache = {}
        self.concept_cache = {}
        
        logger.info("🧠 Neural Thought Engine initialized with enhanced reasoning")
    
    def _setup_redis(self) -> redis.Redis:
        """Setup Redis connection"""
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            redis_password = os.getenv('REDIS_PASSWORD', '')
            
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True
            )
            client.ping()
            logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
            return client
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            # Return a mock Redis client for standalone operation
            return None
    
    async def detect_concepts(self, text: str, detection_depth: str = "deep", context_awareness: bool = True) -> List[ConceptMatch]:
        """🔍 Detect concepts in text with enhanced pattern recognition"""
        try:
            concepts = []
            text_lower = text.lower()
            
            # Multi-layer concept detection
            for category, patterns in self.concept_patterns.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        # Calculate confidence based on context
                        confidence = self._calculate_concept_confidence(text, pattern, category)
                        
                        if confidence >= self.concept_threshold:
                            # Generate reasoning for concept match
                            reasoning = await self._generate_concept_reasoning(text, pattern, category)
                            
                            concept_match = ConceptMatch(
                                concept=f"{category}:{pattern}",
                                confidence=confidence,
                                context=self._extract_context(text, pattern),
                                reasoning=reasoning,
                                timestamp=datetime.now()
                            )
                            concepts.append(concept_match)
            
            # Enhanced concept detection with semantic analysis
            if detection_depth in ["deep", "comprehensive"]:
                semantic_concepts = await self._semantic_concept_detection(text)
                concepts.extend(semantic_concepts)
            
            # Sort by confidence
            concepts.sort(key=lambda x: x.confidence, reverse=True)
            
            # Cache results
            if self.redis_client:
                cache_key = f"concepts:{hash(text)}"
                self.redis_client.setex(cache_key, 3600, json.dumps([
                    {
                        "concept": c.concept,
                        "confidence": c.confidence,
                        "context": c.context,
                        "reasoning": c.reasoning,
                        "timestamp": c.timestamp.isoformat()
                    } for c in concepts
                ]))
            
            return concepts[:10]  # Return top 10 concepts
            
        except Exception as e:
            logger.error(f"❌ Concept detection failed: {e}")
            return []
    
    def _calculate_concept_confidence(self, text: str, pattern: str, category: str) -> float:
        """Calculate confidence score for concept detection"""
        base_confidence = 0.6
        
        # Context analysis
        context_words = text.lower().split()
        pattern_index = -1
        
        for i, word in enumerate(context_words):
            if pattern in word:
                pattern_index = i
                break
        
        if pattern_index == -1:
            return base_confidence
        
        # Boost confidence based on surrounding words
        window_start = max(0, pattern_index - 3)
        window_end = min(len(context_words), pattern_index + 4)
        context_window = context_words[window_start:window_end]
        
        # Check for related terms
        related_terms = self.concept_patterns.get(category, [])
        related_count = sum(1 for word in context_window if any(term in word for term in related_terms))
        
        confidence_boost = min(0.3, related_count * 0.1)
        return min(1.0, base_confidence + confidence_boost)
    
    def _extract_context(self, text: str, pattern: str) -> str:
        """Extract context around the detected pattern"""
        sentences = text.split('.')
        for sentence in sentences:
            if pattern in sentence.lower():
                return sentence.strip()
        return text[:100]  # Fallback to first 100 chars
    
    async def _generate_concept_reasoning(self, text: str, pattern: str, category: str) -> str:
        """Generate reasoning for why a concept was detected"""
        reasoning_prompts = {
            "technical": f"Explain why '{pattern}' indicates technical content in this context.",
            "problem_solving": f"Describe how '{pattern}' relates to problem-solving in this text.",
            "creative": f"Analyze the creative aspects indicated by '{pattern}' in this context.",
            "analytical": f"Explain the analytical nature suggested by '{pattern}' in this text.",
            "philosophical": f"Describe the philosophical implications of '{pattern}' in this context.",
            "scientific": f"Explain the scientific reasoning behind identifying '{pattern}' here."
        }
        
        prompt = reasoning_prompts.get(category, f"Explain the significance of '{pattern}' in this context.")
        
        # Try to get reasoning from phi-2 if available
        try:
            reasoning = await self._query_phi2_for_reasoning(prompt, text[:200])
            return reasoning if reasoning else f"Pattern '{pattern}' detected in {category} context."
        except:
            return f"Pattern '{pattern}' detected in {category} context with high confidence."
    
    async def _semantic_concept_detection(self, text: str) -> List[ConceptMatch]:
        """Advanced semantic concept detection"""
        concepts = []
        
        # Semantic patterns (simplified for this implementation)
        semantic_indicators = {
            "causality": ["because", "therefore", "results in", "leads to", "causes"],
            "comparison": ["compared to", "versus", "unlike", "similar to", "different from"],
            "temporal": ["before", "after", "during", "while", "when", "then"],
            "conditional": ["if", "unless", "provided that", "in case", "assuming"],
            "emphasis": ["importantly", "notably", "significantly", "crucially", "primarily"]
        }
        
        text_lower = text.lower()
        
        for concept_type, indicators in semantic_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    confidence = 0.8  # High confidence for semantic patterns
                    concept_match = ConceptMatch(
                        concept=f"semantic:{concept_type}",
                        confidence=confidence,
                        context=self._extract_context(text, indicator),
                        reasoning=f"Semantic pattern '{indicator}' indicates {concept_type} relationship",
                        timestamp=datetime.now()
                    )
                    concepts.append(concept_match)
        
        return concepts
    
    async def _query_phi2_for_reasoning(self, prompt: str, context: str) -> str:
        """Query phi-2 for enhanced reasoning"""
        try:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nReasoning:"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    f"http://{self.phi2_host}:{self.phi2_port}/generate",
                    json={
                        'prompt': full_prompt,
                        'max_tokens': 100,
                        'temperature': 0.7
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', '').strip()
        except Exception as e:
            logger.warning(f"⚠️ Phi-2 reasoning query failed: {e}")
        
        return ""
    
    async def generate_thought(self, prompt: str, context: str = None, max_tokens: int = 200, 
                            temperature: float = 0.7, concept_detection: bool = True, 
                            reasoning_mode: str = "enhanced") -> Dict[str, Any]:
        """🧠 Generate enhanced thought with concept detection and reasoning"""
        try:
            start_time = time.time()
            
            # Detect concepts in the prompt
            concepts = []
            if concept_detection:
                concepts = await self.detect_concepts(prompt)
            
            # Enhance prompt based on reasoning mode
            enhanced_prompt = await self._enhance_prompt(prompt, context, reasoning_mode, concepts)
            
            # Generate thought using phi-2
            thought_result = await self._generate_with_phi2(enhanced_prompt, max_tokens, temperature)
            
            if not thought_result.get('success', False):
                # Fallback to internal generation
                thought_result = await self._internal_thought_generation(enhanced_prompt, concepts)
            
            processing_time = time.time() - start_time
            
            # Compile comprehensive result
            result = {
                "thought": thought_result.get('output', ''),
                "concepts_detected": [
                    {
                        "concept": c.concept,
                        "confidence": c.confidence,
                        "context": c.context,
                        "reasoning": c.reasoning
                    } for c in concepts
                ],
                "reasoning_mode": reasoning_mode,
                "processing_time_ms": round(processing_time * 1000, 2),
                "enhanced_prompt": enhanced_prompt,
                "generation_stats": thought_result.get('stats', {}),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            if self.redis_client:
                cache_key = f"thought:{hash(prompt + str(context))}"
                self.redis_client.setex(cache_key, 1800, json.dumps(result))
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Thought generation failed: {e}")
            return {
                "thought": "I'm experiencing difficulty generating a thought at the moment.",
                "concepts_detected": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _enhance_prompt(self, prompt: str, context: str = None, reasoning_mode: str = "enhanced", 
                            concepts: List[ConceptMatch] = None) -> str:
        """Enhance prompt based on reasoning mode and detected concepts"""
        
        mode_prefixes = {
            "enhanced": "Think deeply and provide a comprehensive analysis:",
            "basic": "Provide a clear and direct response:",
            "creative": "Think creatively and explore innovative perspectives:",
            "analytical": "Analyze systematically and provide detailed reasoning:"
        }
        
        prefix = mode_prefixes.get(reasoning_mode, mode_prefixes["enhanced"])
        enhanced = f"{prefix}\n\n"
        
        if context:
            enhanced += f"Context: {context}\n\n"
        
        if concepts:
            concept_summary = ", ".join([c.concept.split(':')[-1] for c in concepts[:3]])
            enhanced += f"Key concepts detected: {concept_summary}\n\n"
        
        enhanced += f"Query: {prompt}\n\nResponse:"
        return enhanced
    
    async def _generate_with_phi2(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Generate thought using phi-2 service"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(
                    f"http://{self.phi2_host}:{self.phi2_port}/generate",
                    json={
                        'prompt': prompt,
                        'max_tokens': max_tokens,
                        'temperature': temperature
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'success': True,
                            'output': data.get('response', ''),
                            'stats': {
                                'tokens_generated': data.get('tokens_generated', 0),
                                'generation_time_ms': data.get('generation_time_ms', 0),
                                'model': data.get('model', 'phi-2'),
                                'device': data.get('device', 'unknown')
                            }
                        }
        except Exception as e:
            logger.warning(f"⚠️ Phi-2 generation failed: {e}")
        
        return {'success': False, 'output': '', 'stats': {}}
    
    async def _internal_thought_generation(self, prompt: str, concepts: List[ConceptMatch]) -> Dict[str, Any]:
        """Fallback internal thought generation"""
        
        # Simple rule-based thought generation
        if concepts:
            primary_concept = concepts[0].concept.split(':')[-1]
            thought = f"Based on the concept of '{primary_concept}', I understand this relates to {concepts[0].reasoning.lower()}"
        else:
            thought = "I need to process this request through my reasoning capabilities."
        
        return {
            'success': True,
            'output': thought,
            'stats': {
                'method': 'internal_generation',
                'fallback': True
            }
        }

# FastAPI app
app = FastAPI(title="🧠 Neural Thought Engine", version="10.0.0")

# Global engine instance
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    engine = NeuralThoughtEngine()
    logger.info("🧠 Neural Thought Engine startup complete")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if engine.redis_client:
            engine.redis_client.ping()
        return {"status": "healthy", "service": "neural-thought-engine", "version": "10.0.0"}
    except Exception as e:
        return {"status": "healthy_no_redis", "service": "neural-thought-engine", "warning": str(e)}

@app.post("/generate-thought")
async def generate_thought_endpoint(request: ThoughtRequest):
    """🧠 Generate enhanced thought with concept detection"""
    try:
        result = await engine.generate_thought(
            request.prompt,
            request.context,
            request.max_tokens,
            request.temperature,
            request.concept_detection,
            request.reasoning_mode
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-concepts")
async def detect_concepts_endpoint(request: ConceptDetectionRequest):
    """🔍 Detect concepts in text"""
    try:
        concepts = await engine.detect_concepts(
            request.text,
            request.detection_depth,
            request.context_awareness
        )
        
        result = {
            "concepts": [
                {
                    "concept": c.concept,
                    "confidence": c.confidence,
                    "context": c.context,
                    "reasoning": c.reasoning,
                    "timestamp": c.timestamp.isoformat()
                } for c in concepts
            ],
            "total_concepts": len(concepts),
            "detection_depth": request.detection_depth
        }
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_simple(request: Dict[str, Any]):
    """Simple generation endpoint for compatibility"""
    try:
        prompt = request.get('prompt', '')
        max_tokens = request.get('max_tokens', 200)
        temperature = request.get('temperature', 0.7)
        
        result = await engine.generate_thought(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            concept_detection=True,
            reasoning_mode="enhanced"
        )
        
        return {
            "response": result.get('thought', ''),
            "concepts": result.get('concepts_detected', []),
            "processing_time": result.get('processing_time_ms', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get engine status and metrics"""
    try:
        phi2_status = "unknown"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"http://{engine.phi2_host}:{engine.phi2_port}/health") as response:
                    if response.status == 200:
                        phi2_status = "connected"
                    else:
                        phi2_status = "unavailable"
        except:
            phi2_status = "unreachable"
        
        return {
            "service": "neural-thought-engine",
            "version": "10.0.0",
            "status": "operational",
            "redis_connected": engine.redis_client is not None,
            "phi2_status": phi2_status,
            "reasoning_parameters": {
                "concept_threshold": engine.concept_threshold,
                "reasoning_depth": engine.reasoning_depth,
                "creativity_factor": engine.creativity_factor,
                "analytical_weight": engine.analytical_weight
            },
            "concept_categories": list(engine.concept_patterns.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8890))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info") 