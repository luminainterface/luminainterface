#!/usr/bin/env python3
"""
Enhanced Concept Integration for RAG Coordination
Real-time concept sharing and intelligent routing optimization
"""

import asyncio
import json
import time
import redis
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnhancedConcept:
    name: str
    category: str
    confidence: float
    keywords: List[str]
    suggested_service: str
    service_confidence: float
    timestamp: str
    processing_time_ms: float
    cross_service_validated: bool = False
    routing_history: List[str] = None
    
    def __post_init__(self):
        if self.routing_history is None:
            self.routing_history = []

class EnhancedConceptIntegrator:
    def __init__(self, redis_host="localhost", redis_port=6379):
        """Initialize enhanced concept integration system"""
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Service endpoints for intelligent routing
        self.rag_services = {
            "rag-code": {
                "url": "http://localhost:8922",
                "endpoint": "/search",
                "specialties": ["programming", "algorithms", "code", "python", "javascript", "software"],
                "confidence_boost": 0.15
            },
            "rag-graph": {
                "url": "http://localhost:8921", 
                "endpoint": "/query",
                "specialties": ["relationships", "connections", "networks", "social", "analysis"],
                "confidence_boost": 0.10
            },
            "rag-gpu-long": {
                "url": "http://localhost:8920",
                "endpoint": "/query", 
                "specialties": ["complex", "deep", "analysis", "quantum", "advanced", "scientific"],
                "confidence_boost": 0.20
            },
            "rag-cpu-optimized": {
                "url": "http://localhost:8902",
                "endpoint": "/query",
                "specialties": ["general", "explanation", "overview", "basic", "introduction"],
                "confidence_boost": 0.05
            },
            "rag-router": {
                "url": "http://localhost:8950",
                "endpoint": "/search", 
                "specialties": ["coordination", "routing", "multi-head", "aggregation"],
                "confidence_boost": 0.08
            }
        }
        
        # Concept-to-service mapping cache
        self.concept_cache_ttl = 600  # 10 minutes
        self.routing_history = {}
        
        # Performance metrics
        self.routing_stats = {
            "total_routes": 0,
            "successful_routes": 0,
            "cache_hits": 0,
            "processing_times": []
        }

    async def detect_and_route_concepts(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced concept detection with intelligent routing"""
        start_time = time.time()
        
        try:
            # Step 1: Detect concepts using multi-concept-detector
            concepts = await self._detect_concepts_enhanced(query, user_context)
            
            # Step 2: Enhance concepts with cross-service validation
            enhanced_concepts = await self._enhance_concepts_cross_service(concepts, query)
            
            # Step 3: Intelligent service routing
            routing_plan = await self._generate_intelligent_routing(enhanced_concepts, query)
            
            # Step 4: Cache concepts for future use
            await self._cache_concepts(query, enhanced_concepts, routing_plan)
            
            # Step 5: Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.routing_stats["processing_times"].append(processing_time)
            self.routing_stats["total_routes"] += 1
            
            return {
                "concepts": [asdict(concept) for concept in enhanced_concepts],
                "routing_plan": routing_plan,
                "processing_time_ms": processing_time,
                "cache_status": "updated",
                "intelligence_level": "enhanced"
            }
            
        except Exception as e:
            logger.error(f"Enhanced concept detection failed: {e}")
            return await self._fallback_concept_routing(query)

    async def _detect_concepts_enhanced(self, query: str, user_context: Optional[Dict] = None) -> List[EnhancedConcept]:
        """Enhanced concept detection with multi-concept-detector integration"""
        try:
            # Check cache first
            cache_key = f"concepts:{hash(query)}"
            cached_concepts = await self._get_cached_concepts(cache_key)
            if cached_concepts:
                self.routing_stats["cache_hits"] += 1
                return cached_concepts
            
            # Call multi-concept-detector service
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": query,
                    "user_context": user_context or {},
                    "intelligence_level": "npu"
                }
                
                async with session.post(
                    "http://localhost:8860/detect-concepts",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        concepts_data = data.get("concepts", [])
                        
                        # Convert to EnhancedConcept objects
                        enhanced_concepts = []
                        for concept_data in concepts_data:
                            enhanced_concept = EnhancedConcept(
                                name=concept_data.get("name", "unknown"),
                                category=concept_data.get("category", "general"),
                                confidence=concept_data.get("confidence", 0.5),
                                keywords=concept_data.get("keywords", []),
                                suggested_service="",  # Will be determined by routing
                                service_confidence=0.0,
                                timestamp=datetime.now().isoformat(),
                                processing_time_ms=data.get("processing_time", 0)
                            )
                            enhanced_concepts.append(enhanced_concept)
                        
                        return enhanced_concepts
                    else:
                        logger.warning(f"Multi-concept detector returned status {response.status}")
                        
        except Exception as e:
            logger.warning(f"Enhanced concept detection failed: {e}")
        
        # Fallback to basic concept extraction
        return await self._fallback_concept_detection(query)

    async def _enhance_concepts_cross_service(self, concepts: List[EnhancedConcept], query: str) -> List[EnhancedConcept]:
        """Enhance concepts with cross-service validation and enrichment"""
        enhanced_concepts = []
        
        for concept in concepts:
            # Cross-validate with service specialties
            best_service, service_confidence = self._calculate_service_match(concept)
            
            # Update concept with routing information
            concept.suggested_service = best_service
            concept.service_confidence = service_confidence
            concept.cross_service_validated = True
            
            # Add to routing history
            concept.routing_history.append(f"enhanced_routing:{best_service}:{service_confidence:.3f}")
            
            enhanced_concepts.append(concept)
        
        return enhanced_concepts

    def _calculate_service_match(self, concept: EnhancedConcept) -> Tuple[str, float]:
        """Calculate the best service match for a concept"""
        best_service = "rag-cpu-optimized"  # Default fallback
        best_confidence = 0.3
        
        concept_keywords = [concept.name.lower(), concept.category.lower()] + \
                          [kw.lower() for kw in concept.keywords]
        
        for service_name, service_info in self.rag_services.items():
            # Calculate specialty match score
            specialty_matches = 0
            for keyword in concept_keywords:
                for specialty in service_info["specialties"]:
                    if keyword in specialty or specialty in keyword:
                        specialty_matches += 1
            
            # Calculate confidence score
            if specialty_matches > 0:
                match_ratio = specialty_matches / len(concept_keywords)
                service_confidence = (concept.confidence * match_ratio) + service_info["confidence_boost"]
                
                if service_confidence > best_confidence:
                    best_service = service_name
                    best_confidence = service_confidence
        
        return best_service, min(best_confidence, 1.0)

    async def _generate_intelligent_routing(self, concepts: List[EnhancedConcept], query: str) -> Dict[str, Any]:
        """Generate intelligent routing plan based on enhanced concepts"""
        routing_plan = {
            "primary_service": "rag-cpu-optimized",
            "secondary_services": [],
            "confidence": 0.5,
            "reasoning": "Default routing",
            "parallel_execution": False,
            "expected_response_time": 2.0
        }
        
        if not concepts:
            return routing_plan
        
        # Find highest confidence concept and service
        best_concept = max(concepts, key=lambda c: c.service_confidence)
        
        routing_plan.update({
            "primary_service": best_concept.suggested_service,
            "confidence": best_concept.service_confidence,
            "reasoning": f"Best match for concept '{best_concept.name}' in category '{best_concept.category}'",
            "triggered_by_concept": best_concept.name
        })
        
        # Add secondary services for parallel execution
        secondary_concepts = [c for c in concepts if c != best_concept and c.service_confidence > 0.6]
        if secondary_concepts:
            routing_plan["secondary_services"] = [c.suggested_service for c in secondary_concepts[:2]]
            routing_plan["parallel_execution"] = True
            routing_plan["expected_response_time"] = max(1.5, routing_plan["expected_response_time"] * 0.8)
        
        return routing_plan

    async def _cache_concepts(self, query: str, concepts: List[EnhancedConcept], routing_plan: Dict[str, Any]):
        """Cache concepts and routing information for performance"""
        try:
            cache_key = f"concepts:{hash(query)}"
            cache_data = {
                "concepts": [asdict(concept) for concept in concepts],
                "routing_plan": routing_plan,
                "cached_at": datetime.now().isoformat()
            }
            
            # Store with TTL
            self.redis_client.setex(cache_key, self.concept_cache_ttl, json.dumps(cache_data))
            
            # Update routing history
            primary_service = routing_plan.get("primary_service", "unknown")
            if primary_service not in self.routing_history:
                self.routing_history[primary_service] = []
            
            self.routing_history[primary_service].append({
                "query": query[:100],  # Truncated for storage
                "timestamp": datetime.now().isoformat(),
                "confidence": routing_plan.get("confidence", 0.5)
            })
            
        except Exception as e:
            logger.warning(f"Concept caching failed: {e}")

    async def _get_cached_concepts(self, cache_key: str) -> Optional[List[EnhancedConcept]]:
        """Retrieve cached concepts if available"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                concepts_data = data.get("concepts", [])
                
                # Convert back to EnhancedConcept objects
                concepts = []
                for concept_data in concepts_data:
                    concept = EnhancedConcept(**concept_data)
                    concepts.append(concept)
                
                return concepts
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None

    async def _fallback_concept_detection(self, query: str) -> List[EnhancedConcept]:
        """Fallback concept detection when enhanced detection fails"""
        concepts = []
        words = query.lower().split()
        
        # Simple concept mapping
        concept_indicators = {
            "python": ("programming", ["code", "algorithm", "programming"]),
            "algorithm": ("programming", ["algorithm", "code", "implementation"]),
            "neural": ("ai", ["neural", "network", "machine", "learning"]),
            "quantum": ("physics", ["quantum", "computing", "physics"]),
            "relationship": ("graph", ["relationship", "connection", "network"]),
            "docker": ("devops", ["docker", "container", "deployment"]),
            "react": ("programming", ["react", "javascript", "frontend"]),
        }
        
        for word in words:
            if word in concept_indicators:
                category, keywords = concept_indicators[word]
                concept = EnhancedConcept(
                    name=word,
                    category=category,
                    confidence=0.7,
                    keywords=keywords,
                    suggested_service=self._get_fallback_service(category),
                    service_confidence=0.6,
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=5.0,
                    cross_service_validated=False
                )
                concepts.append(concept)
        
        return concepts[:3]  # Limit to top 3

    def _get_fallback_service(self, category: str) -> str:
        """Get fallback service based on category"""
        category_mapping = {
            "programming": "rag-code",
            "ai": "rag-gpu-long", 
            "physics": "rag-gpu-long",
            "graph": "rag-graph",
            "devops": "rag-code"
        }
        return category_mapping.get(category, "rag-cpu-optimized")

    async def _fallback_concept_routing(self, query: str) -> Dict[str, Any]:
        """Fallback routing when enhanced detection completely fails"""
        return {
            "concepts": [],
            "routing_plan": {
                "primary_service": "rag-cpu-optimized",
                "confidence": 0.3,
                "reasoning": "Fallback routing - enhanced detection failed"
            },
            "processing_time_ms": 50.0,
            "cache_status": "failed",
            "intelligence_level": "fallback"
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        avg_processing_time = sum(self.routing_stats["processing_times"]) / len(self.routing_stats["processing_times"]) if self.routing_stats["processing_times"] else 0
        
        return {
            "total_routes": self.routing_stats["total_routes"],
            "successful_routes": self.routing_stats["successful_routes"],
            "cache_hits": self.routing_stats["cache_hits"],
            "cache_hit_rate": self.routing_stats["cache_hits"] / max(self.routing_stats["total_routes"], 1),
            "average_processing_time_ms": avg_processing_time,
            "routing_history_size": sum(len(history) for history in self.routing_history.values()),
            "active_services": len(self.rag_services)
        } 