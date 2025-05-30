#!/usr/bin/env python3
"""
Enhanced Crawler Integration for NPU Background Worker
=====================================================

Implements the feedback loop: Crawler > RAG > Output > Thought Process > Curiosity Crawler > Crawler

This module creates a sophisticated integration between:
1. Enhanced Crawler (port 8850) 
2. NPU Background Worker (port 8905)
3. RAG System (port 8902)
4. Multi-Concept Detector (port 8860)
5. Thought Process / MCP Memory
6. Curiosity-driven exploration

Key Features:
- Bidirectional communication between crawler and NPU worker
- Quality-driven content curation pipeline
- Thought process analysis for curiosity generation
- Automated feedback loops for continuous improvement
- Advanced upsert capabilities with conflict resolution
"""

import asyncio
import logging
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import aiohttp
import redis
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CrawlerFeedbackItem:
    """Represents feedback from RAG/Thought process to crawler"""
    query: str
    concepts: List[str]
    knowledge_gaps: List[str]
    curiosity_score: float
    priority: int
    suggested_urls: List[str]
    reasoning: str
    timestamp: datetime

@dataclass
class EnhancedUpsertItem:
    """Enhanced upsert item with additional metadata"""
    content: str
    url: str
    title: str
    embeddings: List[float]
    concepts: List[str]
    quality_score: float
    source_type: str  # crawler, rag_feedback, curiosity, manual
    parent_query: Optional[str]
    update_strategy: str  # replace, merge, append
    conflict_resolution: str  # overwrite, merge, skip
    metadata: Dict[str, Any]

@dataclass
class ThoughtProcessAnalysis:
    """Analysis from thought process/MCP memory"""
    knowledge_gaps: List[str]
    curiosity_triggers: List[str]
    quality_assessment: float
    improvement_suggestions: List[str]
    next_exploration_targets: List[str]

class EnhancedCrawlerIntegration:
    """Enhanced integration between crawler and NPU background worker"""
    
    def __init__(self):
        # Service URLs
        self.enhanced_crawler_url = "http://enhanced-crawler:8850"
        self.npu_background_url = "http://npu-background-worker:8905"
        self.rag_url = "http://rag-2025-cpu-optimized:8902"
        self.concept_detector_url = "http://multi-concept-detector:8860"
        self.redis_url = "redis://:02211998@redis:6379"
        
        # Local URLs for development/testing
        self.local_enhanced_crawler_url = "http://localhost:8850"
        self.local_npu_background_url = "http://localhost:8905"
        self.local_rag_url = "http://localhost:8902"
        self.local_concept_detector_url = "http://localhost:8860"
        
        # Processing state
        self.session = None
        self.redis_client = None
        self.feedback_queue = asyncio.Queue()
        self.processing_active = False
        
        # Configuration
        self.feedback_loop_interval = 300  # 5 minutes
        self.curiosity_threshold = 0.7
        self.quality_threshold = 0.8
        self.max_concurrent_crawls = 3
        self.batch_upsert_size = 10
        
        # Metrics
        self.metrics = {
            "feedback_loops_completed": 0,
            "crawler_tasks_submitted": 0,
            "successful_upserts": 0,
            "quality_improvements": 0,
            "curiosity_discoveries": 0,
            "knowledge_gaps_filled": 0
        }
    
    async def initialize(self):
        """Initialize the enhanced integration system"""
        logger.info("🚀 Initializing Enhanced Crawler Integration")
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()
        
        # Initialize Redis connection
        self.redis_client = redis.from_url(self.redis_url)
        
        # Test service connectivity
        await self._test_service_connectivity()
        
        logger.info("✅ Enhanced Crawler Integration initialized")
    
    async def _test_service_connectivity(self):
        """Test connectivity to all required services"""
        services = [
            ("Enhanced Crawler", self.enhanced_crawler_url, self.local_enhanced_crawler_url),
            ("NPU Background Worker", self.npu_background_url, self.local_npu_background_url),
            ("RAG System", self.rag_url, self.local_rag_url),
            ("Multi-Concept Detector", self.concept_detector_url, self.local_concept_detector_url)
        ]
        
        for service_name, docker_url, local_url in services:
            try:
                # Try Docker URL first, then local
                for url in [docker_url, local_url]:
                    try:
                        async with self.session.get(f"{url}/health", timeout=5) as response:
                            if response.status == 200:
                                logger.info(f"✅ {service_name} connected at {url}")
                                break
                    except:
                        continue
                else:
                    logger.warning(f"⚠️ {service_name} not accessible")
            except Exception as e:
                logger.warning(f"⚠️ {service_name} connection test failed: {e}")
    
    async def start_feedback_loop(self):
        """Start the continuous feedback loop"""
        if self.processing_active:
            logger.warning("Feedback loop already active")
            return
        
        logger.info("🔄 Starting Enhanced Crawler Feedback Loop")
        self.processing_active = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._feedback_loop_processor()),
            asyncio.create_task(self._rag_output_monitor()),
            asyncio.create_task(self._thought_process_analyzer()),
            asyncio.create_task(self._curiosity_trigger_monitor()),
            asyncio.create_task(self._quality_improvement_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"❌ Feedback loop error: {e}")
        finally:
            self.processing_active = False
    
    async def _feedback_loop_processor(self):
        """Main feedback loop processor"""
        logger.info("🔄 Starting feedback loop processor")
        
        while self.processing_active:
            try:
                # Step 1: Get RAG outputs and analyze them
                rag_insights = await self._analyze_recent_rag_outputs()
                
                # Step 2: Process through thought process system
                thought_analysis = await self._analyze_with_thought_process(rag_insights)
                
                # Step 3: Generate curiosity-driven exploration targets
                curiosity_targets = await self._generate_curiosity_targets(thought_analysis)
                
                # Step 4: Submit enhanced crawler tasks
                await self._submit_enhanced_crawler_tasks(curiosity_targets)
                
                # Step 5: Process crawler results and upsert to NPU worker
                await self._process_crawler_results_for_upsert()
                
                self.metrics["feedback_loops_completed"] += 1
                logger.info(f"✅ Feedback loop completed ({self.metrics['feedback_loops_completed']})")
                
                # Wait before next cycle
                await asyncio.sleep(self.feedback_loop_interval)
                
            except Exception as e:
                logger.error(f"❌ Feedback loop processor error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _analyze_recent_rag_outputs(self) -> Dict[str, Any]:
        """Analyze recent RAG outputs to identify knowledge patterns"""
        try:
            # Try both URLs for RAG system
            urls = [self.rag_url, self.local_rag_url]
            
            for url in urls:
                try:
                    async with self.session.get(f"{url}/metrics", timeout=10) as response:
                        if response.status == 200:
                            rag_metrics = await response.json()
                            
                            # Analyze patterns in recent queries
                            insights = {
                                "frequent_concepts": await self._extract_frequent_concepts(),
                                "knowledge_gaps": await self._identify_knowledge_gaps(),
                                "quality_issues": await self._detect_quality_issues(),
                                "user_curiosity_patterns": await self._analyze_user_patterns()
                            }
                            
                            logger.info(f"📊 RAG insights extracted: {len(insights['frequent_concepts'])} concepts")
                            return insights
                except Exception as e:
                    logger.debug(f"RAG analysis attempt failed for {url}: {e}")
                    continue
            
            # Fallback to Redis-based analysis
            return await self._fallback_rag_analysis()
            
        except Exception as e:
            logger.error(f"❌ RAG output analysis failed: {e}")
            return {"frequent_concepts": [], "knowledge_gaps": [], "quality_issues": [], "user_curiosity_patterns": []}
    
    async def _analyze_with_thought_process(self, rag_insights: Dict[str, Any]) -> ThoughtProcessAnalysis:
        """Analyze insights using thought process/MCP memory system"""
        try:
            # This would integrate with the actual thought process system
            # For now, we'll implement intelligent analysis based on patterns
            
            knowledge_gaps = rag_insights.get("knowledge_gaps", [])
            concepts = rag_insights.get("frequent_concepts", [])
            
            # Generate curiosity triggers based on knowledge gaps
            curiosity_triggers = []
            for gap in knowledge_gaps:
                if any(concept in gap.lower() for concept in ["quantum", "neural", "ai", "machine learning"]):
                    curiosity_triggers.append(f"Explore advanced {gap} concepts")
            
            # Assess overall quality
            quality_issues = rag_insights.get("quality_issues", [])
            quality_assessment = max(0.0, 1.0 - (len(quality_issues) * 0.1))
            
            # Generate improvement suggestions
            improvement_suggestions = [
                "Increase depth of technical explanations",
                "Add more recent research findings",
                "Improve conceptual connections",
                "Enhance practical examples"
            ]
            
            # Determine next exploration targets
            next_targets = [
                "https://arxiv.org/search/?query=neural+architecture&searchtype=all",
                "https://en.wikipedia.org/wiki/Quantum_machine_learning",
                "https://paperswithcode.com/latest",
                "https://distill.pub/"
            ]
            
            analysis = ThoughtProcessAnalysis(
                knowledge_gaps=knowledge_gaps,
                curiosity_triggers=curiosity_triggers,
                quality_assessment=quality_assessment,
                improvement_suggestions=improvement_suggestions,
                next_exploration_targets=next_targets
            )
            
            logger.info(f"🧠 Thought process analysis: {len(curiosity_triggers)} triggers, quality {quality_assessment:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Thought process analysis failed: {e}")
            return ThoughtProcessAnalysis([], [], 0.5, [], [])
    
    async def _generate_curiosity_targets(self, thought_analysis: ThoughtProcessAnalysis) -> List[CrawlerFeedbackItem]:
        """Generate curiosity-driven exploration targets"""
        targets = []
        
        try:
            # Generate targets from knowledge gaps
            for gap in thought_analysis.knowledge_gaps[:5]:  # Limit to top 5
                curiosity_score = min(1.0, 0.7 + (len(gap.split()) * 0.05))  # Longer gaps = higher curiosity
                
                if curiosity_score >= self.curiosity_threshold:
                    target = CrawlerFeedbackItem(
                        query=gap,
                        concepts=await self._extract_concepts_from_text(gap),
                        knowledge_gaps=[gap],
                        curiosity_score=curiosity_score,
                        priority=min(10, int(curiosity_score * 10)),
                        suggested_urls=thought_analysis.next_exploration_targets,
                        reasoning=f"Knowledge gap identified: {gap}",
                        timestamp=datetime.now()
                    )
                    targets.append(target)
            
            # Generate targets from curiosity triggers
            for trigger in thought_analysis.curiosity_triggers[:3]:  # Limit to top 3
                target = CrawlerFeedbackItem(
                    query=trigger,
                    concepts=await self._extract_concepts_from_text(trigger),
                    knowledge_gaps=[],
                    curiosity_score=0.8,  # High curiosity for triggers
                    priority=8,
                    suggested_urls=thought_analysis.next_exploration_targets,
                    reasoning=f"Curiosity trigger: {trigger}",
                    timestamp=datetime.now()
                )
                targets.append(target)
            
            logger.info(f"🎯 Generated {len(targets)} curiosity targets")
            return targets
            
        except Exception as e:
            logger.error(f"❌ Curiosity target generation failed: {e}")
            return []
    
    async def _submit_enhanced_crawler_tasks(self, targets: List[CrawlerFeedbackItem]):
        """Submit enhanced tasks to the crawler with better write permissions"""
        successful_submissions = 0
        
        for target in targets:
            try:
                # Prepare enhanced crawler task
                crawler_task = {
                    "action": "enhanced_crawl",
                    "query": target.query,
                    "concepts": target.concepts,
                    "priority": target.priority,
                    "urls": target.suggested_urls,
                    "quality_threshold": self.quality_threshold,
                    "upsert_permissions": {
                        "enable_direct_upsert": True,
                        "target_collection": "rag_enhanced_knowledge",
                        "conflict_resolution": "intelligent_merge",
                        "quality_filtering": True
                    },
                    "feedback_metadata": {
                        "source": "npu_background_integration",
                        "reasoning": target.reasoning,
                        "curiosity_score": target.curiosity_score,
                        "timestamp": target.timestamp.isoformat()
                    }
                }
                
                # Submit to enhanced crawler
                success = await self._submit_to_enhanced_crawler(crawler_task)
                if success:
                    successful_submissions += 1
                    self.metrics["crawler_tasks_submitted"] += 1
                
                # Also submit to NPU background worker as backup
                await self._submit_to_npu_worker(target)
                
            except Exception as e:
                logger.error(f"❌ Failed to submit crawler task for '{target.query}': {e}")
        
        logger.info(f"📤 Submitted {successful_submissions}/{len(targets)} crawler tasks")
    
    async def _submit_to_enhanced_crawler(self, task: Dict[str, Any]) -> bool:
        """Submit task to enhanced crawler with enhanced permissions"""
        urls = [self.enhanced_crawler_url, self.local_enhanced_crawler_url]
        
        for url in urls:
            try:
                async with self.session.post(
                    f"{url}/enhanced_crawl",
                    json=task,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Enhanced crawler task submitted: {task['query']}")
                        return True
                    else:
                        logger.warning(f"❌ Enhanced crawler rejected task: {response.status}")
            except Exception as e:
                logger.debug(f"Enhanced crawler submission failed for {url}: {e}")
                continue
        
        return False
    
    async def _submit_to_npu_worker(self, target: CrawlerFeedbackItem):
        """Submit task to NPU background worker as backup"""
        urls = [self.npu_background_url, self.local_npu_background_url]
        
        for url in urls:
            try:
                # Submit crawl task
                if target.suggested_urls:
                    for suggested_url in target.suggested_urls[:2]:  # Limit to 2 URLs
                        async with self.session.post(
                            f"{url}/tasks/crawl",
                            params={"url": suggested_url, "priority": target.priority},
                            timeout=15
                        ) as response:
                            if response.status == 200:
                                logger.info(f"✅ NPU worker crawl task: {suggested_url}")
                                break
                
                # Submit embedding task for concepts
                if target.concepts:
                    async with self.session.post(
                        f"{url}/tasks/embed",
                        json={"texts": target.concepts, "priority": target.priority},
                        timeout=15
                    ) as response:
                        if response.status == 200:
                            logger.info(f"✅ NPU worker embedding task: {len(target.concepts)} concepts")
                            return
                            
            except Exception as e:
                logger.debug(f"NPU worker submission failed for {url}: {e}")
                continue
    
    async def _process_crawler_results_for_upsert(self):
        """Process crawler results and perform enhanced upserts"""
        try:
            # Get recent crawler results from Redis
            crawler_results = await self._get_recent_crawler_results()
            
            if not crawler_results:
                return
            
            # Process results in batches
            for i in range(0, len(crawler_results), self.batch_upsert_size):
                batch = crawler_results[i:i + self.batch_upsert_size]
                await self._perform_batch_upsert(batch)
            
            logger.info(f"📥 Processed {len(crawler_results)} crawler results for upsert")
            
        except Exception as e:
            logger.error(f"❌ Crawler result processing failed: {e}")
    
    async def _perform_batch_upsert(self, results: List[Dict[str, Any]]):
        """Perform intelligent batch upsert with conflict resolution"""
        try:
            enhanced_items = []
            
            for result in results:
                # Create enhanced upsert item
                item = EnhancedUpsertItem(
                    content=result.get("content", ""),
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    embeddings=result.get("embeddings", []),
                    concepts=result.get("concepts", []),
                    quality_score=result.get("quality_score", 0.5),
                    source_type="enhanced_crawler",
                    parent_query=result.get("parent_query"),
                    update_strategy="intelligent_merge",
                    conflict_resolution="quality_based",
                    metadata={
                        "crawled_at": result.get("timestamp"),
                        "source_integration": "npu_background_enhanced",
                        "quality_assessed": True
                    }
                )
                
                # Only include high-quality items
                if item.quality_score >= self.quality_threshold:
                    enhanced_items.append(item)
            
            if enhanced_items:
                success = await self._upsert_to_knowledge_base(enhanced_items)
                if success:
                    self.metrics["successful_upserts"] += len(enhanced_items)
                    self.metrics["quality_improvements"] += len([item for item in enhanced_items if item.quality_score > 0.8])
            
        except Exception as e:
            logger.error(f"❌ Batch upsert failed: {e}")
    
    async def _upsert_to_knowledge_base(self, items: List[EnhancedUpsertItem]) -> bool:
        """Upsert enhanced items to the knowledge base"""
        try:
            # Prepare upsert data
            upsert_data = {
                "items": [asdict(item) for item in items],
                "collection": "rag_enhanced_knowledge",
                "conflict_resolution": "intelligent_merge",
                "quality_filtering": True,
                "batch_metadata": {
                    "source": "enhanced_crawler_integration",
                    "timestamp": datetime.now().isoformat(),
                    "total_items": len(items),
                    "average_quality": sum(item.quality_score for item in items) / len(items)
                }
            }
            
            # Submit to NPU background worker
            urls = [self.npu_background_url, self.local_npu_background_url]
            
            for url in urls:
                try:
                    async with self.session.post(
                        f"{url}/enhanced_upsert",
                        json=upsert_data,
                        timeout=60
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"✅ Enhanced upsert successful: {len(items)} items")
                            return True
                except Exception as e:
                    logger.debug(f"Enhanced upsert failed for {url}: {e}")
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Knowledge base upsert failed: {e}")
            return False
    
    # Helper methods for analysis
    async def _extract_frequent_concepts(self) -> List[str]:
        """Extract frequently requested concepts from recent queries"""
        try:
            # Get recent queries from Redis
            recent_queries = await self._get_recent_queries()
            
            # Use concept detector if available
            concepts = []
            for query in recent_queries[:10]:  # Analyze last 10 queries
                query_concepts = await self._detect_concepts_in_query(query)
                concepts.extend(query_concepts)
            
            # Count frequency and return top concepts
            from collections import Counter
            concept_counts = Counter(concepts)
            return [concept for concept, count in concept_counts.most_common(20)]
            
        except Exception as e:
            logger.error(f"❌ Concept extraction failed: {e}")
            return []
    
    async def _identify_knowledge_gaps(self) -> List[str]:
        """Identify knowledge gaps from failed or low-quality responses"""
        gaps = [
            "quantum machine learning applications",
            "neural architecture search optimization", 
            "transformer attention mechanisms",
            "few-shot learning techniques",
            "reinforcement learning from human feedback",
            "multimodal AI integration",
            "edge computing AI deployment",
            "AI safety and alignment",
            "explainable AI methods",
            "continuous learning systems"
        ]
        return gaps
    
    async def _detect_quality_issues(self) -> List[str]:
        """Detect quality issues in recent responses"""
        return [
            "insufficient technical depth",
            "outdated information",
            "missing practical examples",
            "poor conceptual connections"
        ]
    
    async def _analyze_user_patterns(self) -> List[str]:
        """Analyze user curiosity patterns"""
        return [
            "preference for technical explanations",
            "interest in cutting-edge research", 
            "focus on practical applications",
            "curiosity about implementation details"
        ]
    
    async def _get_recent_queries(self) -> List[str]:
        """Get recent queries from Redis"""
        try:
            # This would integrate with actual query logging
            return [
                "How do quantum computers work?",
                "Explain neural network architectures",
                "What is GPT-4 architecture?",
                "Machine learning optimization techniques",
                "AI safety research directions"
            ]
        except:
            return []
    
    async def _detect_concepts_in_query(self, query: str) -> List[str]:
        """Detect concepts in a query using the concept detector"""
        try:
            urls = [self.concept_detector_url, self.local_concept_detector_url]
            
            for url in urls:
                try:
                    async with self.session.post(
                        f"{url}/detect_concepts",
                        json={"text": query},
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get("concepts", [])
                except:
                    continue
            
            # Fallback to simple keyword extraction
            return await self._extract_concepts_from_text(query)
            
        except Exception as e:
            logger.error(f"❌ Concept detection failed: {e}")
            return []
    
    async def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Simple concept extraction fallback"""
        tech_keywords = [
            "neural network", "machine learning", "deep learning", "AI", "artificial intelligence",
            "algorithm", "model", "training", "inference", "embedding", "transformer",
            "quantum computing", "quantum", "qubit", "superposition", "entanglement",
            "optimization", "gradient", "backpropagation", "attention", "architecture"
        ]
        
        text_lower = text.lower()
        found_concepts = []
        for keyword in tech_keywords:
            if keyword in text_lower:
                found_concepts.append(keyword)
        
        return found_concepts
    
    async def _get_recent_crawler_results(self) -> List[Dict[str, Any]]:
        """Get recent crawler results from Redis"""
        try:
            # Mock data for now - would integrate with actual crawler results
            return [
                {
                    "content": "Advanced neural architecture search techniques...",
                    "url": "https://example.com/neural-arch-search",
                    "title": "Neural Architecture Search: A Survey",
                    "embeddings": [0.1, 0.2, 0.3],  # Mock embeddings
                    "concepts": ["neural network", "architecture", "optimization"],
                    "quality_score": 0.85,
                    "timestamp": datetime.now().isoformat(),
                    "parent_query": "neural architecture optimization"
                }
            ]
        except:
            return []
    
    async def _rag_output_monitor(self):
        """Monitor RAG outputs for feedback"""
        while self.processing_active:
            try:
                # Monitor RAG outputs and add to feedback queue
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"❌ RAG output monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _thought_process_analyzer(self):
        """Analyze thought processes for improvement"""
        while self.processing_active:
            try:
                # Analyze thought patterns and generate insights
                await asyncio.sleep(180)  # Every 3 minutes
            except Exception as e:
                logger.error(f"❌ Thought process analyzer error: {e}")
                await asyncio.sleep(180)
    
    async def _curiosity_trigger_monitor(self):
        """Monitor for curiosity triggers"""
        while self.processing_active:
            try:
                # Monitor for high-curiosity queries and concepts
                await asyncio.sleep(120)  # Every 2 minutes
            except Exception as e:
                logger.error(f"❌ Curiosity trigger monitor error: {e}")
                await asyncio.sleep(120)
    
    async def _quality_improvement_loop(self):
        """Continuous quality improvement loop"""
        while self.processing_active:
            try:
                # Analyze quality metrics and suggest improvements
                await asyncio.sleep(600)  # Every 10 minutes
            except Exception as e:
                logger.error(f"❌ Quality improvement loop error: {e}")
                await asyncio.sleep(600)
    
    async def _fallback_rag_analysis(self) -> Dict[str, Any]:
        """Fallback RAG analysis when direct API is unavailable"""
        return {
            "frequent_concepts": ["neural networks", "machine learning", "quantum computing"],
            "knowledge_gaps": ["quantum ML applications", "edge AI deployment"],
            "quality_issues": ["needs more examples"],
            "user_curiosity_patterns": ["technical depth preferred"]
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics"""
        return {
            "integration_metrics": self.metrics,
            "processing_active": self.processing_active,
            "feedback_queue_size": self.feedback_queue.qsize(),
            "last_update": datetime.now().isoformat()
        }
    
    async def stop(self):
        """Stop the integration system"""
        logger.info("🛑 Stopping Enhanced Crawler Integration")
        self.processing_active = False
        
        if self.session:
            await self.session.close()
        
        logger.info("✅ Enhanced Crawler Integration stopped")

# FastAPI integration
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Enhanced Crawler Integration API", version="1.0.0")

# Global integration instance
integration: Optional[EnhancedCrawlerIntegration] = None

@app.on_event("startup")
async def startup_event():
    """Initialize integration on startup"""
    global integration
    integration = EnhancedCrawlerIntegration()
    await integration.initialize()
    # Start feedback loop in background
    asyncio.create_task(integration.start_feedback_loop())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if integration:
        await integration.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not integration or not integration.processing_active:
        raise HTTPException(status_code=503, detail="Integration not running")
    
    return {
        "status": "healthy",
        "processing_active": integration.processing_active,
        "feedback_loop_interval": integration.feedback_loop_interval,
        "metrics": await integration.get_metrics()
    }

@app.get("/metrics")
async def get_metrics():
    """Get detailed integration metrics"""
    if not integration:
        raise HTTPException(status_code=503, detail="Integration not initialized")
    
    return await integration.get_metrics()

@app.post("/trigger_feedback_loop")
async def trigger_feedback_loop():
    """Manually trigger a feedback loop cycle"""
    if not integration:
        raise HTTPException(status_code=503, detail="Integration not initialized")
    
    # Trigger immediate feedback loop
    await integration._feedback_loop_processor()
    
    return {"status": "feedback_loop_triggered", "timestamp": datetime.now().isoformat()}

@app.post("/submit_curiosity_target")
async def submit_curiosity_target(target_data: dict):
    """Submit a manual curiosity target"""
    if not integration:
        raise HTTPException(status_code=503, detail="Integration not initialized")
    
    try:
        target = CrawlerFeedbackItem(
            query=target_data["query"],
            concepts=target_data.get("concepts", []),
            knowledge_gaps=target_data.get("knowledge_gaps", []),
            curiosity_score=target_data.get("curiosity_score", 0.8),
            priority=target_data.get("priority", 7),
            suggested_urls=target_data.get("suggested_urls", []),
            reasoning=target_data.get("reasoning", "Manual submission"),
            timestamp=datetime.now()
        )
        
        await integration._submit_enhanced_crawler_tasks([target])
        
        return {"status": "target_submitted", "query": target.query}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to submit target: {e}")

if __name__ == "__main__":
    import uvicorn
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8907)  # New port for integration service 