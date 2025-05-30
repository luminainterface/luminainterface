#!/usr/bin/env python3
"""
Interactive Background Coordinator for Enhanced RAG 2025
========================================================

This service creates a continuously engaged background process that:
1. Monitors chat interactions in real-time
2. Triggers immediate feedback loops based on user queries
3. Maintains constant learning and improvement cycles
4. Coordinates all background services through interactive loops

The coordinator ensures that every chat interaction enhances the system:
- User asks question -> RAG responds -> Background analyzes -> Improves knowledge base
- Continuous curiosity generation based on real conversations
- Real-time quality assessment and enhancement
- Interactive learning from every user interaction

Key Features:
- Chat-triggered feedback loops
- Real-time conversation analysis
- Immediate knowledge gap identification
- Interactive curiosity generation
- Continuous quality improvement
- Live system enhancement
"""

import asyncio
import logging
import json
import time
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
import aiohttp
from pathlib import Path
import websockets
import threading
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ChatInteraction:
    """Represents a chat interaction that triggers background processing"""
    user_query: str
    rag_response: str
    concepts_detected: List[str]
    timestamp: datetime
    session_id: str
    quality_score: float
    curiosity_triggers: List[str]
    knowledge_gaps: List[str]
    immediate_actions: List[str]

@dataclass
class InteractiveFeedbackLoop:
    """Represents an interactive feedback loop triggered by chat"""
    trigger_type: str  # user_query, rag_response, concept_gap, curiosity_spike
    source_interaction: ChatInteraction
    priority: int
    actions_taken: List[str]
    results: Dict[str, Any]
    completion_time: Optional[datetime]

class InteractiveBackgroundCoordinator:
    """Coordinates background processes through interactive chat analysis"""
    
    def __init__(self):
        # Service URLs
        self.enhanced_crawler_integration_url = "http://enhanced-crawler-integration:8907"
        self.npu_background_worker_url = "http://npu-background-worker:8905"
        self.rag_url = "http://rag-2025-cpu-optimized:8902"
        self.enhanced_crawler_url = "http://enhanced-crawler:8850"
        self.concept_detector_url = "http://multi-concept-detector:8860"
        self.lightning_chat_url = "http://lightning-npu-chat:5004"
        
        # Local URLs for development
        self.local_enhanced_integration_url = "http://localhost:8907"
        self.local_npu_worker_url = "http://localhost:8905"
        self.local_rag_url = "http://localhost:8902"
        self.local_enhanced_crawler_url = "http://localhost:8850"
        self.local_concept_detector_url = "http://localhost:8860"
        self.local_lightning_chat_url = "http://localhost:5004"
        
        # Redis for real-time coordination
        self.redis_url = "redis://:02211998@redis:6379"
        self.redis_client = None
        
        # Processing state
        self.session = None
        self.active_interactions = deque(maxlen=1000)
        self.active_feedback_loops = {}
        self.conversation_patterns = defaultdict(list)
        self.continuous_learning_active = False
        
        # Interactive configuration
        self.interaction_analysis_interval = 2.0  # Analyze every 2 seconds
        self.immediate_response_threshold = 0.7   # Trigger immediate action
        self.curiosity_spike_threshold = 0.8      # High curiosity triggers
        self.quality_improvement_threshold = 0.6  # Quality needs improvement
        self.knowledge_gap_urgency = 0.75        # Urgent knowledge gaps
        
        # Metrics
        self.metrics = {
            "interactions_processed": 0,
            "immediate_feedback_loops": 0,
            "quality_improvements_triggered": 0,
            "curiosity_explorations_started": 0,
            "knowledge_gaps_identified": 0,
            "background_enhancements": 0,
            "real_time_learning_cycles": 0,
            "chat_triggered_improvements": 0
        }
        
        # Real-time processing queues
        self.immediate_action_queue = asyncio.Queue()
        self.background_enhancement_queue = asyncio.Queue()
        self.curiosity_exploration_queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize the interactive background coordinator"""
        logger.info("🚀 Initializing Interactive Background Coordinator")
        
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()
        
        # Initialize Redis connection
        self.redis_client = redis.from_url(self.redis_url)
        
        # Test service connectivity
        await self._test_service_connectivity()
        
        # Start listening to chat interactions
        await self._setup_chat_monitoring()
        
        logger.info("✅ Interactive Background Coordinator initialized")
    
    async def _test_service_connectivity(self):
        """Test connectivity to all services for both docker and local"""
        services = [
            ("Enhanced Crawler Integration", self.enhanced_crawler_integration_url, self.local_enhanced_integration_url),
            ("NPU Background Worker", self.npu_background_worker_url, self.local_npu_worker_url),
            ("RAG System", self.rag_url, self.local_rag_url),
            ("Enhanced Crawler", self.enhanced_crawler_url, self.local_enhanced_crawler_url),
            ("Multi-Concept Detector", self.concept_detector_url, self.local_concept_detector_url),
            ("Lightning Chat", self.lightning_chat_url, self.local_lightning_chat_url)
        ]
        
        connected_services = []
        
        for service_name, docker_url, local_url in services:
            try:
                # Try Docker URL first, then local
                for url in [docker_url, local_url]:
                    try:
                        async with self.session.get(f"{url}/health", timeout=3) as response:
                            if response.status == 200:
                                logger.info(f"✅ {service_name}: Connected at {url}")
                                connected_services.append(service_name)
                                break
                    except:
                        continue
                else:
                    logger.warning(f"⚠️ {service_name}: Not accessible")
            except Exception as e:
                logger.warning(f"⚠️ {service_name}: {e}")
        
        logger.info(f"📊 Connected Services: {len(connected_services)}/{len(services)}")
        return connected_services
    
    async def _setup_chat_monitoring(self):
        """Setup real-time chat monitoring"""
        logger.info("🔍 Setting up real-time chat monitoring")
        
        # Monitor Redis streams for chat interactions
        try:
            # Create Redis streams for chat monitoring if they don't exist
            streams = [
                "chat_interactions",
                "rag_responses", 
                "user_queries",
                "system_responses"
            ]
            
            for stream in streams:
                try:
                    await asyncio.to_thread(
                        self.redis_client.xgroup_create,
                        stream,
                        "interactive_coordinator",
                        id="0",
                        mkstream=True
                    )
                except Exception:
                    # Group already exists
                    pass
            
            logger.info("✅ Chat monitoring streams ready")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis stream setup failed: {e}")
    
    async def start_interactive_coordination(self):
        """Start the interactive background coordination"""
        if self.continuous_learning_active:
            logger.warning("Interactive coordination already active")
            return
        
        logger.info("🔄 Starting Interactive Background Coordination")
        self.continuous_learning_active = True
        
        # Start all coordination loops
        tasks = [
            asyncio.create_task(self._chat_interaction_monitor()),
            asyncio.create_task(self._immediate_action_processor()),
            asyncio.create_task(self._background_enhancement_processor()),
            asyncio.create_task(self._curiosity_exploration_processor()),
            asyncio.create_task(self._continuous_learning_loop()),
            asyncio.create_task(self._real_time_quality_analyzer()),
            asyncio.create_task(self._pattern_recognition_loop()),
            asyncio.create_task(self._interactive_feedback_coordinator())
        ]
        
        logger.info("✅ All interactive coordination loops started")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"❌ Interactive coordination error: {e}")
        finally:
            self.continuous_learning_active = False
    
    async def _chat_interaction_monitor(self):
        """Monitor chat interactions and trigger immediate analysis"""
        logger.info("👁️ Starting chat interaction monitor")
        
        while self.continuous_learning_active:
            try:
                # Monitor Lightning Chat API for new interactions
                await self._check_lightning_chat_activity()
                
                # Monitor Redis for chat events
                await self._check_redis_chat_streams()
                
                # Monitor RAG system for new responses
                await self._check_rag_activity()
                
                await asyncio.sleep(self.interaction_analysis_interval)
                
            except Exception as e:
                logger.error(f"❌ Chat monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _check_lightning_chat_activity(self):
        """Check Lightning Chat for new interactions"""
        try:
            # Try both URLs
            urls = [self.lightning_chat_url, self.local_lightning_chat_url]
            
            for url in urls:
                try:
                    async with self.session.get(f"{url}/api/recent_interactions", timeout=3) as response:
                        if response.status == 200:
                            interactions = await response.json()
                            
                            for interaction in interactions.get("recent", []):
                                await self._process_chat_interaction(interaction)
                            
                            return  # Success, don't try other URL
                            
                except Exception:
                    continue  # Try next URL
                    
        except Exception as e:
            logger.debug(f"Lightning chat check failed: {e}")
    
    async def _check_redis_chat_streams(self):
        """Check Redis streams for chat activity"""
        try:
            streams = ["chat_interactions", "rag_responses", "user_queries"]
            
            for stream in streams:
                try:
                    messages = await asyncio.to_thread(
                        self.redis_client.xreadgroup,
                        groupname="interactive_coordinator",
                        consumername="coordinator_001",
                        streams={stream: ">"},
                        count=10,
                        block=100
                    )
                    
                    for stream_name, msgs in messages:
                        for msg_id, fields in msgs:
                            await self._process_redis_chat_message(fields)
                            
                except Exception as e:
                    logger.debug(f"Redis stream {stream} check failed: {e}")
                    
        except Exception as e:
            logger.debug(f"Redis chat stream check failed: {e}")
    
    async def _check_rag_activity(self):
        """Check RAG system for new activity"""
        try:
            urls = [self.rag_url, self.local_rag_url]
            
            for url in urls:
                try:
                    async with self.session.get(f"{url}/metrics", timeout=3) as response:
                        if response.status == 200:
                            metrics = await response.json()
                            
                            # Check for new queries or responses
                            if metrics.get("recent_activity"):
                                await self._process_rag_activity(metrics)
                            
                            return  # Success
                            
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"RAG activity check failed: {e}")
    
    async def _process_chat_interaction(self, interaction_data: Dict[str, Any]):
        """Process a chat interaction and trigger appropriate background actions"""
        try:
            # Create ChatInteraction object
            interaction = ChatInteraction(
                user_query=interaction_data.get("user_query", ""),
                rag_response=interaction_data.get("rag_response", ""),
                concepts_detected=interaction_data.get("concepts", []),
                timestamp=datetime.now(),
                session_id=interaction_data.get("session_id", "unknown"),
                quality_score=interaction_data.get("quality_score", 0.5),
                curiosity_triggers=interaction_data.get("curiosity_triggers", []),
                knowledge_gaps=interaction_data.get("knowledge_gaps", []),
                immediate_actions=[]
            )
            
            # Add to active interactions
            self.active_interactions.append(interaction)
            self.metrics["interactions_processed"] += 1
            
            # Analyze interaction for immediate actions
            await self._analyze_interaction_for_actions(interaction)
            
            logger.info(f"📝 Processed chat interaction: {interaction.user_query[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error processing chat interaction: {e}")
    
    async def _analyze_interaction_for_actions(self, interaction: ChatInteraction):
        """Analyze interaction and queue appropriate actions"""
        immediate_actions = []
        
        # Check for immediate quality issues
        if interaction.quality_score < self.quality_improvement_threshold:
            immediate_actions.append("quality_improvement")
            await self.immediate_action_queue.put({
                "type": "quality_improvement",
                "interaction": interaction,
                "priority": "high"
            })
        
        # Check for curiosity spikes
        if len(interaction.curiosity_triggers) > 0:
            curiosity_score = len(interaction.curiosity_triggers) * 0.2
            if curiosity_score >= self.curiosity_spike_threshold:
                immediate_actions.append("curiosity_exploration")
                await self.curiosity_exploration_queue.put({
                    "type": "curiosity_exploration",
                    "interaction": interaction,
                    "triggers": interaction.curiosity_triggers,
                    "priority": "high"
                })
        
        # Check for knowledge gaps
        if len(interaction.knowledge_gaps) > 0:
            immediate_actions.append("knowledge_gap_filling")
            await self.background_enhancement_queue.put({
                "type": "knowledge_gap_filling",
                "interaction": interaction,
                "gaps": interaction.knowledge_gaps,
                "priority": "medium"
            })
        
        # Check for concept exploration opportunities
        if len(interaction.concepts_detected) > 2:  # Rich concept interaction
            immediate_actions.append("concept_enhancement")
            await self.background_enhancement_queue.put({
                "type": "concept_enhancement",
                "interaction": interaction,
                "concepts": interaction.concepts_detected,
                "priority": "low"
            })
        
        # Update interaction with actions
        interaction.immediate_actions = immediate_actions
        
        if immediate_actions:
            logger.info(f"🎯 Triggered actions for interaction: {immediate_actions}")
    
    async def _immediate_action_processor(self):
        """Process immediate actions triggered by chat interactions"""
        logger.info("⚡ Starting immediate action processor")
        
        while self.continuous_learning_active:
            try:
                # Wait for immediate action
                action = await asyncio.wait_for(
                    self.immediate_action_queue.get(),
                    timeout=1.0
                )
                
                await self._execute_immediate_action(action)
                self.metrics["immediate_feedback_loops"] += 1
                
            except asyncio.TimeoutError:
                continue  # No immediate actions, continue
            except Exception as e:
                logger.error(f"❌ Immediate action processing error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_immediate_action(self, action: Dict[str, Any]):
        """Execute an immediate action"""
        action_type = action["type"]
        interaction = action["interaction"]
        
        try:
            if action_type == "quality_improvement":
                await self._trigger_quality_improvement(interaction)
            elif action_type == "knowledge_gap_filling":
                await self._trigger_immediate_learning(interaction)
            elif action_type == "concept_enhancement":
                await self._trigger_concept_exploration(interaction)
            
            logger.info(f"✅ Executed immediate action: {action_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to execute immediate action {action_type}: {e}")
    
    async def _trigger_quality_improvement(self, interaction: ChatInteraction):
        """Trigger immediate quality improvement based on interaction"""
        try:
            # Submit to enhanced crawler integration for quality analysis
            quality_analysis = {
                "user_query": interaction.user_query,
                "rag_response": interaction.rag_response,
                "quality_score": interaction.quality_score,
                "improvement_needed": True,
                "timestamp": interaction.timestamp.isoformat()
            }
            
            # Try enhanced integration service
            urls = [self.enhanced_crawler_integration_url, self.local_enhanced_integration_url]
            
            for url in urls:
                try:
                    async with self.session.post(
                        f"{url}/trigger_quality_improvement",
                        json=quality_analysis,
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            self.metrics["quality_improvements_triggered"] += 1
                            logger.info("🔧 Quality improvement triggered")
                            return
                except Exception:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Quality improvement trigger failed: {e}")
    
    async def _trigger_immediate_learning(self, interaction: ChatInteraction):
        """Trigger immediate learning from knowledge gaps"""
        try:
            # Create learning tasks for identified gaps
            learning_tasks = []
            
            for gap in interaction.knowledge_gaps:
                learning_task = {
                    "query": gap,
                    "concepts": interaction.concepts_detected,
                    "urgency": "high",
                    "source_interaction": {
                        "user_query": interaction.user_query,
                        "session_id": interaction.session_id,
                        "timestamp": interaction.timestamp.isoformat()
                    },
                    "learning_priority": 9
                }
                learning_tasks.append(learning_task)
            
            # Submit to enhanced crawler for immediate learning
            urls = [self.enhanced_crawler_url, self.local_enhanced_crawler_url]
            
            for url in urls:
                try:
                    async with self.session.post(
                        f"{url}/immediate_learning",
                        json={"tasks": learning_tasks},
                        timeout=15
                    ) as response:
                        if response.status == 200:
                            self.metrics["knowledge_gaps_identified"] += len(learning_tasks)
                            logger.info(f"📚 Immediate learning triggered for {len(learning_tasks)} gaps")
                            return
                except Exception:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Immediate learning trigger failed: {e}")
    
    async def _background_enhancement_processor(self):
        """Process background enhancement tasks"""
        logger.info("🔄 Starting background enhancement processor")
        
        while self.continuous_learning_active:
            try:
                # Wait for background enhancement task
                task = await asyncio.wait_for(
                    self.background_enhancement_queue.get(),
                    timeout=5.0
                )
                
                await self._execute_background_enhancement(task)
                self.metrics["background_enhancements"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Background enhancement error: {e}")
                await asyncio.sleep(2)
    
    async def _execute_background_enhancement(self, task: Dict[str, Any]):
        """Execute a background enhancement task"""
        task_type = task["type"]
        interaction = task["interaction"]
        
        try:
            if task_type == "knowledge_gap_filling":
                await self._enhance_knowledge_base(task)
            elif task_type == "concept_enhancement":
                await self._enhance_concept_understanding(task)
            
            logger.info(f"🔧 Executed background enhancement: {task_type}")
            
        except Exception as e:
            logger.error(f"❌ Background enhancement failed {task_type}: {e}")
    
    async def _enhance_knowledge_base(self, task: Dict[str, Any]):
        """Enhance knowledge base based on identified gaps"""
        try:
            gaps = task.get("gaps", [])
            interaction = task["interaction"]
            
            # Submit enhancement to NPU background worker
            enhancement_data = {
                "knowledge_gaps": gaps,
                "concepts": interaction.concepts_detected,
                "quality_threshold": 0.8,
                "enhancement_type": "interactive_learning",
                "source": "chat_interaction",
                "metadata": {
                    "user_query": interaction.user_query,
                    "session_id": interaction.session_id,
                    "timestamp": interaction.timestamp.isoformat()
                }
            }
            
            urls = [self.npu_background_worker_url, self.local_npu_worker_url]
            
            for url in urls:
                try:
                    async with self.session.post(
                        f"{url}/enhance_knowledge_base",
                        json=enhancement_data,
                        timeout=20
                    ) as response:
                        if response.status == 200:
                            logger.info(f"💡 Knowledge base enhanced for {len(gaps)} gaps")
                            return
                except Exception:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Knowledge base enhancement failed: {e}")
    
    async def _curiosity_exploration_processor(self):
        """Process curiosity-driven exploration tasks"""
        logger.info("🎨 Starting curiosity exploration processor")
        
        while self.continuous_learning_active:
            try:
                # Wait for curiosity exploration task
                task = await asyncio.wait_for(
                    self.curiosity_exploration_queue.get(),
                    timeout=10.0
                )
                
                await self._execute_curiosity_exploration(task)
                self.metrics["curiosity_explorations_started"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Curiosity exploration error: {e}")
                await asyncio.sleep(3)
    
    async def _execute_curiosity_exploration(self, task: Dict[str, Any]):
        """Execute curiosity-driven exploration"""
        try:
            triggers = task.get("triggers", [])
            interaction = task["interaction"]
            
            # Create curiosity targets
            curiosity_targets = []
            
            for trigger in triggers:
                target = {
                    "query": trigger,
                    "concepts": interaction.concepts_detected,
                    "curiosity_score": 0.9,  # High curiosity from chat
                    "priority": 8,
                    "reasoning": f"Curiosity triggered by chat interaction: {interaction.user_query[:50]}...",
                    "suggested_urls": [],
                    "exploration_depth": "deep",
                    "source": "interactive_chat"
                }
                curiosity_targets.append(target)
            
            # Submit to enhanced crawler integration
            urls = [self.enhanced_crawler_integration_url, self.local_enhanced_integration_url]
            
            for url in urls:
                try:
                    for target in curiosity_targets:
                        async with self.session.post(
                            f"{url}/submit_curiosity_target",
                            json=target,
                            timeout=15
                        ) as response:
                            if response.status == 200:
                                logger.info(f"🎯 Curiosity exploration started: {target['query']}")
                except Exception:
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Curiosity exploration failed: {e}")
    
    async def _continuous_learning_loop(self):
        """Main continuous learning coordination loop"""
        logger.info("🔄 Starting continuous learning loop")
        
        while self.continuous_learning_active:
            try:
                # Analyze recent interaction patterns
                await self._analyze_interaction_patterns()
                
                # Trigger periodic improvements
                await self._trigger_periodic_improvements()
                
                # Coordinate with all background services
                await self._coordinate_background_services()
                
                # Update learning metrics
                await self._update_learning_metrics()
                
                self.metrics["real_time_learning_cycles"] += 1
                
                await asyncio.sleep(30)  # Continuous learning every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Continuous learning loop error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_interaction_patterns(self):
        """Analyze patterns in recent interactions"""
        try:
            if len(self.active_interactions) < 5:
                return  # Need more interactions for pattern analysis
            
            recent_interactions = list(self.active_interactions)[-20:]  # Last 20 interactions
            
            # Analyze concept patterns
            all_concepts = []
            all_gaps = []
            quality_scores = []
            
            for interaction in recent_interactions:
                all_concepts.extend(interaction.concepts_detected)
                all_gaps.extend(interaction.knowledge_gaps)
                quality_scores.append(interaction.quality_score)
            
            # Identify trending concepts
            from collections import Counter
            concept_counts = Counter(all_concepts)
            trending_concepts = [concept for concept, count in concept_counts.most_common(5)]
            
            # Identify common knowledge gaps
            gap_counts = Counter(all_gaps)
            common_gaps = [gap for gap, count in gap_counts.most_common(3)]
            
            # Calculate average quality
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
            
            # Store patterns for future use
            self.conversation_patterns["trending_concepts"] = trending_concepts
            self.conversation_patterns["common_gaps"] = common_gaps
            self.conversation_patterns["average_quality"] = avg_quality
            self.conversation_patterns["last_analysis"] = datetime.now()
            
            logger.info(f"📊 Pattern analysis: {len(trending_concepts)} trending concepts, {len(common_gaps)} common gaps, avg quality {avg_quality:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Pattern analysis failed: {e}")
    
    async def _coordinate_background_services(self):
        """Coordinate with all background services for optimal performance"""
        try:
            # Check service health and coordination
            coordination_data = {
                "coordinator_metrics": self.metrics,
                "conversation_patterns": dict(self.conversation_patterns),
                "active_interactions": len(self.active_interactions),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update enhanced crawler integration
            urls = [self.enhanced_crawler_integration_url, self.local_enhanced_integration_url]
            
            for url in urls:
                try:
                    async with self.session.post(
                        f"{url}/coordination_update",
                        json=coordination_data,
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            logger.debug("🤝 Coordination update sent to integration service")
                            break
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"Background service coordination update failed: {e}")
    
    async def get_coordinator_metrics(self) -> Dict[str, Any]:
        """Get detailed coordinator metrics"""
        return {
            "coordinator_metrics": self.metrics,
            "continuous_learning_active": self.continuous_learning_active,
            "active_interactions": len(self.active_interactions),
            "active_feedback_loops": len(self.active_feedback_loops),
            "conversation_patterns": dict(self.conversation_patterns),
            "queue_sizes": {
                "immediate_actions": self.immediate_action_queue.qsize(),
                "background_enhancements": self.background_enhancement_queue.qsize(),
                "curiosity_explorations": self.curiosity_exploration_queue.qsize()
            },
            "last_update": datetime.now().isoformat()
        }
    
    async def stop(self):
        """Stop the interactive background coordinator"""
        logger.info("🛑 Stopping Interactive Background Coordinator")
        self.continuous_learning_active = False
        
        if self.session:
            await self.session.close()
        
        logger.info("✅ Interactive Background Coordinator stopped")

# Additional helper methods for pattern recognition and real-time analysis
    async def _real_time_quality_analyzer(self):
        """Real-time quality analysis of interactions"""
        while self.continuous_learning_active:
            try:
                if self.active_interactions:
                    recent = list(self.active_interactions)[-5:]  # Last 5 interactions
                    avg_quality = sum(i.quality_score for i in recent) / len(recent)
                    
                    if avg_quality < 0.6:  # Quality dropping
                        await self._trigger_emergency_quality_improvement()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Real-time quality analysis error: {e}")
                await asyncio.sleep(30)
    
    async def _pattern_recognition_loop(self):
        """Advanced pattern recognition in conversations"""
        while self.continuous_learning_active:
            try:
                # Advanced pattern analysis every 2 minutes
                await self._detect_conversation_themes()
                await self._detect_learning_opportunities()
                await self._detect_curiosity_clusters()
                
                await asyncio.sleep(120)
                
            except Exception as e:
                logger.error(f"❌ Pattern recognition error: {e}")
                await asyncio.sleep(180)
    
    async def _interactive_feedback_coordinator(self):
        """Coordinate feedback loops triggered by interactions"""
        while self.continuous_learning_active:
            try:
                # Coordinate multiple feedback loops
                await self._prioritize_feedback_loops()
                await self._merge_similar_loops()
                await self._optimize_loop_execution()
                
                await asyncio.sleep(60)  # Coordinate every minute
                
            except Exception as e:
                logger.error(f"❌ Interactive feedback coordination error: {e}")
                await asyncio.sleep(90)
    
    # Placeholder methods for advanced functionality
    async def _trigger_emergency_quality_improvement(self):
        """Trigger emergency quality improvement when quality drops significantly"""
        logger.warning("⚠️ Emergency quality improvement triggered")
        # Implementation would trigger immediate quality enhancement
    
    async def _detect_conversation_themes(self):
        """Detect emerging themes in conversations"""
        # Implementation would analyze conversation themes
        pass
    
    async def _detect_learning_opportunities(self):
        """Detect new learning opportunities from conversation patterns"""
        # Implementation would identify learning opportunities
        pass
    
    async def _detect_curiosity_clusters(self):
        """Detect clusters of curiosity in conversations"""
        # Implementation would identify curiosity patterns
        pass
    
    async def _prioritize_feedback_loops(self):
        """Prioritize multiple feedback loops for optimal execution"""
        # Implementation would prioritize loops
        pass
    
    async def _merge_similar_loops(self):
        """Merge similar feedback loops for efficiency"""
        # Implementation would merge loops
        pass
    
    async def _optimize_loop_execution(self):
        """Optimize execution of feedback loops"""
        # Implementation would optimize execution
        pass
    
    async def _trigger_periodic_improvements(self):
        """Trigger periodic system improvements"""
        # Implementation would trigger improvements
        pass
    
    async def _update_learning_metrics(self):
        """Update learning and improvement metrics"""
        # Implementation would update metrics
        pass
    
    async def _process_redis_chat_message(self, fields: Dict):
        """Process a message from Redis chat streams"""
        # Implementation would process Redis messages
        pass
    
    async def _process_rag_activity(self, metrics: Dict):
        """Process RAG system activity"""
        # Implementation would process RAG activity
        pass
    
    async def _enhance_concept_understanding(self, task: Dict):
        """Enhance understanding of specific concepts"""
        # Implementation would enhance concept understanding
        pass
    
    async def _trigger_concept_exploration(self, interaction: ChatInteraction):
        """Trigger exploration of concepts from interaction"""
        # Implementation would trigger concept exploration
        pass

# FastAPI integration for interactive background coordinator
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Interactive Background Coordinator API", version="1.0.0")

# Global coordinator instance
coordinator: Optional[InteractiveBackgroundCoordinator] = None

@app.on_event("startup")
async def startup_event():
    """Initialize coordinator on startup"""
    global coordinator
    coordinator = InteractiveBackgroundCoordinator()
    await coordinator.initialize()
    # Start coordination in background
    asyncio.create_task(coordinator.start_interactive_coordination())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if coordinator:
        await coordinator.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not coordinator or not coordinator.continuous_learning_active:
        raise HTTPException(status_code=503, detail="Coordinator not running")
    
    return {
        "status": "healthy",
        "continuous_learning_active": coordinator.continuous_learning_active,
        "active_interactions": len(coordinator.active_interactions),
        "metrics": coordinator.metrics
    }

@app.get("/metrics")
async def get_metrics():
    """Get detailed coordinator metrics"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    return await coordinator.get_coordinator_metrics()

@app.post("/trigger_interaction_analysis")
async def trigger_interaction_analysis(interaction_data: dict):
    """Manually trigger analysis of a chat interaction"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    try:
        await coordinator._process_chat_interaction(interaction_data)
        return {"status": "interaction_analyzed", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze interaction: {e}")

@app.post("/force_quality_improvement")
async def force_quality_improvement():
    """Force immediate quality improvement trigger"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    try:
        await coordinator._trigger_emergency_quality_improvement()
        return {"status": "quality_improvement_triggered", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality improvement failed: {e}")

@app.get("/conversation_patterns")
async def get_conversation_patterns():
    """Get current conversation patterns"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    return {
        "patterns": dict(coordinator.conversation_patterns),
        "active_interactions": len(coordinator.active_interactions),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8908)  # New port for interactive coordinator 