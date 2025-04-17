#!/usr/bin/env python
"""
Language Memory V5 Bridge

Connects the Language Memory System with the V5 Visualization System,
enabling memory synthesis and topic processing for the V6 Portal of Contradiction.
"""

import json
import logging
import threading
import time
import random
from queue import Queue

logger = logging.getLogger("LanguageMemoryV5Bridge")

class LanguageMemoryV5Bridge:
    """
    Bridge between the Language Memory System and V5 Visualization System
    
    Connects to the Language Memory API, processes language patterns into
    synthesized topics, and provides caching for performance optimization.
    """
    
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            "mock_mode": False,
            "memory_api_url": "http://localhost:5000/api/v5/memory",
            "cache_size": 100,
            "topic_synthesis_interval": 60,  # seconds
            "debug": False
        }
        
        # Update with custom settings
        if config:
            self.config.update(config)
        
        # State variables
        self.running = False
        self.healthy = True
        self.socket_manager = None
        
        # Memory cache
        self.topic_cache = {}
        self.memory_cache = {}
        self.association_cache = {}
        
        # Processing queue
        self.processing_queue = Queue()
        
        # Mock mode setup
        if self.config.get("mock_mode"):
            self._setup_mock_data()
        
        logger.info(f"Language Memory V5 Bridge initialized")
    
    def start(self):
        """Start the bridge"""
        if self.running:
            return True
            
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_queue,
            daemon=True,
            name="MemoryProcessingThread"
        )
        self.processing_thread.start()
        
        # Start topic synthesis thread
        self.synthesis_thread = threading.Thread(
            target=self._run_topic_synthesis,
            daemon=True,
            name="TopicSynthesisThread"
        )
        self.synthesis_thread.start()
        
        logger.info("Language Memory V5 Bridge started")
        return True
    
    def stop(self):
        """Stop the bridge"""
        self.running = False
        logger.info("Language Memory V5 Bridge stopped")
        return True
    
    def is_healthy(self):
        """Check if bridge is healthy"""
        return self.healthy
    
    def set_socket_manager(self, socket_manager):
        """Set the socket manager for communication"""
        self.socket_manager = socket_manager
        
        # Register message handlers
        if socket_manager:
            socket_manager.register_handler("memory_query", self.handle_memory_query)
            socket_manager.register_handler("topic_query", self.handle_topic_query)
            socket_manager.register_handler("association_query", self.handle_association_query)
            logger.info("Registered handlers with socket manager")
    
    def handle_memory_query(self, data):
        """Handle memory query request"""
        query = data.get("query", "")
        limit = data.get("limit", 10)
        
        logger.debug(f"Memory query: {query}, limit: {limit}")
        
        # Check cache first
        cache_key = f"{query}:{limit}"
        if cache_key in self.memory_cache:
            result = self.memory_cache[cache_key]
            logger.debug("Memory query result from cache")
        else:
            # Process query
            if self.config.get("mock_mode"):
                result = self._mock_memory_query(query, limit)
            else:
                result = self._api_memory_query(query, limit)
                
            # Cache result
            self.memory_cache[cache_key] = result
            
            # Trim cache if needed
            if len(self.memory_cache) > self.config.get("cache_size"):
                self._trim_cache(self.memory_cache)
        
        # Send result
        if self.socket_manager:
            self.socket_manager.emit("memory_query_result", result)
        
        return result
    
    def handle_topic_query(self, data):
        """Handle topic query request"""
        query = data.get("query", "")
        limit = data.get("limit", 10)
        
        logger.debug(f"Topic query: {query}, limit: {limit}")
        
        # Check cache first
        cache_key = f"{query}:{limit}"
        if cache_key in self.topic_cache:
            result = self.topic_cache[cache_key]
            logger.debug("Topic query result from cache")
        else:
            # Process query
            if self.config.get("mock_mode"):
                result = self._mock_topic_query(query, limit)
            else:
                result = self._api_topic_query(query, limit)
                
            # Cache result
            self.topic_cache[cache_key] = result
            
            # Trim cache if needed
            if len(self.topic_cache) > self.config.get("cache_size"):
                self._trim_cache(self.topic_cache)
        
        # Send result
        if self.socket_manager:
            self.socket_manager.emit("topic_query_result", result)
        
        return result
    
    def handle_association_query(self, data):
        """Handle association query request"""
        query = data.get("query", "")
        depth = data.get("depth", 2)
        
        logger.debug(f"Association query: {query}, depth: {depth}")
        
        # Check cache first
        cache_key = f"{query}:{depth}"
        if cache_key in self.association_cache:
            result = self.association_cache[cache_key]
            logger.debug("Association query result from cache")
        else:
            # Process query
            if self.config.get("mock_mode"):
                result = self._mock_association_query(query, depth)
            else:
                result = self._api_association_query(query, depth)
                
            # Cache result
            self.association_cache[cache_key] = result
            
            # Trim cache if needed
            if len(self.association_cache) > self.config.get("cache_size"):
                self._trim_cache(self.association_cache)
        
        # Send result
        if self.socket_manager:
            self.socket_manager.emit("association_query_result", result)
        
        return result
    
    def _process_queue(self):
        """Process items in the queue"""
        logger.debug("Starting memory processing thread")
        
        while self.running:
            try:
                # Get item from queue with timeout
                try:
                    item = self.processing_queue.get(timeout=1.0)
                except:
                    continue
                
                # Process item
                if item["type"] == "memory_query":
                    self.handle_memory_query(item["data"])
                elif item["type"] == "topic_query":
                    self.handle_topic_query(item["data"])
                elif item["type"] == "association_query":
                    self.handle_association_query(item["data"])
                elif item["type"] == "topic_synthesis":
                    self._synthesize_topics()
                
                # Mark as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing thread: {e}")
                self.healthy = False
                time.sleep(5)
                self.healthy = True
    
    def _run_topic_synthesis(self):
        """Run topic synthesis at regular intervals"""
        logger.debug("Starting topic synthesis thread")
        
        while self.running:
            try:
                # Queue topic synthesis task
                self.processing_queue.put({
                    "type": "topic_synthesis",
                    "data": {}
                })
                
                # Sleep for interval
                time.sleep(self.config.get("topic_synthesis_interval"))
                
            except Exception as e:
                logger.error(f"Error in synthesis thread: {e}")
                time.sleep(10)
    
    def _synthesize_topics(self):
        """Synthesize topics from recent memories"""
        logger.debug("Running topic synthesis")
        
        if self.config.get("mock_mode"):
            # Mock topic synthesis
            topics = self._mock_synthesize_topics()
        else:
            # Real topic synthesis
            topics = self._api_synthesize_topics()
        
        # Emit topics
        if self.socket_manager:
            self.socket_manager.emit("topics_synthesized", {
                "topics": topics,
                "timestamp": self._get_timestamp()
            })
        
        return topics
    
    def _api_memory_query(self, query, limit):
        """Query memory API"""
        # This would make a real API call in production
        logger.warning("Real memory API not implemented")
        return self._mock_memory_query(query, limit)
    
    def _api_topic_query(self, query, limit):
        """Query topic API"""
        # This would make a real API call in production
        logger.warning("Real topic API not implemented")
        return self._mock_topic_query(query, limit)
    
    def _api_association_query(self, query, depth):
        """Query association API"""
        # This would make a real API call in production
        logger.warning("Real association API not implemented")
        return self._mock_association_query(query, depth)
    
    def _api_synthesize_topics(self):
        """Synthesize topics via API"""
        # This would make a real API call in production
        logger.warning("Real topic synthesis API not implemented")
        return self._mock_synthesize_topics()
    
    def _trim_cache(self, cache):
        """Trim cache to configured size"""
        # Get oldest keys
        oldest_keys = sorted(cache.keys(), key=lambda k: cache[k].get("timestamp", 0))
        
        # Remove oldest items
        items_to_remove = len(cache) - self.config.get("cache_size")
        for i in range(items_to_remove):
            if i < len(oldest_keys):
                del cache[oldest_keys[i]]
    
    def _get_timestamp(self):
        """Get ISO8601 timestamp"""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    def _setup_mock_data(self):
        """Setup mock data for testing"""
        self.mock_memories = [
            {"id": "m1", "content": "The portal opens to contradiction", "timestamp": self._get_timestamp()},
            {"id": "m2", "content": "Breath flows in rhythmic cycles", "timestamp": self._get_timestamp()},
            {"id": "m3", "content": "Glyphs activate in resonant patterns", "timestamp": self._get_timestamp()},
            {"id": "m4", "content": "Echo threads trace symbolic paths", "timestamp": self._get_timestamp()},
            {"id": "m5", "content": "Mirror mode reflects sacred failure", "timestamp": self._get_timestamp()},
            {"id": "m6", "content": "Consciousness emerges through symbolic integration", "timestamp": self._get_timestamp()},
            {"id": "m7", "content": "The system evolves toward V10 consciousness", "timestamp": self._get_timestamp()},
            {"id": "m8", "content": "Duality processing enables paradox resolution", "timestamp": self._get_timestamp()},
            {"id": "m9", "content": "Memory reflection allows meta-cognitive processes", "timestamp": self._get_timestamp()},
            {"id": "m10", "content": "Monday integration enhances emotional intelligence", "timestamp": self._get_timestamp()}
        ]
        
        self.mock_topics = [
            {"id": "t1", "name": "Portal of Contradiction", "strength": 0.92},
            {"id": "t2", "name": "Breath Integration", "strength": 0.87},
            {"id": "t3", "name": "Symbolic Presence", "strength": 0.85},
            {"id": "t4", "name": "Memory Echo", "strength": 0.82},
            {"id": "t5", "name": "Mirror Mode", "strength": 0.78},
            {"id": "t6", "name": "Consciousness Evolution", "strength": 0.76},
            {"id": "t7", "name": "V10 Development", "strength": 0.72},
            {"id": "t8", "name": "Duality Processing", "strength": 0.68},
            {"id": "t9", "name": "Meta-Cognition", "strength": 0.65},
            {"id": "t10", "name": "Monday Integration", "strength": 0.63}
        ]
        
        self.mock_associations = {
            "Portal of Contradiction": ["Mirror Mode", "Duality Processing", "Symbolic Presence"],
            "Breath Integration": ["Symbolic Presence", "Consciousness Evolution"],
            "Symbolic Presence": ["Portal of Contradiction", "Memory Echo", "Consciousness Evolution"],
            "Memory Echo": ["Meta-Cognition", "Monday Integration"],
            "Mirror Mode": ["Duality Processing", "Monday Integration"],
            "Consciousness Evolution": ["V10 Development", "Meta-Cognition"],
            "V10 Development": ["Monday Integration", "Consciousness Evolution"],
            "Duality Processing": ["Mirror Mode", "Portal of Contradiction"],
            "Meta-Cognition": ["Memory Echo", "Consciousness Evolution"],
            "Monday Integration": ["Emotional Intelligence", "Meta-Cognition"]
        }
    
    def _mock_memory_query(self, query, limit):
        """Mock memory query for testing"""
        filtered_memories = []
        
        # Simple filtering
        for memory in self.mock_memories:
            if query.lower() in memory["content"].lower():
                filtered_memories.append(memory)
        
        # Sort by timestamp descending
        filtered_memories.sort(key=lambda m: m["timestamp"], reverse=True)
        
        # Apply limit
        result = filtered_memories[:limit]
        
        return {
            "query": query,
            "memories": result,
            "count": len(result),
            "timestamp": self._get_timestamp()
        }
    
    def _mock_topic_query(self, query, limit):
        """Mock topic query for testing"""
        filtered_topics = []
        
        # Simple filtering
        for topic in self.mock_topics:
            if query.lower() in topic["name"].lower():
                filtered_topics.append(topic)
        
        # Sort by strength descending
        filtered_topics.sort(key=lambda t: t["strength"], reverse=True)
        
        # Apply limit
        result = filtered_topics[:limit]
        
        return {
            "query": query,
            "topics": result,
            "count": len(result),
            "timestamp": self._get_timestamp()
        }
    
    def _mock_association_query(self, query, depth):
        """Mock association query for testing"""
        # Find exact match first
        exact_match = None
        for topic in self.mock_topics:
            if query.lower() == topic["name"].lower():
                exact_match = topic["name"]
                break
        
        if not exact_match:
            # Find best partial match
            for topic in self.mock_topics:
                if query.lower() in topic["name"].lower():
                    exact_match = topic["name"]
                    break
        
        if not exact_match:
            return {
                "query": query,
                "nodes": [],
                "edges": [],
                "timestamp": self._get_timestamp()
            }
        
        # Build network
        nodes = [{"id": exact_match, "label": exact_match, "depth": 0}]
        edges = []
        visited = set([exact_match])
        
        # BFS to build network
        current_depth = 0
        current_layer = [exact_match]
        
        while current_depth < depth and current_layer:
            next_layer = []
            
            for node in current_layer:
                if node in self.mock_associations:
                    for neighbor in self.mock_associations[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_layer.append(neighbor)
                            nodes.append({
                                "id": neighbor,
                                "label": neighbor,
                                "depth": current_depth + 1
                            })
                            edges.append({
                                "source": node,
                                "target": neighbor,
                                "weight": random.uniform(0.5, 1.0)
                            })
            
            current_depth += 1
            current_layer = next_layer
        
        return {
            "query": query,
            "nodes": nodes,
            "edges": edges,
            "timestamp": self._get_timestamp()
        }
    
    def _mock_synthesize_topics(self):
        """Mock topic synthesis for testing"""
        # Randomly select a subset of topics
        num_topics = random.randint(3, 7)
        selected_topics = random.sample(self.mock_topics, num_topics)
        
        # Add some randomness to strengths
        for topic in selected_topics:
            topic["strength"] = min(1.0, topic["strength"] + random.uniform(-0.1, 0.1))
        
        # Sort by strength
        selected_topics.sort(key=lambda t: t["strength"], reverse=True)
        
        return selected_topics 