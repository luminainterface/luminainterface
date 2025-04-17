"""
Memory Consolidator Module for V7 Dream Mode

This module implements the Memory Consolidation component of the Dream Mode system,
which processes and strengthens recently acquired information during dream states.
"""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
import threading

# Set up logging
logger = logging.getLogger("lumina_v7.memory_consolidator")

class MemoryConsolidator:
    """
    Processes and strengthens recently acquired memories during dream state
    
    Key features:
    - Recency bias for prioritizing recent memories
    - Emotional tagging for deeper processing of emotional content
    - Pattern reinforcement for strengthening frequently accessed patterns
    - Connection creation between related concepts
    - Contradiction resolution attempts
    """
    
    def __init__(self, node_manager=None, recency_bias: float = 0.8, 
                 emotional_tag_weight: float = 0.6, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Memory Consolidator
        
        Args:
            node_manager: NodeConsciousnessManager instance (optional)
            recency_bias: Weight factor for recency (0.0-1.0)
            emotional_tag_weight: Weight factor for emotional content (0.0-1.0)
            config: Configuration dictionary (optional)
        """
        # Default configuration
        self.config = {
            "max_consolidation_batch": 20,  # Maximum memories to process in one batch
            "memory_threshold_hours": 72,   # Consider memories from last 72 hours
            "consolidation_interval": 5.0,  # Seconds between consolidation batches
            "contradiction_resolution_enabled": True,  # Attempt to resolve contradictions
            "min_connection_strength": 0.4, # Minimum strength for new connections
            "max_consolidation_time": 300,  # Maximum seconds to spend on consolidation
            "connection_creation_probability": 0.65  # Probability of creating new connections
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # Parameters
        self.recency_bias = max(0.0, min(1.0, recency_bias))
        self.emotional_tag_weight = max(0.0, min(1.0, emotional_tag_weight))
        
        # External components
        self.node_manager = node_manager
        
        # State
        self.processing_stats = {
            "total_memories_consolidated": 0,
            "total_connections_created": 0,
            "total_patterns_reinforced": 0,
            "total_contradictions_resolved": 0,
            "last_consolidation_time": None
        }
        
        # Locking
        self.consolidation_lock = threading.Lock()
        
        logger.info(f"Memory Consolidator initialized with recency_bias={self.recency_bias}, "
                   f"emotional_tag_weight={self.emotional_tag_weight}")
    
    def consolidate_memories(self, intensity: float = 0.7, 
                            time_limit: Optional[float] = None) -> Dict[str, Any]:
        """
        Consolidate memories with the given intensity
        
        Args:
            intensity: Processing intensity (0.0-1.0)
            time_limit: Maximum time to spend in seconds (None for default)
            
        Returns:
            Dict with consolidation results
        """
        # Use default time limit if not specified
        if time_limit is None:
            time_limit = self.config["max_consolidation_time"]
        
        # Ensure reasonable intensity
        intensity = max(0.1, min(1.0, intensity))
        
        # Use lock to prevent concurrent consolidation
        if not self.consolidation_lock.acquire(blocking=False):
            logger.info("Memory consolidation already in progress, skipping")
            return {"status": "busy", "message": "Consolidation already in progress"}
        
        try:
            start_time = time.time()
            logger.info(f"Starting memory consolidation with intensity {intensity}")
            
            # Track results for this consolidation run
            results = {
                "start_time": datetime.now().isoformat(),
                "intensity": intensity,
                "memories_consolidated": 0,
                "connections_created": 0,
                "patterns_reinforced": 0,
                "contradictions_resolved": 0,
                "status": "completed",
                "memory_details": []
            }
            
            # Retrieve recent memories to consolidate
            recent_memories = self._get_recent_memories()
            
            # Skip if no memories
            if not recent_memories:
                logger.info("No recent memories to consolidate")
                results["status"] = "no_memories"
                return results
            
            # Prioritize memories based on recency and emotional content
            prioritized_memories = self._prioritize_memories(recent_memories)
            
            # Calculate batch size based on intensity
            batch_size = int(self.config["max_consolidation_batch"] * intensity)
            batch_size = max(1, min(batch_size, len(prioritized_memories)))
            
            # Process each memory in the prioritized order
            for i, memory in enumerate(prioritized_memories[:batch_size]):
                # Check if we've exceeded the time limit
                if time.time() - start_time > time_limit:
                    logger.info(f"Time limit reached after processing {i} memories")
                    results["status"] = "time_limit_reached"
                    break
                
                # Process this memory
                memory_result = self._process_memory(memory, intensity)
                
                # Update statistics
                results["memories_consolidated"] += 1
                results["connections_created"] += memory_result.get("connections_created", 0)
                results["patterns_reinforced"] += memory_result.get("patterns_reinforced", 0)
                results["contradictions_resolved"] += memory_result.get("contradictions_resolved", 0)
                
                # Add memory details if interesting
                if memory_result.get("interesting", False):
                    results["memory_details"].append(memory_result)
                
                # Small sleep between memories to not overload the system
                time.sleep(self.config["consolidation_interval"] / batch_size)
            
            # Update processing stats
            self.processing_stats["total_memories_consolidated"] += results["memories_consolidated"]
            self.processing_stats["total_connections_created"] += results["connections_created"]
            self.processing_stats["total_patterns_reinforced"] += results["patterns_reinforced"]
            self.processing_stats["total_contradictions_resolved"] += results["contradictions_resolved"]
            self.processing_stats["last_consolidation_time"] = datetime.now().isoformat()
            
            # Log results
            logger.info(f"Memory consolidation completed: {results['memories_consolidated']} memories, "
                       f"{results['connections_created']} connections, "
                       f"{results['patterns_reinforced']} patterns, "
                       f"{results['contradictions_resolved']} contradictions resolved")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
            return {
                "status": "error",
                "error": str(e),
                "start_time": datetime.now().isoformat(),
                "intensity": intensity
            }
        finally:
            # Always release the lock
            self.consolidation_lock.release()
    
    def _get_recent_memories(self) -> List[Dict[str, Any]]:
        """
        Get recent memories for consolidation
        
        Returns:
            List of memory objects to consolidate
        """
        # If we have a node manager, try to get memories from it
        if self.node_manager:
            try:
                # Get memory nodes from node manager
                memory_nodes = self.node_manager.get_nodes_by_type("memory")
                
                # If we have memory nodes, collect memories
                if memory_nodes:
                    all_memories = []
                    for node_id, node in memory_nodes.items():
                        if hasattr(node, "get_recent_memories"):
                            # Calculate threshold time
                            hours = self.config["memory_threshold_hours"]
                            threshold_time = datetime.now() - timedelta(hours=hours)
                            
                            # Get memories from this node
                            node_memories = node.get_recent_memories(
                                since=threshold_time,
                                limit=100  # Reasonable limit per node
                            )
                            
                            if node_memories:
                                all_memories.extend(node_memories)
                    
                    return all_memories
            except Exception as e:
                logger.error(f"Error getting memories from node manager: {e}")
        
        # If no memories from node manager or error, use mock memories
        return self._generate_mock_memories()
    
    def _generate_mock_memories(self) -> List[Dict[str, Any]]:
        """
        Generate mock memories for testing
        
        Returns:
            List of mock memory objects
        """
        mock_memories = []
        
        # Generate some mock memories
        memory_count = random.randint(10, 30)
        
        # Topics for mock memories
        topics = ["neural networks", "consciousness", "language", "dreams", 
                 "patterns", "learning", "communication", "creativity"]
        
        # Emotions for tagging
        emotions = ["neutral", "interest", "joy", "surprise", "confusion", "concern"]
        emotion_weights = [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]
        
        # Generate memories
        for i in range(memory_count):
            # Create random time within threshold
            hours_ago = random.uniform(0, self.config["memory_threshold_hours"])
            timestamp = datetime.now() - timedelta(hours=hours_ago)
            
            # Select random topic and emotion
            topic = random.choice(topics)
            emotion = random.choices(emotions, weights=emotion_weights)[0]
            
            # Create memory
            memory = {
                "id": f"mock_memory_{i}",
                "timestamp": timestamp.isoformat(),
                "topic": topic,
                "content": f"Mock memory content about {topic}",
                "source": "mock_generator",
                "emotional_tag": emotion,
                "emotional_intensity": random.uniform(0.1, 0.9) if emotion != "neutral" else 0.1,
                "access_count": random.randint(1, 10),
                "last_access": (datetime.now() - timedelta(hours=random.uniform(0, hours_ago))).isoformat(),
                "connections": []
            }
            
            # Add some connections to other topics
            connection_count = random.randint(0, 3)
            for j in range(connection_count):
                other_topic = random.choice([t for t in topics if t != topic])
                memory["connections"].append({
                    "topic": other_topic,
                    "strength": random.uniform(0.3, 0.8),
                    "type": random.choice(["association", "causation", "similarity", "contrast"])
                })
            
            mock_memories.append(memory)
        
        logger.info(f"Generated {len(mock_memories)} mock memories for consolidation")
        return mock_memories
    
    def _prioritize_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize memories based on recency and emotional content
        
        Args:
            memories: List of memory objects
            
        Returns:
            List of memories sorted by priority
        """
        # Calculate priority score for each memory
        for memory in memories:
            # Calculate recency score (1.0 = now, 0.0 = threshold_hours ago)
            timestamp = datetime.fromisoformat(memory["timestamp"])
            hours_since = (datetime.now() - timestamp).total_seconds() / 3600
            recency_score = max(0.0, 1.0 - (hours_since / self.config["memory_threshold_hours"]))
            
            # Get emotional intensity (default to 0.1 if not present)
            emotional_intensity = memory.get("emotional_intensity", 0.1)
            
            # Calculate access frequency score (more accesses = higher score)
            access_count = memory.get("access_count", 1)
            access_score = min(1.0, access_count / 10)  # Cap at 10 accesses for a 1.0 score
            
            # Calculate overall priority score
            # Combine recency, emotional intensity, and access frequency
            priority = (
                (recency_score * self.recency_bias) + 
                (emotional_intensity * self.emotional_tag_weight) +
                (access_score * (1.0 - self.recency_bias - self.emotional_tag_weight))
            )
            
            # Store priority score
            memory["_priority_score"] = priority
        
        # Sort by priority score (descending)
        sorted_memories = sorted(memories, key=lambda m: m.get("_priority_score", 0), reverse=True)
        
        return sorted_memories
    
    def _process_memory(self, memory: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """
        Process a single memory for consolidation
        
        Args:
            memory: Memory object to process
            intensity: Processing intensity (0.0-1.0)
            
        Returns:
            Dict with processing results
        """
        # Initialize result
        result = {
            "memory_id": memory.get("id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "connections_created": 0,
            "patterns_reinforced": 0,
            "contradictions_resolved": 0,
            "interesting": False
        }
        
        # Simulate memory processing
        # In a real implementation, this would interact with actual memory systems
        
        # 1. Pattern reinforcement
        if random.random() < intensity:
            pattern_count = random.randint(1, 3)
            result["patterns_reinforced"] = pattern_count
            
            # More patterns reinforced makes this interesting
            if pattern_count >= 2:
                result["interesting"] = True
        
        # 2. Connection creation (more likely with higher intensity)
        if random.random() < (self.config["connection_creation_probability"] * intensity):
            connection_count = random.randint(1, 2)
            result["connections_created"] = connection_count
            
            # Any new connection makes this interesting
            if connection_count > 0:
                result["interesting"] = True
        
        # 3. Contradiction resolution (if enabled)
        if self.config["contradiction_resolution_enabled"] and random.random() < (intensity * 0.3):
            # Contradictions are rare
            if random.random() < 0.2:
                result["contradictions_resolved"] = 1
                result["interesting"] = True
                
                # Add details about contradiction
                result["contradiction_details"] = {
                    "type": random.choice(["logical", "temporal", "contextual"]),
                    "resolution_method": random.choice(["context_binding", "separation", "synthesis"]),
                    "resolution_confidence": random.uniform(0.5, 0.9)
                }
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get consolidation statistics
        
        Returns:
            Dict with consolidation statistics
        """
        return self.processing_stats.copy() 