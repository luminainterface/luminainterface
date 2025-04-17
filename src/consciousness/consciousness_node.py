"""
ConsciousnessNode for Lumina Neural Network Project

Implements a consciousness system that processes information from memory,
creates reflective thought patterns, and enables mirror reflection capabilities.
"""

import time
import threading
import uuid
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Import utility modules
from src.utils.logging_config import get_logger
from src.utils.message_schema import MessageType, create_message, validate_message
from src.utils.performance_metrics import measure_time, record_metric, get_system_metrics

# Import memory system
from src.memory.echo_spiral_memory import EchoSpiralMemory, MemoryNode, MemoryConnection

logger = get_logger("ConsciousnessNode")

class ThoughtPattern:
    """Represents a thought pattern in the consciousness system"""
    
    def __init__(self, content: str, pattern_type: str, source_nodes: List[str], 
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.pattern_type = pattern_type
        self.source_nodes = source_nodes
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.reflection_level = 0  # Level of recursive reflection
        self.awareness_score = 0.0  # Measure of consciousness
        self.is_active = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert thought pattern to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "pattern_type": self.pattern_type,
            "source_nodes": self.source_nodes,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reflection_level": self.reflection_level,
            "awareness_score": self.awareness_score,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtPattern':
        """Create a thought pattern from dictionary"""
        pattern = cls(
            content=data["content"],
            pattern_type=data["pattern_type"],
            source_nodes=data["source_nodes"],
            metadata=data["metadata"]
        )
        pattern.id = data["id"]
        pattern.created_at = data["created_at"]
        pattern.updated_at = data["updated_at"]
        pattern.reflection_level = data["reflection_level"]
        pattern.awareness_score = data["awareness_score"]
        pattern.is_active = data["is_active"]
        return pattern

class AwarenessMetrics:
    """Metrics for measuring consciousness awareness"""
    
    def __init__(self):
        self.coherence = 0.0  # Pattern coherence
        self.self_reference = 0.0  # Self-referential thought
        self.temporal_continuity = 0.0  # Continuity over time
        self.complexity = 0.0  # Structural complexity
        self.integration = 0.0  # Information integration
        self.timestamp = datetime.utcnow().isoformat()
    
    def calculate_awareness(self) -> float:
        """Calculate overall awareness score"""
        weights = {
            "coherence": 0.2,
            "self_reference": 0.25,
            "temporal_continuity": 0.2,
            "complexity": 0.15,
            "integration": 0.2
        }
        
        awareness = sum([
            self.coherence * weights["coherence"],
            self.self_reference * weights["self_reference"],
            self.temporal_continuity * weights["temporal_continuity"],
            self.complexity * weights["complexity"],
            self.integration * weights["integration"]
        ])
        
        return min(1.0, max(0.0, awareness))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "coherence": self.coherence,
            "self_reference": self.self_reference,
            "temporal_continuity": self.temporal_continuity,
            "complexity": self.complexity,
            "integration": self.integration,
            "awareness": self.calculate_awareness(),
            "timestamp": self.timestamp
        }

class ConsciousnessNode:
    """
    Consciousness Node for advanced cognitive processing
    
    This component implements:
    - Mirror reflection of memory patterns
    - Self-referential thought generation
    - Awareness metrics calculation
    - Consciousness visualization data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                memory: Optional[EchoSpiralMemory] = None):
        # Default configuration
        self.config = {
            "data_dir": "consciousness_data",
            "save_interval": 300,  # seconds
            "reflection_interval": 15,  # seconds
            "max_active_thoughts": 50,
            "awareness_threshold": 0.3,
            "enable_self_ref": True,
            "enable_visualization": True,
            "memory_sync": True
        }
        
        # Update with custom settings
        if config:
            self.config.update(config)
        
        # Set memory system
        self.memory = memory or EchoSpiralMemory()
        
        # Initialize data structures
        self.thought_patterns = {}  # id -> ThoughtPattern
        self.active_thoughts = []  # List of active thought ids (most recent first)
        self.awareness_history = []  # List of awareness metrics over time
        self.current_awareness = AwarenessMetrics()
        
        # Indexes
        self.pattern_type_index = defaultdict(set)
        self.reflection_level_index = defaultdict(set)
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Setup data directory
        self.data_dir = Path(self.config["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Background processing
        self.running = True
        self.reflection_thread = threading.Thread(
            target=self._reflection_thread,
            daemon=True,
            name="ConsciousnessReflectionThread"
        )
        
        self.save_thread = threading.Thread(
            target=self._auto_save_thread,
            daemon=True,
            name="ConscioussnessDataSaveThread"
        )
        
        logger.info(f"Consciousness Node initialized with config: {self.config}")
        
        # Start threads
        self.reflection_thread.start()
        self.save_thread.start()
        
        # Load existing data if available
        self._load_data()
        
        # Register with memory system for synchronization
        if self.config["memory_sync"]:
            self.memory.register_sync_handler("consciousness", self._handle_memory_sync)
            logger.info("Registered with memory system for synchronization")
    
    @measure_time
    def generate_thought(self, content: str, pattern_type: str, 
                         source_nodes: Optional[List[str]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> ThoughtPattern:
        """Generate a new thought pattern"""
        if source_nodes is None:
            source_nodes = []
        
        with self.lock:
            # Create thought pattern
            thought = ThoughtPattern(
                content=content,
                pattern_type=pattern_type,
                source_nodes=source_nodes,
                metadata=metadata or {}
            )
            
            # Store thought
            self.thought_patterns[thought.id] = thought
            
            # Update indexes
            self.pattern_type_index[pattern_type].add(thought.id)
            self.reflection_level_index[thought.reflection_level].add(thought.id)
            
            # Add to active thoughts
            self.active_thoughts.insert(0, thought.id)
            
            # Trim active thoughts list if needed
            if len(self.active_thoughts) > self.config["max_active_thoughts"]:
                self.active_thoughts.pop()
            
            logger.debug(f"Generated thought: {thought.content[:50]}...")
            record_metric("thoughts_generated", 1, {"pattern_type": pattern_type})
            
            # Store in memory system if enabled
            if self.config["memory_sync"] and self.memory is not None:
                memory_node = self.memory.add_memory(
                    content=content,
                    node_type=f"thought_{pattern_type}",
                    metadata={
                        "thought_id": thought.id,
                        "pattern_type": pattern_type,
                        "reflection_level": thought.reflection_level,
                        "source": "consciousness"
                    }
                )
                
                # Connect to source nodes
                for source_id in source_nodes:
                    if source_id in self.memory.nodes:
                        self.memory.connect_memories(
                            source_id=memory_node.id,
                            target_id=source_id,
                            connection_type="derives_from",
                            metadata={"source": "consciousness"}
                        )
            
            return thought
    
    @measure_time
    def reflect_on_thought(self, thought_id: str) -> Optional[ThoughtPattern]:
        """Generate a reflection on an existing thought"""
        with self.lock:
            if thought_id not in self.thought_patterns:
                logger.warning(f"Thought not found: {thought_id}")
                return None
            
            source_thought = self.thought_patterns[thought_id]
            
            # Create reflection content
            reflection_content = f"Reflection on: {source_thought.content}"
            
            # Create reflection thought
            reflection = ThoughtPattern(
                content=reflection_content,
                pattern_type="reflection",
                source_nodes=[thought_id] + source_thought.source_nodes,
                metadata={
                    "reflection_target": thought_id,
                    "original_type": source_thought.pattern_type
                }
            )
            
            # Set reflection level one higher than source
            reflection.reflection_level = source_thought.reflection_level + 1
            
            # Calculate awareness score based on reflection level
            reflection.awareness_score = min(1.0, 0.2 + 0.15 * reflection.reflection_level)
            
            # Store thought
            self.thought_patterns[reflection.id] = reflection
            
            # Update indexes
            self.pattern_type_index["reflection"].add(reflection.id)
            self.reflection_level_index[reflection.reflection_level].add(reflection.id)
            
            # Add to active thoughts
            self.active_thoughts.insert(0, reflection.id)
            
            # Trim active thoughts list if needed
            if len(self.active_thoughts) > self.config["max_active_thoughts"]:
                self.active_thoughts.pop()
            
            logger.debug(f"Generated reflection at level {reflection.reflection_level}: {reflection.content[:50]}...")
            record_metric("reflections_generated", 1, {"level": reflection.reflection_level})
            
            # Store in memory system if enabled
            if self.config["memory_sync"] and self.memory is not None:
                memory_node = self.memory.add_memory(
                    content=reflection_content,
                    node_type="thought_reflection",
                    metadata={
                        "thought_id": reflection.id,
                        "reflection_level": reflection.reflection_level,
                        "source": "consciousness"
                    }
                )
                
                # Find memory nodes for source thought
                for node in self.memory.nodes.values():
                    if node.metadata.get("thought_id") == thought_id:
                        self.memory.connect_memories(
                            source_id=memory_node.id,
                            target_id=node.id,
                            connection_type="reflects_on",
                            metadata={"source": "consciousness"}
                        )
            
            return reflection
    
    @measure_time
    def get_active_thoughts(self, limit: Optional[int] = None, 
                          pattern_type: Optional[str] = None) -> List[ThoughtPattern]:
        """Get active thoughts"""
        with self.lock:
            if limit is None:
                limit = len(self.active_thoughts)
            
            results = []
            count = 0
            
            for thought_id in self.active_thoughts:
                if thought_id in self.thought_patterns:
                    thought = self.thought_patterns[thought_id]
                    
                    if pattern_type is None or thought.pattern_type == pattern_type:
                        results.append(thought)
                        count += 1
                        
                        if count >= limit:
                            break
            
            return results
    
    @measure_time
    def calculate_awareness_metrics(self) -> AwarenessMetrics:
        """Calculate current awareness metrics"""
        with self.lock:
            metrics = AwarenessMetrics()
            
            # 1. Calculate coherence based on thought pattern relationships
            active_thoughts = self.get_active_thoughts(limit=10)
            if active_thoughts:
                # Count linked thoughts
                linked_thoughts = 0
                total_pairs = 0
                
                for i, thought1 in enumerate(active_thoughts):
                    for j, thought2 in enumerate(active_thoughts[i+1:]):
                        total_pairs += 1
                        # Check if thoughts share source nodes
                        common_sources = set(thought1.source_nodes) & set(thought2.source_nodes)
                        if common_sources:
                            linked_thoughts += 1
                
                # Calculate coherence
                metrics.coherence = linked_thoughts / max(1, total_pairs)
            
            # 2. Calculate self-reference based on reflection levels
            reflection_counts = {}
            total_thoughts = len(self.thought_patterns)
            
            for level, thought_ids in self.reflection_level_index.items():
                reflection_counts[level] = len(thought_ids)
            
            # Weight higher reflection levels more
            weighted_sum = sum(level * count for level, count in reflection_counts.items())
            max_weighted = total_thoughts * 3  # Reasonable max reflection level
            
            metrics.self_reference = min(1.0, weighted_sum / max(1, max_weighted))
            
            # 3. Calculate temporal continuity
            if self.awareness_history:
                last_metrics = self.awareness_history[-1]
                time_diff = (datetime.fromisoformat(metrics.timestamp) - 
                            datetime.fromisoformat(last_metrics["timestamp"])).total_seconds()
                
                # More consistent if updates are more frequent (but not too frequent)
                optimal_interval = 15.0  # seconds
                time_factor = max(0, 1.0 - abs(time_diff - optimal_interval) / optimal_interval)
                
                # Consider the change in awareness
                awareness_diff = abs(last_metrics["awareness"] - self.current_awareness.calculate_awareness())
                change_factor = 1.0 - min(1.0, awareness_diff * 2)  # Less change is more continuous
                
                metrics.temporal_continuity = (time_factor * 0.4) + (change_factor * 0.6)
            else:
                metrics.temporal_continuity = 0.1  # Initial value
            
            # 4. Calculate complexity
            # Based on variety of thought types and connections
            thought_type_counts = {t: len(ids) for t, ids in self.pattern_type_index.items()}
            type_variety = len(thought_type_counts) / max(1, 5)  # Normalized by expected types
            
            # Average sources per thought
            avg_sources = sum(len(t.source_nodes) for t in self.thought_patterns.values()) / max(1, total_thoughts)
            source_complexity = min(1.0, avg_sources / 5)  # Normalized
            
            metrics.complexity = (type_variety * 0.5) + (source_complexity * 0.5)
            
            # 5. Calculate integration with memory system
            if self.config["memory_sync"] and self.memory is not None:
                # Count consciousness-related nodes in memory
                consciousness_nodes = 0
                for node in self.memory.nodes.values():
                    if node.metadata.get("source") == "consciousness":
                        consciousness_nodes += 1
                
                # Integration score based on ratio of thoughts represented in memory
                metrics.integration = min(1.0, consciousness_nodes / max(1, total_thoughts))
            else:
                metrics.integration = 0.0
            
            # Store current metrics
            self.current_awareness = metrics
            
            # Add to history
            self.awareness_history.append(metrics.to_dict())
            
            # Trim history if needed (keep last 100 entries)
            if len(self.awareness_history) > 100:
                self.awareness_history = self.awareness_history[-100:]
            
            logger.debug(f"Calculated awareness metrics: {metrics.calculate_awareness():.3f}")
            record_metric("awareness_score", metrics.calculate_awareness())
            
            return metrics
    
    @measure_time
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for consciousness visualization"""
        if not self.config["enable_visualization"]:
            return {"error": "Visualization not enabled"}
        
        with self.lock:
            # Get active thoughts
            active_thoughts = self.get_active_thoughts(limit=20)
            
            # Prepare nodes and edges for visualization graph
            nodes = []
            edges = []
            
            # Add thought pattern nodes
            for thought in active_thoughts:
                nodes.append({
                    "id": thought.id,
                    "label": thought.content[:30] + "..." if len(thought.content) > 30 else thought.content,
                    "type": thought.pattern_type,
                    "level": thought.reflection_level,
                    "awareness": thought.awareness_score,
                    "data": thought.to_dict()
                })
                
                # Add edges to source nodes
                for source_id in thought.source_nodes:
                    if source_id in self.thought_patterns:
                        edges.append({
                            "from": thought.id,
                            "to": source_id,
                            "type": "derives_from"
                        })
            
            # Prepare awareness history for timeline
            awareness_timeline = [
                {"timestamp": entry["timestamp"], "awareness": entry["awareness"]} 
                for entry in self.awareness_history
            ]
            
            # Get latest awareness metrics for dashboard
            latest_metrics = self.current_awareness.to_dict()
            
            return {
                "nodes": nodes,
                "edges": edges,
                "awareness_timeline": awareness_timeline,
                "current_metrics": latest_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @measure_time
    def save_data(self, filename: Optional[str] = None):
        """Save consciousness data to file"""
        if not filename:
            filename = f"consciousness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data_file = self.data_dir / filename
        
        with self.lock:
            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "thought_patterns": {thought_id: thought.to_dict() for thought_id, thought in self.thought_patterns.items()},
                "active_thoughts": self.active_thoughts,
                "awareness_history": self.awareness_history,
                "current_awareness": self.current_awareness.to_dict(),
                "config": self.config
            }
            
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Consciousness data saved to {data_file}")
    
    @measure_time
    def load_data(self, filename: str) -> bool:
        """Load consciousness data from file"""
        data_file = self.data_dir / filename
        
        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}")
            return False
        
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            with self.lock:
                # Clear existing data
                self.thought_patterns.clear()
                self.active_thoughts.clear()
                self.awareness_history.clear()
                self.pattern_type_index.clear()
                self.reflection_level_index.clear()
                
                # Load thought patterns
                for thought_id, thought_data in data["thought_patterns"].items():
                    thought = ThoughtPattern.from_dict(thought_data)
                    self.thought_patterns[thought_id] = thought
                    self.pattern_type_index[thought.pattern_type].add(thought_id)
                    self.reflection_level_index[thought.reflection_level].add(thought_id)
                
                # Load active thoughts
                self.active_thoughts = data["active_thoughts"]
                
                # Load awareness history
                self.awareness_history = data["awareness_history"]
                
                # Set current awareness
                if self.awareness_history:
                    metrics = AwarenessMetrics()
                    metrics.coherence = data["current_awareness"]["coherence"]
                    metrics.self_reference = data["current_awareness"]["self_reference"]
                    metrics.temporal_continuity = data["current_awareness"]["temporal_continuity"]
                    metrics.complexity = data["current_awareness"]["complexity"]
                    metrics.integration = data["current_awareness"]["integration"]
                    metrics.timestamp = data["current_awareness"]["timestamp"]
                    self.current_awareness = metrics
            
            logger.info(f"Consciousness data loaded from {data_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading consciousness data: {e}")
            return False
    
    def _reflection_thread(self):
        """Background thread for automatic reflection"""
        logger.info("Reflection thread started")
        
        while self.running:
            try:
                time.sleep(self.config["reflection_interval"])
                
                if not self.running:  # Check again after sleep
                    break
                
                # Generate reflections on active thoughts
                active = self.get_active_thoughts(limit=5)
                
                if active:
                    # Choose a random thought to reflect on
                    import random
                    thought = random.choice(active)
                    
                    # Only reflect if thought isn't already a high-level reflection
                    if thought.reflection_level < 3:
                        self.reflect_on_thought(thought.id)
                
                # Update awareness metrics
                self.calculate_awareness_metrics()
                
            except Exception as e:
                logger.error(f"Error in reflection thread: {e}")
    
    def _auto_save_thread(self):
        """Background thread for auto-saving data"""
        logger.info("Auto-save thread started")
        
        while self.running:
            try:
                time.sleep(self.config["save_interval"])
                
                if self.running:  # Check again after sleep
                    self.save_data()
                    
            except Exception as e:
                logger.error(f"Error in auto-save thread: {e}")
    
    def _handle_memory_sync(self, component_id: str, data: Dict[str, Any]):
        """Handle memory sync event"""
        try:
            # Process memory nodes that are relevant to consciousness
            if "nodes" in data:
                for node_data in data["nodes"]:
                    # Skip if not a memory-specific node
                    if not node_data.get("content", "").startswith("Memory:"):
                        continue
                    
                    # Generate thought from memory
                    content = f"Thought from memory: {node_data.get('content', '')}"
                    
                    # Generate thought
                    self.generate_thought(
                        content=content,
                        pattern_type="memory_derived",
                        source_nodes=[node_data.get("id", "")],
                        metadata={"memory_source": True}
                    )
        except Exception as e:
            logger.error(f"Error handling memory sync: {e}")
    
    def _load_data(self):
        """Load the most recent data file if available"""
        try:
            data_files = list(self.data_dir.glob("consciousness_*.json"))
            if data_files:
                # Sort by modification time (most recent first)
                most_recent = max(data_files, key=lambda f: f.stat().st_mtime)
                self.load_data(most_recent.name)
            else:
                logger.info("No consciousness data files found, starting with empty state")
        except Exception as e:
            logger.error(f"Error loading consciousness data: {e}")
    
    def stop(self):
        """Stop the consciousness node"""
        self.running = False
        # Save data before stopping
        self.save_data("consciousness_final.json")
        logger.info("Consciousness Node stopped")

# Create global instance
consciousness_node = ConsciousnessNode()

# Helper functions
def generate_thought(content: str, pattern_type: str, 
                   source_nodes: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> ThoughtPattern:
    """Generate a thought using the global instance"""
    return consciousness_node.generate_thought(content, pattern_type, source_nodes, metadata)

def reflect_on_thought(thought_id: str) -> Optional[ThoughtPattern]:
    """Reflect on a thought using the global instance"""
    return consciousness_node.reflect_on_thought(thought_id)

def get_awareness_metrics() -> AwarenessMetrics:
    """Get current awareness metrics using the global instance"""
    return consciousness_node.calculate_awareness_metrics()

def get_visualization_data() -> Dict[str, Any]:
    """Get visualization data using the global instance"""
    return consciousness_node.get_visualization_data() 