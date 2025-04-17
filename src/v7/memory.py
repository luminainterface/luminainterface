"""
Simple Memory Module for LUMINA V7

This module provides a minimal memory implementation to address the missing
'src.v7.memory' module error.
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleMemory:
    """
    Simple memory implementation for LUMINA V7
    
    This class provides basic memory storage and retrieval functionality
    for LUMINA V7 systems that don't have the full memory implementation.
    """
    
    def __init__(self, file_path=None):
        """
        Initialize the memory system
        
        Args:
            file_path: Path to the memory file (optional)
        """
        self.data_dir = Path("data") / "memory"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_path = file_path or self.data_dir / "simple_memory.json"
        self.memories = self._load_memories()
        
        logger.info(f"SimpleMemory initialized at {self.file_path}")
    
    def _load_memories(self):
        """Load memories from disk"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading memories: {e}")
        
        # Return default structure if file doesn't exist or has errors
        return {
            "facts": [],
            "concepts": [],
            "experiences": [],
            "meta": {
                "last_saved": time.time(),
                "version": "1.0"
            }
        }
    
    def save(self):
        """Save memories to disk"""
        try:
            self.memories["meta"]["last_saved"] = time.time()
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
            return False
    
    def add_fact(self, content, source=None, importance=0.5):
        """
        Add a fact to memory
        
        Args:
            content: Fact content
            source: Source of the fact (optional)
            importance: Importance score (0.0 to 1.0)
            
        Returns:
            str: Memory ID
        """
        memory_id = f"fact_{len(self.memories['facts'])}"
        
        fact = {
            "id": memory_id,
            "content": content,
            "source": source,
            "importance": importance,
            "created": time.time(),
            "last_accessed": time.time()
        }
        
        self.memories["facts"].append(fact)
        self.save()
        return memory_id
    
    def add_concept(self, name, description, relations=None, importance=0.5):
        """
        Add a concept to memory
        
        Args:
            name: Concept name
            description: Concept description
            relations: Related concepts (optional)
            importance: Importance score (0.0 to 1.0)
            
        Returns:
            str: Memory ID
        """
        memory_id = f"concept_{len(self.memories['concepts'])}"
        
        concept = {
            "id": memory_id,
            "name": name,
            "description": description,
            "relations": relations or [],
            "importance": importance,
            "created": time.time(),
            "last_accessed": time.time()
        }
        
        self.memories["concepts"].append(concept)
        self.save()
        return memory_id
    
    def add_experience(self, content, context=None, importance=0.5):
        """
        Add an experience to memory
        
        Args:
            content: Experience content
            context: Experience context (optional)
            importance: Importance score (0.0 to 1.0)
            
        Returns:
            str: Memory ID
        """
        memory_id = f"experience_{len(self.memories['experiences'])}"
        
        experience = {
            "id": memory_id,
            "content": content,
            "context": context,
            "importance": importance,
            "created": time.time(),
            "last_accessed": time.time()
        }
        
        self.memories["experiences"].append(experience)
        self.save()
        return memory_id
    
    def get_memories(self, memory_type=None, limit=10):
        """
        Get memories of a specific type
        
        Args:
            memory_type: Type of memories to get (facts, concepts, experiences, or None for all)
            limit: Maximum number of memories to return
            
        Returns:
            list: Memories
        """
        if memory_type == "facts":
            memories = sorted(self.memories["facts"], key=lambda x: x["importance"], reverse=True)
            return memories[:limit]
        elif memory_type == "concepts":
            memories = sorted(self.memories["concepts"], key=lambda x: x["importance"], reverse=True)
            return memories[:limit]
        elif memory_type == "experiences":
            memories = sorted(self.memories["experiences"], key=lambda x: x["importance"], reverse=True)
            return memories[:limit]
        else:
            # Return all types
            all_memories = []
            all_memories.extend(self.memories["facts"])
            all_memories.extend(self.memories["concepts"])
            all_memories.extend(self.memories["experiences"])
            all_memories = sorted(all_memories, key=lambda x: x.get("importance", 0), reverse=True)
            return all_memories[:limit]
    
    def search_memories(self, query, limit=10):
        """
        Search memories for a query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            list: Matching memories
        """
        results = []
        query_lower = query.lower()
        
        # Search facts
        for fact in self.memories["facts"]:
            if query_lower in fact["content"].lower():
                results.append(fact)
        
        # Search concepts
        for concept in self.memories["concepts"]:
            if query_lower in concept["name"].lower() or query_lower in concept["description"].lower():
                results.append(concept)
        
        # Search experiences
        for experience in self.memories["experiences"]:
            if query_lower in experience["content"].lower():
                results.append(experience)
        
        # Sort by importance and limit results
        results = sorted(results, key=lambda x: x.get("importance", 0), reverse=True)
        return results[:limit]
    
    def get_memory_by_id(self, memory_id):
        """
        Get a memory by ID
        
        Args:
            memory_id: Memory ID
            
        Returns:
            dict: Memory or None if not found
        """
        # Check facts
        for fact in self.memories["facts"]:
            if fact["id"] == memory_id:
                fact["last_accessed"] = time.time()
                return fact
        
        # Check concepts
        for concept in self.memories["concepts"]:
            if concept["id"] == memory_id:
                concept["last_accessed"] = time.time()
                return concept
        
        # Check experiences
        for experience in self.memories["experiences"]:
            if experience["id"] == memory_id:
                experience["last_accessed"] = time.time()
                return experience
        
        return None
    
    def update_memory_importance(self, memory_id, importance):
        """
        Update memory importance
        
        Args:
            memory_id: Memory ID
            importance: New importance value (0.0 to 1.0)
            
        Returns:
            bool: Success status
        """
        memory = self.get_memory_by_id(memory_id)
        
        if memory:
            memory["importance"] = max(0.0, min(1.0, importance))
            self.save()
            return True
        
        return False
    
    def decay_memories(self, decay_rate=0.01):
        """
        Apply decay to all memories
        
        Args:
            decay_rate: Rate of decay (0.0 to 1.0)
            
        Returns:
            int: Number of memories affected
        """
        now = time.time()
        count = 0
        
        # Decay all memory types
        for memory_type in ["facts", "concepts", "experiences"]:
            for memory in self.memories[memory_type]:
                # Calculate time factor (higher for older memories)
                time_factor = (now - memory["last_accessed"]) / (60 * 60 * 24)  # Days
                decay = decay_rate * time_factor
                
                # Apply decay
                memory["importance"] = max(0.1, memory["importance"] - decay)
                count += 1
        
        self.save()
        return count


class OnsiteMemory:
    """
    Onsite Memory System for LUMINA V7
    
    Enhanced memory system that stores and organizes knowledge directly onsite.
    Handles conversations, knowledge facts, and user preferences.
    """
    
    def __init__(self, memory_file=None):
        """
        Initialize the onsite memory system
        
        Args:
            memory_file: Path to the memory file
        """
        self.memory_file = memory_file or "data/onsite_memory/memory_store.json"
        self.storage_path = os.path.dirname(self.memory_file)
        
        # Ensure storage path exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize memory structure
        self.memory = {
            "conversations": [],
            "knowledge": {},
            "user_preferences": {},
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        # Load existing memory if available
        self.load_memory()
        
        logger.info(f"Onsite Memory initialized with storage at {self.storage_path}")
    
    def load_memory(self):
        """Load memory from disk if available"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    loaded_memory = json.load(f)
                    # Update with loaded memory
                    self.memory.update(loaded_memory)
                logger.info(f"Loaded memory from {self.memory_file}")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
    
    def save_memory(self):
        """Save memory to disk"""
        try:
            # Update timestamp
            self.memory["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2)
            
            logger.debug(f"Saved memory to {self.memory_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False
    
    def add_conversation(self, conversation, metadata=None):
        """
        Add a conversation to memory
        
        Args:
            conversation: Dictionary with conversation data
            metadata: Additional metadata
            
        Returns:
            str: Conversation ID
        """
        # Generate ID
        conversation_id = f"conv_{int(time.time())}_{len(self.memory['conversations']) + 1}"
        
        # Create conversation object
        conv_obj = {
            "id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "content": conversation,
            "metadata": metadata or {}
        }
        
        # Add to memory
        self.memory["conversations"].append(conv_obj)
        
        # Save changes
        self.save_memory()
        
        return conversation_id
    
    def add_knowledge(self, topic, content, source=None, importance=0.5):
        """
        Add knowledge to a specific topic
        
        Args:
            topic: Knowledge topic/category
            content: The knowledge content
            source: Source of the knowledge
            importance: Importance score (0.0-1.0)
            
        Returns:
            str: Knowledge entry ID
        """
        # Initialize topic if not exists
        if topic not in self.memory["knowledge"]:
            self.memory["knowledge"][topic] = []
        
        # Generate ID
        knowledge_id = f"know_{int(time.time())}_{len(self.memory['knowledge'][topic]) + 1}"
        
        # Create knowledge entry
        entry = {
            "id": knowledge_id,
            "content": content,
            "created": datetime.now().isoformat(),
            "source": source,
            "importance": importance,
            "access_count": 0,
            "last_accessed": None
        }
        
        # Add to memory
        self.memory["knowledge"][topic].append(entry)
        
        # Save changes
        self.save_memory()
        
        return knowledge_id
    
    def search_knowledge(self, query, limit=5):
        """
        Search knowledge entries
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            list: Matching knowledge entries
        """
        results = []
        query = query.lower()
        
        # Search through all topics
        for topic, entries in self.memory["knowledge"].items():
            for entry in entries:
                content = entry.get("content", "").lower()
                
                # Check if content matches query
                if query in content:
                    # Create a copy with topic info
                    result = entry.copy()
                    result["topic"] = topic
                    results.append(result)
                    
                    # Update access stats
                    entry["access_count"] = entry.get("access_count", 0) + 1
                    entry["last_accessed"] = datetime.now().isoformat()
        
        # Sort by importance
        results.sort(key=lambda x: x.get("importance", 0), reverse=True)
        
        # Save access stats
        if results:
            self.save_memory()
            
        return results[:limit]
    
    def update_preferences(self, user_id, preferences):
        """
        Update user preferences
        
        Args:
            user_id: User identifier
            preferences: Dictionary of preferences
            
        Returns:
            bool: Success status
        """
        # Initialize user preferences if not exists
        if user_id not in self.memory["user_preferences"]:
            self.memory["user_preferences"][user_id] = {}
        
        # Update preferences
        self.memory["user_preferences"][user_id].update(preferences)
        
        # Save changes
        return self.save_memory()
    
    def get_preferences(self, user_id):
        """
        Get user preferences
        
        Args:
            user_id: User identifier
            
        Returns:
            dict: User preferences
        """
        return self.memory["user_preferences"].get(user_id, {})
    
    def get_stats(self):
        """
        Get statistics about the memory system
        
        Returns:
            dict: Memory statistics
        """
        # Count conversations
        conversation_count = len(self.memory["conversations"])
        
        # Count knowledge entries
        knowledge_count = 0
        topics = []
        for topic, entries in self.memory["knowledge"].items():
            knowledge_count += len(entries)
            topics.append(topic)
        
        # Count user preferences
        user_count = len(self.memory["user_preferences"])
        
        # Get storage stats
        storage_size = 0
        if os.path.exists(self.memory_file):
            storage_size = os.path.getsize(self.memory_file)
        
        return {
            "conversation_count": conversation_count,
            "knowledge_count": knowledge_count,
            "topics": topics,
            "user_count": user_count,
            "storage_path": self.storage_path,
            "storage_size": storage_size,
            "last_updated": self.memory["metadata"].get("last_updated")
        }
    
    def clear_memory(self, confirm=False):
        """
        Clear all memory data
        
        Args:
            confirm: Confirmation flag
            
        Returns:
            bool: Success status
        """
        if not confirm:
            logger.warning("Memory clear attempted without confirmation")
            return False
        
        # Reset memory structure
        self.memory = {
            "conversations": [],
            "knowledge": {},
            "user_preferences": {},
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        # Save empty memory
        return self.save_memory()


class MemoryAnalyzer:
    """
    Memory Analysis System for LUMINA V7
    
    This class provides tools to analyze and derive insights from the memory system.
    """
    
    def __init__(self, memory_system=None):
        """
        Initialize the memory analyzer
        
        Args:
            memory_system: The memory system to analyze (SimpleMemory or OnsiteMemory)
        """
        self.memory_system = memory_system
        logger.info(f"Memory Analyzer initialized")
    
    def analyze_conversation(self, conversation_text):
        """
        Analyze conversation text and extract key insights
        
        Args:
            conversation_text: The conversation text to analyze
            
        Returns:
            dict: Analysis results
        """
        # Simple implementation - in production would use NLP
        words = conversation_text.lower().split()
        word_count = len(words)
        
        # Extract potential key points (sentences ending with period)
        sentences = conversation_text.split('.')
        key_points = [s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 100]
        
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'positive', 'helpful']
        negative_words = ['bad', 'terrible', 'awful', 'negative', 'unhelpful', 'problem', 'issue']
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Extract potential topics (most frequent non-stop words)
        stop_words = ['the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'that', 'for', 'on', 'with']
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        
        topics = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        return {
            "key_points": key_points[:3],  # Top 3 key points
            "sentiment": sentiment,
            "topics": topics,
            "word_count": word_count
        }
    
    def suggest_memories(self, context, limit=3):
        """
        Suggest relevant memories based on context
        
        Args:
            context: The context to match memories against
            limit: Maximum number of memories to suggest
            
        Returns:
            list: Suggested memories
        """
        if not self.memory_system:
            return []
        
        # For SimpleMemory
        if hasattr(self.memory_system, 'search_memories'):
            # Extract keywords from context
            words = context.lower().split()
            keywords = [word for word in words if len(word) > 4]
            
            results = []
            # Search for each keyword
            for keyword in keywords[:3]:  # Use first 3 keywords
                memories = self.memory_system.search_memories(keyword, limit=2)
                results.extend(memories)
            
            # Remove duplicates and limit results
            unique_results = []
            seen_ids = set()
            for memory in results:
                memory_id = memory.get('id', '')
                if memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    unique_results.append(memory)
            
            return unique_results[:limit]
        
        # For OnsiteMemory
        elif hasattr(self.memory_system, 'search_knowledge'):
            return self.memory_system.search_knowledge(context, limit=limit)
        
        return []
    
    def compute_importance(self, memory):
        """
        Compute importance score for a memory
        
        Args:
            memory: The memory to score
            
        Returns:
            float: Importance score (0.0 to 1.0)
        """
        # Base importance from memory
        importance = memory.get('importance', 0.5)
        
        # Adjust based on access count (more accessed = more important)
        access_count = memory.get('access_count', 0)
        if isinstance(access_count, int):
            access_boost = min(0.3, access_count * 0.02)
            importance += access_boost
        
        # Adjust based on recency
        try:
            if 'last_accessed' in memory:
                last_accessed = memory['last_accessed']
                if isinstance(last_accessed, str):
                    from datetime import datetime
                    try:
                        # Try ISO format
                        timestamp = datetime.fromisoformat(last_accessed)
                    except:
                        # Try timestamp
                        timestamp = datetime.fromtimestamp(float(last_accessed))
                elif isinstance(last_accessed, (int, float)):
                    from datetime import datetime
                    timestamp = datetime.fromtimestamp(last_accessed)
                else:
                    timestamp = None
                
                if timestamp:
                    from datetime import datetime
                    days_old = (datetime.now() - timestamp).days
                    recency_boost = max(0, 0.2 - (days_old * 0.01))
                    importance += recency_boost
        except Exception as e:
            logger.error(f"Error calculating recency boost: {e}")
        
        # Cap importance at 1.0
        return min(1.0, importance)
    
    def analyze_memory_network(self):
        """
        Analyze the memory network structure
        
        Returns:
            dict: Network analysis
        """
        if not self.memory_system:
            return {
                "node_count": 0,
                "edge_count": 0,
                "density": 0.0,
                "clusters": []
            }
        
        # Number of memories
        node_count = 0
        
        # For SimpleMemory
        if hasattr(self.memory_system, 'memories'):
            facts_count = len(self.memory_system.memories.get('facts', []))
            concepts_count = len(self.memory_system.memories.get('concepts', []))
            experiences_count = len(self.memory_system.memories.get('experiences', []))
            node_count = facts_count + concepts_count + experiences_count
            
            # Count relationships between concepts
            edge_count = 0
            for concept in self.memory_system.memories.get('concepts', []):
                edge_count += len(concept.get('relations', []))
            
            # Identify clusters (simple implementation)
            clusters = [
                {"name": "Facts", "size": facts_count},
                {"name": "Concepts", "size": concepts_count},
                {"name": "Experiences", "size": experiences_count}
            ]
            
            # Calculate network density
            if node_count > 1:
                max_edges = node_count * (node_count - 1) / 2
                density = edge_count / max_edges if max_edges > 0 else 0
            else:
                density = 0
            
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "density": density,
                "clusters": clusters
            }
        
        # For OnsiteMemory
        elif hasattr(self.memory_system, 'memory'):
            conversations_count = len(self.memory_system.memory.get('conversations', []))
            knowledge_count = sum(len(entries) for entries in self.memory_system.memory.get('knowledge', {}).values())
            user_prefs_count = len(self.memory_system.memory.get('user_preferences', {}))
            node_count = conversations_count + knowledge_count + user_prefs_count
            
            # Simple estimate of edges
            edge_count = knowledge_count  # Assume one edge per knowledge entry
            
            # Identify clusters
            clusters = [
                {"name": "Conversations", "size": conversations_count},
                {"name": "Knowledge", "size": knowledge_count},
                {"name": "User Preferences", "size": user_prefs_count}
            ]
            
            # Calculate network density
            if node_count > 1:
                max_edges = node_count * (node_count - 1) / 2
                density = edge_count / max_edges if max_edges > 0 else 0
            else:
                density = 0
            
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "density": density,
                "clusters": clusters
            }
        
        return {
            "node_count": 0,
            "edge_count": 0,
            "density": 0.0,
            "clusters": []
        }


def get_memory_system(file_path=None):
    """
    Get a memory system instance
    
    Args:
        file_path: Path to the memory file (optional)
        
    Returns:
        SimpleMemory: Memory system instance
    """
    return SimpleMemory(file_path)


def get_onsite_memory(memory_file=None):
    """
    Get an onsite memory instance
    
    Args:
        memory_file: Path to the memory file (optional)
        
    Returns:
        OnsiteMemory: Onsite memory instance
    """
    return OnsiteMemory(memory_file) 