"""
ConversationMemory - Memory system for language processing

This module provides a memory system specifically designed for storing and 
retrieving conversation history with enhanced semantic understanding capabilities.
"""

import json
import os
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable
import random

try:
    from memory_manager import memory_manager
    memory_manager_available = True
except ImportError:
    memory_manager_available = False
    logging.getLogger(__name__).warning("Memory Manager import failed, collaborative memory features disabled")

logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Enhanced memory system for conversation history with semantic retrieval capabilities.
    
    This class provides mechanisms to:
    1. Store conversations with metadata (emotions, topics, etc.)
    2. Retrieve conversations by semantic similarity
    3. Build memory connections between related conversations
    4. Support collaborative work with other AI agents through memory sharing
    """
    
    def __init__(self, component_name: str = "conversation_memory", 
                 memory_file: str = "data/memory/conversation_memory.jsonl"):
        """
        Initialize the conversation memory system
        
        Args:
            component_name: Name of this component for memory manager registration
            memory_file: Path to the memory storage file
        """
        self.component_name = component_name
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Memory storage
        self.memories = []
        self.memory_capabilities_version = "1.1.0"
        self.memory_enabled = True
        
        # Memory indices for fast retrieval
        self.topic_index = {}      # Topic -> list of memory IDs
        self.emotion_index = {}    # Emotion -> list of memory IDs
        self.keyword_index = {}    # Keyword -> list of memory IDs
        self.timestamp_index = []  # Sorted list of (timestamp, memory_id) tuples
        self.entity_index = {}     # Entity -> list of memory IDs
        self.importance_index = [] # Sorted list of (importance_score, memory_id) tuples
        
        # Subscription tracking
        self.subscribed_to = set()  # Components we're subscribed to
        self.sharing_preferences = {}  # Component -> sharing filter function
        
        # Register with memory manager if available
        self._register_with_memory_manager()
        
        # Load existing memories
        self._load_memories()
        
        logger.info(f"ConversationMemory initialized with {len(self.memories)} existing memories")
    
    def _register_with_memory_manager(self):
        """Register this component with the memory manager if available"""
        if memory_manager_available:
            try:
                memory_manager.register_component(self.component_name, self)
                logger.info(f"Successfully registered {self.component_name} with Memory Manager")
            except Exception as e:
                logger.error(f"Failed to register with Memory Manager: {str(e)}")
    
    def _load_memories(self):
        """Load existing memories from file"""
        if not self.memory_file.exists():
            logger.info(f"No memory file found at {self.memory_file}, starting fresh")
            return
            
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        memory = json.loads(line)
                        self.memories.append(memory)
                        self._index_memory(memory)
                        
            logger.info(f"Loaded {len(self.memories)} memories from {self.memory_file}")
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
    
    def _index_memory(self, memory: Dict[str, Any]):
        """
        Add a memory to the various indices for fast retrieval
        
        Args:
            memory: Memory data dictionary to index
        """
        memory_id = memory["id"]
        
        # Index by topics
        for topic in memory.get("metadata", {}).get("topics", []):
            if topic not in self.topic_index:
                self.topic_index[topic] = []
            self.topic_index[topic].append(memory_id)
        
        # Index by emotion
        emotion = memory.get("metadata", {}).get("emotion")
        if emotion:
            if emotion not in self.emotion_index:
                self.emotion_index[emotion] = []
            self.emotion_index[emotion].append(memory_id)
        
        # Index by keywords
        for keyword in memory.get("metadata", {}).get("keywords", []):
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = []
            self.keyword_index[keyword].append(memory_id)
        
        # Index by entities
        for entity in memory.get("metadata", {}).get("entities", []):
            if entity not in self.entity_index:
                self.entity_index[entity] = []
            self.entity_index[entity].append(memory_id)
        
        # Index by importance
        importance = memory.get("metadata", {}).get("importance", 0)
        if importance:
            self.importance_index.append((importance, memory_id))
            # Keep importance index sorted (highest first)
            self.importance_index.sort(reverse=True)
        
        # Index by timestamp
        timestamp = memory.get("timestamp", datetime.datetime.now().isoformat())
        self.timestamp_index.append((timestamp, memory_id))
        # Keep timestamp index sorted
        self.timestamp_index.sort(reverse=True)
    
    def store(self, user_input: str, system_response: str, 
              metadata: Dict[str, Any] = None) -> str:
        """
        Store a conversation exchange in memory
        
        Args:
            user_input: User's input text
            system_response: System's response text  
            metadata: Additional metadata about the conversation
            
        Returns:
            Memory ID of the stored conversation
        """
        if not self.memory_enabled:
            logger.warning("Memory storage skipped: Memory system disabled")
            return None
        
        try:
            # Generate memory ID
            timestamp = datetime.datetime.now()
            memory_id = f"conv_{timestamp.strftime('%Y%m%d%H%M%S')}_{len(self.memories)}"
            
            # Prepare metadata
            metadata = metadata or {}
            
            # Create memory entry
            memory = {
                "id": memory_id,
                "user_input": user_input,
                "system_response": system_response,
                "timestamp": timestamp.isoformat(),
                "metadata": metadata
            }
            
            # Add component information
            memory["metadata"]["source"] = self.component_name
            memory["metadata"]["memory_version"] = self.memory_capabilities_version
            
            # Store memory
            self.memories.append(memory)
            self._index_memory(memory)
            
            # Save to file
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(memory) + "\n")
            
            # Share with subscribers if using memory manager
            if memory_manager_available:
                subscribers = memory_manager.get_subscribers(self.component_name)
                if subscribers:
                    memory_manager.share_memories([memory], self.component_name)
            
            logger.debug(f"Stored conversation in memory with ID: {memory_id}")
            return memory_id
        
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            return None
    
    def retrieve_by_topic(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories by topic
        
        Args:
            topic: Topic to search for
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries
        """
        if not self.memory_enabled:
            return []
        
        try:
            # Look up memory IDs in topic index
            memory_ids = self.topic_index.get(topic, [])
            
            # Get memories by ID
            memories = [mem for mem in self.memories if mem["id"] in memory_ids]
            
            # Sort by timestamp (newest first)
            memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return memories[:limit]
        
        except Exception as e:
            logger.error(f"Error retrieving by topic: {str(e)}")
            return []
    
    def retrieve_by_emotion(self, emotion: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories by emotion
        
        Args:
            emotion: Emotion to search for
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries
        """
        if not self.memory_enabled:
            return []
        
        try:
            # Look up memory IDs in emotion index
            memory_ids = self.emotion_index.get(emotion, [])
            
            # Get memories by ID
            memories = [mem for mem in self.memories if mem["id"] in memory_ids]
            
            # Sort by timestamp (newest first)
            memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return memories[:limit]
        
        except Exception as e:
            logger.error(f"Error retrieving by emotion: {str(e)}")
            return []
    
    def retrieve_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories by keyword
        
        Args:
            keyword: Keyword to search for
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries
        """
        if not self.memory_enabled:
            return []
        
        try:
            # Look up memory IDs in keyword index
            memory_ids = self.keyword_index.get(keyword, [])
            
            # Get memories by ID
            memories = [mem for mem in self.memories if mem["id"] in memory_ids]
            
            # Sort by timestamp (newest first)
            memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return memories[:limit]
        
        except Exception as e:
            logger.error(f"Error retrieving by keyword: {str(e)}")
            return []
    
    def retrieve_by_entity(self, entity: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories by entity
        
        Args:
            entity: Entity to search for (person, place, etc.)
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries
        """
        if not self.memory_enabled:
            return []
        
        try:
            # Look up memory IDs in entity index
            memory_ids = self.entity_index.get(entity, [])
            
            # Get memories by ID
            memories = [mem for mem in self.memories if mem["id"] in memory_ids]
            
            # Sort by timestamp (newest first)
            memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return memories[:limit]
        
        except Exception as e:
            logger.error(f"Error retrieving by entity: {str(e)}")
            return []
    
    def retrieve_important(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve most important memories
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries
        """
        if not self.memory_enabled:
            return []
        
        try:
            # Get most important memory IDs from importance index
            important_ids = [memory_id for _, memory_id in self.importance_index[:limit]]
            
            # Get memories by ID
            memories = [mem for mem in self.memories if mem["id"] in important_ids]
            
            # Sort by importance (highest first)
            memories.sort(
                key=lambda x: x.get("metadata", {}).get("importance", 0), 
                reverse=True
            )
            
            return memories[:limit]
        
        except Exception as e:
            logger.error(f"Error retrieving important memories: {str(e)}")
            return []
    
    def retrieve_recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve most recent memories
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries
        """
        if not self.memory_enabled:
            return []
        
        try:
            # Get most recent memory IDs from timestamp index
            recent_ids = [memory_id for _, memory_id in self.timestamp_index[:limit]]
            
            # Get memories by ID
            memories = [mem for mem in self.memories if mem["id"] in recent_ids]
            
            # Sort by timestamp (newest first)
            memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return memories[:limit]
        
        except Exception as e:
            logger.error(f"Error retrieving recent memories: {str(e)}")
            return []
    
    def search_text(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories by text content (simple substring match)
        
        Args:
            query: Text to search for
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries
        """
        if not self.memory_enabled:
            return []
        
        try:
            query = query.lower()
            matches = []
            
            for memory in self.memories:
                user_input = memory.get("user_input", "").lower()
                system_response = memory.get("system_response", "").lower()
                
                if query in user_input or query in system_response:
                    matches.append(memory)
            
            # Sort by timestamp (newest first)
            matches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return matches[:limit]
        
        except Exception as e:
            logger.error(f"Error searching text: {str(e)}")
            return []
    
    def subscribe_to(self, source_component: str) -> bool:
        """
        Subscribe to memories from another component
        
        Args:
            source_component: Name of the component to subscribe to
            
        Returns:
            Success flag
        """
        if not memory_manager_available:
            logger.warning("Subscription skipped: Memory manager unavailable")
            return False
            
        try:
            # Register subscription with memory manager
            success = memory_manager.subscribe(source_component, self.component_name)
            
            if success:
                self.subscribed_to.add(source_component)
                logger.info(f"Subscribed to memories from {source_component}")
            else:
                logger.warning(f"Failed to subscribe to {source_component}")
                
            return success
        
        except Exception as e:
            logger.error(f"Error subscribing to component: {str(e)}")
            return False
    
    def unsubscribe_from(self, source_component: str) -> bool:
        """
        Unsubscribe from memories of another component
        
        Args:
            source_component: Name of the component to unsubscribe from
            
        Returns:
            Success flag
        """
        if not memory_manager_available:
            logger.warning("Unsubscription skipped: Memory manager unavailable")
            return False
            
        try:
            # Unregister subscription with memory manager
            success = memory_manager.unsubscribe(source_component, self.component_name)
            
            if success and source_component in self.subscribed_to:
                self.subscribed_to.remove(source_component)
                logger.info(f"Unsubscribed from memories of {source_component}")
            else:
                logger.warning(f"Failed to unsubscribe from {source_component}")
                
            return success
        
        except Exception as e:
            logger.error(f"Error unsubscribing from component: {str(e)}")
            return False
    
    def set_sharing_preference(self, target_component: str, 
                             filter_func: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Set a filter function for sharing memories with a specific component
        
        Args:
            target_component: Target component name
            filter_func: Function that takes a memory dict and returns True if it should be shared
        """
        self.sharing_preferences[target_component] = filter_func
        logger.info(f"Set sharing filter for {target_component}")
    
    def share_with_component(self, target_component: str = None, 
                           filter_func = None) -> List[Dict[str, Any]]:
        """
        Share memories with another component
        
        Args:
            target_component: Name of the target component (None = broadcast to all subscribers)
            filter_func: Optional function to filter memories to share
            
        Returns:
            List of shared memories
        """
        if not self.memory_enabled or not memory_manager_available:
            logger.warning("Memory sharing skipped: Memory system disabled or manager unavailable")
            return []
        
        try:
            # Use component-specific filter if available and none provided
            if target_component and not filter_func:
                filter_func = self.sharing_preferences.get(target_component)
                
            # Filter memories if filter function provided
            if filter_func:
                filtered = [m for m in self.memories if filter_func(m)]
            else:
                filtered = self.memories
            
            # Add sharing metadata
            for memory in filtered:
                if "metadata" not in memory:
                    memory["metadata"] = {}
                memory["metadata"]["shared_from"] = self.component_name
                memory["metadata"]["shared_at"] = datetime.datetime.now().isoformat()
            
            if target_component:
                logger.info(f"Sharing {len(filtered)} memories with {target_component}")
            else:
                logger.info(f"Broadcasting {len(filtered)} memories to all subscribers")
            
            # Share memories through memory manager
            memory_manager.share_memories(filtered, self.component_name, target_component)
            
            return filtered
        
        except Exception as e:
            logger.error(f"Error sharing memories: {str(e)}")
            return []
    
    def receive_shared_memories(self, memories: List[Dict[str, Any]], 
                             source_component: str) -> int:
        """
        Receive shared memories from another component
        
        Args:
            memories: List of memory items
            source_component: Name of the source component
            
        Returns:
            Number of memories received
        """
        if not self.memory_enabled:
            logger.warning("Memory receiving skipped: Memory system disabled")
            return 0
        
        try:
            count = 0
            for memory in memories:
                # Add source component if not present
                if "metadata" not in memory:
                    memory["metadata"] = {}
                
                memory["metadata"]["received_from"] = source_component
                memory["metadata"]["received_at"] = datetime.datetime.now().isoformat()
                
                # Add to memory store
                self.memories.append(memory)
                self._index_memory(memory)
                count += 1
            
            logger.info(f"Received {count} memories from {source_component}")
            
            return count
        
        except Exception as e:
            logger.error(f"Error receiving shared memories: {str(e)}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            return {
                "total_memories": len(self.memories),
                "topics": len(self.topic_index),
                "emotions": len(self.emotion_index),
                "keywords": len(self.keyword_index),
                "entities": len(self.entity_index),
                "importance_entries": len(self.importance_index),
                "oldest_memory": self.timestamp_index[-1][0] if self.timestamp_index else None,
                "newest_memory": self.timestamp_index[0][0] if self.timestamp_index else None,
                "subscribed_to": list(self.subscribed_to),
                "memory_enabled": self.memory_enabled,
                "memory_version": self.memory_capabilities_version
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {}
    
    def clear_memories(self) -> bool:
        """
        Clear all memories from the system
        
        Returns:
            Success flag
        """
        try:
            self.memories = []
            self.topic_index = {}
            self.emotion_index = {}
            self.keyword_index = {}
            self.entity_index = {}
            self.importance_index = []
            self.timestamp_index = []
            
            # Backup existing memory file
            if self.memory_file.exists():
                backup_path = self.memory_file.with_suffix(f".bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
                self.memory_file.rename(backup_path)
                
            # Create new empty file
            with open(self.memory_file, "w") as f:
                pass
            
            logger.info("Memory system cleared")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing memories: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize memory system
    memory = ConversationMemory()
    
    # Store a conversation
    memory.store(
        "What's the weather like today?",
        "It's sunny with a high of 75°F.",
        metadata={
            "topics": ["weather", "conversation"],
            "emotion": "neutral",
            "keywords": ["weather", "sunny"],
            "entities": ["weather", "temperature"],
            "importance": 0.3
        }
    )
    
    # Store another with higher importance
    memory.store(
        "Remember to call mom at 5pm for her birthday!",
        "I've set a reminder for you to call your mom at 5pm.",
        metadata={
            "topics": ["reminder", "birthday", "family"],
            "emotion": "happy",
            "keywords": ["reminder", "birthday", "mom"],
            "entities": ["mom", "birthday", "5pm"],
            "importance": 0.9
        }
    )
    
    # Retrieve important memories
    important_memories = memory.retrieve_important()
    print(f"Found {len(important_memories)} important memories")
    
    # Retrieve by entity
    mom_memories = memory.retrieve_by_entity("mom")
    print(f"Found {len(mom_memories)} memories about mom")
    
    # Check memory stats
    stats = memory.get_memory_stats()
    print(f"Memory stats: {stats}") 