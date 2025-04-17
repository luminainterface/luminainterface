#!/usr/bin/env python3
"""
Conversation Memory Module

This module provides a memory system for storing and retrieving conversation history
with enhanced semantic understanding capabilities.
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("conversation_memory")


class ConversationMemory:
    """
    A memory system for storing and retrieving conversation history.
    
    This class provides functionality to:
    - Store conversation snippets with metadata
    - Retrieve memories by topic, emotion, keywords
    - Search through memory content
    - Share memories with other components
    """
    
    def __init__(self, memory_file_path=None):
        """
        Initialize the Conversation Memory system.
        
        Args:
            memory_file_path: Path to memory storage file (default: data/memory/conversation_memory.jsonl)
        """
        # Initialize memory storage
        self.memories = []
        self.memory_file_path = memory_file_path or "data/memory/conversation_memory.jsonl"
        self.memory_capabilities_version = "1.0"
        
        # Create indices for fast retrieval
        self.topic_index = {}     # topic -> list of memory indices
        self.emotion_index = {}   # emotion -> list of memory indices
        self.keyword_index = {}   # keyword -> list of memory indices
        self.timestamp_index = [] # sorted list of (timestamp, memory_index) tuples
        
        # Register with memory manager if available
        self._try_register_with_memory_manager()
        
        # Load existing memories if available
        self._load_existing_memories()
        
        logger.info(f"Conversation Memory initialized (version {self.memory_capabilities_version})")
    
    def _try_register_with_memory_manager(self):
        """Attempt to register this component with the memory manager if it exists"""
        try:
            # This would be implemented to connect with a central memory manager
            # For now it's a placeholder
            pass
        except Exception as e:
            logger.warning(f"Could not register with memory manager: {str(e)}")
    
    def _load_existing_memories(self):
        """Load existing memories from storage"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
        
        # Load memories if file exists
        if os.path.exists(self.memory_file_path):
            try:
                with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            memory = json.loads(line.strip())
                            self._add_to_memory(memory, skip_save=True)
                
                logger.info(f"Loaded {len(self.memories)} existing memories")
            except Exception as e:
                logger.error(f"Error loading existing memories: {str(e)}")
                # Continue with empty memories
    
    def store(self, content, metadata=None):
        """
        Store a new memory.
        
        Args:
            content: The conversation content to store
            metadata: Dictionary of metadata about the conversation
                      (topic, emotion, keywords, etc.)
        
        Returns:
            Dictionary containing the stored memory with its ID
        """
        # Ensure metadata exists
        metadata = metadata or {}
        
        # Create the memory object
        memory = {
            "id": str(uuid.uuid4()),
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        # Add to memory storage and indices
        self._add_to_memory(memory)
        
        logger.info(f"Stored new memory with ID: {memory['id']}")
        return memory
    
    def _add_to_memory(self, memory, skip_save=False):
        """
        Add a memory to storage and update indices
        
        Args:
            memory: The memory object to store
            skip_save: If True, don't save to disk (used during loading)
        """
        # Add to main memory store
        memory_index = len(self.memories)
        self.memories.append(memory)
        
        # Update topic index
        topic = memory.get("metadata", {}).get("topic")
        if topic:
            if topic not in self.topic_index:
                self.topic_index[topic] = []
            self.topic_index[topic].append(memory_index)
        
        # Update emotion index
        emotion = memory.get("metadata", {}).get("emotion")
        if emotion:
            if emotion not in self.emotion_index:
                self.emotion_index[emotion] = []
            self.emotion_index[emotion].append(memory_index)
        
        # Update keyword index
        keywords = memory.get("metadata", {}).get("keywords", [])
        for keyword in keywords:
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = []
            self.keyword_index[keyword].append(memory_index)
        
        # Update timestamp index
        timestamp = memory.get("timestamp")
        if timestamp:
            self.timestamp_index.append((timestamp, memory_index))
            # Keep timestamp index sorted
            self.timestamp_index.sort(key=lambda x: x[0])
        
        # Save to disk if not loading
        if not skip_save:
            self._save_memory(memory)
    
    def _save_memory(self, memory):
        """
        Save a memory to disk
        
        Args:
            memory: The memory to save
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
            
            # Append to file
            with open(self.memory_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(memory) + '\n')
        except Exception as e:
            logger.error(f"Error saving memory to disk: {str(e)}")
    
    def retrieve_by_topic(self, topic):
        """
        Retrieve memories by topic
        
        Args:
            topic: The topic to search for
        
        Returns:
            List of memories related to the topic
        """
        if topic not in self.topic_index:
            return []
        
        # Get memories from the topic index
        memory_indices = self.topic_index[topic]
        memories = [self.memories[i] for i in memory_indices]
        
        logger.info(f"Retrieved {len(memories)} memories for topic: {topic}")
        return memories
    
    def retrieve_by_emotion(self, emotion):
        """
        Retrieve memories by emotion
        
        Args:
            emotion: The emotion to search for
        
        Returns:
            List of memories with the specified emotion
        """
        if emotion not in self.emotion_index:
            return []
        
        # Get memories from the emotion index
        memory_indices = self.emotion_index[emotion]
        memories = [self.memories[i] for i in memory_indices]
        
        logger.info(f"Retrieved {len(memories)} memories for emotion: {emotion}")
        return memories
    
    def retrieve_by_keyword(self, keyword):
        """
        Retrieve memories by keyword
        
        Args:
            keyword: The keyword to search for
        
        Returns:
            List of memories with the specified keyword
        """
        if keyword not in self.keyword_index:
            return []
        
        # Get memories from the keyword index
        memory_indices = self.keyword_index[keyword]
        memories = [self.memories[i] for i in memory_indices]
        
        logger.info(f"Retrieved {len(memories)} memories for keyword: {keyword}")
        return memories
    
    def retrieve_recent(self, count=10):
        """
        Retrieve the most recent memories
        
        Args:
            count: Number of recent memories to retrieve
        
        Returns:
            List of recent memories
        """
        # Use timestamp index to get the most recent memories
        if not self.timestamp_index:
            return []
        
        # Get the most recent indices
        recent_indices = [idx for _, idx in self.timestamp_index[-count:]]
        memories = [self.memories[i] for i in recent_indices]
        
        logger.info(f"Retrieved {len(memories)} recent memories")
        return memories
    
    def search_text(self, query):
        """
        Search through memory content for matching text
        
        Args:
            query: The text to search for
        
        Returns:
            List of memories containing the query text
        """
        # Simple text search implementation
        query = query.lower()
        matching_memories = []
        
        for memory in self.memories:
            content = memory.get("content", "").lower()
            if query in content:
                matching_memories.append(memory)
        
        logger.info(f"Found {len(matching_memories)} memories matching text: {query}")
        return matching_memories
    
    def share_with_component(self, component_id, memories=None):
        """
        Share memories with another component
        
        Args:
            component_id: ID of the component to share with
            memories: Specific memories to share (None = all)
        
        Returns:
            Dictionary with status and count of shared memories
        """
        # This would be implemented to share memories with other components
        # For now it's a placeholder
        to_share = memories if memories is not None else self.memories
        
        logger.info(f"Shared {len(to_share)} memories with component: {component_id}")
        return {"status": "success", "shared_count": len(to_share)}
    
    def receive_shared_memories(self, memories, source_component):
        """
        Receive memories shared from another component
        
        Args:
            memories: List of memories to receive
            source_component: ID of the source component
        
        Returns:
            Dictionary with status and count of received memories
        """
        count = 0
        for memory in memories:
            # Add source component to metadata
            if "metadata" not in memory:
                memory["metadata"] = {}
            memory["metadata"]["source_component"] = source_component
            
            # Add to memory
            self._add_to_memory(memory)
            count += 1
        
        logger.info(f"Received {count} memories from component: {source_component}")
        return {"status": "success", "received_count": count}
    
    def get_memory_stats(self):
        """
        Get statistics about the memory system
        
        Returns:
            Dictionary of statistics
        """
        # Compute top topics
        topic_counts = {}
        for topic, indices in self.topic_index.items():
            topic_counts[topic] = len(indices)
        
        # Get top topics sorted by count
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Count sentences (simplistic implementation)
        total_sentences = 0
        for memory in self.memories:
            content = memory.get("content", "")
            # Count periods, question marks, and exclamation points as sentence endings
            sentence_count = content.count('.') + content.count('?') + content.count('!')
            total_sentences += max(1, sentence_count)  # At least 1 sentence per memory
        
        stats = {
            "total_memories": len(self.memories),
            "top_topics": top_topics,
            "total_keywords": len(self.keyword_index),
            "total_emotions": len(self.emotion_index),
            "total_sentences": total_sentences,
            "memory_capabilities_version": self.memory_capabilities_version
        }
        
        return stats
    
    def clear_memories(self, older_than=None):
        """
        Clear memories, optionally only those older than a specific timestamp
        
        Args:
            older_than: ISO format timestamp, memories older than this will be cleared
        
        Returns:
            Dictionary with status and count of cleared memories
        """
        if older_than is None:
            # Clear all memories
            cleared_count = len(self.memories)
            self.memories = []
            self.topic_index = {}
            self.emotion_index = {}
            self.keyword_index = {}
            self.timestamp_index = []
            
            # Recreate the memory file
            try:
                os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
                with open(self.memory_file_path, 'w') as f:
                    pass  # Just create an empty file
            except Exception as e:
                logger.error(f"Error clearing memory file: {str(e)}")
        else:
            # Clear memories older than the specified timestamp
            new_memories = []
            cleared_count = 0
            
            for memory in self.memories:
                if memory.get("timestamp", "") >= older_than:
                    new_memories.append(memory)
                else:
                    cleared_count += 1
            
            # Rebuild indices
            self.memories = new_memories
            self.topic_index = {}
            self.emotion_index = {}
            self.keyword_index = {}
            self.timestamp_index = []
            
            # Re-add each memory to rebuild indices
            for memory in new_memories:
                self._add_to_memory(memory, skip_save=True)
            
            # Rewrite the memory file
            try:
                os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
                with open(self.memory_file_path, 'w') as f:
                    for memory in new_memories:
                        f.write(json.dumps(memory) + '\n')
            except Exception as e:
                logger.error(f"Error rewriting memory file: {str(e)}")
        
        logger.info(f"Cleared {cleared_count} memories")
        return {"status": "success", "cleared_count": cleared_count}


# Example usage
if __name__ == "__main__":
    # Initialize memory system
    memory_system = ConversationMemory()
    
    # Store a conversation
    memory = memory_system.store(
        content="Today we discussed the weather and how it might affect our plans for the weekend.",
        metadata={
            "topic": "weather",
            "emotion": "neutral",
            "keywords": ["weather", "weekend", "plans"]
        }
    )
    
    # Retrieve memories by topic
    weather_memories = memory_system.retrieve_by_topic("weather")
    print(f"Found {len(weather_memories)} memories about weather") 