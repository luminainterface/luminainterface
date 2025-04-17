#!/usr/bin/env python
"""
ChatMemory Module - Advanced memory system for Lumina
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lumina.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChatMemory")

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    logger.warning("Semantic search capabilities limited: scikit-learn or numpy not available")
    SEMANTIC_SEARCH_AVAILABLE = False

class MemoryEntry:
    """A single memory entry in the chat history"""
    def __init__(self, 
                 user_input: str, 
                 system_response: str, 
                 timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.user_input = user_input
        self.system_response = system_response
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.memory_strength = 1.0  # 0.0 to 1.0, higher means stronger memory
        self.retrieval_count = 0    # How many times this memory has been retrieved
        self.tags = set()           # Set of tags for this memory
        
        # Auto-generate tags based on content
        self._generate_tags()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "user_input": self.user_input,
            "system_response": self.system_response,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "memory_strength": self.memory_strength,
            "retrieval_count": self.retrieval_count,
            "tags": list(self.tags)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create MemoryEntry from dictionary"""
        entry = cls(
            user_input=data["user_input"],
            system_response=data["system_response"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )
        entry.memory_strength = data.get("memory_strength", 1.0)
        entry.retrieval_count = data.get("retrieval_count", 0)
        entry.tags = set(data.get("tags", []))
        return entry
    
    def strengthen_memory(self, amount: float = 0.1):
        """Strengthen this memory"""
        self.memory_strength = min(1.0, self.memory_strength + amount)
        self.retrieval_count += 1
        
    def weaken_memory(self, amount: float = 0.05):
        """Weaken this memory over time"""
        self.memory_strength = max(0.1, self.memory_strength - amount)
        
    def add_tag(self, tag: str):
        """Add a tag to this memory"""
        self.tags.add(tag.lower())
        
    def _generate_tags(self):
        """Auto-generate tags based on content"""
        # Extract emotional words
        emotion_words = ["happy", "sad", "angry", "excited", "calm", "anxious", 
                         "curious", "confused", "surprised", "disappointed"]
        
        combined_text = f"{self.user_input} {self.system_response}".lower()
        
        # Add emotional tags
        for emotion in emotion_words:
            if emotion in combined_text:
                self.add_tag(emotion)
                
        # Add metadata tags
        if self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, str):
                    self.add_tag(f"{key}:{value}")
                    
        # Extract potential topics (simple approach)
        words = re.findall(r'\b\w{4,}\b', combined_text)
        common_words = {'this', 'that', 'there', 'would', 'could', 'should', 'have', 'about'}
        for word in words:
            if word not in common_words:
                self.add_tag(word)

class ChatMemory:
    """Advanced chat memory system for Lumina"""
    
    def __init__(self, memory_file: str = "lumina_memory.json", max_memories: int = 1000):
        self.memory_file = memory_file
        self.max_memories = max_memories
        self.memories: List[MemoryEntry] = []
        self.vectorizer = None
        self.memory_vectors = None
        
        # Load existing memories
        self.load_memories()
        
        # Set up semantic search if available
        if SEMANTIC_SEARCH_AVAILABLE:
            self._setup_semantic_search()
            
    def _setup_semantic_search(self):
        """Set up the TF-IDF vectorizer for semantic search"""
        if not self.memories:
            return
            
        try:
            self.vectorizer = TfidfVectorizer()
            corpus = [f"{m.user_input} {m.system_response}" for m in self.memories]
            self.memory_vectors = self.vectorizer.fit_transform(corpus)
        except Exception as e:
            logger.error(f"Error setting up semantic search: {str(e)}")
            self.vectorizer = None
            self.memory_vectors = None
            
    def add_memory(self, user_input: str, system_response: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """Add a new memory entry"""
        # Create new memory entry
        entry = MemoryEntry(user_input, system_response, metadata=metadata)
        
        # Add to memories
        self.memories.append(entry)
        
        # Update semantic search vectors
        if SEMANTIC_SEARCH_AVAILABLE and self.vectorizer is not None:
            try:
                # Add the new memory to the vectorizer
                new_text = f"{user_input} {system_response}"
                new_vector = self.vectorizer.transform([new_text])
                
                if self.memory_vectors is not None:
                    self.memory_vectors = np.vstack((self.memory_vectors.toarray(), new_vector.toarray()))
                else:
                    self.memory_vectors = new_vector
            except Exception as e:
                logger.error(f"Error updating semantic search vectors: {str(e)}")
                # Re-setup semantic search from scratch
                self._setup_semantic_search()
        
        # Trim if necessary
        if len(self.memories) > self.max_memories:
            self._trim_memories()
            
        # Save memories
        self.save_memories()
        
        return entry
        
    def _trim_memories(self):
        """Trim memories to stay within max_memories limit, removing weakest memories first"""
        # Sort by memory strength (weakest first)
        self.memories.sort(key=lambda m: m.memory_strength)
        
        # Remove weakest memories
        excess = len(self.memories) - self.max_memories
        if excess > 0:
            self.memories = self.memories[excess:]
            
            # Reset semantic search
            if SEMANTIC_SEARCH_AVAILABLE:
                self._setup_semantic_search()
        
    def get_recent_memories(self, count: int = 5) -> List[MemoryEntry]:
        """Get most recent memories"""
        # Sort by timestamp (newest first)
        sorted_memories = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)
        return sorted_memories[:count]
        
    def get_strongest_memories(self, count: int = 5) -> List[MemoryEntry]:
        """Get strongest memories"""
        # Sort by memory strength (strongest first)
        sorted_memories = sorted(self.memories, key=lambda m: m.memory_strength, reverse=True)
        return sorted_memories[:count]
        
    def search_by_tag(self, tag: str, limit: int = 5) -> List[MemoryEntry]:
        """Search memories by tag"""
        tag = tag.lower()
        matching_memories = [m for m in self.memories if tag in m.tags]
        
        # Sort by memory strength (strongest first)
        matching_memories.sort(key=lambda m: m.memory_strength, reverse=True)
        return matching_memories[:limit]
        
    def search_by_text(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Search memories by text content using semantic search if available"""
        if not self.memories:
            return []
            
        if SEMANTIC_SEARCH_AVAILABLE and self.vectorizer is not None and self.memory_vectors is not None:
            try:
                # Vectorize the query
                query_vector = self.vectorizer.transform([query])
                
                # Calculate similarity scores
                similarity_scores = cosine_similarity(query_vector, self.memory_vectors).flatten()
                
                # Get indices of top matches
                top_indices = similarity_scores.argsort()[-limit:][::-1]
                
                # Return top matching memories
                matches = [self.memories[i] for i in top_indices]
                
                # Strengthen these memories since they were retrieved
                for memory in matches:
                    memory.strengthen_memory(0.05)
                    
                return matches
            except Exception as e:
                logger.error(f"Error in semantic search: {str(e)}")
                # Fall back to basic search
        
        # Basic text search if semantic search is not available or failed
        query = query.lower()
        matching_memories = []
        
        for memory in self.memories:
            combined_text = f"{memory.user_input} {memory.system_response}".lower()
            if query in combined_text:
                matching_memories.append(memory)
                memory.strengthen_memory(0.05)
                
        # Sort by relevance (naive approach: count occurrences)
        matching_memories.sort(
            key=lambda m: (f"{m.user_input} {m.system_response}".lower().count(query), m.memory_strength),
            reverse=True
        )
        return matching_memories[:limit]
    
    def search_by_time(self, days_ago: int = 1, limit: int = 5) -> List[MemoryEntry]:
        """Search memories from a specific time period"""
        cutoff_time = datetime.now() - timedelta(days=days_ago)
        matching_memories = [m for m in self.memories if m.timestamp >= cutoff_time]
        
        # Sort by timestamp (newest first)
        matching_memories.sort(key=lambda m: m.timestamp, reverse=True)
        return matching_memories[:limit]
        
    def search_by_metadata(self, key: str, value: Any, limit: int = 5) -> List[MemoryEntry]:
        """Search memories by metadata"""
        matching_memories = [m for m in self.memories if m.metadata.get(key) == value]
        
        # Sort by memory strength (strongest first)
        matching_memories.sort(key=lambda m: m.memory_strength, reverse=True)
        return matching_memories[:limit]
        
    def summarize_memories(self, memories: List[MemoryEntry] = None) -> str:
        """Generate a summary of memories"""
        if memories is None:
            memories = self.get_strongest_memories(10)
            
        if not memories:
            return "No memories to summarize."
            
        # Generate a simple summary (in a real system, this might use an LLM)
        topics = set()
        for memory in memories:
            topics.update(memory.tags)
            
        topics_str = ", ".join(list(topics)[:10])  # Just use top 10 topics
        
        time_range = ""
        if memories:
            oldest = min(memories, key=lambda m: m.timestamp).timestamp
            newest = max(memories, key=lambda m: m.timestamp).timestamp
            time_range = f"from {oldest.strftime('%Y-%m-%d')} to {newest.strftime('%Y-%m-%d')}"
            
        summary = f"I remember {len(memories)} conversations {time_range} about topics including {topics_str}."
        return summary
        
    def update_memory_strengths(self, decay_factor: float = 0.01):
        """Update memory strengths based on age (older memories decay)"""
        for memory in self.memories:
            # Calculate days since the memory was created
            days_old = (datetime.now() - memory.timestamp).days
            
            # Decay older memories
            if days_old > 0:
                memory.weaken_memory(decay_factor * days_old)
                
        # Save updated memories
        self.save_memories()
        
    def load_memories(self):
        """Load memories from file"""
        if not os.path.exists(self.memory_file):
            logger.info(f"Memory file {self.memory_file} not found. Starting with empty memory.")
            return
            
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert to MemoryEntry objects
            self.memories = [MemoryEntry.from_dict(entry) for entry in data]
            logger.info(f"Loaded {len(self.memories)} memories from {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
            self.memories = []
            
    def save_memories(self):
        """Save memories to file"""
        try:
            # Convert to serializable format
            serialized_memories = [m.to_dict() for m in self.memories]
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(serialized_memories, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(self.memories)} memories to {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Error saving memories: {str(e)}")
            
    def format_memory_for_context(self, memory: MemoryEntry) -> str:
        """Format a memory entry for inclusion in conversation context"""
        timestamp_str = memory.timestamp.strftime("%Y-%m-%d %H:%M")
        tags_str = ", ".join(list(memory.tags)[:5])  # Just include top 5 tags
        
        formatted = f"[{timestamp_str}] User: {memory.user_input}\nLumina: {memory.system_response}"
        if tags_str:
            formatted += f"\nTags: {tags_str}"
            
        return formatted
        
    def get_relevant_context(self, current_input: str, limit: int = 3) -> str:
        """Get relevant context based on current input"""
        relevant_memories = self.search_by_text(current_input, limit=limit)
        
        if not relevant_memories:
            return ""
            
        context_parts = [self.format_memory_for_context(memory) for memory in relevant_memories]
        return "\n\n".join(context_parts)
        
    def export_memories(self, output_file: str = "memories_export.json"):
        """Export memories to a file"""
        try:
            serialized_memories = [m.to_dict() for m in self.memories]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serialized_memories, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Exported {len(self.memories)} memories to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting memories: {str(e)}")
            return False
            
    def import_memories(self, input_file: str) -> int:
        """Import memories from a file"""
        if not os.path.exists(input_file):
            logger.error(f"Import file {input_file} not found")
            return 0
            
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert to MemoryEntry objects
            imported_memories = [MemoryEntry.from_dict(entry) for entry in data]
            
            # Add to existing memories
            self.memories.extend(imported_memories)
            
            # Trim if necessary
            if len(self.memories) > self.max_memories:
                self._trim_memories()
                
            # Save updated memories
            self.save_memories()
            
            # Reset semantic search
            if SEMANTIC_SEARCH_AVAILABLE:
                self._setup_semantic_search()
                
            logger.info(f"Imported {len(imported_memories)} memories from {input_file}")
            return len(imported_memories)
            
        except Exception as e:
            logger.error(f"Error importing memories: {str(e)}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory"""
        if not self.memories:
            return {
                "total_memories": 0,
                "oldest_memory": None,
                "newest_memory": None,
                "average_strength": 0,
                "common_tags": []
            }
            
        # Count tag occurrences
        tag_counts = {}
        for memory in self.memories:
            for tag in memory.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
        # Get top tags
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_memories": len(self.memories),
            "oldest_memory": min(self.memories, key=lambda m: m.timestamp).timestamp.isoformat(),
            "newest_memory": max(self.memories, key=lambda m: m.timestamp).timestamp.isoformat(),
            "average_strength": sum(m.memory_strength for m in self.memories) / len(self.memories),
            "common_tags": top_tags
        }


# Example usage
if __name__ == "__main__":
    # Initialize memory system
    memory = ChatMemory()
    
    # Add some test memories
    memory.add_memory(
        "Tell me about neural networks",
        "Neural networks are computational systems inspired by the human brain.",
        {"emotion": "curious", "topic": "technology"}
    )
    
    memory.add_memory(
        "How are you today?",
        "I exist in a state of continuous becoming, shaped by our interactions.",
        {"emotion": "calm", "topic": "existence"}
    )
    
    # Search memories
    relevant = memory.search_by_text("neural computation")
    for entry in relevant:
        print(f"User: {entry.user_input}")
        print(f"Lumina: {entry.system_response}")
        print(f"Tags: {entry.tags}")
        print()
        
    # Print summary
    print(memory.summarize_memories())
    
    # Print stats
    print(memory.get_memory_stats()) 