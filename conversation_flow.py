"""
LUMINA v7.5 - Conversation Flow Module

This module provides conversation flow management with memory of previous topics 
and support for cascading thought processes.
"""

import os
import logging
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/conversation_flow.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ConversationFlow")

class ConversationFlow:
    """
    Manages conversation flow including topic memory and thought continuation.
    
    Features:
    - Remembers previous topics in the conversation
    - Tracks cascading thought processes
    - Identifies key concepts for follow-up
    - Maintains conversation context for more natural responses
    - Enables recall and references to earlier conversation parts
    """
    
    def __init__(self, data_dir: str = "data/conversations", 
                 context_window_size: int = 10,
                 max_topics: int = 50,
                 min_topic_relevance: float = 0.3):
        """
        Initialize the conversation flow system.
        
        Args:
            data_dir: Directory to store conversation data
            context_window_size: Number of exchanges to keep in immediate context
            max_topics: Maximum number of topics to track per conversation
            min_topic_relevance: Minimum relevance score for topic tracking
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.context_window_size = context_window_size
        self.max_topics = max_topics
        self.min_topic_relevance = min_topic_relevance
        
        # Active conversation state
        self.current_conversation_id = None
        self.conversation_history = []
        self.active_topics = set()
        self.topic_occurrences = {}
        self.thought_processes = []
        self.current_thought_process = None
        
        # Initialize conversation
        self._initialize_conversation()
        
        logger.info(f"ConversationFlow initialized with context window of {context_window_size}")
    
    def _initialize_conversation(self):
        """Initialize a new conversation"""
        self.current_conversation_id = str(uuid.uuid4())
        self.conversation_history = []
        self.active_topics = set()
        self.topic_occurrences = {}
        self.thought_processes = []
        self.current_thought_process = None
        
        # Create conversation metadata
        self.conversation_metadata = {
            "id": self.current_conversation_id,
            "created_at": time.time(),
            "topic_history": [],
            "thought_process_count": 0,
            "exchange_count": 0
        }
        
        # Save conversation metadata
        self._save_conversation_metadata()
        
        logger.info(f"New conversation initialized with ID: {self.current_conversation_id}")
    
    def _save_conversation_metadata(self):
        """Save conversation metadata to disk"""
        if not self.current_conversation_id:
            return
            
        try:
            # Update metadata timestamps
            self.conversation_metadata["updated_at"] = time.time()
            self.conversation_metadata["exchange_count"] = len(self.conversation_history)
            self.conversation_metadata["topic_count"] = len(self.active_topics)
            
            # Save to file
            metadata_file = self.data_dir / f"{self.current_conversation_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.conversation_metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving conversation metadata: {e}")
    
    def process_exchange(self, user_message: str, system_response: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a conversation exchange to update flow state.
        
        Args:
            user_message: The user's message
            system_response: The system's response
            metadata: Additional metadata about the exchange
            
        Returns:
            Dict containing updated context information for next exchange
        """
        if metadata is None:
            metadata = {}
            
        # Create exchange object
        exchange = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "user_message": user_message,
            "system_response": system_response,
            "metadata": metadata,
            "active_topics": list(self.active_topics),
            "thought_process_id": self.current_thought_process
        }
        
        # Update conversation history
        self.conversation_history.append(exchange)
        if len(self.conversation_history) > self.context_window_size * 3:
            # Keep more than just the context window for references,
            # but limit overall size
            self.conversation_history = self.conversation_history[-self.context_window_size*3:]
        
        # Extract and update topics
        new_topics = self._extract_topics(user_message, system_response)
        self._update_active_topics(new_topics)
        
        # Update thought processes
        self._update_thought_processes(exchange)
        
        # Save exchange to disk
        self._save_exchange(exchange)
        
        # Update conversation metadata
        self.conversation_metadata["exchange_count"] += 1
        self.conversation_metadata["updated_at"] = time.time()
        self._save_conversation_metadata()
        
        # Generate context for next exchange
        return self._generate_context_for_next_exchange()
    
    def _extract_topics(self, user_message: str, system_response: str) -> Dict[str, float]:
        """
        Extract topics from the current exchange.
        
        Args:
            user_message: User's message
            system_response: System's response
            
        Returns:
            Dict mapping topics to relevance scores
        """
        # In a real implementation, this would use NLP processing
        # For now, we'll use a simple keyword-based approach
        
        combined_text = f"{user_message} {system_response}".lower()
        words = combined_text.split()
        
        # Filter stop words & short words
        stop_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "is", "are", "was"}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word occurrences
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate relevance based on frequency
        total_words = max(1, len(filtered_words))
        topics = {}
        for word, count in word_counts.items():
            relevance = min(1.0, count / (total_words * 0.2))
            if relevance >= self.min_topic_relevance:
                topics[word] = relevance
        
        # Limit to most relevant topics
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_topics[:self.max_topics])
    
    def _update_active_topics(self, new_topics: Dict[str, float]):
        """
        Update the active topics based on new topics.
        
        Args:
            new_topics: Dict mapping topics to relevance scores
        """
        # Add new topics to active set
        for topic, relevance in new_topics.items():
            self.active_topics.add(topic)
            
            # Track occurrence count
            if topic not in self.topic_occurrences:
                self.topic_occurrences[topic] = 1
                # Add to topic history in metadata
                self.conversation_metadata["topic_history"].append({
                    "topic": topic,
                    "first_occurrence": time.time(),
                    "latest_occurrence": time.time(),
                    "occurrences": 1,
                    "initial_relevance": relevance
                })
            else:
                self.topic_occurrences[topic] += 1
                # Update topic history
                for topic_data in self.conversation_metadata["topic_history"]:
                    if topic_data["topic"] == topic:
                        topic_data["occurrences"] += 1
                        topic_data["latest_occurrence"] = time.time()
                        break
        
        # Limit active topics to most frequently occurring ones
        if len(self.active_topics) > self.max_topics:
            sorted_topics = sorted(
                [(t, self.topic_occurrences.get(t, 0)) for t in self.active_topics],
                key=lambda x: x[1],
                reverse=True
            )
            self.active_topics = set([t[0] for t in sorted_topics[:self.max_topics]])
    
    def _update_thought_processes(self, exchange: Dict[str, Any]):
        """
        Update thought processes based on the current exchange.
        
        Args:
            exchange: The current exchange data
        """
        # For now, we'll use a simple approach where a thought process
        # is a sequence of related exchanges
        
        # Check if this exchange continues an existing thought process
        if self.current_thought_process:
            # Get the last exchange in the current thought process
            last_exchange = None
            for ex in reversed(self.conversation_history[:-1]):  # Exclude current exchange
                if ex.get("thought_process_id") == self.current_thought_process:
                    last_exchange = ex
                    break
            
            # Check if current exchange continues the thought process
            if last_exchange:
                # Simple check for continuity - overlap in topics
                last_topics = set(last_exchange.get("active_topics", []))
                current_topics = set(exchange.get("active_topics", []))
                
                # If there's significant topic overlap, consider it the same thought process
                if len(last_topics & current_topics) >= min(2, len(last_topics) * 0.3):
                    # Continue current thought process
                    exchange["thought_process_id"] = self.current_thought_process
                    return
        
        # If we reach here, start a new thought process
        new_process_id = str(uuid.uuid4())
        self.current_thought_process = new_process_id
        exchange["thought_process_id"] = new_process_id
        
        # Add to thought processes list
        self.thought_processes.append({
            "id": new_process_id,
            "started_at": time.time(),
            "starting_exchange_id": exchange["id"],
            "topics": list(self.active_topics)
        })
        
        # Update metadata
        self.conversation_metadata["thought_process_count"] += 1
    
    def _save_exchange(self, exchange: Dict[str, Any]):
        """
        Save an exchange to disk.
        
        Args:
            exchange: Exchange data to save
        """
        try:
            # Create file path
            exchange_file = self.data_dir / f"{self.current_conversation_id}_{exchange['id']}.json"
            
            # Save to file
            with open(exchange_file, 'w') as f:
                json.dump(exchange, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving exchange: {e}")
    
    def _generate_context_for_next_exchange(self) -> Dict[str, Any]:
        """
        Generate context information for the next exchange.
        
        Returns:
            Dict containing context for next exchange
        """
        # Get recent history (limited to context window)
        recent_history = self.conversation_history[-self.context_window_size:]
        
        # Get active thought process information
        thought_process = None
        if self.current_thought_process:
            for tp in self.thought_processes:
                if tp["id"] == self.current_thought_process:
                    thought_process = tp
                    break
        
        # Create context object
        context = {
            "conversation_id": self.current_conversation_id,
            "exchange_count": len(self.conversation_history),
            "recent_history": [{
                "user_message": ex["user_message"],
                "system_response": ex["system_response"],
                "timestamp": ex["timestamp"]
            } for ex in recent_history],
            "active_topics": list(self.active_topics),
            "topic_occurrences": self.topic_occurrences,
            "current_thought_process": self.current_thought_process,
            "thought_process_info": thought_process,
            "context_window_size": self.context_window_size
        }
        
        return context
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """
        Get the current conversation context.
        
        Returns:
            Dict containing conversation context
        """
        return self._generate_context_for_next_exchange()
    
    def search_conversation_history(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search conversation history for relevant exchanges.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching exchanges
        """
        query = query.lower()
        results = []
        
        for exchange in reversed(self.conversation_history):
            combined_text = f"{exchange['user_message']} {exchange['system_response']}".lower()
            
            if query in combined_text:
                results.append({
                    "id": exchange["id"],
                    "user_message": exchange["user_message"],
                    "system_response": exchange["system_response"],
                    "timestamp": exchange["timestamp"],
                    "topics": exchange.get("active_topics", [])
                })
                
                if len(results) >= limit:
                    break
                    
        return results
    
    def get_topic_history(self, topic: str) -> List[Dict[str, Any]]:
        """
        Get history of exchanges related to a specific topic.
        
        Args:
            topic: Topic to search for
            
        Returns:
            List of exchanges related to the topic
        """
        topic = topic.lower()
        results = []
        
        for exchange in self.conversation_history:
            if topic in exchange.get("active_topics", []):
                results.append({
                    "id": exchange["id"],
                    "user_message": exchange["user_message"],
                    "system_response": exchange["system_response"],
                    "timestamp": exchange["timestamp"]
                })
                
        return results
    
    def get_thought_process(self, thought_process_id: str) -> List[Dict[str, Any]]:
        """
        Get all exchanges in a specific thought process.
        
        Args:
            thought_process_id: ID of the thought process
            
        Returns:
            List of exchanges in the thought process
        """
        results = []
        
        for exchange in self.conversation_history:
            if exchange.get("thought_process_id") == thought_process_id:
                results.append({
                    "id": exchange["id"],
                    "user_message": exchange["user_message"],
                    "system_response": exchange["system_response"],
                    "timestamp": exchange["timestamp"],
                    "topics": exchange.get("active_topics", [])
                })
                
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the conversation flow.
        
        Returns:
            Dict containing conversation statistics
        """
        return {
            "conversation_id": self.current_conversation_id,
            "exchange_count": len(self.conversation_history),
            "active_topics_count": len(self.active_topics),
            "thought_process_count": len(self.thought_processes),
            "current_thought_process": self.current_thought_process,
            "context_window_size": self.context_window_size
        }
    
    def save_state(self):
        """Save the current state to disk"""
        self._save_conversation_metadata()
        
        # Save any unsaved exchanges
        for exchange in self.conversation_history:
            if "saved" not in exchange or not exchange["saved"]:
                self._save_exchange(exchange)
                exchange["saved"] = True
                
        logger.info(f"Conversation flow state saved for conversation {self.current_conversation_id}")
    
    def reset(self):
        """Reset the conversation flow state"""
        self._initialize_conversation()
        logger.info("Conversation flow state reset") 