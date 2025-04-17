#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Onsite Memory System for Mistral Integration

This module provides a persistent, local storage system for conversation history
and knowledge retrieval that works with the Mistral integration.
"""

import os
import json
import time
import logging
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class OnsiteMemory:
    """
    Onsite Memory System for storing conversation history and knowledge items
    
    This class provides persistent storage for:
    - Chat conversations and history
    - Dictionary-based knowledge entries
    - Learned topics and information
    - User preferences and settings
    """
    
    def __init__(
        self,
        data_dir: str = "data/onsite_memory",
        memory_file: str = "memory.json",
        auto_save: bool = True,
        save_interval: int = 60,  # seconds
        max_conversations: int = 100,
        max_entries: int = 1000
    ):
        """
        Initialize the onsite memory system
        
        Args:
            data_dir: Directory for storing memory files
            memory_file: Filename for the main memory file
            auto_save: Whether to automatically save changes
            save_interval: How often to auto-save (in seconds)
            max_conversations: Maximum number of conversation entries to keep
            max_entries: Maximum number of dictionary entries to keep
        """
        self.data_dir = Path(data_dir)
        self.memory_file = self.data_dir / memory_file
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.max_conversations = max_conversations
        self.max_entries = max_entries
        
        # Memory storage
        self.memory = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "conversations": [],
            "knowledge": {},
            "preferences": {},
            "stats": {
                "total_conversations": 0,
                "total_messages": 0,
                "total_knowledge_entries": 0,
                "last_conversation": None
            }
        }
        
        # Threading
        self.save_lock = threading.RLock()
        self.save_thread = None
        self.stop_auto_save = threading.Event()
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load memory if file exists
        self._load_memory()
        
        # Start auto-save thread if enabled
        if auto_save:
            self._start_auto_save()
        
        logger.info(f"Onsite Memory initialized at {self.memory_file}")
    
    def _load_memory(self) -> None:
        """Load memory from file if it exists"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    loaded_memory = json.load(f)
                    self.memory = loaded_memory
                    logger.info(f"Loaded memory from {self.memory_file}")
                    
                    # Update stats
                    self.memory["stats"]["total_conversations"] = len(self.memory.get("conversations", []))
                    self.memory["stats"]["total_knowledge_entries"] = len(self.memory.get("knowledge", {}))
            else:
                logger.info(f"No memory file found at {self.memory_file}, creating new memory")
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
    
    def save_memory(self) -> bool:
        """
        Save memory to disk
        
        Returns:
            bool: True if successful, False otherwise
        """
        with self.save_lock:
            try:
                # Update metadata
                self.memory["metadata"]["updated"] = datetime.now().isoformat()
                
                # Update stats
                self.memory["stats"]["total_conversations"] = len(self.memory.get("conversations", []))
                self.memory["stats"]["total_knowledge_entries"] = len(self.memory.get("knowledge", {}))
                
                # Save to disk
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(self.memory, f, indent=2)
                
                logger.debug(f"Saved memory to {self.memory_file}")
                return True
            except Exception as e:
                logger.error(f"Error saving memory: {str(e)}")
                return False
    
    def _start_auto_save(self) -> None:
        """Start background thread for auto-saving memory"""
        def auto_save_thread():
            while not self.stop_auto_save.is_set():
                time.sleep(self.save_interval)
                self.save_memory()
        
        self.save_thread = threading.Thread(target=auto_save_thread, daemon=True)
        self.save_thread.start()
        logger.debug(f"Started auto-save thread (interval: {self.save_interval}s)")
    
    def stop(self) -> None:
        """Stop the auto-save thread and save memory one last time"""
        if self.save_thread:
            self.stop_auto_save.set()
            self.save_thread.join(timeout=2.0)
        
        # Final save
        self.save_memory()
        logger.info("Onsite Memory stopped")
    
    # Conversation Methods
    
    def add_conversation(
        self, 
        user_message: str, 
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a conversation to memory
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            metadata: Optional metadata for the conversation
            
        Returns:
            str: Conversation ID
        """
        # Generate conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Create timestamp
        timestamp = datetime.now().isoformat()
        
        # Create conversation entry
        conversation = {
            "id": conversation_id,
            "timestamp": timestamp,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "metadata": metadata or {}
        }
        
        # Add to conversations
        self.memory["conversations"].append(conversation)
        
        # Update stats
        self.memory["stats"]["total_messages"] += 1
        self.memory["stats"]["last_conversation"] = timestamp
        
        # Limit size if needed
        if len(self.memory["conversations"]) > self.max_conversations:
            self.memory["conversations"] = self.memory["conversations"][-self.max_conversations:]
        
        # Save if not auto-saving
        if not self.auto_save:
            self.save_memory()
        
        return conversation_id
    
    def get_conversation_history(
        self, 
        limit: int = 10, 
        offset: int = 0,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            limit: Maximum number of conversations to return
            offset: Starting offset in the conversation list
            include_metadata: Whether to include metadata
            
        Returns:
            List of conversation entries
        """
        conversations = self.memory.get("conversations", [])
        
        # Sort by timestamp (newest first)
        sorted_conversations = sorted(
            conversations, 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        
        # Apply offset and limit
        result = sorted_conversations[offset:offset+limit]
        
        # Remove metadata if not requested
        if not include_metadata:
            result = [{k: v for k, v in conv.items() if k != "metadata"} 
                     for conv in result]
        
        return result
    
    def search_conversations(
        self, 
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversations for the query term
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching conversations
        """
        if not query:
            return []
        
        query = query.lower()
        matches = []
        
        for conv in self.memory.get("conversations", []):
            user_msg = conv.get("user_message", "").lower()
            assistant_msg = conv.get("assistant_response", "").lower()
            
            if query in user_msg or query in assistant_msg:
                # Calculate a simple relevance score
                relevance = 0
                if query in user_msg:
                    relevance += user_msg.count(query) * 2
                if query in assistant_msg:
                    relevance += assistant_msg.count(query)
                
                matches.append({
                    "conversation": conv,
                    "relevance": relevance
                })
        
        # Sort by relevance
        matches.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Return just the conversations, limited
        return [m["conversation"] for m in matches[:limit]]
    
    # Knowledge Dictionary Methods
    
    def add_knowledge(
        self, 
        topic: str, 
        content: str,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add or update knowledge in the dictionary
        
        Args:
            topic: The topic/key for the knowledge
            content: The content/definition
            source: Optional source reference
            metadata: Optional additional metadata
            
        Returns:
            bool: True if successful
        """
        if not topic or not content:
            return False
        
        try:
            # Check if we need to update an existing entry
            is_update = topic.lower() in {k.lower() for k in self.memory.get("knowledge", {}).keys()}
            
            # Find exact key if case-insensitive match
            if is_update:
                existing_key = next(k for k in self.memory.get("knowledge", {}).keys() 
                                   if k.lower() == topic.lower())
                
                # Get existing entry
                existing_entry = self.memory["knowledge"][existing_key]
                
                # Update entry
                entry = {
                    "content": content,
                    "sources": existing_entry.get("sources", []),
                    "created": existing_entry.get("created", datetime.now().isoformat()),
                    "updated": datetime.now().isoformat(),
                    "metadata": metadata or existing_entry.get("metadata", {})
                }
                
                # Add source if provided and not already in sources
                if source and source not in entry["sources"]:
                    entry["sources"].append(source)
                
                # Update entry
                self.memory["knowledge"][existing_key] = entry
            else:
                # Create new entry
                entry = {
                    "content": content,
                    "sources": [source] if source else [],
                    "created": datetime.now().isoformat(),
                    "updated": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
                
                # Add entry
                self.memory["knowledge"][topic] = entry
            
            # Save if not auto-saving
            if not self.auto_save:
                self.save_memory()
            
            return True
        except Exception as e:
            logger.error(f"Error adding knowledge: {str(e)}")
            return False
    
    def get_knowledge(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge by topic
        
        Args:
            topic: The topic to retrieve
            
        Returns:
            Optional[Dict]: The knowledge entry if found, None otherwise
        """
        # Check for exact match first
        if topic in self.memory.get("knowledge", {}):
            return self.memory["knowledge"][topic]
        
        # Try case-insensitive match
        for key in self.memory.get("knowledge", {}).keys():
            if key.lower() == topic.lower():
                return self.memory["knowledge"][key]
        
        return None
    
    def search_knowledge(
        self, 
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge entries
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching entries with their topics
        """
        if not query:
            return []
        
        query = query.lower()
        matches = []
        
        for topic, entry in self.memory.get("knowledge", {}).items():
            content = entry.get("content", "").lower()
            
            if query in topic.lower() or query in content:
                # Calculate a simple relevance score
                relevance = 0
                if query in topic.lower():
                    relevance += 3
                    if query == topic.lower():
                        relevance += 5
                
                if query in content:
                    relevance += 1
                    relevance += 0.1 * content.count(query)
                
                matches.append({
                    "topic": topic,
                    "entry": entry,
                    "relevance": relevance
                })
        
        # Sort by relevance
        matches.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Return limited results
        return matches[:limit]
    
    def delete_knowledge(self, topic: str) -> bool:
        """
        Delete knowledge entry
        
        Args:
            topic: Topic to delete
            
        Returns:
            bool: True if deleted
        """
        # Check for exact match first
        if topic in self.memory.get("knowledge", {}):
            del self.memory["knowledge"][topic]
            
            # Save if not auto-saving
            if not self.auto_save:
                self.save_memory()
            
            return True
        
        # Try case-insensitive match
        for key in list(self.memory.get("knowledge", {}).keys()):
            if key.lower() == topic.lower():
                del self.memory["knowledge"][key]
                
                # Save if not auto-saving
                if not self.auto_save:
                    self.save_memory()
                
                return True
        
        return False
    
    def get_all_topics(self) -> List[str]:
        """
        Get all knowledge topics
        
        Returns:
            List of all topics
        """
        return list(self.memory.get("knowledge", {}).keys())
    
    # Preference Methods
    
    def set_preference(self, key: str, value: Any) -> bool:
        """
        Set a user preference
        
        Args:
            key: Preference key
            value: Preference value
            
        Returns:
            bool: True if successful
        """
        try:
            self.memory["preferences"][key] = value
            
            # Save if not auto-saving
            if not self.auto_save:
                self.save_memory()
            
            return True
        except Exception as e:
            logger.error(f"Error setting preference: {str(e)}")
            return False
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference
        
        Args:
            key: Preference key
            default: Default value if not found
            
        Returns:
            Preference value or default
        """
        return self.memory.get("preferences", {}).get(key, default)
    
    # Stats Methods
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics
        
        Returns:
            Dict with statistics
        """
        return self.memory.get("stats", {}) 