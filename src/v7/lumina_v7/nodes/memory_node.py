"""
Memory Node

This module implements the Memory Consciousness Node for the V7 system, providing
persistent, decay-aware memory capabilities.
"""

import os
import time
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime

# Import base node class
from src.v7.lumina_v7.core.node_consciousness_manager import BaseConsciousnessNode, NodeState

# Set up logging
logger = logging.getLogger("lumina_v7.memory_node")

class MemoryStore:
    """
    Memory storage system with persistence and decay capabilities.
    """
    
    def __init__(self, persistence_file: str = None, store_type: str = "json"):
        """
        Initialize the memory store.
        
        Args:
            persistence_file: Path to the persistence file (optional)
            store_type: Type of storage ('json' or 'sqlite')
        """
        self.store_type = store_type
        self.persistence_file = persistence_file
        self.memories = {}  # id -> memory dict
        self.memory_index = {
            "type": {},      # type -> [id, id, ...]
            "source": {},    # source -> [id, id, ...]
            "importance": {} # importance range -> [id, id, ...]
        }
        self.associations = {}  # (id1, id2) -> association dict
        
        # Load existing memories if persistence file exists
        if persistence_file and os.path.exists(persistence_file):
            self._load_memories()
        
        logger.debug(f"Memory store initialized with store_type={store_type}")
    
    def store(self, memory: Dict[str, Any]) -> str:
        """
        Store a memory in the memory store.
        
        Args:
            memory: The memory to store
            
        Returns:
            The memory ID
        """
        memory_id = memory.get("id")
        if not memory_id:
            memory_id = str(uuid.uuid4())
            memory["id"] = memory_id
        
        # Update timestamps if not present
        if "created_at" not in memory:
            memory["created_at"] = time.time()
        
        memory["last_accessed"] = time.time()
        
        # Store the memory
        self.memories[memory_id] = memory
        
        # Update indexes
        self._update_indexes(memory)
        
        # Persist if configured
        if self.persistence_file:
            self._persist_memories()
        
        return memory_id
    
    def retrieve(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: The ID of the memory to retrieve
            
        Returns:
            The memory dict, or None if not found
        """
        memory = self.memories.get(memory_id)
        if memory:
            # Update last accessed time
            memory["last_accessed"] = time.time()
            return memory.copy()
        return None
    
    def search(self, criteria: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search memories based on criteria.
        
        Args:
            criteria: Search criteria
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        # Start with all memory IDs
        result_ids = set(self.memories.keys())
        
        # Filter by type
        if "memory_type" in criteria:
            type_ids = set(self.memory_index["type"].get(criteria["memory_type"], []))
            result_ids &= type_ids
        
        # Filter by source
        if "source" in criteria:
            source_ids = set(self.memory_index["source"].get(criteria["source"], []))
            result_ids &= source_ids
        
        # Filter by importance
        if "min_importance" in criteria:
            min_importance = criteria["min_importance"]
            result_ids = {mid for mid in result_ids 
                         if self.memories[mid]["importance"] >= min_importance}
        
        # Filter by content
        if "content_contains" in criteria:
            search_term = criteria["content_contains"].lower()
            result_ids = {mid for mid in result_ids 
                         if isinstance(self.memories[mid]["content"], str) and
                         search_term in self.memories[mid]["content"].lower()}
        
        # Retrieve memories, sorted by importance
        results = [self.retrieve(mid) for mid in result_ids]
        results = [mem for mem in results if mem is not None]
        results.sort(key=lambda m: m["importance"], reverse=True)
        
        return results[:limit]
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory from the store.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        if memory_id not in self.memories:
            return False
        
        # Remove from indexes
        memory = self.memories[memory_id]
        self._remove_from_indexes(memory)
        
        # Remove the memory
        del self.memories[memory_id]
        
        # Remove any associations
        self._remove_associations(memory_id)
        
        # Persist if configured
        if self.persistence_file:
            self._persist_memories()
        
        return True
    
    def associate(self, memory_id1: str, memory_id2: str, 
                 association_type: str = "related", strength: float = 0.5,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create an association between two memories.
        
        Args:
            memory_id1: First memory ID
            memory_id2: Second memory ID
            association_type: Type of association
            strength: Association strength (0.0 to 1.0)
            metadata: Additional association metadata
            
        Returns:
            True if association created, False otherwise
        """
        if memory_id1 not in self.memories or memory_id2 not in self.memories:
            return False
        
        # Create unique identifier for the association
        assoc_key = (memory_id1, memory_id2)
        
        # Create the association
        association = {
            "memory_id1": memory_id1,
            "memory_id2": memory_id2,
            "type": association_type,
            "strength": max(0.0, min(1.0, strength)),  # Clamp to 0.0-1.0
            "created_at": time.time(),
            "metadata": metadata or {}
        }
        
        # Store the association
        self.associations[assoc_key] = association
        
        # Persist if configured
        if self.persistence_file:
            self._persist_memories()
        
        return True
    
    def get_associations(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Get all associations for a memory.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            List of association dicts
        """
        results = []
        
        for (id1, id2), assoc in self.associations.items():
            if id1 == memory_id or id2 == memory_id:
                results.append(assoc.copy())
        
        return results
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Get all memories in the store.
        
        Returns:
            List of all memories
        """
        return [mem.copy() for mem in self.memories.values()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.
        
        Returns:
            Dictionary with statistics
        """
        memory_types = {}
        for mem in self.memories.values():
            mem_type = mem.get("memory_type", "unknown")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
        
        avg_importance = 0.0
        if self.memories:
            avg_importance = sum(mem.get("importance", 0.0) for mem in self.memories.values()) / len(self.memories)
        
        return {
            "total_memories": len(self.memories),
            "total_associations": len(self.associations),
            "memory_types": memory_types,
            "average_importance": avg_importance
        }
    
    def _update_indexes(self, memory: Dict[str, Any]) -> None:
        """Update search indexes for a memory"""
        memory_id = memory["id"]
        
        # Index by type
        memory_type = memory.get("memory_type")
        if memory_type:
            if memory_type not in self.memory_index["type"]:
                self.memory_index["type"][memory_type] = []
            if memory_id not in self.memory_index["type"][memory_type]:
                self.memory_index["type"][memory_type].append(memory_id)
        
        # Index by source
        source = memory.get("source")
        if source:
            if source not in self.memory_index["source"]:
                self.memory_index["source"][source] = []
            if memory_id not in self.memory_index["source"][source]:
                self.memory_index["source"][source].append(memory_id)
        
        # Index by importance
        importance = memory.get("importance", 0.0)
        importance_range = str(int(importance * 10) / 10)  # Round to nearest 0.1
        if importance_range not in self.memory_index["importance"]:
            self.memory_index["importance"][importance_range] = []
        if memory_id not in self.memory_index["importance"][importance_range]:
            self.memory_index["importance"][importance_range].append(memory_id)
    
    def _remove_from_indexes(self, memory: Dict[str, Any]) -> None:
        """Remove a memory from search indexes"""
        memory_id = memory["id"]
        
        # Remove from type index
        memory_type = memory.get("memory_type")
        if memory_type and memory_type in self.memory_index["type"]:
            if memory_id in self.memory_index["type"][memory_type]:
                self.memory_index["type"][memory_type].remove(memory_id)
        
        # Remove from source index
        source = memory.get("source")
        if source and source in self.memory_index["source"]:
            if memory_id in self.memory_index["source"][source]:
                self.memory_index["source"][source].remove(memory_id)
        
        # Remove from importance index
        importance = memory.get("importance", 0.0)
        importance_range = str(int(importance * 10) / 10)  # Round to nearest 0.1
        if importance_range in self.memory_index["importance"]:
            if memory_id in self.memory_index["importance"][importance_range]:
                self.memory_index["importance"][importance_range].remove(memory_id)
    
    def _remove_associations(self, memory_id: str) -> None:
        """Remove all associations involving a memory"""
        to_remove = []
        
        for (id1, id2) in self.associations.keys():
            if id1 == memory_id or id2 == memory_id:
                to_remove.append((id1, id2))
        
        for key in to_remove:
            del self.associations[key]
    
    def _persist_memories(self) -> None:
        """Persist memories to storage"""
        if not self.persistence_file:
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.persistence_file)), exist_ok=True)
            
            # Prepare data
            data = {
                "memories": self.memories,
                "associations": {str(k): v for k, v in self.associations.items()},
                "version": "1.0.0",
                "timestamp": time.time()
            }
            
            # Write to file
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Persisted {len(self.memories)} memories to {self.persistence_file}")
        except Exception as e:
            logger.error(f"Error persisting memories: {str(e)}")
    
    def _load_memories(self) -> None:
        """Load memories from storage"""
        if not self.persistence_file or not os.path.exists(self.persistence_file):
            return
        
        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
            
            # Load memories
            self.memories = data.get("memories", {})
            
            # Load associations (convert string tuple keys back to actual tuples)
            associations_data = data.get("associations", {})
            self.associations = {}
            for k_str, v in associations_data.items():
                # Parse the string tuple back to an actual tuple
                parts = k_str.strip("()").split(", ")
                if len(parts) == 2:
                    key = (parts[0].strip("'\""), parts[1].strip("'\""))
                    self.associations[key] = v
            
            # Rebuild indexes
            self.memory_index = {"type": {}, "source": {}, "importance": {}}
            for memory in self.memories.values():
                self._update_indexes(memory)
            
            logger.info(f"Loaded {len(self.memories)} memories from {self.persistence_file}")
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")

class MemoryNode(BaseConsciousnessNode):
    """
    Memory Consciousness Node for the V7 system.
    
    This node provides persistent, decay-aware memory capabilities with support for
    different memory types, importance levels, and rich search.
    """
    
    def __init__(self, node_id: Optional[str] = None, 
                persistence_file: Optional[str] = None,
                store_type: str = "json", 
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Memory Node.
        
        Args:
            node_id: Unique identifier for the node (optional)
            persistence_file: Path to the persistence file (optional)
            store_type: Type of storage ('json' or 'sqlite')
            config: Configuration dictionary (optional)
        """
        super().__init__(node_id=node_id, node_type="memory")
        
        # Configuration
        self.config = {
            "decay_rate": 0.1,
            "decay_interval": 3600,  # seconds (1 hour)
            "minimum_importance": 0.1,
            "decay_enabled": True,
            "reinforce_on_access": True,
            "reinforcement_strength": 0.1,
            "memory_types": ["fact", "experience", "concept", "belief", "goal"]
        }
        
        # Update with custom config
        if config:
            self.config.update(config)
        
        # Initialize memory store
        self.memory_store = MemoryStore(persistence_file, store_type)
        
        # Decay management
        self.last_decay_time = time.time()
        self.decay_thread = None
        self.stop_decay = threading.Event()
        
        # Update node state
        self.node_state.update({
            "memory_capability": "persistent" if persistence_file else "transient",
            "decay_enabled": self.config["decay_enabled"],
            "decay_rate": self.config["decay_rate"],
            "store_type": store_type
        })
        
        # Set personality traits for memory node
        self.personality.update({
            "communication_style": "precise",
            "areas_of_interest": ["knowledge", "patterns", "connections", "temporal_relations"],
            "processing_biases": {
                "recency": 0.6,  # Bias toward recent memories
                "importance": 0.8,  # Bias toward important memories
                "connectedness": 0.7  # Bias toward memories with many associations
            }
        })
        
        logger.info(f"Memory Node initialized with {len(self.memory_store.get_all_memories())} memories")
    
    def activate(self) -> bool:
        """
        Activate the Memory Node.
        
        Returns:
            True if activation successful, False otherwise
        """
        success = super().activate()
        if success and self.config["decay_enabled"]:
            # Start decay thread if decay is enabled
            self._start_decay_thread()
        return success
    
    def deactivate(self) -> bool:
        """
        Deactivate the Memory Node.
        
        Returns:
            True if deactivation successful, False otherwise
        """
        # Stop decay thread if running
        if self.decay_thread and self.decay_thread.is_alive():
            self.stop_decay.set()
            self.decay_thread.join(timeout=2.0)
        
        return super().deactivate()
    
    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a message received from another node.
        
        Args:
            message: The message to process
            
        Returns:
            Optional response message
        """
        # First call the parent method to handle basic processing
        super().process_message(message)
        
        message_type = message.get("type", "")
        content = message.get("content", {})
        sender = message.get("sender")
        
        response = {
            "type": "response",
            "recipient": sender,
            "sender": self.node_id,
            "timestamp": time.time(),
            "message_id": str(uuid.uuid4()),
            "in_response_to": message.get("message_id"),
            "content": {}
        }
        
        # Process different message types
        if message_type == "store_memory":
            # Store a new memory
            memory_id = self.add_memory(
                content=content.get("content"),
                memory_type=content.get("memory_type"),
                importance=content.get("importance", 0.5),
                source=content.get("source", sender),
                metadata=content.get("metadata", {})
            )
            response["content"] = {"success": True, "memory_id": memory_id}
        
        elif message_type == "retrieve_memory":
            # Retrieve a specific memory
            memory_id = content.get("memory_id")
            memory = self.get_memory(memory_id)
            if memory:
                response["content"] = {"success": True, "memory": memory}
            else:
                response["content"] = {"success": False, "error": "Memory not found"}
        
        elif message_type == "search_memories":
            # Search for memories
            criteria = content.get("criteria", {})
            limit = content.get("limit", 100)
            memories = self.search_memories(criteria, limit)
            response["content"] = {
                "success": True, 
                "count": len(memories),
                "memories": memories
            }
        
        elif message_type == "update_memory":
            # Update an existing memory
            memory_id = content.get("memory_id")
            updates = content.get("updates", {})
            success = self.update_memory(memory_id, **updates)
            response["content"] = {"success": success}
            
            if not success:
                response["content"]["error"] = "Memory update failed"
        
        elif message_type == "delete_memory":
            # Delete a memory
            memory_id = content.get("memory_id")
            success = self.delete_memory(memory_id)
            response["content"] = {"success": success}
            
            if not success:
                response["content"]["error"] = "Memory deletion failed"
        
        elif message_type == "associate_memories":
            # Create an association between memories
            memory_id1 = content.get("memory_id1")
            memory_id2 = content.get("memory_id2")
            association_type = content.get("association_type", "related")
            strength = content.get("strength", 0.5)
            metadata = content.get("metadata", {})
            
            success = self.associate_memories(
                memory_id1, memory_id2, association_type, strength, metadata
            )
            response["content"] = {"success": success}
            
            if not success:
                response["content"]["error"] = "Association failed"
        
        elif message_type == "get_statistics":
            # Get memory store statistics
            stats = self.get_statistics()
            response["content"] = {"success": True, "statistics": stats}
        
        elif message_type == "force_decay":
            # Force memory decay
            self.process_decay()
            response["content"] = {"success": True}
        
        else:
            # Unknown message type
            logger.warning(f"Unknown message type: {message_type}")
            response["content"] = {
                "success": False, 
                "error": f"Unknown message type: {message_type}"
            }
        
        return response
    
    def add_memory(self, content: Any, memory_type: str = "fact", 
                  importance: float = 0.5, source: str = "system",
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new memory to the memory store.
        
        Args:
            content: The memory content
            memory_type: Type of memory
            importance: Importance level (0.0 to 1.0)
            source: Source of the memory
            metadata: Additional metadata
            
        Returns:
            The memory ID
        """
        # Validate memory type
        if memory_type not in self.config["memory_types"]:
            logger.warning(f"Unknown memory type: {memory_type}, using 'fact' instead")
            memory_type = "fact"
        
        # Create memory object
        memory = {
            "id": str(uuid.uuid4()),
            "content": content,
            "memory_type": memory_type,
            "importance": max(0.0, min(1.0, importance)),  # Clamp to 0.0-1.0
            "source": source,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "metadata": metadata or {}
        }
        
        # Store the memory
        memory_id = self.memory_store.store(memory)
        
        # Update node statistics
        stats = self.get_statistics()
        self.node_state.update({
            "memory_count": stats["total_memories"],
            "memory_types": list(stats["memory_types"].keys())
        })
        
        logger.debug(f"Added memory {memory_id} of type {memory_type}")
        return memory_id
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            The memory dict, or None if not found
        """
        memory = self.memory_store.retrieve(memory_id)
        
        if memory and self.config["reinforce_on_access"]:
            # Reinforce the memory by increasing its importance
            current_importance = memory.get("importance", 0.5)
            reinforcement = self.config["reinforcement_strength"]
            new_importance = min(1.0, current_importance + reinforcement)
            
            if new_importance > current_importance:
                self.update_memory(memory_id, importance=new_importance)
        
        return memory
    
    def search_memories(self, criteria: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for memories based on criteria.
        
        Args:
            criteria: Search criteria
            limit: Maximum number of results
            
        Returns:
            List of matching memories
        """
        return self.memory_store.search(criteria, limit)
    
    def update_memory(self, memory_id: str, **updates) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: The memory ID
            **updates: Updates to apply
            
        Returns:
            True if updated, False if not found
        """
        memory = self.memory_store.retrieve(memory_id)
        if not memory:
            return False
        
        # Apply updates
        for key, value in updates.items():
            if key in ["content", "memory_type", "importance", "source", "metadata"]:
                memory[key] = value
        
        # Clamp importance to 0.0-1.0 if provided
        if "importance" in updates:
            memory["importance"] = max(0.0, min(1.0, memory["importance"]))
        
        # Update last accessed time
        memory["last_accessed"] = time.time()
        
        # Store the updated memory
        self.memory_store.store(memory)
        
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            True if deleted, False if not found
        """
        return self.memory_store.delete(memory_id)
    
    def associate_memories(self, memory_id1: str, memory_id2: str, 
                          association_type: str = "related", 
                          strength: float = 0.5,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create an association between two memories.
        
        Args:
            memory_id1: First memory ID
            memory_id2: Second memory ID
            association_type: Type of association
            strength: Association strength (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            True if association created, False otherwise
        """
        return self.memory_store.associate(
            memory_id1, memory_id2, association_type, strength, metadata
        )
    
    def get_associations(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Get all associations for a memory.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            List of association dicts
        """
        return self.memory_store.get_associations(memory_id)
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Get all memories in the store.
        
        Returns:
            List of all memories
        """
        return self.memory_store.get_all_memories()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.
        
        Returns:
            Dictionary with statistics
        """
        stats = self.memory_store.get_statistics()
        stats.update({
            "decay_rate": self.config["decay_rate"],
            "decay_enabled": self.config["decay_enabled"],
            "last_decay_time": self.last_decay_time
        })
        return stats
    
    def process_decay(self) -> None:
        """
        Process memory decay based on configured decay rate.
        """
        if not self.config["decay_enabled"]:
            return
        
        current_time = time.time()
        self.last_decay_time = current_time
        
        # Get all memories
        memories = self.get_all_memories()
        min_importance = self.config["minimum_importance"]
        decay_rate = self.config["decay_rate"]
        
        for memory in memories:
            memory_id = memory["id"]
            importance = memory.get("importance", 0.5)
            last_accessed = memory.get("last_accessed", memory.get("created_at", current_time))
            
            # Calculate time since last access
            time_since_access = current_time - last_accessed
            
            # Skip newly created or recently accessed memories
            if time_since_access < 60:  # 1 minute
                continue
            
            # Calculate decay based on time and importance
            # Higher importance memories decay more slowly
            time_factor = time_since_access / self.config["decay_interval"]
            importance_factor = 1.0 - importance  # Higher importance = slower decay
            
            # Calculate new importance
            decay_amount = decay_rate * time_factor * importance_factor
            new_importance = max(0.0, importance - decay_amount)
            
            # Update if importance changed significantly
            if abs(new_importance - importance) > 0.01:
                # Delete if below minimum importance
                if new_importance < min_importance:
                    self.delete_memory(memory_id)
                    logger.debug(f"Deleted memory {memory_id} due to decay (importance: {new_importance:.2f})")
                else:
                    self.update_memory(memory_id, importance=new_importance)
                    logger.debug(f"Decayed memory {memory_id} from {importance:.2f} to {new_importance:.2f}")
    
    def _start_decay_thread(self) -> None:
        """Start the background decay thread"""
        self.stop_decay.clear()
        self.decay_thread = threading.Thread(
            target=self._decay_loop,
            daemon=True,
            name="MemoryDecay"
        )
        self.decay_thread.start()
    
    def _decay_loop(self) -> None:
        """Background thread for periodic memory decay"""
        # Smaller time interval for more responsive stopping
        check_interval = min(300, self.config["decay_interval"] / 12)  # Check at most every 5 minutes
        
        logger.debug(f"Memory decay thread started (interval: {check_interval}s)")
        
        accumulated_time = 0
        while not self.stop_decay.is_set():
            # Sleep for the check interval
            time.sleep(check_interval)
            accumulated_time += check_interval
            
            # Check if it's time to decay
            if accumulated_time >= self.config["decay_interval"]:
                try:
                    self.process_decay()
                    accumulated_time = 0
                except Exception as e:
                    logger.error(f"Error in memory decay process: {str(e)}")
        
        logger.debug("Memory decay thread stopped")
    
    def get_memories_by_type(self, memory_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get memories of a specific type.
        
        Args:
            memory_type: The memory type
            limit: Maximum number of results
            
        Returns:
            List of memories of the specified type
        """
        return self.search_memories({"memory_type": memory_type}, limit)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the Memory Node.
        
        Returns:
            Status dictionary
        """
        status = super().get_status()
        
        # Add memory-specific status information
        memory_stats = self.get_statistics()
        status["memory_status"] = {
            "memory_capability": self.node_state.get("memory_capability", "transient"),
            "store_type": self.node_state.get("store_type", "json"),
            "decay_enabled": self.node_state.get("decay_enabled", False),
            "decay_rate": self.node_state.get("decay_rate", 0.1),
            "stats": memory_stats
        }
        
        return status

def create_memory_node(persistence_file: Optional[str] = None, 
                     store_type: str = "json", 
                     decay_enabled: bool = True) -> MemoryNode:
    """
    Create a Memory Node instance.
    
    Args:
        persistence_file: Path to the persistence file (optional)
        store_type: Type of storage ('json' or 'sqlite')
        decay_enabled: Whether to enable memory decay
        
    Returns:
        The Memory Node instance
    """
    config = {
        "decay_enabled": decay_enabled,
        "decay_rate": 0.1 if decay_enabled else 0.0,
        "decay_interval": 3600,  # 1 hour
        "minimum_importance": 0.1,
        "reinforce_on_access": True,
        "reinforcement_strength": 0.1
    }
    
    return MemoryNode(
        persistence_file=persistence_file,
        store_type=store_type,
        config=config
    )

if __name__ == "__main__":
    # If run directly, initialize and test the Memory Node
    logging.basicConfig(level=logging.INFO)
    
    # Create a test memory node
    test_node = create_memory_node(persistence_file="memory_test.json", decay_enabled=True)
    
    # Activate the node
    test_node.activate()
    
    try:
        # Add some test memories
        memory_id1 = test_node.add_memory(
            content="The sky is blue because of Rayleigh scattering.",
            memory_type="fact",
            importance=0.8,
            source="test"
        )
        
        memory_id2 = test_node.add_memory(
            content="Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences.",
            memory_type="concept",
            importance=0.9,
            source="test"
        )
        
        # Create an association
        test_node.associate_memories(memory_id1, memory_id2, "related", 0.7)
        
        # Search for memories
        results = test_node.search_memories({"memory_type": "fact"})
        print(f"Found {len(results)} fact memories:")
        for mem in results:
            print(f"  - {mem['content'][:50]}... (importance: {mem['importance']:.2f})")
        
        # Wait a bit to test decay
        print("\nWaiting to test memory decay...")
        time.sleep(5)
        
        # Force decay
        test_node.process_decay()
        
        # Check memories after decay
        print("\nAfter decay:")
        for memory_id in [memory_id1, memory_id2]:
            memory = test_node.get_memory(memory_id)
            if memory:
                print(f"  - {memory['content'][:50]}... (importance: {memory['importance']:.2f})")
        
        # Print statistics
        stats = test_node.get_statistics()
        print("\nMemory Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    finally:
        # Deactivate the node
        test_node.deactivate() 