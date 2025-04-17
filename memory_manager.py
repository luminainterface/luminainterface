#!/usr/bin/env python
"""
Memory Manager - Central coordination for memory systems

This module provides a central system for managing and coordinating memory 
operations between different components and AI agents working on the project.
"""

import logging
import json
import os
import datetime
import threading
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Central coordination system for memory sharing and integration.
    
    The MemoryManager acts as a hub where different components can:
    1. Register themselves to participate in memory sharing
    2. Share memories with specific components or broadcast to all
    3. Subscribe to memory events from specific components
    4. Query and search across all available memories
    5. Synchronize memory operations between parallel processes/agents
    """
    
    def __init__(self, log_directory: str = "data/memory_logs"):
        """
        Initialize the memory manager
        
        Args:
            log_directory: Directory for memory operation logs
        """
        self.components = {}  # name -> component instance
        self.subscribers = {}  # source_component -> list of target components
        self.activity_log = []  # Log of memory operations
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Memory sharing statistics
        self.stats = {
            "total_shares": 0,
            "total_broadcasts": 0,
            "total_memories_shared": 0,
            "component_shares": {},
        }
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Memory Manager initialized")
        
    def register_component(self, name: str, component_instance: Any) -> bool:
        """
        Register a component with the memory manager
        
        Args:
            name: Unique name for the component
            component_instance: Instance of the component
            
        Returns:
            Success flag
        """
        with self.lock:
            if name in self.components:
                logger.warning(f"Component '{name}' already registered")
                return False
                
            self.components[name] = component_instance
            self.subscribers[name] = set()
            
            if "component_shares" not in self.stats:
                self.stats["component_shares"] = {}
            self.stats["component_shares"][name] = 0
            
            logger.info(f"Component '{name}' registered with Memory Manager")
            self._log_activity("register", name, None, {})
            
            return True
    
    def unregister_component(self, name: str) -> bool:
        """
        Unregister a component from the memory manager
        
        Args:
            name: Name of the component to unregister
            
        Returns:
            Success flag
        """
        with self.lock:
            if name not in self.components:
                logger.warning(f"Component '{name}' not registered")
                return False
                
            # Remove component
            del self.components[name]
            
            # Remove subscriptions to this component
            if name in self.subscribers:
                del self.subscribers[name]
                
            # Remove component from other components' subscriber lists
            for subscribers in self.subscribers.values():
                if name in subscribers:
                    subscribers.remove(name)
            
            logger.info(f"Component '{name}' unregistered from Memory Manager")
            self._log_activity("unregister", name, None, {})
            
            return True
    
    def subscribe(self, source: str, target: str) -> bool:
        """
        Subscribe target component to receive memories from source component
        
        Args:
            source: Name of the source component 
            target: Name of the target component
            
        Returns:
            Success flag
        """
        with self.lock:
            if source not in self.components:
                logger.warning(f"Source component '{source}' not registered")
                return False
                
            if target not in self.components:
                logger.warning(f"Target component '{target}' not registered") 
                return False
                
            if source not in self.subscribers:
                self.subscribers[source] = set()
                
            self.subscribers[source].add(target)
            
            logger.info(f"Component '{target}' subscribed to memories from '{source}'")
            self._log_activity("subscribe", source, target, {})
            
            return True
    
    def unsubscribe(self, source: str, target: str) -> bool:
        """
        Unsubscribe target component from receiving memories from source component
        
        Args:
            source: Name of the source component
            target: Name of the target component
            
        Returns:
            Success flag
        """
        with self.lock:
            if source not in self.subscribers:
                logger.warning(f"Source component '{source}' has no subscribers")
                return False
                
            if target not in self.subscribers[source]:
                logger.warning(f"Target component '{target}' not subscribed to '{source}'")
                return False
                
            self.subscribers[source].remove(target)
            
            logger.info(f"Component '{target}' unsubscribed from memories of '{source}'")
            self._log_activity("unsubscribe", source, target, {})
            
            return True
    
    def share_memories(self, memories: List[Dict[str, Any]], 
                      source: str, target: str = None) -> int:
        """
        Share memories from source component to target component(s)
        
        Args:
            memories: List of memory items to share
            source: Name of the source component
            target: Name of the target component (None = broadcast to all subscribers)
            
        Returns:
            Number of components memories were shared with
        """
        with self.lock:
            if source not in self.components:
                logger.warning(f"Source component '{source}' not registered")
                return 0
                
            # Track for stats
            self.stats["total_memories_shared"] += len(memories)
            self.stats["component_shares"][source] = (
                self.stats["component_shares"].get(source, 0) + len(memories)
            )
            
            # Target specific component
            if target is not None:
                if target not in self.components:
                    logger.warning(f"Target component '{target}' not registered")
                    return 0
                    
                try:
                    # Get the component instance
                    component = self.components[target]
                    
                    # Call receive_shared_memories method if it exists
                    if hasattr(component, "receive_shared_memories"):
                        component.receive_shared_memories(memories, source)
                    else:
                        logger.warning(f"Component '{target}' does not implement receive_shared_memories")
                        
                    # Log the sharing activity
                    self._log_activity("share", source, target, {"count": len(memories)})
                    self.stats["total_shares"] += 1
                    
                    return 1
                    
                except Exception as e:
                    logger.error(f"Error sharing memories from '{source}' to '{target}': {str(e)}")
                    return 0
            
            # Broadcast to all subscribers
            else:
                count = 0
                # Get subscribers for this source
                subscribers = self.subscribers.get(source, set())
                
                for target in subscribers:
                    try:
                        # Get the component instance
                        component = self.components[target]
                        
                        # Call receive_shared_memories method if it exists
                        if hasattr(component, "receive_shared_memories"):
                            component.receive_shared_memories(memories, source)
                            count += 1
                        else:
                            logger.warning(f"Component '{target}' does not implement receive_shared_memories")
                            
                    except Exception as e:
                        logger.error(f"Error broadcasting memories from '{source}' to '{target}': {str(e)}")
                
                # Log the broadcasting activity
                self._log_activity("broadcast", source, None, {
                    "count": len(memories),
                    "recipients": list(subscribers)
                })
                
                self.stats["total_broadcasts"] += 1
                
                return count
    
    def get_component_names(self) -> List[str]:
        """
        Get names of all registered components
        
        Returns:
            List of component names
        """
        with self.lock:
            return list(self.components.keys())
    
    def get_component(self, name: str) -> Any:
        """
        Get a component instance by name
        
        Args:
            name: Name of the component
            
        Returns:
            Component instance or None if not found
        """
        with self.lock:
            return self.components.get(name)
    
    def get_subscribers(self, source: str) -> Set[str]:
        """
        Get names of components subscribed to a source component
        
        Args:
            source: Name of the source component
            
        Returns:
            Set of subscriber component names
        """
        with self.lock:
            return self.subscribers.get(source, set()).copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory sharing statistics
        
        Returns:
            Dictionary with memory statistics
        """
        with self.lock:
            return self.stats.copy()
    
    def _log_activity(self, action: str, source: str, target: str, details: Dict[str, Any]):
        """
        Log memory activity for auditing and debugging
        
        Args:
            action: Type of action (register, share, etc.)
            source: Source component
            target: Target component (may be None)
            details: Additional action details
        """
        timestamp = datetime.datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "action": action,
            "source": source,
            "target": target,
            "details": details
        }
        
        # Add to in-memory log
        self.activity_log.append(entry)
        
        # Limit in-memory log size
        if len(self.activity_log) > 1000:
            self.activity_log = self.activity_log[-1000:]
        
        # Write to log file
        try:
            log_file = self.log_directory / f"memory_log_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing to memory log: {str(e)}")
    
    def get_activity_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent memory activity log
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of log entries
        """
        with self.lock:
            return self.activity_log[-limit:]


# Create a singleton instance
memory_manager = MemoryManager()

# Main function for standalone testing
if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test memory manager
    manager = MemoryManager()
    
    # Display registered components
    components = manager.get_component_names()
    print(f"Registered components: {components}")
    
    # Create a test memory
    test_memory = {
        "user": "Test user input",
        "system": "Test system response",
        "emotion": "curiosity",
        "glyph": "✧",
        "metadata": {
            "test": True,
            "timestamp": datetime.datetime.now().isoformat()
        }
    }
    
    # Store in shared pool
    memory_id = manager.share_memories([test_memory], "test_component")
    print(f"Stored memory with ID: {memory_id}")
    
    # Retrieve from shared pool
    retrieved = manager.get_component("test_component")
    print(f"Retrieved memory: {retrieved}")
    
    # Create global snapshot
    snapshot_path = manager.create_global_memory_snapshot("test_snapshot")
    print(f"Created global snapshot at: {snapshot_path}")
