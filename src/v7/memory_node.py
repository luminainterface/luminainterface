#!/usr/bin/env python
"""
Memory Consciousness Node for V7

This module provides a specialized consciousness node for persistent memory
capabilities within the V7 Node Consciousness system, supporting the storage
and retrieval of various types of memories with temporal awareness.
"""

import logging
import threading
import time
import os
import json
import sqlite3
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from src.v7.node_consciousness_manager import ConsciousnessNode, NodeState

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Represents a memory item in the memory consciousness node."""
    id: str
    content: Any
    memory_type: str
    created_at: float
    last_accessed: float
    tags: List[str] = field(default_factory=list)
    strength: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    source_node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage."""
        return {
            'id': self.id,
            'content': self.content,
            'memory_type': self.memory_type,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'tags': self.tags,
            'strength': self.strength,
            'context': self.context,
            'source_node_id': self.source_node_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary."""
        return cls(
            id=data['id'],
            content=data['content'],
            memory_type=data['memory_type'],
            created_at=data['created_at'],
            last_accessed=data['last_accessed'],
            tags=data.get('tags', []),
            strength=data.get('strength', 1.0),
            context=data.get('context', {}),
            source_node_id=data.get('source_node_id'),
            metadata=data.get('metadata', {})
        )


class MemoryStore:
    """Base class for memory storage implementations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the memory store."""
        self.config = config or {}
    
    def store(self, memory: Memory) -> bool:
        """
        Store a memory.
        
        Args:
            memory: The memory to store
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement store()")
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: The ID of the memory to retrieve
            
        Returns:
            The memory if found, None otherwise
        """
        raise NotImplementedError("Subclasses must implement retrieve()")
    
    def search(self, query: Dict[str, Any], limit: int = 10) -> List[Memory]:
        """
        Search for memories matching the query.
        
        Args:
            query: Query parameters
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        raise NotImplementedError("Subclasses must implement search()")
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a memory.
        
        Args:
            memory_id: The ID of the memory to update
            updates: The fields to update
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement delete()")
    
    def list_all(self, memory_type: Optional[str] = None, limit: int = 100) -> List[Memory]:
        """
        List all memories, optionally filtered by type.
        
        Args:
            memory_type: Optional type to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of memories
        """
        raise NotImplementedError("Subclasses must implement list_all()")
    
    def clear(self) -> bool:
        """
        Clear all memories.
        
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement clear()")


class JSONMemoryStore(MemoryStore):
    """Memory store implementation using JSON files."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the JSON memory store."""
        super().__init__(config)
        self.memory_path = Path(self.config.get('memory_path', './data/memories'))
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.memories: Dict[str, Memory] = {}
        self.memory_file = self.memory_path / 'memories.json'
        self._load_memories()
        self._lock = threading.RLock()
    
    def _load_memories(self) -> None:
        """Load memories from file."""
        if not self.memory_file.exists():
            self.memories = {}
            return
        
        try:
            with open(self.memory_file, 'r') as f:
                memory_dicts = json.load(f)
                self.memories = {
                    mem_id: Memory.from_dict(mem_dict)
                    for mem_id, mem_dict in memory_dicts.items()
                }
            logger.info(f"Loaded {len(self.memories)} memories from {self.memory_file}")
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
            self.memories = {}
    
    def _save_memories(self) -> bool:
        """Save memories to file."""
        try:
            memory_dicts = {
                mem_id: memory.to_dict()
                for mem_id, memory in self.memories.items()
            }
            with open(self.memory_file, 'w') as f:
                json.dump(memory_dicts, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving memories: {str(e)}")
            return False
    
    def store(self, memory: Memory) -> bool:
        """Store a memory."""
        with self._lock:
            self.memories[memory.id] = memory
            return self._save_memories()
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        with self._lock:
            memory = self.memories.get(memory_id)
            if memory:
                # Update last accessed time
                memory.last_accessed = time.time()
                self._save_memories()
            return memory
    
    def search(self, query: Dict[str, Any], limit: int = 10) -> List[Memory]:
        """Search for memories matching the query."""
        with self._lock:
            results = []
            
            for memory in self.memories.values():
                # Match all query parameters
                match = True
                for key, value in query.items():
                    if key == 'tags' and isinstance(value, list):
                        # Check if any of the specified tags are in the memory's tags
                        if not any(tag in memory.tags for tag in value):
                            match = False
                            break
                    elif key == 'memory_type' and isinstance(value, list):
                        # Check if memory type is in the list
                        if memory.memory_type not in value:
                            match = False
                            break
                    elif key == 'before' and isinstance(value, (int, float)):
                        # Check if memory was created before the specified time
                        if memory.created_at >= value:
                            match = False
                            break
                    elif key == 'after' and isinstance(value, (int, float)):
                        # Check if memory was created after the specified time
                        if memory.created_at <= value:
                            match = False
                            break
                    elif key == 'min_strength' and isinstance(value, (int, float)):
                        # Check if memory strength is at least the specified value
                        if memory.strength < value:
                            match = False
                            break
                    elif key == 'source_node_id':
                        # Check if memory came from the specified node
                        if memory.source_node_id != value:
                            match = False
                            break
                    elif key == 'content_contains' and isinstance(value, str):
                        # Check if content contains the specified string
                        if isinstance(memory.content, str) and value.lower() not in memory.content.lower():
                            match = False
                            break
                
                if match:
                    results.append(memory)
                    if len(results) >= limit:
                        break
            
            # Update last accessed time for all retrieved memories
            for memory in results:
                memory.last_accessed = time.time()
            self._save_memories()
            
            return results
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory."""
        with self._lock:
            memory = self.memories.get(memory_id)
            if not memory:
                return False
            
            # Apply updates
            for key, value in updates.items():
                if key in ['content', 'tags', 'strength', 'context', 'metadata']:
                    setattr(memory, key, value)
            
            # Update last accessed time
            memory.last_accessed = time.time()
            
            return self._save_memories()
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        with self._lock:
            if memory_id in self.memories:
                del self.memories[memory_id]
                return self._save_memories()
            return False
    
    def list_all(self, memory_type: Optional[str] = None, limit: int = 100) -> List[Memory]:
        """List all memories, optionally filtered by type."""
        with self._lock:
            if memory_type:
                results = [m for m in self.memories.values() if m.memory_type == memory_type]
            else:
                results = list(self.memories.values())
            
            # Sort by creation time, newest first
            results.sort(key=lambda m: m.created_at, reverse=True)
            
            return results[:limit]
    
    def clear(self) -> bool:
        """Clear all memories."""
        with self._lock:
            self.memories = {}
            return self._save_memories()


class SQLiteMemoryStore(MemoryStore):
    """Memory store implementation using SQLite."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the SQLite memory store."""
        super().__init__(config)
        self.memory_path = Path(self.config.get('memory_path', './data/memories'))
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.db_file = self.memory_path / 'memories.db'
        self._lock = threading.RLock()
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize the database schema."""
        try:
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            
            # Create memories table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                tags TEXT,
                strength REAL NOT NULL,
                context TEXT,
                source_node_id TEXT,
                metadata TEXT
            )
            ''')
            
            # Create index on memory_type for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories (memory_type)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Initialized SQLite memory store at {self.db_file}")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {str(e)}")
    
    def _dict_to_memory(self, row: tuple) -> Memory:
        """Convert a database row to a Memory object."""
        id, content, memory_type, created_at, last_accessed, tags_str, strength, context_str, source_node_id, metadata_str = row
        
        # Parse JSON fields
        tags = json.loads(tags_str) if tags_str else []
        context = json.loads(context_str) if context_str else {}
        metadata = json.loads(metadata_str) if metadata_str else {}
        
        # Parse content if it's a JSON string
        try:
            content_obj = json.loads(content)
            content = content_obj
        except:
            # Keep as string if not valid JSON
            pass
        
        return Memory(
            id=id,
            content=content,
            memory_type=memory_type,
            created_at=created_at,
            last_accessed=last_accessed,
            tags=tags,
            strength=strength,
            context=context,
            source_node_id=source_node_id,
            metadata=metadata
        )
    
    def store(self, memory: Memory) -> bool:
        """Store a memory."""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_file))
                cursor = conn.cursor()
                
                # Convert complex objects to JSON strings
                content = json.dumps(memory.content) if not isinstance(memory.content, str) else memory.content
                tags_str = json.dumps(memory.tags)
                context_str = json.dumps(memory.context)
                metadata_str = json.dumps(memory.metadata)
                
                cursor.execute('''
                INSERT OR REPLACE INTO memories 
                (id, content, memory_type, created_at, last_accessed, tags, strength, context, source_node_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory.id,
                    content,
                    memory.memory_type,
                    memory.created_at,
                    memory.last_accessed,
                    tags_str,
                    memory.strength,
                    context_str,
                    memory.source_node_id,
                    metadata_str
                ))
                
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                logger.error(f"Error storing memory: {str(e)}")
                return False
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_file))
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
                row = cursor.fetchone()
                
                if not row:
                    conn.close()
                    return None
                
                memory = self._dict_to_memory(row)
                
                # Update last accessed time
                memory.last_accessed = time.time()
                cursor.execute('UPDATE memories SET last_accessed = ? WHERE id = ?', 
                              (memory.last_accessed, memory_id))
                
                conn.commit()
                conn.close()
                
                return memory
            except Exception as e:
                logger.error(f"Error retrieving memory: {str(e)}")
                return None
    
    def search(self, query: Dict[str, Any], limit: int = 10) -> List[Memory]:
        """Search for memories matching the query."""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_file))
                cursor = conn.cursor()
                
                # Build query parts
                query_parts = []
                query_params = []
                
                if 'memory_type' in query:
                    if isinstance(query['memory_type'], list):
                        placeholders = ', '.join(['?' for _ in query['memory_type']])
                        query_parts.append(f'memory_type IN ({placeholders})')
                        query_params.extend(query['memory_type'])
                    else:
                        query_parts.append('memory_type = ?')
                        query_params.append(query['memory_type'])
                
                if 'source_node_id' in query:
                    query_parts.append('source_node_id = ?')
                    query_params.append(query['source_node_id'])
                
                if 'before' in query:
                    query_parts.append('created_at < ?')
                    query_params.append(query['before'])
                
                if 'after' in query:
                    query_parts.append('created_at > ?')
                    query_params.append(query['after'])
                
                if 'min_strength' in query:
                    query_parts.append('strength >= ?')
                    query_params.append(query['min_strength'])
                
                # Build the final query
                sql = 'SELECT * FROM memories'
                if query_parts:
                    sql += ' WHERE ' + ' AND '.join(query_parts)
                
                # Add order and limit
                sql += ' ORDER BY created_at DESC LIMIT ?'
                query_params.append(limit)
                
                cursor.execute(sql, query_params)
                rows = cursor.fetchall()
                
                memories = [self._dict_to_memory(row) for row in rows]
                
                # Filter memories based on tags and content_contains if needed
                # These are harder to do in SQL so we filter in Python
                filtered_memories = []
                for memory in memories:
                    include = True
                    
                    # Filter by tags
                    if 'tags' in query and isinstance(query['tags'], list):
                        if not any(tag in memory.tags for tag in query['tags']):
                            include = False
                    
                    # Filter by content contains
                    if include and 'content_contains' in query:
                        search_text = query['content_contains'].lower()
                        content_text = str(memory.content).lower()
                        if search_text not in content_text:
                            include = False
                    
                    if include:
                        filtered_memories.append(memory)
                
                # Update last accessed time for all retrieved memories
                for memory in filtered_memories:
                    memory.last_accessed = time.time()
                    cursor.execute('UPDATE memories SET last_accessed = ? WHERE id = ?', 
                                  (memory.last_accessed, memory.id))
                
                conn.commit()
                conn.close()
                
                return filtered_memories[:limit]
                
            except Exception as e:
                logger.error(f"Error searching memories: {str(e)}")
                return []
    
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory."""
        with self._lock:
            try:
                # First retrieve the memory
                memory = self.retrieve(memory_id)
                if not memory:
                    return False
                
                # Apply updates
                for key, value in updates.items():
                    if key in ['content', 'tags', 'strength', 'context', 'metadata']:
                        setattr(memory, key, value)
                
                # Store the updated memory
                return self.store(memory)
                
            except Exception as e:
                logger.error(f"Error updating memory: {str(e)}")
                return False
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_file))
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
                
                conn.commit()
                conn.close()
                
                return True
            except Exception as e:
                logger.error(f"Error deleting memory: {str(e)}")
                return False
    
    def list_all(self, memory_type: Optional[str] = None, limit: int = 100) -> List[Memory]:
        """List all memories, optionally filtered by type."""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_file))
                cursor = conn.cursor()
                
                if memory_type:
                    cursor.execute('SELECT * FROM memories WHERE memory_type = ? ORDER BY created_at DESC LIMIT ?', 
                                 (memory_type, limit))
                else:
                    cursor.execute('SELECT * FROM memories ORDER BY created_at DESC LIMIT ?', (limit,))
                
                rows = cursor.fetchall()
                
                memories = [self._dict_to_memory(row) for row in rows]
                
                conn.close()
                
                return memories
                
            except Exception as e:
                logger.error(f"Error listing memories: {str(e)}")
                return []
    
    def clear(self) -> bool:
        """Clear all memories."""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_file))
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM memories')
                
                conn.commit()
                conn.close()
                
                return True
            except Exception as e:
                logger.error(f"Error clearing memories: {str(e)}")
                return False


class MemoryConsciousnessNode(ConsciousnessNode):
    """
    Specialized consciousness node for persistent memory capabilities.
    
    This node provides storage and retrieval of memories with temporal
    awareness, supporting the memory needs of the V7 Node Consciousness system.
    """
    
    def __init__(self, node_id: Optional[str] = None, 
                 name: str = 'Memory Node', 
                 config: Optional[Dict[str, Any]] = None,
                 node_type: str = 'memory'):
        """Initialize the Memory Consciousness Node."""
        super().__init__(node_id=node_id, name=name, node_type=node_type)
        
        self.config = config or {}
        
        # Initialize with default settings
        self.memory_path = self.config.get('memory_path', './data/memories')
        self.memory_persistence = self.config.get('memory_persistence', True)
        self.store_type = self.config.get('store_type', 'sqlite')  # 'sqlite' or 'json'
        self.decay_enabled = self.config.get('decay_enabled', True)
        self.decay_rate = self.config.get('decay_rate', 0.05)  # 5% per day
        self.decay_interval = self.config.get('decay_interval', 86400)  # 24 hours
        
        # Components and state
        self.memory_store = None
        self.decay_thread = None
        self._running = False
        self.processing_lock = threading.RLock()
        
        # Stats tracking
        self.stats = {
            'memories_stored': 0,
            'memories_retrieved': 0,
            'memories_updated': 0,
            'memories_deleted': 0,
            'searches_performed': 0
        }
        
        # Specialized attributes for memory node
        self.attributes.update({
            'memory_version': '0.1.0',
            'memory_capability': 'persistent',
            'decay_enabled': self.decay_enabled,
            'temporal_awareness': True
        })
        
        logger.info(f"Memory Consciousness Node initialized: {self.name}")
    
    def _initialize(self) -> None:
        """Initialize the memory node components."""
        try:
            # Create memory directory if it doesn't exist
            if self.memory_persistence:
                os.makedirs(self.memory_path, exist_ok=True)
            
            # Initialize memory store
            store_config = {
                'memory_path': self.memory_path,
                'persistence': self.memory_persistence
            }
            
            if self.store_type == 'sqlite':
                self.memory_store = SQLiteMemoryStore(config=store_config)
                self.attributes['memory_capability'] = 'advanced'
            else:
                self.memory_store = JSONMemoryStore(config=store_config)
                self.attributes['memory_capability'] = 'basic'
            
            # Start the decay process if enabled
            if self.decay_enabled:
                self._running = True
                self.decay_thread = threading.Thread(
                    target=self._memory_decay_process,
                    name="MemoryDecayProcess",
                    daemon=True
                )
                self.decay_thread.start()
            
            logger.info(f"Memory node {self.name} fully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing memory node {self.name}: {str(e)}")
            raise
    
    def _cleanup(self) -> None:
        """Clean up resources used by the memory node."""
        # Stop the decay thread
        self._running = False
        if self.decay_thread and self.decay_thread.is_alive():
            self.decay_thread.join(timeout=2.0)
        
        # Clear resources
        self.memory_store = None
        
        logger.info(f"Memory node {self.name} cleaned up")
    
    def _process_impl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of memory processing logic."""
        start_time = time.time()
        
        # Process different input types
        if 'store' in data:
            return self._store_memory(data['store'], data.get('metadata', {}))
        elif 'retrieve' in data:
            return self._retrieve_memory(data['retrieve'], data.get('params', {}))
        elif 'search' in data:
            return self._search_memories(data['search'], data.get('params', {}))
        elif 'update' in data:
            return self._update_memory(data['update'], data.get('updates', {}))
        elif 'delete' in data:
            return self._delete_memory(data['delete'])
        elif 'list' in data:
            return self._list_memories(data.get('list', {}), data.get('params', {}))
        else:
            logger.warning(f"Unsupported data format in memory node {self.name}")
            return {
                'success': False,
                'error': 'Unsupported data format',
                'expected': ['store', 'retrieve', 'search', 'update', 'delete', 'list'],
                'received': list(data.keys())
            }
    
    def _memory_decay_process(self) -> None:
        """Background thread for memory decay processing."""
        logger.info(f"Starting memory decay process for node {self.name}")
        
        while self._running:
            try:
                # Sleep for a while (check _running periodically)
                for _ in range(int(self.decay_interval / 10)):
                    if not self._running:
                        break
                    time.sleep(10)
                
                if not self._running:
                    break
                
                # Process memory decay
                self._apply_memory_decay()
                
            except Exception as e:
                logger.error(f"Error in memory decay process for node {self.name}: {str(e)}")
                time.sleep(60)  # Wait a bit longer after an error
    
    def _apply_memory_decay(self) -> None:
        """Apply decay to memories based on their age and access frequency."""
        if not self.memory_store:
            return
        
        logger.debug(f"Applying memory decay for node {self.name}")
        
        try:
            # Get all memories
            memories = self.memory_store.list_all(limit=1000)
            
            # Current time
            current_time = time.time()
            
            # Update strengths based on decay
            updated_count = 0
            
            for memory in memories:
                # Calculate time since last access in days
                days_since_access = (current_time - memory.last_accessed) / 86400
                
                # Calculate decay factor
                decay_factor = 1.0 - (self.decay_rate * days_since_access)
                decay_factor = max(0.1, decay_factor)  # Don't let it go below 0.1
                
                # Calculate new strength
                new_strength = memory.strength * decay_factor
                
                # Update if strength changed significantly
                if abs(new_strength - memory.strength) > 0.01:
                    memory.strength = new_strength
                    self.memory_store.update(memory.id, {'strength': new_strength})
                    updated_count += 1
            
            if updated_count > 0:
                logger.debug(f"Applied decay to {updated_count} memories")
                
        except Exception as e:
            logger.error(f"Error applying memory decay: {str(e)}")
    
    def _generate_memory_id(self, content: Any, memory_type: str) -> str:
        """Generate a unique ID for a memory."""
        timestamp = int(time.time())
        content_hash = hash(str(content))
        return f"mem_{memory_type}_{timestamp}_{content_hash % 10000}"
    
    def _store_memory(self, data: Dict[str, Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store a new memory."""
        with self.processing_lock:
            try:
                # Extract memory data
                content = data.get('content')
                if content is None:
                    return {'success': False, 'error': 'Content is required'}
                
                memory_type = data.get('memory_type', 'generic')
                tags = data.get('tags', [])
                strength = data.get('strength', 1.0)
                context = data.get('context', {})
                source_node_id = data.get('source_node_id')
                
                # Generate ID if not provided
                memory_id = data.get('id', self._generate_memory_id(content, memory_type))
                
                # Create memory object
                memory = Memory(
                    id=memory_id,
                    content=content,
                    memory_type=memory_type,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    tags=tags,
                    strength=strength,
                    context=context,
                    source_node_id=source_node_id,
                    metadata=metadata or {}
                )
                
                # Store the memory
                success = self.memory_store.store(memory)
                
                if success:
                    self.stats['memories_stored'] += 1
                
                return {
                    'success': success,
                    'memory_id': memory_id if success else None,
                    'created_at': memory.created_at if success else None
                }
                
            except Exception as e:
                logger.error(f"Error storing memory in node {self.name}: {str(e)}")
                return {'success': False, 'error': str(e)}
    
    def _retrieve_memory(self, memory_id: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve a memory by ID."""
        with self.processing_lock:
            try:
                memory = self.memory_store.retrieve(memory_id)
                
                if not memory:
                    return {'success': False, 'error': f"Memory not found: {memory_id}"}
                
                self.stats['memories_retrieved'] += 1
                
                return {
                    'success': True,
                    'memory': memory.to_dict()
                }
                
            except Exception as e:
                logger.error(f"Error retrieving memory in node {self.name}: {str(e)}")
                return {'success': False, 'error': str(e)}
    
    def _search_memories(self, query: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for memories matching the query."""
        with self.processing_lock:
            try:
                limit = params.get('limit', 10) if params else 10
                memories = self.memory_store.search(query, limit=limit)
                
                self.stats['searches_performed'] += 1
                
                return {
                    'success': True,
                    'memories': [m.to_dict() for m in memories],
                    'count': len(memories),
                    'query': query
                }
                
            except Exception as e:
                logger.error(f"Error searching memories in node {self.name}: {str(e)}")
                return {'success': False, 'error': str(e)}
    
    def _update_memory(self, memory_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a memory."""
        with self.processing_lock:
            try:
                success = self.memory_store.update(memory_id, updates)
                
                if success:
                    self.stats['memories_updated'] += 1
                
                return {
                    'success': success,
                    'memory_id': memory_id,
                    'updates': updates
                }
                
            except Exception as e:
                logger.error(f"Error updating memory in node {self.name}: {str(e)}")
                return {'success': False, 'error': str(e)}
    
    def _delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory."""
        with self.processing_lock:
            try:
                success = self.memory_store.delete(memory_id)
                
                if success:
                    self.stats['memories_deleted'] += 1
                
                return {
                    'success': success,
                    'memory_id': memory_id
                }
                
            except Exception as e:
                logger.error(f"Error deleting memory in node {self.name}: {str(e)}")
                return {'success': False, 'error': str(e)}
    
    def _list_memories(self, query: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """List memories, optionally filtered by type."""
        with self.processing_lock:
            try:
                memory_type = query.get('memory_type')
                limit = params.get('limit', 100) if params else 100
                
                memories = self.memory_store.list_all(memory_type=memory_type, limit=limit)
                
                return {
                    'success': True,
                    'memories': [m.to_dict() for m in memories],
                    'count': len(memories),
                    'memory_type': memory_type
                }
                
            except Exception as e:
                logger.error(f"Error listing memories in node {self.name}: {str(e)}")
                return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of this node, with memory-specific additions."""
        base_status = super().get_status()
        
        # Add memory-specific status information
        with self.processing_lock:
            memory_status = {
                'memory_capability': self.attributes.get('memory_capability', 'basic'),
                'store_type': self.store_type,
                'decay_enabled': self.decay_enabled,
                'stats': self.stats
            }
        
        # Update attributes dictionary
        self.attributes.update({
            'memory_stats': {
                'stored': self.stats['memories_stored'],
                'retrieved': self.stats['memories_retrieved']
            }
        })
        
        # Merge with base status
        base_status.update({
            'memory_status': memory_status
        })
        
        return base_status


# Factory function for easy creation
def create_memory_node(config: Optional[Dict[str, Any]] = None) -> MemoryConsciousnessNode:
    """
    Create and initialize a memory consciousness node.
    
    Args:
        config: Configuration options for the node
        
    Returns:
        An initialized MemoryConsciousnessNode
    """
    node = MemoryConsciousnessNode(config=config)
    node.activate()
    return node 