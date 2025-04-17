"""
Echo Spiral Memory System for Lumina Neural Network Project

Implements a hyperdimensional memory system that enables recursive thought patterns,
multidimensional associations, and temporal awareness for the consciousness components.
"""

import time
import threading
import uuid
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from collections import defaultdict
import heapq

# Import utility modules
from src.utils.logging_config import get_logger
from src.utils.message_schema import MessageType, create_message, validate_message
from src.utils.performance_metrics import measure_time, record_metric, get_system_metrics

logger = get_logger("EchoSpiralMemory")

class MemoryNode:
    """Represents a single node in the Echo Spiral Memory network"""
    
    def __init__(self, content: str, node_type: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.node_type = node_type
        self.metadata = metadata or {}
        self.connections = []
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.access_count = 0
        self.activation_level = 0.0
        self.temporal_markers = []
        self.vector = None  # For embeddings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type,
            "metadata": self.metadata,
            "connections": [conn.to_dict() for conn in self.connections],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "activation_level": self.activation_level,
            "temporal_markers": self.temporal_markers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create a node from dictionary"""
        node = cls(content=data["content"], node_type=data["node_type"], metadata=data["metadata"])
        node.id = data["id"]
        node.created_at = data["created_at"]
        node.updated_at = data["updated_at"]
        node.access_count = data["access_count"]
        node.activation_level = data["activation_level"]
        node.temporal_markers = data["temporal_markers"]
        return node

class MemoryConnection:
    """Represents a connection between two memory nodes"""
    
    def __init__(self, source_id: str, target_id: str, connection_type: str, 
                 strength: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.source_id = source_id
        self.target_id = target_id
        self.connection_type = connection_type
        self.strength = strength
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow().isoformat()
        self.access_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert connection to dictionary"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "connection_type": self.connection_type,
            "strength": self.strength,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryConnection':
        """Create a connection from dictionary"""
        conn = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            connection_type=data["connection_type"],
            strength=data["strength"],
            metadata=data["metadata"]
        )
        conn.id = data["id"]
        conn.created_at = data["created_at"]
        conn.access_count = data["access_count"]
        return conn

class EchoSpiralMemory:
    """
    Echo Spiral Memory System for hyperdimensional thought components
    
    This memory system enables:
    - Recursive thought patterns through spiral connections
    - Hyperdimensional associations across memory nodes
    - Temporal awareness and decay through time-based activation
    - Bidirectional memory synchronization with other components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default configuration
        self.config = {
            "memory_dir": "echo_memory",
            "save_interval": 300,  # seconds
            "max_nodes": 10000,
            "decay_rate": 0.01,
            "activation_threshold": 0.3,
            "temporal_awareness": True,
            "vector_dimensions": 384,
            "enable_embeddings": True,
            "mock_mode": False
        }
        
        # Update with custom settings
        if config:
            self.config.update(config)
        
        # Initialize memory structures
        self.nodes = {}  # id -> MemoryNode
        self.node_by_content = {}  # content hash -> id
        self.connections = {}  # id -> MemoryConnection
        self.node_vectors = {}  # id -> numpy vector
        
        # Indexes for fast retrieval
        self.node_type_index = defaultdict(set)
        self.temporal_index = []  # heap for temporal ordering
        self.activation_index = []  # heap for activation level
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Bidirectional sync
        self.sync_handlers = {}
        
        # Setup memory directory
        self.memory_dir = Path(self.config["memory_dir"])
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Start background threads
        self.running = True
        self.save_thread = threading.Thread(
            target=self._auto_save_thread,
            daemon=True,
            name="EchoMemorySaveThread"
        )
        self.decay_thread = threading.Thread(
            target=self._activation_decay_thread,
            daemon=True,
            name="EchoMemoryDecayThread"
        )
        
        logger.info(f"Echo Spiral Memory initialized with config: {self.config}")
        
        # Start threads
        self.save_thread.start()
        self.decay_thread.start()
        
        # Load existing memory if available
        self._load_memory()
    
    @measure_time
    def add_memory(self, content: str, node_type: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryNode:
        """Add a new memory node"""
        with self.lock:
            # Check if similar content already exists
            content_hash = hash(content)
            if content_hash in self.node_by_content:
                node_id = self.node_by_content[content_hash]
                node = self.nodes[node_id]
                # Update access count and activation
                node.access_count += 1
                node.activation_level = 1.0
                node.updated_at = datetime.utcnow().isoformat()
                # Add temporal marker
                if self.config["temporal_awareness"]:
                    node.temporal_markers.append(node.updated_at)
                # Update indexes
                self._update_indexes(node)
                logger.debug(f"Existing memory node accessed: {node.id}")
                return node
            
            # Create new node
            node = MemoryNode(content=content, node_type=node_type, metadata=metadata)
            
            # Generate vector embedding if enabled
            if self.config["enable_embeddings"]:
                node.vector = self._generate_embedding(content)
                self.node_vectors[node.id] = node.vector
            
            # Store node
            self.nodes[node.id] = node
            self.node_by_content[content_hash] = node.id
            
            # Update indexes
            self.node_type_index[node_type].add(node.id)
            if self.config["temporal_awareness"]:
                heapq.heappush(self.temporal_index, (node.created_at, node.id))
            heapq.heappush(self.activation_index, (-node.activation_level, node.id))
            
            logger.debug(f"New memory node created: {node.id}")
            record_metric("memory_nodes_created", 1, {"node_type": node_type})
            
            return node
    
    @measure_time
    def connect_memories(self, source_id: str, target_id: str, connection_type: str, 
                        strength: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> MemoryConnection:
        """Create a connection between two memory nodes"""
        with self.lock:
            # Verify nodes exist
            if source_id not in self.nodes or target_id not in self.nodes:
                raise ValueError(f"Source or target node not found: {source_id}, {target_id}")
            
            # Create connection
            connection = MemoryConnection(
                source_id=source_id,
                target_id=target_id,
                connection_type=connection_type,
                strength=strength,
                metadata=metadata
            )
            
            # Store connection
            self.connections[connection.id] = connection
            
            # Add to node connections
            source_node = self.nodes[source_id]
            source_node.connections.append(connection)
            
            logger.debug(f"Memory connection created: {connection.id}")
            record_metric("memory_connections_created", 1, {
                "connection_type": connection_type,
                "source_type": source_node.node_type,
                "target_type": self.nodes[target_id].node_type
            })
            
            return connection
    
    @measure_time
    def search_by_content(self, query: str, limit: int = 10, threshold: float = 0.6) -> List[MemoryNode]:
        """Search memory nodes by content similarity"""
        results = []
        
        if self.config["enable_embeddings"]:
            # Generate query embedding
            query_vector = self._generate_embedding(query)
            
            # Find similar nodes by vector similarity
            similarities = []
            for node_id, vector in self.node_vectors.items():
                similarity = self._vector_similarity(query_vector, vector)
                if similarity >= threshold:
                    similarities.append((similarity, node_id))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            # Get top results
            for similarity, node_id in similarities[:limit]:
                node = self.nodes[node_id]
                # Update node access
                with self.lock:
                    node.access_count += 1
                    node.activation_level = max(node.activation_level, similarity)
                    node.updated_at = datetime.utcnow().isoformat()
                results.append(node)
        else:
            # Simple text search when embeddings not enabled
            query_lower = query.lower()
            matches = []
            
            for node in self.nodes.values():
                if query_lower in node.content.lower():
                    # Calculate rough similarity
                    similarity = len(query_lower) / (len(node.content.lower()) + 0.1)
                    if similarity >= threshold:
                        matches.append((similarity, node))
            
            # Sort by similarity
            matches.sort(reverse=True)
            
            # Get top results
            for similarity, node in matches[:limit]:
                # Update node access
                with self.lock:
                    node.access_count += 1
                    node.activation_level = max(node.activation_level, similarity)
                    node.updated_at = datetime.utcnow().isoformat()
                results.append(node)
        
        record_metric("memory_searches", 1, {"query_length": len(query), "results": len(results)})
        return results
    
    @measure_time
    def get_connected_memories(self, node_id: str, connection_types: Optional[List[str]] = None, 
                             max_depth: int = 2) -> List[Tuple[MemoryNode, List[MemoryConnection]]]:
        """Get connected memories with their connection paths"""
        if node_id not in self.nodes:
            return []
        
        results = []
        visited = set()
        
        def traverse(current_id, depth, path):
            if depth > max_depth or current_id in visited:
                return
            
            visited.add(current_id)
            current_node = self.nodes[current_id]
            
            # Skip root node in results
            if current_id != node_id:
                results.append((current_node, path.copy()))
            
            # Continue traversal
            for conn in current_node.connections:
                if connection_types and conn.connection_type not in connection_types:
                    continue
                
                target_id = conn.target_id
                if target_id not in visited:
                    new_path = path.copy()
                    new_path.append(conn)
                    traverse(target_id, depth + 1, new_path)
        
        # Start traversal
        traverse(node_id, 0, [])
        
        # Update access counts
        with self.lock:
            self.nodes[node_id].access_count += 1
            self.nodes[node_id].activation_level = 1.0
            self.nodes[node_id].updated_at = datetime.utcnow().isoformat()
        
        return results
    
    @measure_time
    def get_active_memories(self, threshold: Optional[float] = None, limit: int = 10) -> List[MemoryNode]:
        """Get most active memory nodes"""
        if threshold is None:
            threshold = self.config["activation_threshold"]
        
        active_nodes = []
        
        with self.lock:
            # Create a copy of the activation index to avoid modifying the original during iteration
            temp_index = list(self.activation_index)
            
            # Get nodes with activation above threshold
            for _, node_id in temp_index:
                node = self.nodes.get(node_id)
                if node and node.activation_level >= threshold:
                    active_nodes.append(node)
                    if len(active_nodes) >= limit:
                        break
        
        return active_nodes
    
    @measure_time
    def save_memory(self, filename: Optional[str] = None):
        """Save memory state to file"""
        if not filename:
            filename = f"echo_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        memory_file = self.memory_dir / filename
        
        with self.lock:
            memory_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
                "connections": {conn_id: conn.to_dict() for conn_id, conn in self.connections.items()},
                "config": self.config
            }
            
            with open(memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
            
            logger.info(f"Memory saved to {memory_file}")
    
    @measure_time
    def load_memory(self, filename: str):
        """Load memory state from file"""
        memory_file = self.memory_dir / filename
        
        if not memory_file.exists():
            logger.warning(f"Memory file not found: {memory_file}")
            return False
        
        try:
            with open(memory_file, 'r') as f:
                memory_data = json.load(f)
            
            with self.lock:
                # Clear existing memory
                self.nodes.clear()
                self.connections.clear()
                self.node_type_index.clear()
                self.temporal_index.clear()
                self.activation_index.clear()
                self.node_vectors.clear()
                self.node_by_content.clear()
                
                # Load nodes
                for node_id, node_data in memory_data["nodes"].items():
                    node = MemoryNode.from_dict(node_data)
                    self.nodes[node_id] = node
                    self.node_type_index[node.node_type].add(node_id)
                    content_hash = hash(node.content)
                    self.node_by_content[content_hash] = node_id
                    
                    # Regenerate vectors if needed
                    if self.config["enable_embeddings"] and node.vector is None:
                        node.vector = self._generate_embedding(node.content)
                        self.node_vectors[node_id] = node.vector
                    
                    # Update indexes
                    if self.config["temporal_awareness"]:
                        heapq.heappush(self.temporal_index, (node.created_at, node_id))
                    heapq.heappush(self.activation_index, (-node.activation_level, node_id))
                
                # Load connections
                for conn_id, conn_data in memory_data["connections"].items():
                    conn = MemoryConnection.from_dict(conn_data)
                    self.connections[conn_id] = conn
                    
                    # Add to source node
                    if conn.source_id in self.nodes:
                        source_node = self.nodes[conn.source_id]
                        source_node.connections.append(conn)
            
            logger.info(f"Memory loaded from {memory_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return False
    
    def register_sync_handler(self, handler_id: str, handler_func):
        """Register a synchronization handler"""
        self.sync_handlers[handler_id] = handler_func
        logger.info(f"Sync handler registered: {handler_id}")
    
    def unregister_sync_handler(self, handler_id: str):
        """Unregister a synchronization handler"""
        if handler_id in self.sync_handlers:
            del self.sync_handlers[handler_id]
            logger.info(f"Sync handler unregistered: {handler_id}")
    
    @measure_time
    def sync_with_component(self, component_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize memory with another component"""
        response = {
            "status": "success",
            "synced_nodes": 0,
            "synced_connections": 0,
            "errors": []
        }
        
        try:
            if "nodes" in data:
                for node_data in data["nodes"]:
                    try:
                        # Check if node already exists
                        if "id" in node_data and node_data["id"] in self.nodes:
                            # Update existing node
                            existing_node = self.nodes[node_data["id"]]
                            existing_node.metadata.update(node_data.get("metadata", {}))
                            existing_node.updated_at = datetime.utcnow().isoformat()
                        else:
                            # Create new node
                            self.add_memory(
                                content=node_data["content"],
                                node_type=node_data["node_type"],
                                metadata=node_data.get("metadata", {})
                            )
                        response["synced_nodes"] += 1
                    except Exception as e:
                        response["errors"].append(f"Error syncing node: {e}")
            
            if "connections" in data:
                for conn_data in data["connections"]:
                    try:
                        # Check if source and target exist
                        if conn_data["source_id"] in self.nodes and conn_data["target_id"] in self.nodes:
                            # Create connection
                            self.connect_memories(
                                source_id=conn_data["source_id"],
                                target_id=conn_data["target_id"],
                                connection_type=conn_data["connection_type"],
                                strength=conn_data.get("strength", 1.0),
                                metadata=conn_data.get("metadata", {})
                            )
                            response["synced_connections"] += 1
                        else:
                            response["errors"].append("Source or target node not found")
                    except Exception as e:
                        response["errors"].append(f"Error syncing connection: {e}")
            
            # Trigger registered sync handlers
            for handler_id, handler_func in self.sync_handlers.items():
                try:
                    handler_func(component_id, data)
                except Exception as e:
                    logger.error(f"Error in sync handler {handler_id}: {e}")
            
            logger.info(f"Memory synced with {component_id}: {response['synced_nodes']} nodes, {response['synced_connections']} connections")
            
        except Exception as e:
            response["status"] = "error"
            response["errors"].append(f"Global sync error: {e}")
            logger.error(f"Error syncing with {component_id}: {e}")
        
        return response
    
    def stop(self):
        """Stop the memory system"""
        self.running = False
        # Save memory before stopping
        self.save_memory("echo_memory_final.json")
        logger.info("Echo Spiral Memory stopped")
    
    def _auto_save_thread(self):
        """Background thread for auto-saving memory"""
        logger.info("Auto-save thread started")
        while self.running:
            try:
                time.sleep(self.config["save_interval"])
                if self.running:  # Check again after sleep
                    self.save_memory()
            except Exception as e:
                logger.error(f"Error in auto-save thread: {e}")
    
    def _activation_decay_thread(self):
        """Background thread for activation decay"""
        logger.info("Activation decay thread started")
        while self.running:
            try:
                time.sleep(10)  # Decay every 10 seconds
                if not self.running:  # Check again after sleep
                    break
                
                with self.lock:
                    # Apply decay to all nodes
                    decay_rate = self.config["decay_rate"]
                    for node in self.nodes.values():
                        if node.activation_level > 0:
                            node.activation_level = max(0, node.activation_level - decay_rate)
                    
                    # Rebuild activation index periodically
                    if random.random() < 0.1:  # ~10% chance each cycle
                        self.activation_index = [(-node.activation_level, node_id) for node_id, node in self.nodes.items()]
                        heapq.heapify(self.activation_index)
                
            except Exception as e:
                logger.error(f"Error in activation decay thread: {e}")
    
    def _load_memory(self):
        """Load the most recent memory file if available"""
        try:
            memory_files = list(self.memory_dir.glob("echo_memory_*.json"))
            if memory_files:
                # Sort by modification time (most recent first)
                most_recent = max(memory_files, key=lambda f: f.stat().st_mtime)
                self.load_memory(most_recent.name)
            else:
                logger.info("No memory files found, starting with empty memory")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
    
    def _update_indexes(self, node: MemoryNode):
        """Update indexes for a node"""
        # Update node type index
        self.node_type_index[node.node_type].add(node.id)
        
        # Rebuild activation index is expensive, so we don't do it here
        # It will be rebuilt periodically in the decay thread
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate vector embedding for text content"""
        if self.config["mock_mode"]:
            # Generate random embedding for testing
            return np.random.randn(self.config["vector_dimensions"])
        
        try:
            # Try to use sentence transformers if available
            import sentence_transformers
            model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(text)
        except ImportError:
            # Fallback to random if sentence transformers not available
            logger.warning("sentence_transformers not available, using random embeddings")
            return np.random.randn(self.config["vector_dimensions"])
    
    def _vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)

# Create global instance
echo_spiral_memory = EchoSpiralMemory()

# Helper functions
def add_memory(content: str, node_type: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryNode:
    """Add a memory using the global instance"""
    return echo_spiral_memory.add_memory(content, node_type, metadata)

def connect_memories(source_id: str, target_id: str, connection_type: str, 
                    strength: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> MemoryConnection:
    """Connect memories using the global instance"""
    return echo_spiral_memory.connect_memories(source_id, target_id, connection_type, strength, metadata)

def search_memory(query: str, limit: int = 10, threshold: float = 0.6) -> List[MemoryNode]:
    """Search memory by content using the global instance"""
    return echo_spiral_memory.search_by_content(query, limit, threshold) 