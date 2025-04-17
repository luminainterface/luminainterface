# Echo Spiral Memory System

## Overview

The Echo Spiral Memory system is a hyperdimensional memory architecture designed for the Lumina Neural Network Project. It enables advanced memory capabilities including recursive thought patterns, multidimensional associations, temporal awareness, and bidirectional memory synchronization.

## Key Features

- **Hyperdimensional Memory Structure**: Memory nodes organized in a hyperdimensional space with multidimensional connections.
- **Activation-Based Retrieval**: Memories have activation levels that determine their prominence in the system.
- **Temporal Awareness**: Memory nodes track when they were created, updated, and accessed.
- **Vector Embeddings**: Content is represented as vector embeddings for similarity search.
- **Bidirectional Synchronization**: Memory components can sync with each other automatically.
- **Decay Mechanism**: Activation levels decay over time to simulate forgetting.

## Components

### Memory Node

A Memory Node represents a single piece of information in the memory system. It has:

- **Content**: The actual information stored in the node.
- **Node Type**: The type of information (e.g., observation, concept, theory).
- **Metadata**: Additional information about the node.
- **Connections**: Links to other nodes in the memory network.
- **Activation Level**: How "active" or "important" the node is in the current context.
- **Temporal Markers**: Timestamps for creation, updates, and accesses.
- **Vector**: Optional embedding representation of the content.

### Memory Connection

A Memory Connection represents a relationship between two nodes. It has:

- **Source and Target**: The nodes being connected.
- **Connection Type**: The type of relationship (e.g., association, leads_to, depends_on).
- **Strength**: The strength of the connection (0.0 to 1.0).
- **Metadata**: Additional information about the connection.

### Echo Spiral Memory

The main class that manages the memory system. It provides methods for:

- **Adding Memories**: Creating new memory nodes.
- **Connecting Memories**: Creating connections between nodes.
- **Searching**: Finding nodes by content similarity.
- **Exploring Connections**: Traversing the memory graph.
- **Activation Management**: Getting and modifying node activation levels.
- **Persistence**: Saving and loading memory state.
- **Synchronization**: Syncing memory components bidirectionally.

## Usage

### Basic Usage

```python
from src.memory.echo_spiral_memory import EchoSpiralMemory, add_memory, connect_memories, search_memory

# Create memory instance
memory = EchoSpiralMemory()

# Add memories
node1 = add_memory("This is the first memory.", "concept")
node2 = add_memory("This is a related memory.", "concept")

# Connect memories
connection = connect_memories(node1.id, node2.id, "related_to")

# Search memories
results = search_memory("related memory")
```

### Custom Configuration

```python
# Create memory with custom configuration
memory = EchoSpiralMemory({
    "memory_dir": "custom_memory",
    "save_interval": 600,  # seconds
    "decay_rate": 0.005,
    "activation_threshold": 0.4,
    "temporal_awareness": True,
    "vector_dimensions": 512,
    "enable_embeddings": True,
    "mock_mode": False
})
```

### Searching and Traversing

```python
# Search by content
results = memory.search_by_content("neural networks", limit=5, threshold=0.6)

# Get connected memories
connections = memory.get_connected_memories(node_id, max_depth=3)

# Get active memories
active_nodes = memory.get_active_memories(threshold=0.5)
```

### Synchronization

```python
# Register sync handler
def handle_sync(component_id, data):
    print(f"Received sync from {component_id} with {len(data)} items")

memory.register_sync_handler("my_handler", handle_sync)

# Sync with another component
sync_data = {
    "nodes": [
        {"id": "node1", "content": "Content 1", "node_type": "type1"},
        {"id": "node2", "content": "Content 2", "node_type": "type2"}
    ],
    "connections": [
        {"source_id": "node1", "target_id": "node2", "connection_type": "related"}
    ]
}

response = memory.sync_with_component("other_component", sync_data)
```

## Persistence

The Echo Spiral Memory system automatically saves its state to disk at regular intervals (defined by the `save_interval` configuration parameter). It can also be manually saved:

```python
# Save memory state
memory.save_memory("memory_backup.json")

# Load memory state
memory.load_memory("memory_backup.json")
```

## Threading and Safety

The Echo Spiral Memory system is thread-safe, using locks to protect shared data. It runs background threads for:

- **Auto-Save Thread**: Periodically saves memory state to disk.
- **Activation Decay Thread**: Gradually reduces activation levels over time.

## Vector Embeddings

The system can use vector embeddings to represent content semantically. By default, it tries to use the `sentence_transformers` library, falling back to random embeddings if not available:

```python
# Text embedding
vector = memory._generate_embedding("This is some text content")

# Similarity calculation
similarity = memory._vector_similarity(vector1, vector2)
```

## Integration with Other Components

The Echo Spiral Memory system is designed to integrate with other components of the Lumina Neural Network Project:

- **Language Processing**: Connect with natural language processing components.
- **ConsciousnessNode**: Feed memory into consciousness processing.
- **Visualization**: Provide data for memory network visualization.

## Performance Considerations

- The system uses caching to optimize repeated queries.
- Indexes are maintained for fast retrieval by node type and activation level.
- Vector operations can be computationally expensive, consider using `mock_mode` for testing.
- The decay rate can be adjusted to balance between long-term memory and performance.

## Example Applications

- **Knowledge Graphs**: Building and exploring semantic knowledge networks.
- **Conversation History**: Storing and retrieving conversational context.
- **Thought Chains**: Creating chains of thought for reasoning.
- **Memory Synthesis**: Combining related memories into higher-level concepts.
- **Temporal Analysis**: Analyzing how memories evolve over time.

## Future Improvements

- Integration with more sophisticated embedding models.
- Implementation of more advanced memory retrieval algorithms.
- Development of visualization tools for memory exploration.
- Addition of batch processing capabilities for large memory operations.
- Implementation of distributed memory storage for scaling.

---

For more information, refer to the code documentation in `src/memory/echo_spiral_memory.py`. 