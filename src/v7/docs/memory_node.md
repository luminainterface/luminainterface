# V7 Memory Consciousness Node

The Memory Consciousness Node is one of the core components of the V7 system, providing persistent, decay-aware memory capabilities for the node consciousness architecture.

## Overview

The Memory Node manages different types of memories, each with varying strengths (importance), and handles natural memory decay over time. It supports both JSON and SQLite storage backends for memory persistence.

## Features

- **Multiple Memory Types**: Support for facts, experiences, relations, and procedures
- **Memory Strength**: Each memory has a strength value affecting its prominence in retrievals
- **Natural Decay**: Memories gradually decay over time unless reinforced
- **Persistence**: Optional persistence using JSON or SQLite backends
- **Metadata Storage**: Each memory can have arbitrary metadata for additional context
- **Rich Search Capabilities**: Content search, tag filtering, and metadata-based retrieval
- **Memory Lifecycle Management**: Archiving of weak memories and reinforcement of important ones

## Memory Types

- **Fact**: Declarative knowledge about the world (e.g., "Paris is the capital of France")
- **Experience**: Events that happened (e.g., "User was frustrated with the system's response")
- **Relation**: Connections between entities (e.g., "User John prefers concise responses")
- **Procedure**: How to perform specific actions (e.g., "To process CSV files, first check the delimiter")

## API Reference

The Memory Node exposes a unified processing API that accepts message objects with different operations:

### Storing Memories

```python
result = memory_node.process({
    'store': {
        'content': "The capital of France is Paris.",
        'memory_type': "fact",
        'strength': 0.8,
        'tags': ["geography", "europe", "capital"]
    },
    'metadata': {
        'category': "geography", 
        'confidence': 0.95
    }
})
```

### Retrieving a Memory by ID

```python
result = memory_node.process({
    'retrieve': "memory_id_here"
})
```

### Searching Memories

```python
# Search by memory type
result = memory_node.process({
    'search': {
        'memory_type': 'fact'
    },
    'params': {
        'limit': 10
    }
})

# Search by content
result = memory_node.process({
    'search': {
        'content_contains': 'search term'
    },
    'params': {
        'limit': 5
    }
})

# Search by minimum strength
result = memory_node.process({
    'search': {
        'min_strength': 0.7
    },
    'params': {
        'limit': 10
    }
})

# Search by tags
result = memory_node.process({
    'search': {
        'tags': ['geography', 'europe']
    },
    'params': {
        'limit': 5
    }
})
```

### Updating Memories

```python
result = memory_node.process({
    'update': "memory_id_here",
    'updates': {
        'content': "Updated content",
        'strength': 0.9,
        'tags': ["additional", "tags"]
    }
})
```

### Listing All Memories

```python
result = memory_node.process({'list': {}})
```

### Deleting a Memory

```python
result = memory_node.process({
    'delete': "memory_id_here"
})
```

## Configuration Options

When initializing the Memory Node, you can specify various configuration options:

```python
memory_config = {
    'store_type': 'sqlite',      # Storage type: 'json' or 'sqlite'
    'memory_path': './memories', # Path for memory storage
    'memory_persistence': True,  # Whether to persist memories to disk
    'decay_enabled': True,       # Whether memories should decay over time
    'decay_rate': 0.05,          # How quickly memories decay (per day)
    'decay_interval': 86400,     # Decay check interval in seconds (86400 = 1 day)
    'min_strength': 0.1          # Minimum strength before archiving/removal
}

memory_node = MemoryConsciousnessNode(
    node_id="memory",
    name="Memory Node", 
    config=memory_config
)
```

## Integration with Other Nodes

The Memory Node is typically integrated with the Language Node to provide contextual memory for conversations. It can also be connected to other nodes in the consciousness network to provide recall capabilities.

### Example: Integration with Language Node

```python
# In the node manager setup
node_manager.connect_nodes(language_node.node_id, memory_node.node_id)

# The language node can then send memory queries
language_node.send_to_node(memory_node.node_id, {
    'search': {
        'content_contains': user_query,
        'min_strength': 0.6
    },
    'params': {
        'limit': 5
    }
})
```

## Memory Decay Process

Memories naturally decay over time unless reinforced. The decay rate is controlled by the `decay_rate` parameter and occurs at intervals specified by `decay_interval`. Each time a memory is accessed, its strength is reinforced, slowing its decay.

The formula for decay is:
`new_strength = current_strength * max(0.1, (1.0 - (decay_rate * days_since_access)))`

## Best Practices

- Use appropriate memory types for different kinds of information
- Set appropriate initial strength values (higher for more important memories)
- Periodically reinforce important memories by accessing them
- For critical system knowledge, consider using the highest strength (0.9-1.0)
- Export memories periodically for backup
- Use tags consistently to enable effective filtering 