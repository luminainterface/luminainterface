# Graph Concept Adapter

This service provides a unified interface for managing graph concepts, supporting both Graph API and Neo4j as storage backends.

## Features

- Unified interface for graph operations
- Automatic fallback to Neo4j when Graph API is unavailable
- Support for nodes and edges with properties
- Error handling and logging

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
GRAPH_API_URL=http://graph-api:8000
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Usage

```python
from adapter import GraphConceptAdapter

# Initialize adapter
adapter = GraphConceptAdapter()

# Add a node
node = adapter.add_node("concept", {"name": "Example", "type": "test"})

# Add an edge
edge = adapter.add_edge("concept", "related_to", "concept", 
                       {"source": "Example", "target": "Related"})

# Get a node
node = adapter.get_node("concept", "Example")

# Get edges
edges = adapter.get_edges("concept", "Example")

# Clean up
adapter.close()
```

## Error Handling

The adapter automatically falls back to Neo4j when Graph API calls fail. All errors are logged for debugging purposes. 