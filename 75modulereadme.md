# LUMINA v7.5 Mistral Integration Module

## Overview
The LUMINA v7.5 Mistral Integration Module provides a sophisticated neural network-enhanced interface to the Mistral AI API, featuring bidirectional information flow through a spiderweb architecture and dynamic parameter optimization.

## Features
- Neural network-based parameter weighting
- Bidirectional memory management system
- Dynamic embedding computation and resizing
- Conversation state tracking
- Spiderweb architecture integration
- Automatic parameter optimization

## Prerequisites
- Python 3.8+
- PyTorch
- MistralAI API access
- CUDA-capable GPU (optional, but recommended)

## Installation

1. Set up your environment:
```bash
pip install torch mistralai numpy
```

2. Set your Mistral API key:
```bash
export MISTRAL_API_KEY='your-api-key-here'
```

## Basic Usage

```python
from src.v7_5.mistral_integration import MistralIntegration

# Initialize the integration
mistral = MistralIntegration()

# Process a conversation
conversation = [
    {"role": "user", "content": "Hello, how are you?"}
]
response = mistral.process_message(conversation)

# Access the response
print(response["response"])
```

## Advanced Features

### Neural Network Parameter Control
```python
# Adjust parameters manually
mistral.adjust_parameters(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    llm_weight=0.8,
    nn_weight=0.2
)

# Save/load neural network state
mistral.save_neural_network("path/to/save.pt")
mistral.load_neural_network("path/to/save.pt")
```

### Memory Management
```python
# Start a new conversation
conversation_id = mistral.start_conversation()

# Access conversation history
history = mistral.get_conversation_history(conversation_id)

# Create and connect memory nodes
node_id = mistral.create_memory_node({
    "type": "context",
    "content": "Important information"
})

# Search memory
results = mistral.search_memory({"type": "context"})
```

### Embedding Operations
```python
# Compute embeddings for text
embedding = mistral.compute_embedding("Sample text")

# Resize embeddings
resized = mistral.resize_embedding(embedding, target_size=1024)
```

## Configuration

### Model Selection
The module supports different Mistral models:
- mistral-tiny
- mistral-small
- mistral-medium
- mistral-large

```python
mistral = MistralIntegration(model_name="mistral-medium")
```

### Neural Network Architecture
The default architecture includes:
- Input size: 1024 (adjustable)
- Hidden layers: 512 → 256 → 128
- Parameter heads for temperature, top_p, top_k, and LLM weight
- Dropout rate: 0.2

## Memory System

### Node Types
1. Conversation nodes
2. Message nodes
3. Context nodes
4. State nodes

### Data Persistence
Memory is automatically saved to disk and can be loaded on initialization:
```python
mistral.save_memory()  # Manual save
mistral.load_memory()  # Manual load
```

## Spiderweb Architecture Integration

The module implements the spiderweb architecture for version compatibility:
- Bidirectional information flow
- Version-aware message handling
- State preservation
- Dynamic routing

## Error Handling

The module includes comprehensive error handling:
- API call failures
- Embedding computation errors
- Memory operations
- Neural network operations

## Logging

Logging is configured by default:
```python
import logging
logging.getLogger("mistral_integration").setLevel(logging.DEBUG)
```

## Performance Optimization

### GPU Acceleration
The neural network automatically uses CUDA if available:
```python
# Check if using GPU
print(mistral.neural_network.device)
```

### Embedding Cache
Embeddings are cached in memory nodes for improved performance.

## Development Guidelines

1. Follow the existing error handling patterns
2. Maintain type hints for all functions
3. Update docstrings for new features
4. Add logging for significant operations
5. Preserve the spiderweb architecture principles

## Troubleshooting

Common issues and solutions:
1. API Key not found: Set MISTRAL_API_KEY environment variable
2. CUDA out of memory: Reduce batch size or model size
3. Memory persistence errors: Check file permissions
4. Embedding size mismatch: Use resize_embedding method

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License
MIT License - See LICENSE file for details 