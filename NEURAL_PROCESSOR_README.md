# Neural Processor

A sophisticated neural processor implementation for text embedding and concept mapping with configurable models.

## Overview

The Neural Processor (`neural_processor.py`) is a powerful component that combines sentence embeddings with concept mapping for enhanced text understanding and generation. It serves as a bridge between raw text input and higher-level concept representations.

## Features

- **Sentence Transformer Integration**: Leverages the power of SentenceTransformers for high-quality text embeddings
- **Concept Mapping**: Maps embeddings to a concept space through a neural network
- **State Tracking**: Comprehensive tracking of neural state including activations and confidence
- **Resonance History**: Records and analyzes resonance scores for generated text
- **Flexible Architecture**: Configurable dimensions and model selection
- **Background Processing**: Thread-based workers for non-blocking operations

## Architecture

### Core Components

1. **NeuralProcessor**: Main class that orchestrates the text processing pipeline
2. **ConceptMapper**: Neural network module that maps embeddings to concept space
3. **NeuralState**: Data class for representing and persisting neural network state
4. **ProcessingState**: Named tuple for holding the state of text processing

### Processing Pipeline

The Neural Processor follows this processing pipeline:

1. **Text Input**: Raw text is received for processing
2. **Embedding Generation**: Text is encoded into embeddings using SentenceTransformer
3. **Concept Mapping**: Embeddings are mapped to concept space through the ConceptMapper
4. **State Tracking**: The neural state is recorded for future reference
5. **Optional Response Generation**: Text responses can be generated based on the neural state

## Usage

### Basic Usage

```python
from neural_processor import NeuralProcessor

# Initialize the processor
processor = NeuralProcessor(model_dir="model_output")

# Process text
result = processor.process_text("Neural networks process language patterns.")

# Generate text response
response = processor.generate_text("Tell me about neural networks")

# Update resonance score for generated text
processor.update_resonance(response, score=4.5)
```

### Advanced Configuration

```python
# Initialize with custom parameters
processor = NeuralProcessor(
    model_dir="custom_models",
    embedding_dim=768,  
    output_dim=512,
    num_concepts=300,
    vocab_size=30000,
    temperature=0.8,
    embedding_model="all-mpnet-base-v2"
)
```

## Technical Implementation

### Sentence Transformers

The Neural Processor uses SentenceTransformer for generating high-quality text embeddings. The default model is 'paraphrase-mpnet-base-v2', which provides 768-dimensional embeddings.

### ConceptMapper Architecture

The ConceptMapper is a PyTorch neural network with two main components:

1. **Dimension Adapter**: Adapts the input dimension to the desired output dimension
2. **Concept Layer**: Maps the adapted dimensions to concept space

```python
class ConceptMapper(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_concepts: int):
        super().__init__()
        self.dimension_adapter = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.concept_layer = nn.Linear(output_dim, num_concepts)
        self.activation = nn.Tanh()
```

### Neural State Management

The system uses a comprehensive approach to state management:

- **NeuralState**: Tracks activations, attention weights, concept scores, and confidence
- **State History**: Maintains a history of states for analysis and learning
- **Background Saving**: Uses a worker thread for non-blocking state persistence

## Integration with Other Components

The Neural Processor is designed to integrate with:

1. **Neural Network Executable**: For system-wide orchestration
2. **Language Components**: For enhanced language processing
3. **PySide6 GUI**: For visualization of neural state and concepts

## Future Directions

1. **Enhanced Concept Mapping**: More sophisticated mapping between embeddings and concepts
2. **Dynamic Concept Space**: Concepts that can evolve based on usage patterns
3. **Multi-modal Processing**: Integration with image and audio processing
4. **Self-Optimization**: Neural architectures that adapt based on resonance scores
5. **Cross-domain Integration**: Better integration with domain-specific subnets

## References

- [SentenceTransformers Documentation](https://www.sbert.net/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- For a comprehensive overview of all neural network components, see [docs/NEURAL_ARCHITECTURE.md](docs/NEURAL_ARCHITECTURE.md) 