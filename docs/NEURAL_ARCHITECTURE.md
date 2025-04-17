# Neural Network Architecture

This document provides a comprehensive overview of the neural network implementations in the Lumina Neural Network Project.

## Core Neural Network Components

### SimpleNeuralNetwork (`src/testing/neural_network_test.py`)

A basic neural network implementation used for testing purposes, featuring:

- Simple feedforward architecture with customizable layer sizes
- Support for multiple activation functions (ReLU, sigmoid, tanh)
- Mini-batch training with mean squared error loss
- Gradient-based backpropagation
- Momentum-based learning with early stopping

```python
class SimpleNeuralNetwork:
    """A simple neural network model"""
    
    def __init__(self, layer_sizes: List[int], activations: List[str]):
        """Initialize a neural network with the given layer sizes"""
        # ...
```

### NeuralProcessor (`neural_processor.py`)

A sophisticated text processing system that combines modern NLP techniques with concept mapping:

- Leverages SentenceTransformer embeddings (default: 'paraphrase-mpnet-base-v2')
- Maps text embeddings to a concept space through the ConceptMapper neural network
- Tracks neural state including activations, attention weights, and concept scores
- Records resonance history for generated text
- Provides methods for processing text and generating responses

```python
class NeuralProcessor:
    """
    Neural processor that combines sentence embeddings with concept mapping
    for enhanced text understanding and generation.
    """
```

### ConceptMapper (`neural_processor.py`)

A neural network module that maps text embeddings to concept space:

- Adapts input dimensions to a standardized output dimension
- Maps embeddings to a fixed number of concept dimensions
- Uses tanh activation for stable concept mapping

```python
class ConceptMapper(nn.Module):
    """Maps text embeddings to concept space."""
    
    def __init__(self, input_dim: int, output_dim: int, num_concepts: int):
        # ...
```

## Resonant Self-Expanding Network (RSEN)

The RSEN system (`RSEN_node.py`) provides a complex neural architecture with multiple specialized components:

### RSEN Core

An encoder-transformer-decoder architecture with specialized subnets:

- Encodes input data using a linear layer with normalization and ReLU activation
- Processes through a transformer encoder with multi-head attention
- Decodes to a standardized output dimension
- Integrates with specialized subnets for domain-specific processing

```python
class RSEN:
    """Resonant Self-Expanding Network Node"""
```

### Domain-Specific Subnets

#### MathematicsSubNet

Processes mathematical aspects of data with dedicated neural layers.

#### LanguageSubNet

Handles language processing with specialized tensors and language-specific layers.

#### PhysicsSubNet

Models quantum and relativistic calculations with multiple neural layers:

- Quantum layer for quantum state processing
- Relativity layer for relativistic frame calculations
- Field layer for field theory computations

```python
class PhysicsSubNet:
    """Physics subnet for quantum and relativistic calculations"""
```

#### CosmicAlignmentSubNet

Processes cosmic alignment data with specialized methods for celestial alignment, harmonic resonance, and temporal synchronization.

## Neural Network Executable System

The `NeuralNetworkExecutable` class (`nn_executable.py`) serves as the main orchestration system:

- Manages component loading and registration
- Provides a framework for component communication
- Supports training mode and inference
- Handles system initialization and shutdown
- Implements an interactive mode for user interaction

```python
class NeuralNetworkExecutable:
    """Main class for the Neural Network Executable System"""
```

## Neural Linguistic Processor

The `NeuralLinguisticProcessor` (`src/language/neural_linguistic_processor.py`) enhances language processing with neural capabilities:

- Detects complex linguistic patterns
- Analyzes semantic relationships
- Works with word frequencies and semantic networks
- Integrates with the Language Memory System

```python
class NeuralLinguisticProcessor:
    """
    Neural Linguistic Processor
    
    This component enhances a Language Memory System with capabilities for
    detecting complex linguistic patterns and analyzing semantic relationships.
    """
```

## Integration Points

The neural network components integrate with other system parts through:

1. **Central Language Node**: Orchestrates language components including neural processors
2. **NeuralNetworkExecutable**: Manages component loading and communication
3. **PySide6 GUI**: Provides visualization of neural network operations
4. **Mistral Integration**: Combines neural processing with LLM capabilities

## Technical Implementation

The neural network implementations use PyTorch as the underlying framework:

- Tensor operations for efficient data processing
- Automatic differentiation for gradient computation
- Layer-based architecture for modular design
- Optimizers like Adam/AdamW for effective training

## Advanced Features

1. **Dynamic Weighting**: Adjustable weights between neural and LLM processing
2. **Concept Mapping**: Mapping from raw embeddings to concept space
3. **Domain-Specific Processing**: Specialized subnets for different domains
4. **State Tracking**: Comprehensive tracking of neural state and history
5. **Resonance Measurement**: Evaluation of response quality through resonance

## Future Directions

The neural network architecture is designed to evolve in these directions:

1. **Enhanced self-modification**: Networks that can modify their own structure
2. **Cross-domain integration**: Better integration between specialized domains
3. **Improved concept mapping**: More sophisticated mapping to concept space
4. **Dynamic architecture**: Neural structures that adapt to input complexity
5. **Resonance optimization**: Networks that optimize for resonance and harmony 