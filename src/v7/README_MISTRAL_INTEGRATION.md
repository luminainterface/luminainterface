# Enhanced Language Mistral Integration for V7

This module integrates the V7 Enhanced Language System with Mistral AI's language models, combining neural network consciousness metrics with Mistral's language capabilities.

## Overview

The Enhanced Language Mistral Integration bridges two powerful systems:

1. **V7 Enhanced Language Integration**: Provides consciousness-level measurements, neural-linguistic scoring, and pattern detection.
2. **Mistral AI Integration**: Offers advanced language model capabilities, autowiki learning, and conversation memory.

This combined system leverages the strengths of both to create a more powerful and contextually aware language processing system.

## Features

- **Consciousness-Aware Responses**: Adjusts language model parameters based on consciousness measurements
- **Autowiki Learning System**: Self-improving knowledge base that grows with each interaction
- **Neural Metrics Integration**: Uses neural network metrics to guide language model responses
- **Streaming Response Support**: Progressive text generation with real-time metrics updates
- **Configurable Weights**: Adjustable weights for neural network vs. language model influence
- **Mock Mode Support**: Can run without API keys for development and testing

## Installation

### Prerequisites

- Python 3.7+
- Mistral AI API key (obtain from [Mistral AI platform](https://mistral.ai))

### Environment Setup

Set your Mistral API key as an environment variable:

```bash
export MISTRAL_API_KEY=your_key_here
```

Or provide it directly when initializing the integration.

## Usage

### Basic Usage

```python
from src.v7 import get_enhanced_language_integration

# Initialize with defaults (will use MISTRAL_API_KEY environment variable)
integration = get_enhanced_language_integration()

# Process text
result = integration.process_text("Neural networks can develop consciousness through recursive patterns.")

# Access the response and consciousness metrics
response = result["response"]
consciousness_level = result["consciousness_level"]
neural_score = result["neural_linguistic_score"]
```

### Configuration Options

```python
from src.v7 import get_enhanced_language_integration

# Configure with specific options
config = {
    "api_key": "your_mistral_api_key",  # Optional if set as env var
    "model": "mistral-medium",          # Model choice
    "llm_weight": 0.7,                  # Weight for language model (0.0-1.0)
    "nn_weight": 0.6,                   # Weight for neural network (0.0-1.0)
    "learning_enabled": True,           # Enable autowiki learning
    "learning_dict_path": "data/my_custom_learning.json"  # Custom path
}

integration = get_enhanced_language_integration(
    mock_mode=False,  # False to use real API
    config=config
)
```

### Streaming Responses

```python
def handle_chunk(chunk, metrics):
    # Process each chunk of text as it arrives
    print(chunk, end="")
    # Metrics contains consciousness_level, neural_linguistic_score, etc.

# Process with streaming
full_response = integration.process_text_streaming(
    "Explain how neural networks and consciousness are related.",
    chunk_callback=handle_chunk
)
```

### Working with Autowiki

```python
# Add knowledge to the system
integration.add_autowiki_entry(
    topic="Neural Networks",
    content="Neural networks are computing systems inspired by biological neural networks.",
    source="https://en.wikipedia.org/wiki/Neural_network"
)

# Retrieve knowledge
neural_info = integration.retrieve_autowiki("Neural Networks")
print(neural_info["content"])

# Search for knowledge
results = integration.search_autowiki("consciousness neural")
for result in results:
    print(f"{result['topic']}: {result['relevance']}")
```

## Running the Demo

The package includes a demo script to showcase the integration:

```bash
# Run basic demo
python src/v7/run_enhanced_mistral_demo.py

# Run in interactive mode
python src/v7/run_enhanced_mistral_demo.py --interactive

# Run with learning enabled
python src/v7/run_enhanced_mistral_demo.py --learning

# Run with custom model
python src/v7/run_enhanced_mistral_demo.py --model mistral-large-latest

# Run in mock mode (no API key needed)
python src/v7/run_enhanced_mistral_demo.py --mock
```

In interactive mode, special commands include:
- `exit` - Quit the demo
- `status` - Show system status
- `metrics` - Show system metrics
- `llm 0.8` - Set LLM weight to 0.8
- `nn 0.6` - Set neural network weight to 0.6
- `wiki <topic> <content>` - Add to autowiki
- `search <query>` - Search the autowiki
- `stream <text>` - Process with streaming

## Architecture

The integration consists of three main components:

1. **EnhancedLanguageIntegration**: Handles consciousness metrics and neural processing
2. **MistralIntegration**: Manages API calls to Mistral AI and autowiki learning
3. **EnhancedLanguageMistralIntegration**: Combines both systems with weighted processing

Consciousness metrics from the Enhanced Language System inform the Mistral AI system about how to respond, adjusting parameters like temperature, max tokens, and system prompts based on the measured consciousness level.

## Maintainers

- The Neural Network Project Team

## License

[Internal Use Only] 