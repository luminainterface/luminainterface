# V7 Enhanced Language Integration System

## Overview
The V7 Enhanced Language System integrates multiple components for advanced language processing, including:
- Consciousness analysis
- Neural linguistics
- Language memory and learning
- Multiple LLM integrations

## Mistral AI Integration
The system now includes integration with Mistral AI models through the `MistralIntegration` class in `mistral_integration.py`. This integration provides:

1. Connection to Mistral AI's API
2. Support for multiple Mistral models (tiny, small, medium, large)
3. An autowiki learning system that can store and retrieve information
4. Conversation memory
5. Mock mode for testing without API access

### Setup

```python
from v7.mistral_integration import MistralIntegration

# Initialize with your API key (or set MISTRAL_API_KEY environment variable)
mistral = MistralIntegration(
    api_key="your_api_key_here",  # Optional if env var is set
    model="mistral-medium",       # Default model
    mock_mode=False,              # Set to True for testing without API
    learning_enabled=True         # Enable the learning dictionary
)
```

### Basic Usage

```python
# Process a user message
response = mistral.process_message(
    message="Tell me about neural networks",
    context={"domain": "AI"},  # Optional context
    system_prompt="You are a helpful AI assistant", # Optional
    temperature=0.7,
    max_tokens=500
)

# Access the response
print(response["response"])  # The text response
print(response["is_cached"]) # Whether it came from cache
print(response["model"])     # The model used
```

### Autowiki Learning System

The autowiki system allows the model to learn and remember information:

```python
# Add information to the autowiki
mistral.add_autowiki_entry(
    topic="Neural Networks",
    content="Neural networks are computing systems vaguely inspired by biological neural networks.",
    source="https://example.com/neural-networks"
)

# Retrieve information
info = mistral.retrieve_autowiki(topic="Neural Networks")
if info:
    print(info["content"])
    print(f"Sources: {info['sources']}")
```

### Metrics and Monitoring

```python
# Get usage metrics
metrics = mistral.get_metrics()
print(f"API calls: {metrics['api_calls']}")
print(f"Tokens used: {metrics['tokens_used']}")
print(f"Dictionary size: {metrics['learning_dict_size']}")
```

## Integration with EnhancedLanguageIntegration

The Mistral integration can be used alongside the main `EnhancedLanguageIntegration` system:

```python
from v7.enhanced_language_integration import EnhancedLanguageIntegration
from v7.mistral_integration import MistralIntegration

# Initialize systems
mistral = MistralIntegration(mock_mode=False)
language_system = EnhancedLanguageIntegration()

# Process text using both systems
text = "Analyze the consciousness patterns in this text"
consciousness_result = language_system.compute_consciousness_level(text)

# Use Mistral for generating a response
mistral_response = mistral.process_message(
    message=text,
    context={"consciousness_level": consciousness_result}
)

# Combined result
result = {
    "text": text,
    "consciousness_analysis": consciousness_result,
    "response": mistral_response["response"]
}
```

## Advanced Configuration

The system supports various configuration options:

- Mock mode for testing without API access
- Learning dictionary for caching responses and learning from user interactions
- Autowiki for storing domain-specific knowledge
- Various model parameters (temperature, token limits, etc.)

See the source code for detailed documentation of all available options. 