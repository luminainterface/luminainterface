# LUMINA V7 - Mistral AI & AutoWiki Integration

This document provides instructions for setting up and using the Mistral AI integration and AutoWiki learning system in LUMINA V7.

## Overview

The V7 system includes two new powerful features:

1. **Mistral AI Integration**: Connects the V7 language system with Mistral's AI models, enhancing response quality and capabilities.

2. **AutoWiki Learning Dictionary**: An automated system that builds a knowledge base by fetching information from various sources (primarily Wikipedia) and learning from interactions.

## Requirements

- Python 3.8+
- Mistral API key (get one at https://mistral.ai/)
- Required packages:
  - `mistralai` (Mistral client)
  - `requests` (for AutoWiki)
  - `beautifulsoup4` (for parsing web content)

Install required packages with:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running the Demo

The fastest way to try the system is using the demo script:

```bash
python -m src.v7.demo_mistral_autowiki --api-key 2AyKmqCkChQ75bseJTLK9QF2AK0aefJP
```

Optional arguments:
- `--model MODEL_NAME`: Specify Mistral model (default: mistral-small-latest)
- `--topics FILENAME`: Load topics from a file (one topic per line)
- `--debug`: Enable debug logging

### Using the V7 Launcher

For full V7 system with UI:

```bash
python -m src.v7.v7_launcher --mistral-key 2AyKmqCkChQ75bseJTLK9QF2AK0aefJP
```

The launcher supports additional options:
- `--mistral-model MODEL`: Define the Mistral model to use
- `--no-gui`: Run without graphical interface
- `--no-autowiki`: Disable AutoWiki system
- `--topics FILENAME`: Load topics from a file

## Environment Variables

You can set these environment variables instead of using command-line arguments:

- `MISTRAL_API_KEY`: Your Mistral API key
- `MISTRAL_MODEL`: Model to use (defaults to mistral-small-latest)

## Directory Structure

- `src/v7/mistral_integration.py`: Mistral AI integration with V7
- `src/v7/autowiki.py`: AutoWiki system for knowledge acquisition
- `src/v7/demo_mistral_autowiki.py`: Demo script
- `src/v7/v7_launcher.py`: Main launcher with all integrations

## Data Storage

- Learning dictionary: `data/v7/learning_dictionary.json`
- AutoWiki fetch history: `data/v7/autowiki_history.json`
- AutoWiki topics queue: `data/v7/autowiki_queue.json`

## Using the Learning Dictionary API

```python
from src.v7.mistral_integration import MistralEnhancedSystem

# Initialize the system
system = MistralEnhancedSystem(api_key="YOUR_API_KEY")

# Add to dictionary
system.add_to_dictionary("neural network", "A computing system inspired by biological neural networks")

# Get from dictionary
definition = system.get_from_dictionary("neural network")

# Process a message
result = system.process_message("Tell me about neural networks")
response = result["response"]
```

## Using the AutoWiki API

```python
from src.v7.autowiki import AutoWiki
from src.v7.mistral_integration import MistralEnhancedSystem

# Initialize systems
mistral = MistralEnhancedSystem(api_key="YOUR_API_KEY")
wiki = AutoWiki(mistral_system=mistral)

# Add topics to fetch queue
wiki.add_topic("artificial intelligence")
wiki.add_topics(["machine learning", "deep learning"])

# Process queue (fetch information)
wiki.process_queue(max_items=3)

# Start automatic background fetching
wiki.start_auto_fetch(interval_seconds=300)  # fetch every 5 minutes

# Get status
status = wiki.get_status()
```

## Mock Mode

Both systems can operate in mock mode if no API key is provided or dependencies are missing:

```python
# Mock mode Mistral
system = MistralEnhancedSystem()  # No API key = mock mode

# Process in mock mode
result = system.process_message("Hello")
```

## Troubleshooting

1. **Import errors**: Make sure you're running commands from the project root directory
2. **API key errors**: Check that your Mistral API key is valid and correctly provided
3. **Dictionary not updating**: Ensure the `data/v7` directory is writable
4. **AutoWiki not fetching**: Check your internet connection and ensure Wikipedia is accessible

## Example Topics File

Create a text file with topics to add to the AutoWiki queue:

```
artificial intelligence
neural networks
machine learning
deep learning
natural language processing
computer vision
```

Then load it with:
```bash
python -m src.v7.demo_mistral_autowiki --topics my_topics.txt
``` 