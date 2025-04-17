# Lumina v6.5

A minimalist implementation of the Enhanced Language System with a clean, functional UI.

![Lumina v6.5 Interface](diagrams/lumina_v6_5_interface.png)

## Overview

Lumina v6.5 provides a streamlined interface for interacting with the Enhanced Language System. It features:

- Simple chatbox interface for natural language interaction
- Integration with consciousness, neural linguistics, and memory systems
- Process panel with breathing, resonance, and echo functions
- Glyph panel for symbolic interaction with the system
- Neural network visualization
- Clean, modern UI design

## Features

- **Chatbox Interface**: Enter text and receive responses processed through the language system
- **Consciousness Integration**: Language processing with consciousness awareness
- **Neural Linguistic Processing**: Advanced pattern recognition and language analysis
- **Memory System**: Persistent storage and recall of language data
- **Symbolic Interaction**: Use glyphs to activate different system modes

## Installation

### Requirements

- Python 3.8 or higher
- PySide6

Install dependencies:

```bash
pip install PySide6
pip install -r requirements.txt
```

### Directory Structure

Ensure the following directory structure exists:

```
data/
  ├── memory/
  │    └── language_memory/
  ├── neural_linguistic/
  ├── central_language/
  └── v10/
```

These directories will be created automatically on first run if they don't exist.

## Usage

Run the application with:

```bash
python lumina_v6_5.py
```

### Interface Components

1. **Chatbox Panel**
   - Type messages in the input field at the bottom
   - Press Enter or click Send to submit
   - View conversation history in the main area

2. **Process Panel**
   - Breathe: Activate breathing patterns
   - Resonance: Engage with resonance patterns
   - Echo: Create echo patterns in the system

3. **Glyphs Panel**
   - Neural Network visualization
   - Glyph buttons for symbolic interaction
   - Click glyphs to activate them

### Messaging

Type your message in the input field and press Enter or click the Send button. The system will process your message through the Enhanced Language System, analyzing it with:

- Neural linguistic pattern recognition
- Consciousness level evaluation
- Memory integration
- Language pattern analysis

## Integration

Lumina v6.5 integrates with the following components:

- **Language Memory System**: Stores and recalls language patterns
- **Conscious Mirror Language**: Processes language with consciousness awareness
- **Neural Linguistic Processor**: Identifies patterns and builds semantic networks
- **Central Language Node**: Coordinates all language components

## Troubleshooting

If you encounter issues:

1. Check that PySide6 is properly installed
2. Verify that the required data directories exist
3. Look for errors in the `lumina_v6_5.log` file

## Development

To extend the system:

1. Additional glyphs can be added to the `create_glyphs_panel` method
2. New process functions can be added to the `create_process_panel` method
3. Language processing can be customized in the `on_processing_complete` method 