# Mock Lumina v6.5

A simplified implementation of the Lumina interface with minimal dependencies.

## Overview

Mock Lumina provides a clean interface that mimics the functionality of the full Lumina v6.5 system without requiring complex dependencies like numpy or other libraries that might be difficult to install. This version gives you the same visual experience with predefined responses and interactions.

## Features

- **Clean Chat Interface**: Same 16:9 layout and aesthetic as the full version
- **Process Panel**: Interactive buttons for Breathe, Resonance, and Echo processes
- **Glyph Panel**: Full set of interactive glyphs with custom responses
- **Neural Network Visualization**: Static visualization of neural network structure
- **Predefined Responses**: Set of philosophical responses that cycle through on interaction

## Requirements

- Python 3.8 or higher
- PySide6 (the only external dependency)

## Installation

1. Install PySide6:

```bash
pip install PySide6
```

2. No other dependencies required!

## Usage

Run the application with:

```bash
python mock_lumina.py
```

### Interface Components

1. **Chatbox Panel**
   - Type messages in the input field at the bottom
   - Press Enter or click Send to submit
   - View conversation history in the main area

2. **Process Panel**
   - Breathe: Activates breathing pattern simulation
   - Resonance: Simulates resonance patterns
   - Echo: Activates echo function

3. **Glyphs Panel**
   - Neural Network visualization (static)
   - Glyph buttons for symbolic interaction
   - Each glyph has a unique philosophical response

## Why Use Mock Lumina?

- **Simplicity**: No complex dependencies to install
- **Compatibility**: Works with newer Python versions (including Python 3.13)
- **Easy Setup**: Just install PySide6 and run
- **Same Look and Feel**: Provides the same visual experience as the full version
- **Predictable Behavior**: Predefined responses make testing and demos easier

## Differences from Full Lumina

- Uses predefined responses instead of language processing
- No actual language memory system or neural linguistic processing
- No consciousness mirror integration
- No persistent memory storage between sessions
- Process and glyph functions are simulated rather than integrated with backend systems

## Troubleshooting

If you encounter issues:

1. Make sure PySide6 is properly installed
2. Verify that you're using Python 3.8 or higher
3. Check that no other applications are using port 9000 (if using networked features)

## Next Steps

Once you have Mock Lumina working, you might consider:

1. Setting up a Python virtual environment for the full Lumina v6.5
2. Installing specific versions of dependencies required by the full system
3. Gradually adding components from the full system as you get them working 