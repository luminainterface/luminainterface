# Chat-Language Memory Integration

This integration connects the V5 Neural Network/LLM Weighted Conversation Panel with the Language Memory System, allowing for a seamless interaction between conversation components and memory systems.

## Overview

The integration enables conversations to be:
- Processed with variable neural network weighting
- Stored in the Language Memory System
- Retrieved with context-aware memory recall
- Enhanced with neural linguistic processing

## Components

### 1. Chat Memory Interface

The `ChatMemoryInterface` class (`src/chat_memory_interface.py`) serves as the primary connection point between conversation components and the Memory API. It provides methods for:

- Processing messages with different neural weights
- Generating responses based on memory modes
- Storing conversations in the memory system
- Retrieving relevant memories based on context
- Obtaining memory statistics

### 2. Language Memory V5 Bridge

The `LanguageMemoryV5Bridge` class (`language_memory_v5_bridge.py`) creates a bridge between the Language Memory System and the V5 Visualization System. It:

- Connects to the Memory API Socket
- Processes messages from both systems
- Handles memory retrieval and storage
- Generates fractal patterns based on memory content
- Provides caching for performance optimization

### 3. Test Script

The `run_chat_language_integration.py` script allows testing of the integration with three UI modes:

- **Text UI**: Console-based interface for testing
- **Qt UI**: Graphical interface using either PySide6 or PyQt5
- **Headless**: No UI, just logs processing results

## Usage

### Basic Usage

```bash
# Run with text UI (default)
python run_chat_language_integration.py

# Run with Qt UI
python run_chat_language_integration.py --ui qt

# Run in mock mode for testing without actual backend
python run_chat_language_integration.py --mock

# Run with custom test messages
python run_chat_language_integration.py --test-messages "Hello,How are neural networks related to memory"

# Run with custom neural network weights
python run_chat_language_integration.py --weights "0.1,0.5,0.9"
```

### Interactive Mode

In Text UI mode, after the predefined tests run, you can interact with the system using this format:

```
message | neural_weight | memory_mode
```

For example:
```
Tell me about neural networks | 0.7 | combined
```

Where:
- `neural_weight` is a float between 0 and 1 (default: 0.5)
- `memory_mode` is one of: "contextual", "synthesized", "combined" (default: "combined")

## Memory Modes

- **Contextual (NN weight < 0.3)**: Primarily uses language model with minimal neural network influence
- **Combined (NN weight 0.3-0.7)**: Balanced approach using both language model and neural network
- **Synthesized (NN weight > 0.7)**: Heavily relies on neural network processing with memory synthesis

## Integration Architecture

```
+---------------------+        +-------------------------+
| Conversation Panel  |<------>| Chat Memory Interface   |
| (V5 UI Component)   |        | (Integration Layer)     |
+---------------------+        +------------|------------+
                                            |
                               +------------v------------+
                               | Language Memory System  |
                               | (Memory API)            |
                               +------------|------------+
                                            |
                               +------------v------------+
                               | Language Memory Bridge  |
                               | (V5 Integration)        |
                               +------------|------------+
                                            |
                               +------------v------------+
                               | V5 Visualization System |
                               | (Fractal Echo)          |
                               +-------------------------+
```

## Development

### Adding New Memory Processing Modes

1. Extend the `process_message` method in `ChatMemoryInterface`
2. Implement the new mode logic
3. Update the memory mode selection in the test script

### Extending UI Support

The integration supports both text and graphical interfaces. To add new UI:

1. Create a new panel class that accepts the `chat_interface` parameter
2. Implement the user interaction flow
3. Add the new UI option to the test script's argument parser

## Troubleshooting

Common issues:

1. **Connection failures**: Ensure the Memory API is running and accessible
2. **Import errors**: Check that all dependencies are installed
3. **UI rendering issues**: Try switching between PyQt5 and PySide6 using `--ui qt`
4. **Slow response times**: Enable mock mode for testing without backend dependencies

## License

This software is part of the Lumina Neural Network System and is subject to the same licensing as the core system. 