# Mistral Chat with Onsite Memory

This application provides a GUI interface for interacting with Mistral, enhanced with memory capabilities. The system stores conversations and knowledge entries locally, providing context-aware responses based on past interactions.

## Features

- **Conversation Memory**: The application remembers past conversations and uses them to provide more relevant responses
- **Knowledge Base**: Users can add knowledge entries that the system can reference in future conversations
- **Context-Aware Responses**: When memory is enabled, the system searches for relevant context before generating responses
- **Neural-Linguistic Processing**: When available, the system processes text through neural-linguistic models to enhance understanding
- **Database Integration**: Can integrate with a wider language database system if available

## Running the Application

Two batch files are provided for running the application:

1. **run_mistral_memory.bat**: Runs the application in mock mode (no API key needed)
2. **run_mistral_memory_with_key.bat**: Runs the application with a provided API key

To run with your own API key:
```
run_mistral_memory_with_key.bat YOUR_API_KEY_HERE
```

## System Requirements

- Python 3.8 or higher
- PySide6 (will be installed automatically if missing)

## Interface

The application provides two main tabs:

### Chat Tab
- Message input area for conversing with the assistant
- Options to enable/disable memory usage
- Ability to add knowledge entries directly from the chat interface
- View memory statistics

### Memory Tab
- View and manage knowledge entries
- View and manage conversation history
- Add, view, and delete entries

## Memory Integration

The application uses a two-tier memory system:

1. **File-based Storage**: All conversations and knowledge are stored in JSON files in the `data/onsite_memory` directory
2. **Database Integration** (if available): Can synchronize with a central language database system

## Environment Variables

The following environment variables can be set to configure the application:

- `MISTRAL_API_KEY`: Your Mistral API key
- `MOCK_MODE`: Set to "True" to use mock responses instead of calling the Mistral API
- `NN_WEIGHT`: Neural network weight (0.0-1.0) for processing
- `LLM_WEIGHT`: LLM weight (0.0-1.0) for processing

## Adding Knowledge

Knowledge can be added in several ways:

1. Through the "Add Knowledge Entry" button in the Chat tab
2. Through the "Add Entry" button in the Memory tab
3. Programmatically through the memory system API

Each knowledge entry includes:
- Topic
- Content
- Source (optional)
- Timestamp

## Directory Structure

- `data/onsite_memory`: Storage for memory files
- `data/memory`: Additional memory storage for language system
- `data/logs`: Application logs
- `data/neural_linguistic`: Neural-linguistic processing data

## Advanced Integration

The system can integrate with:

1. Language Database system (if available)
2. Central Language Node (if available)
3. Neural-Linguistic Processing (if available)

When these systems are available, they enhance the memory capabilities and provide metrics like consciousness level and neural-linguistic score. 