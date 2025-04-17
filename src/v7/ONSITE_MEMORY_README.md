# Onsite Memory System for Mistral Integration

## Overview

The Onsite Memory System provides persistent, local storage for conversation history and knowledge in the Mistral AI integration. It enables the Mistral AI Chat application to remember past conversations, store important knowledge, and use this memory to provide more contextually relevant responses.

## Key Features

- **Conversation History Storage**: Automatically stores all chat conversations for future reference
- **Knowledge Base**: Maintains a dictionary of knowledge entries that can be searched and retrieved
- **Context Enhancement**: Uses stored memories to enhance responses to user queries
- **User Preferences**: Stores user settings and preferences
- **Persistent Storage**: All data is stored locally on disk and persists between sessions

## Components

### OnsiteMemory Class

The core memory system that provides:

- Memory storage and retrieval
- Conversation history management
- Knowledge dictionary operations
- Search capabilities
- Preference storage
- Automatic saving

### OnsiteMemoryIntegration

Integration between the memory system and the PySide6 UI:

- Connects the memory system to the Mistral Chat application
- Provides methods to add conversations and knowledge
- Implements context enhancement for queries
- Manages the memory UI panel

### OnsiteMemoryPanel

UI component for the memory system:

- Displays and manages knowledge entries
- Shows conversation history
- Provides search functionality
- Configures memory settings
- Displays memory statistics

## Usage

### Running the Application

To run the Mistral Chat application with the onsite memory system:

```bash
python run_memory_app.py
```

### Using Memory in Chat

1. When the "Use Memory for Context" option is enabled, the system will:
   - Search for relevant knowledge and past conversations when you ask a question
   - Enhance your query with context from memory when relevant matches are found
   - Add your conversations to memory automatically

2. To extract knowledge from conversations:
   - Click the "Extract Knowledge from Last Chat" button
   - Enter a topic/keyword for the knowledge
   - The knowledge will be added to your memory for future reference

3. To manually add knowledge:
   - Go to the Memory tab
   - Click "Add Knowledge" on the Knowledge panel
   - Enter the topic, content, and optional source
   - Click OK to save

### Memory Tab

The Memory tab provides access to:

- **Knowledge Panel**: View, search, add, and delete knowledge entries
- **Conversations Panel**: View and search your conversation history
- **Preferences Panel**: Configure memory settings like auto-save

## File Structure

- `src/v7/onsite_memory.py`: The core memory system implementation
- `src/v7/ui/onsite_memory_integration.py`: Integration with the PySide6 UI
- `src/v7/ui/mistral_pyside_app.py`: Mistral Chat application with memory integration

## Storage Location

By default, memory is stored in:
- `data/onsite_memory/mistral_memory.json`

This file contains conversation history, knowledge entries, and preferences in JSON format.

## Configuration

Memory settings that can be configured:

- **Auto-save**: Enable/disable automatic saving of memory
- **Save interval**: How often to auto-save (in seconds)
- **Max conversations**: Maximum number of conversations to keep
- **Max knowledge entries**: Maximum number of knowledge entries to store

These settings can be adjusted in the Preferences panel on the Memory tab.

## Development

### Adding to the Memory System

To add functionality to the memory system:

1. Extend the `OnsiteMemory` class in `src/v7/onsite_memory.py`
2. Update the integration in `src/v7/ui/onsite_memory_integration.py`
3. Add UI components as needed in the `OnsiteMemoryPanel` class

### Testing

Use the provided test scripts to verify functionality:

- `test_onsite_memory.py`: Comprehensive test of all memory features
- `simple_memory_test.py`: Simple test of basic functionality

## Technical Details

- **Threading**: The memory system uses a background thread for auto-saving
- **Search Algorithm**: Uses relevance scoring based on keyword matches
- **Storage Format**: JSON with UTF-8 encoding
- **Error Handling**: Comprehensive error handling with logging

## Requirements

- Python 3.6+
- PySide6
- mistralai client library 