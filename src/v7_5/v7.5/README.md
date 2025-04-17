# LUMINA v7.5 - Integrated Neural Interface

LUMINA v7.5 is a comprehensive PySide6-based frontend that integrates with the CI/CD pipeline, providing a unified interface for interaction with the Lumina v8 system's capabilities.

## Overview

The Lumina v7.5 chat interface provides:

1. A clean, modern interface for communicating with the Lumina system
2. Conversation memory that tracks topics across exchanges
3. Integration with the CI/CD pipeline status and components
4. System state monitoring in real-time
5. Support for both connected and mock modes

## Features

- **Complete System Integration**: Connects with all v7 components including consciousness nodes, autowiki, breath detection, and memory systems
- **Intuitive Chat Interface**: Simple yet powerful chat interface for communicating with the system
- **Glyph Activation**: Activate special glyphs to trigger specific neural patterns and system responses
- **Process Controls**: Access to system processes like Breathe, Resonance, and Echo
- **Neural Network Visualization**: Visual representation of the active neural network
- **Persistent Memory**: Integration with onsite memory for continuous learning

## Components

The system consists of the following core components:

- **Conversation Flow**: Manages conversation context and topic tracking
- **System Integration**: Interfaces with all system components with graceful degradation
- **Frontend UI**: PySide6-based interface with chat and system monitoring

## CI/CD Integration

The v7.5 chat interface has been integrated into the Lumina v8 CI/CD pipeline to provide:

1. Real-time status updates on pipeline processes
2. Access to system component information
3. Communication about test results and system state
4. Command-based interaction with pipeline components

## Usage

The chat interface can be launched directly from the CI/CD pipeline by selecting option 5 in the run_complete_knowledge_cycle.bat script:

```
Choose your startup configuration:
 [1] Start all components in integrated mode
 [2] Start root connection system only
 [3] Start complete cycle with visualization
 [4] Run full test suite only (no component startup)
 [5] Start Lumina v7.5 Chat Interface
```

Alternatively, you can start it directly with:

```bash
python src/v7.5/lumina_frontend.py
```

## API Key Configuration

The system will attempt to use a Mistral API key if available, looking in:

1. Environment variables (`MISTRAL_API_KEY`)
2. The `.env` file in the project root

If no key is found, the system will run in mock mode, providing simulated responses.

## Directory Structure

- `src/v7.5/`
  - `lumina_frontend.py`: Main PySide6 application
  - `conversation_flow.py`: Conversation context and topic tracking
  - `system_integration.py`: System component integration with fallbacks
  - `config.py`: Configuration settings

## Development

To extend the chat functionality:

1. Add new components to `system_integration.py`
2. Update the UI in `lumina_frontend.py` to display new functionality
3. Extend conversation capabilities in `conversation_flow.py`

## Troubleshooting

If you encounter issues:

1. Check the log files in `logs/v7.5_frontend.log`
2. Verify the Mistral API key if connection is failing
3. Ensure all required Python packages are installed
4. Check if PySide6 is properly installed

## System Requirements

- Python 3.8+
- PySide6
- Additional dependencies: pyqtgraph, matplotlib, numpy, pandas

## Integration

LUMINA v7.5 integrates with the following system components:

- Enhanced Language Mistral Integration
- Onsite Memory System
- AutoWiki Plugin
- V7 Breath Detector
- Node Consciousness System

Each component is loaded dynamically, allowing the system to function even if some components are unavailable, though with limited functionality.

## Future Enhancements

- Advanced neural visualization with real-time updates
- Integration with external APIs and data sources
- Expanded glyph library and effects
- Voice interaction capabilities
- Advanced consciousness metrics display 