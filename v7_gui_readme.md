# LUMINA V7 GUI Setup

This README provides instructions for setting up and running the LUMINA V7 GUI with Enhanced Language Integration and Mistral capabilities.

## Quick Start

To launch the V7 GUI with Enhanced Mistral Integration:

```
.\run_v7_enhanced_ui.bat
```

To use with your own Mistral API key:

```
.\run_v7_enhanced_ui.bat YOUR_API_KEY
```

## Available Launch Options

### 1. Enhanced Mistral Demo (Recommended)
- **File**: `run_v7_enhanced_ui.bat`
- **Description**: Launches the Enhanced Language Integration with Mistral capabilities in interactive mode
- **Features**: Interactive chat, learning capabilities, memory integration

### 2. Standard V7 GUI (May require additional configuration)
- **File**: `run_v7_gui.bat`
- **Description**: Attempts to launch the core V7 GUI system
- **Note**: This may require additional module setup as it depends on the full V7 system being available

### 3. V7 with Mistral Integration (May require additional configuration)
- **File**: `run_v7_mistral.bat`
- **Description**: Launches V7 with Mistral integration and autowiki features
- **Optional parameter**: Mistral API key

## Directory Structure

The V7 system relies on the following directory structure:

```
/
├── src/
│   └── v7/               - V7 system components
│       ├── ui/           - UI components
│       ├── data/         - Data files
│       └── docs/         - Documentation
├── data/                 - Runtime data
│   ├── onsite_memory/    - Memory storage
│   ├── neural_linguistic/ - Neural linguistic data
│   └── demo/             - Demo data
├── logs/                 - Log files
```

## Requirements

- Python 3.8 or higher
- PySide6 (installed automatically if missing)
- Internet connection for Mistral API (if using API key)

## UI Features

The Enhanced Mistral Integration provides:

1. **Chat Interface**: Interactive communication with the language system
2. **Memory Integration**: Remembers conversation history and context
3. **Learning Capabilities**: Can learn and improve over time
4. **Neural Linguistic Analysis**: Analyzes language patterns and consciousness levels
5. **AutoWiki Support**: Can access and learn from knowledge sources

## Troubleshooting

If you encounter issues when launching the V7 GUI:

1. Check the log files in the `logs/` directory
2. Verify Python and PySide6 are properly installed
3. Ensure your Mistral API key is valid (if using one)
4. Make sure all required directories exist
5. Check that the PYTHONPATH includes both the current directory and the src directory

## Advanced Configuration

For advanced configuration, you can modify:

- **Neural Network Weight**: Controls the influence of neural components (default: 0.6)
- **Language Model Weight**: Controls the influence of language models (default: 0.7)
- **Model Selection**: Different Mistral models can be specified (default: mistral-medium)

These settings can be adjusted in the batch files or passed as parameters to the scripts.

## Additional Resources

For more information about specific components:

- See `src/v7/languageReadme.md` for details on the language integration
- See `src/v7/README_MISTRAL_INTEGRATION.md` for Mistral integration details
- See `src/v7/ONSITE_MEMORY_README.md` for memory system documentation 