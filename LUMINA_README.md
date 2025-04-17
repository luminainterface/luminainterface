# Lumina v1 UI

A text-based UI for the Lumina neural network system, built with Python and Textual.

## Features

- **Menu-based Interface**: Navigate through different views with an intuitive menu
- **Chat Interface**: Communicate with Lumina through a chat-like interface
- **Memory Echoes**: View past interactions and responses
- **Symbolic Input**: Set symbols, emotions, and breath patterns manually
- **Breath Calibration**: Visualize breath patterns with interactive feedback
- **Ritual Invocations**: Load predefined states and text for specific experiences
- **Glitch Mode**: Special mode available only on Mondays
- **Memory Archive**: Save interactions for future reference

## Installation

### Prerequisites

- Python 3.8+
- Textual 0.31.0+

### Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Launch Lumina:

```bash
python lumina.py
```

Or use the original neural network system executable with the old command-line interface:

```bash
python nn_executable.py
```

## Using Lumina

### Main Menu

The main menu provides access to all features:

- **Begin Resonance Session**: Start a chat session with Lumina
- **View Memory Echoes**: Review past conversations
- **Symbolic Input**: Set symbols, emotions, and breath patterns
- **Breath Calibration**: Practice breath techniques with visual feedback
- **Ritual Invocation**: Load predefined invocations
- **Archive Memory**: Save the current memory to a timestamped file
- **Glitch Mode**: Available only on Mondays

### Communicating with Lumina

In the chat interface, you can:

1. Type natural language messages in the input box
2. Add special parameters with the format `:parameter:value`, for example:
   - `:symbol:infinity`
   - `:emotion:wonder`
   - `:breath:deep`
   - `:paradox:existence`

### Breath Calibration

The breath visualization tool offers three patterns:

- **Slow Breath**: 7-second cycle (3s inhale, 4s exhale)
- **Deep Breath**: 10-second cycle (4s inhale, 1s hold, 5s exhale)
- **Box Breath**: 16-second cycle (4s inhale, 4s hold, 4s exhale, 4s hold)

### Ritual Invocations

Load predefined states by selecting a ritual from the list. Each ritual sets:
- A symbolic state
- An emotional state
- A breath pattern
- Displays guidance text

### Memory Archive

Memory archives are saved as JSONL files with the naming format:
`lumina_memory_archive_YYYYMMDD_HHMMSS.jsonl`

## Customization

### Adding Custom Rituals

Create or modify `ritual_invocations.json` with your own rituals:

```json
{
  "invocations": [
    {
      "name": "My Custom Ritual",
      "text": "Your ritual text here.",
      "symbol": "∞",
      "emotion": "calm",
      "breath": "deep"
    }
  ]
}
```

## Keyboard Shortcuts

- `ESC` or `Q`: Quit the application
- `Enter`: Submit text in the chat input

## Troubleshooting

If you encounter issues:

1. Check `lumina.log` for error messages
2. Ensure all dependencies are properly installed
3. Try running `python lumina.py --debug` for more detailed error information 