# Enhanced Language System - PySide6 Integration

This document explains how to run and use the Enhanced Language System with the PySide6 GUI interface.

## Overview

The Enhanced Language System has been adapted to work with PySide6, providing a modern and responsive GUI interface for interacting with the system's components:

- **Language Memory**: Stores and retrieves word associations and language patterns
- **Neural Linguistic Processor**: Analyzes text for patterns and semantic networks
- **Conscious Mirror Language**: Models consciousness in language processing
- **Central Language Node**: Integrates all components with unified LLM weighing

## Prerequisites

To use the PySide6 version of the Enhanced Language System, you'll need:

1. Python 3.7 or higher
2. PySide6
3. Matplotlib (optional, for visualizations)
4. NetworkX (optional, for graph visualizations)

## Installation

1. Install PySide6 and other required packages:

```bash
pip install PySide6 matplotlib networkx
```

2. Make sure your `.env` file is properly configured (should include any API keys or settings needed)

## Running the Application

To start the PySide6 GUI application, run:

```bash
python -m src.language.main_pyside_app
```

Or from the project root:

```bash
python src/language/main_pyside_app.py
```

## Using the GUI

The application provides several tabs for interacting with the Enhanced Language System:

### Text Processing Tab

This tab allows you to process text through the language system:

1. Enter text in the input field
2. Choose whether to use consciousness and neural linguistic processing
3. Click "Process Text" to analyze the text
4. View the results, including unified language score, neural linguistic score, consciousness level, and detailed analysis

### LLM Weight Control Tab

This tab allows you to adjust the LLM weight across all components:

1. Use the slider to set the desired LLM weight (0.0 to 1.0)
2. Click "Apply Weight to All Components" to synchronize the weight
3. View the current weights for each component
4. Experiment with different weights to see how they affect the system's behavior

## LLM Weight Effects

The LLM weight controls the influence of Large Language Model suggestions on the system's operation:

- **0.0**: No LLM influence, pure algorithmic processing
- **0.2**: Minimal LLM influence, primarily algorithmic
- **0.5**: Balanced between algorithmic and LLM processing
- **0.8**: Strong LLM influence with algorithmic grounding
- **1.0**: Maximum LLM influence, minimal algorithmic constraints

## Visualizations

The system includes several visualization tools:

1. **Semantic Network Visualization**: Displays word associations and connections
2. **Consciousness Level Chart**: Shows consciousness levels over time
3. **LLM Weight Effects Chart**: Illustrates how different LLM weights affect scores

To access visualizations, process text through the system and view the results panel.

## Troubleshooting

If you encounter issues:

1. **PySide6 not found**: Make sure PySide6 is installed with `pip install PySide6`
2. **Components not initializing**: Check your `.env` file and ensure data directories exist
3. **Visualization errors**: Install matplotlib and networkx with `pip install matplotlib networkx`
4. **Slow performance**: Consider reducing the LLM weight to decrease external API calls

## Development

To extend the PySide6 application:

1. The adapter classes in `src/language/pyside6_adapter.py` bridge between the core components and the GUI
2. Add new visualizations in `src/language/visualization_utils.py`
3. Extend the main application in `src/language/main_pyside_app.py`

Each adapter handles:
- Non-blocking processing using worker threads
- Signal-based communication for UI updates
- Data conversion for visualization
- Error handling and status reporting

## Architecture

The PySide6 integration follows this architecture:

```
                  +-------------------+
                  | PySide6 GUI       |
                  |   (main_pyside_app)|
                  +--------+----------+
                           |
                  +--------v----------+
                  | PySide6 Adapters  |
                  | (pyside6_adapter) |
                  +--------+----------+
                           |
           +---------------+----------------+
           |               |                |
+----------v------+ +------v--------+ +----v-------------+
| Language Memory | | Neural        | | Conscious Mirror |
|                 | | Linguistic    | | Language         |
+-----------------+ +---------------+ +------------------+
           |               |                |
           +---------------v----------------+
                           |
                  +--------v----------+
                  | Central Language  |
                  | Node              |
                  +-------------------+
```

## License

This software is provided under the same license as the core Enhanced Language System. 