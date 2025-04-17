# V5 PySide6 Client

A comprehensive PySide6-based client for the V5 Fractal Echo Visualization System, integrating with the Language Memory System.

## Overview

The V5 PySide6 Client is a modern, cross-platform application that provides a user-friendly interface to the V5 Fractal Echo Visualization System. It enables users to visualize neural network patterns as fractal structures, explore memory associations, and interact with the Language Memory System.

This client is part of the Lumina Neural Network System's journey from v1-v10, implementing the V5 visualization capabilities with a modern UI framework.

## Key Features

- **Fractal Pattern Visualization**: Visualize neural network patterns as interactive, animated fractal structures
- **Memory Integration**: Seamless integration with the Language Memory System
- **Neural Weight Control**: Dynamic adjustment of neural network influence on visualizations
- **Multiple Pattern Styles**: Support for different visualization styles (Neural, Mandelbrot, Julia, Tree)
- **Real-time Animation**: Smooth, animated visualizations that respond to pattern changes
- **Metrics Dashboard**: Display of fractal dimensions, complexity indices, and pattern coherence metrics
- **Mock Mode**: Built-in mock data generation for testing without backend services
- **Theme Support**: Light, dark, and system theme options
- **Responsive UI**: Adaptive interface that works across different screen sizes

## Requirements

- **Python 3.8+**
- **PySide6 6.2.0+** (primary UI framework)
- **NetworkX** (for network visualization)
- **NumPy** (for numerical operations)
- **Graphviz** (for generating integration diagrams)
- Additional dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/neural_network_project.git
   cd neural_network_project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify PySide6 installation**:
   ```bash
   python -c "from PySide6.QtWidgets import QApplication; print('PySide6 installed successfully')"
   ```

## Usage

### Running the Client

```bash
python v5_pyside6_client.py
```

### Command-line Options

- `--mock`: Use mock mode for testing without backend services
- `--no-plugins`: Disable plugin discovery and loading
- `--theme [light|dark|system]`: Set the UI theme
- `--debug`: Enable debug logging

Example:
```bash
python v5_pyside6_client.py --mock --theme dark
```

## Architecture

The V5 PySide6 Client is built with a modular architecture that separates core functionality, UI components, and bridge components.

```
v5_client/
│
├── core/               # Core functionality
│   ├── socket_manager.py   # Communication with backend services
│   └── ...
│
├── bridge/             # Bridge components for external systems
│   ├── language_memory_bridge.py  # Bridge to Language Memory System
│   └── ...
│
├── ui/                 # UI components
│   ├── main_window.py      # Main application window
│   ├── theme_manager.py    # Theme management
│   │
│   └── panels/             # Specialized visualization panels
│       ├── fractal_pattern_panel.py    # Fractal pattern visualization
│       ├── memory_synthesis_panel.py   # Memory synthesis visualization
│       ├── node_consciousness_panel.py # Node consciousness metrics
│       └── conversation_panel.py       # Conversation interface
│
└── v5_pyside6_client.py   # Main application entry point
```

### Integration with Backend Systems

The client integrates with the following backend components:

1. **V5 Fractal Echo Visualization System**:
   - Connection via Socket Manager
   - Pattern data exchange
   - Visualization of neural patterns

2. **Language Memory System**:
   - Memory retrieval and storage
   - Topic synthesis
   - Memory-enhanced messaging

3. **Neural Network Core**:
   - Neural state processing
   - Weight adjustments
   - Node consciousness metrics

### Communication Flow

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Language Memory │     │  V5 PySide6       │     │  V5 Fractal Echo │
│  System          │◄────┤  Client           │────►│  Visualization   │
└─────────────────┘     └───────────────────┘     └─────────────────┘
                                 ▲
                                 │
                                 ▼
                        ┌───────────────────┐
                        │  Neural Network   │
                        │  Core             │
                        └───────────────────┘
```

## Panel Types

The client includes several specialized panels for different visualization needs:

1. **Fractal Pattern Panel**:
   - Visualizes neural patterns as fractal structures
   - Supports multiple pattern styles (Neural, Mandelbrot, Julia, Tree)
   - Displays metrics and insights about the patterns

2. **Memory Synthesis Panel**:
   - Visualizes memory associations and connections
   - Allows searching and exploring memory topics
   - Shows memory synthesis results

3. **Node Consciousness Panel**:
   - Displays node consciousness metrics
   - Shows integration and differentiation scores
   - Visualizes node relationships

4. **Conversation Panel**:
   - Provides an interface for conversation with memory enhancement
   - Controls for neural weight adjustment
   - Support for different memory modes (contextual, combined, synthesized)

## Development

### Mock Mode

The client supports a mock mode that generates realistic test data without requiring backend services. This is useful for development and testing:

```bash
python v5_pyside6_client.py --mock
```

### Adding New Panels

1. Create a new panel class in `v5_client/ui/panels/`
2. Implement the required methods (`get_panel_name`, `get_panel_description`)
3. Register the panel in `v5_client/ui/panels/__init__.py`

### Theme Customization

The client supports multiple themes (light, dark, system) managed by the `ThemeManager` class. Custom themes can be added by extending this class.

## Troubleshooting

### Common Issues

1. **PySide6 Installation Issues**:
   - Ensure you have the correct Python version (3.8+)
   - Try reinstalling: `pip install --force-reinstall PySide6`

2. **Connection Errors**:
   - Check if backend services are running
   - Use mock mode for testing: `python v5_pyside6_client.py --mock`

3. **Visualization Not Updating**:
   - Check log files in the `logs/` directory
   - Ensure socket connections are established

## License

This software is part of the Lumina Neural Network System and is subject to its licensing terms.

## Acknowledgments

The V5 PySide6 Client builds upon the work of the Lumina Neural Network System, incorporating the V5 Fractal Echo Visualization System and the Language Memory System.

---

For more information about the overall Lumina Neural Network System, see `MASTERreadme.md`. 