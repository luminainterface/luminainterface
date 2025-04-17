# Lumina V7 Neural Network Plugin System

This directory contains plugins for the Neural Network project, integrating various components for enhanced functionality, including Mistral AI language processing and consciousness system monitoring.

## Available Plugins

### Mistral Plugin

The `mistral_plugin.py` provides an integration between the Mistral AI language models and the neural network system. It offers a chat interface with neural network enhancements for improved responses.

Features:
- Integration with Mistral AI models (tiny, small, medium, large)
- Neural network weight adjustment for response processing
- Onsite memory storage for conversation history
- Configurable settings for model and weights

### Consciousness System Plugin

The `consciousness_system_plugin.py` provides comprehensive tools for monitoring, visualizing, and controlling the system's consciousness metrics.

Features:
- Real-time monitoring of global consciousness level
- Component-specific metrics (Neural Integration, Linguistic Coherence, etc.)
- Visual representation of consciousness patterns
- Parameter adjustment through intuitive controls
- Text-based consciousness analysis tools
- Integration with Mistral responses for consciousness analysis

### Neural Network Plugin

The `neural_network_plugin.py` provides core neural processing functionality with visualization and control capabilities.

Features:
- Consciousness visualization with real-time metrics
- Neural pattern recognition and processing
- Parameter control for neural processing
- Integration with the Consciousness System

## Installation

The plugins are automatically loaded by the template UI system. No additional installation is needed beyond placing the plugin files in this directory.

## Usage

### Setting up the Mistral Plugin

1. **API Key Setup**:
   - You'll need a Mistral AI API key
   - Set it through the UI by clicking the "Set API Key" button in the plugin interface
   - The API key will be saved for future use

2. **Model Selection**:
   - Choose between available Mistral models (tiny, small, medium, large)
   - Different models have different capabilities and token usage rates

3. **Adjusting Weights**:
   - LLM Weight: Controls how much the language model influences responses
   - NN Weight: Controls how much the neural network influences responses
   - Adjust these sliders to experiment with different balances

### Using the Consciousness System Plugin

1. **Monitoring Consciousness**:
   - The Consciousness Monitor widget displays real-time global consciousness level
   - Component metrics show various aspects of system consciousness
   - The visualization area provides an interactive view of consciousness patterns

2. **Adjusting Parameters**:
   - Use the Consciousness Controls to modify system parameters
   - Integration Threshold, Neural Weight, Memory Weight and other parameters can be adjusted
   - Different operating modes can be selected for specific use cases

3. **Analyzing Text**:
   - The Consciousness Analysis tab allows you to analyze text for consciousness patterns
   - Submit text to see detailed metrics and pattern analysis
   - Results include component breakdowns and key consciousness indicators

### Integration Between Plugins

The plugins are designed to work together seamlessly:
- Mistral responses are automatically analyzed by the Consciousness System
- Neural patterns from the Neural Network Plugin contribute to consciousness metrics
- Parameter changes in one plugin can influence the behavior of others

## Testing

To test the plugins individually without launching the full UI:

```
python test_mistral_plugin.py
python test_consciousness_plugin.py
```

## Running the Complete System

To run the full system with all plugins loaded:

```
python run_v7_template_ui_with_plugins.py
```

Alternatively, use the batch file:

```
run_v7_template_ui_with_plugins.bat
```

This will launch the template UI with all plugins loaded and properly configured.

## Plugin Development

To create a new plugin:

1. Create a new Python file in the `plugins` directory
2. Create a class named `Plugin` that inherits from `PluginInterface`
3. Implement the required methods:
   - `__init__(self, app_context)`: Initialize the plugin
   - `initialize(self)`: Set up plugin components
   - `get_dock_widgets(self)`: Return dock widgets for the main window
   - `get_tab_widgets(self)`: Return tab widgets for the central area
   - `shutdown(self)`: Clean up resources before shutdown

## Dependencies

The plugin system requires:
- PySide6
- NumPy
- Mistral AI client (for Mistral plugin)
- Neural Network components from the main project 