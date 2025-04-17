# V5 Language Memory Integration Guide

## Overview

The V5 Language Memory System provides a framework for visualizing neural network activity, language pattern processing, and real-time memory integration. This guide outlines how to integrate with the system components and leverage its visualization capabilities.

## Architecture

The V5 system consists of the following core components:

- **Central Language Node**: Main coordination point for language processing
- **Socket Manager**: Communication hub between components
- **UI Framework**: Qt-based visualization panels and widgets
- **Pattern Processors**: Specialized modules for processing different types of patterns
- **API Services**: External service integrations

## Socket Manager Integration

The Socket Manager acts as the central communication hub:

```python
from src.v5.socket_manager import SocketManager

# Create socket manager instance
socket_manager = SocketManager()

# Register a plugin
socket_manager.register_plugin(my_plugin)

# Send messages
socket_manager.send_message(message_type="pattern_update", data=pattern_data)
```

## Creating Plugins

Plugins allow you to extend the system's functionality:

```python
class MyPlugin:
    def __init__(self, plugin_id="my_plugin"):
        self.plugin_id = plugin_id
        
    def get_plugin_id(self):
        return self.plugin_id
        
    def process_message(self, message_type, data):
        if message_type == "pattern_update":
            # Process the pattern update
            pass
            
    def get_status(self):
        # Return plugin status
        return {
            "status": "active",
            "metrics": {
                "processed_patterns": 150,
                "active_connections": 3
            }
        }
```

## UI Integration

The V5 UI system supports both PyQt5 and PySide6 through a compatibility layer:

```python
# Import Qt compatibility layer
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt, Signal, Slot
from src.v5.ui.qt_compat import get_widgets, get_gui, get_core

# Get specific widgets
QSplitter = get_widgets().QSplitter
QPainter = get_gui().QPainter
```

To create a new panel:

```python
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Signal, Slot
from src.v5.ui.panels.base_panel import BasePanel

class MyCustomPanel(BasePanel):
    def __init__(self, socket_manager, parent=None):
        super().__init__(socket_manager, parent)
        self.setObjectName("my_custom_panel")
        
        # Initialize UI
        self._init_ui()
        
        # Connect to socket manager
        self._connect_signals()
        
    def _init_ui(self):
        # Create layout
        self.main_layout = QtWidgets.QVBoxLayout(self)
        
        # Add widgets
        self.label = QtWidgets.QLabel("My Custom Panel")
        self.main_layout.addWidget(self.label)
        
    def _connect_signals(self):
        # Connect to socket manager signals
        self.socket_manager.message_received.connect(self._handle_message)
        
    def _handle_message(self, message_type, data):
        if message_type == "my_update_type":
            # Update UI based on message
            pass
```

## Pattern Visualization

### Fractal Pattern Visualization

The system includes support for various fractal patterns:

```python
from src.v5.ui.panels.fractal_pattern_panel import FractalPatternPanel

# Create panel instance
panel = FractalPatternPanel(socket_manager)

# Add to main window
main_window.addPanel(panel)

# Send pattern update
pattern_data = {
    "pattern_type": "neural",
    "nodes": [...],  # Node data
    "connections": [...],  # Connection data
    "parameters": {
        "depth": 5,
        "scale": 1.2,
        "animation_speed": 0.8
    }
}
socket_manager.send_message("pattern_update", pattern_data)
```

Available pattern types:
- `neural`: Neural network-inspired patterns
- `mandelbrot`: Mandelbrot set visualization
- `julia`: Julia set patterns
- `tree`: Tree-based fractal patterns

## Message Types

The V5 system defines several message types:

- `pattern_update`: Update to a pattern visualization
- `language_node_status`: Status update from language processing
- `memory_integration`: Memory integration event
- `api_service_result`: Results from API service calls
- `consciousness_metric`: Analytics on language consciousness

Example of sending a pattern update:

```python
socket_manager.send_message(
    message_type="pattern_update",
    data={
        "pattern_type": "neural",
        "nodes": [
            {"id": "node1", "activation": 0.8, "position": [0.5, 0.3]},
            {"id": "node2", "activation": 0.4, "position": [0.7, 0.6]}
        ],
        "connections": [
            {"source": "node1", "target": "node2", "strength": 0.6}
        ],
        "parameters": {
            "depth": 4,
            "scale": 1.0,
            "animation_speed": 0.5
        }
    }
)
```

## Language Memory Integration

To integrate with the language memory system:

```python
from src.language_memory_synthesis_integration import LanguageMemorySynthesisIntegration

# Create integration instance
memory_integration = LanguageMemorySynthesisIntegration(socket_manager)

# Process text
memory_integration.process_text("Sample text for processing")

# Get memory patterns
patterns = memory_integration.get_memory_patterns()
```

## API Service Integration

To integrate with external API services:

```python
from src.v5.api_service_plugin import APIServicePlugin

# Create API service plugin
api_service = APIServicePlugin(
    plugin_id="my_api_service",
    base_url="https://api.example.com/v1",
    auth_token="your_auth_token"
)

# Register with socket manager
socket_manager.register_plugin(api_service)

# Make API request
api_service.make_request(
    endpoint="/process",
    method="POST",
    data={"text": "Sample text for processing"}
)
```

## Consciousness Analytics

To use the consciousness analytics features:

```python
from src.v5.consciousness_analytics_plugin import ConsciousnessAnalyticsPlugin

# Create plugin instance
analytics = ConsciousnessAnalyticsPlugin()

# Register with socket manager
socket_manager.register_plugin(analytics)

# Get consciousness metrics
metrics = analytics.get_metrics()
```

## Main Application Integration

To integrate all components into a main application:

```python
from src.v5.ui.qt_compat import QtWidgets, QtCore
from src.v5.ui.main_widget import V5MainWidget
from src.v5.socket_manager import SocketManager

# Create application
app = QtWidgets.QApplication([])

# Create socket manager
socket_manager = SocketManager()

# Register plugins
socket_manager.register_plugin(my_plugin)
socket_manager.register_plugin(api_service)
socket_manager.register_plugin(analytics)

# Create main widget
main_widget = V5MainWidget(socket_manager)
main_widget.show()

# Run application
app.exec_()
```

## Pattern Data Format

Pattern data should be structured as follows:

```python
{
    "pattern_type": "neural",  # or "mandelbrot", "julia", "tree"
    "nodes": [
        {
            "id": "node_id", 
            "activation": 0.8,  # 0.0 to 1.0
            "position": [0.5, 0.3],  # x, y in 0.0 to 1.0 range
            "attributes": {...}  # Additional attributes
        },
        # More nodes...
    ],
    "connections": [
        {
            "source": "source_node_id",
            "target": "target_node_id",
            "strength": 0.6,  # 0.0 to 1.0
            "attributes": {...}  # Additional attributes
        },
        # More connections...
    ],
    "parameters": {
        "depth": 5,  # Fractal recursion depth
        "scale": 1.2,  # Scale factor
        "animation_speed": 0.8,  # Animation speed (0.0 to 1.0)
        "color_scheme": "neural",  # Color scheme name
        # Additional pattern-specific parameters
    }
}
```

## Best Practices

1. **Message Handling**: Register for specific message types to avoid processing irrelevant messages
2. **UI Responsiveness**: Use background processing for heavy computations
3. **Plugin Design**: Keep plugins focused on a single responsibility
4. **Error Handling**: Implement robust error handling in all components
5. **Resource Management**: Release resources when panels are closed
6. **Configuration**: Use configuration files for customizable settings

## Troubleshooting

- **Socket Connection Issues**: Check socket manager initialization and plugin registration
- **UI Rendering Problems**: Verify Qt compatibility layer initialization
- **Pattern Processing Errors**: Check pattern data format and parameter values
- **API Service Failures**: Verify network connectivity and API credentials
- **Performance Issues**: Monitor processing time in plugins and UI updates

## Qt Compatibility

The V5 system supports both PyQt5 and PySide6 through its compatibility layer:

```python
# Check active Qt framework
from src.v5.ui.qt_compat import ACTIVE_BINDING
print(f"Using Qt framework: {ACTIVE_BINDING}")

# Import specific modules
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui
from src.v5.ui.qt_compat import get_widgets, get_gui, get_core

# Get specific widget classes
QSplitter = get_widgets().QSplitter
QPainter = get_gui().QPainter
QPoint = get_core().QPoint
``` 