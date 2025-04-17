# V5 Fractal Echo Visualization System: PySide6 Integration

## Overview

This documentation describes the integration of the V5 Fractal Echo Visualization system with PySide6, providing a seamless transition path from PyQt5 to PySide6. This system is a key component of the Lumina Neural Network journey toward v10, as described in the MASTERreadme.md.

## Key Features

- **Framework Compatibility**: Unified support for both PyQt5 and PySide6
- **Memory Visualization**: Visual representation of Language Memory System data
- **Fractal Pattern Display**: Neural network visualization using fractal patterns
- **Consciousness Metrics**: Real-time node consciousness metrics display
- **Unified API**: Consistent API across both frameworks
- **Thread-Safe Communication**: Thread-safe messaging between UI and backend

## System Architecture

The system architecture has been refined to support both PyQt5 and PySide6:

```
V5 Visualization System
├── Compatibility Layer
│   ├── QtCompat - Framework detection and abstraction
│   ├── UI Component Factory - Widget creation
│   └── Signal/Slot Bridges - Event handling
├── Core Components
│   ├── Language Memory Bridge - Integration with Language Memory System
│   ├── Node Socket System - Communication protocol
│   └── Frontend Socket Manager - Plugin management
├── UI Components
│   ├── Memory Synthesis Panel - Language memory visualization
│   ├── Fractal Pattern Panel - Pattern visualization
│   └── Consciousness Metrics Panel - Node consciousness display
└── Plugins
    ├── Neural State Plugin - Neural network state management
    ├── Pattern Processor Plugin - Fractal pattern processing
    └── Language Memory Integration Plugin - Memory synthesis
```

## Getting Started

### Prerequisites

- Python 3.8+
- PySide6 6.4.0+ or PyQt5 5.15.4+
- Other dependencies listed in requirements.txt

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural_network_project.git
   cd neural_network_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the unified launcher:
   ```bash
   python src/ui/v5_unified_run.py
   ```

### Framework Selection

You can specify which framework to use:

```bash
# Use PySide6 (default)
python src/ui/v5_unified_run.py --framework PySide6

# Use PyQt5
python src/ui/v5_unified_run.py --framework PyQt5
```

Or set the environment variable:

```bash
# Windows
set V5_QT_FRAMEWORK=PySide6
python src/ui/v5_unified_run.py

# Linux/macOS
export V5_QT_FRAMEWORK=PySide6
python src/ui/v5_unified_run.py
```

### Testing PySide6 Integration

To verify the PySide6 integration:

```bash
python test_v5_pyside6.py
```

This will run the test application using PySide6 and display diagnostic information.

## Framework Compatibility

### Qt Compatibility Layer

The compatibility layer provides seamless transitions between PyQt5 and PySide6:

```python
from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Signal

# Framework-agnostic widget creation
button = QtWidgets.QPushButton("Click Me")

# Framework-agnostic signals
class MyWidget(QtWidgets.QWidget):
    my_signal = Signal(str)
```

### Signal and Slot Handling

Signals and slots are handled differently in PyQt5 and PySide6. The compatibility layer provides a consistent interface:

```python
from src.v5.ui.qt_compat import Signal, Slot

class MyWidget(QtWidgets.QWidget):
    # Define signals
    value_changed = Signal(int)
    
    @Slot(int)
    def on_value_changed(self, value):
        # Handle value change
        pass
```

## Language Memory Integration

### Memory API Compatibility Layer

The Language Memory API compatibility layer provides a consistent interface for accessing the Language Memory System:

```python
from src.language_memory_api_compat import memory_api

# Process a message with memory context
response = memory_api.process_message(
    "Tell me about neural networks",
    user_id="user123",
    async_mode=True
)

# Synthesize a topic
synthesis = memory_api.synthesize_topic(
    "consciousness",
    depth=3,
    async_mode=True
)
```

### Memory Visualization

The Memory Synthesis Panel provides visualization of synthesized memories:

```python
from src.v5.ui.panels.memory_synthesis_panel import MemorySynthesisPanel

# Create panel
panel = MemorySynthesisPanel(socket_manager)

# Connect to signals
panel.memory_selected.connect(handle_memory_selection)

# Update visualization
panel.synthesize_memory("consciousness", depth=3)
```

## Node Socket System

The Node Socket system has been enhanced to support both PyQt5 and PySide6:

```python
from src.v5.node_socket import NodeSocket

# Create socket
socket = NodeSocket("my_plugin", "plugin")

# Register message handler
socket.register_message_handler("my_message_type", handle_message)

# Send message
socket.send_message({
    "type": "my_message_type",
    "content": {"key": "value"}
})
```

## Best Practices

1. **Use the Compatibility Layer**: Always import from the compatibility layer rather than directly from PyQt5 or PySide6.

2. **Framework-Agnostic Widgets**: Create widgets using the QtWidgets module from the compatibility layer.

3. **Signal/Slot Connection**: Use the standard connect method for connecting signals to slots.

4. **Thread Safety**: Use the thread-safe messaging system for communication between threads.

5. **Testing**: Test your components with both PyQt5 and PySide6 to ensure compatibility.

## Common Issues and Solutions

### Signal Type Errors

If you encounter type errors with signals, ensure you're using the Signal class from the compatibility layer:

```python
from src.v5.ui.qt_compat import Signal
```

### Import Errors

If you encounter import errors, ensure the compatibility layer is properly initialized before importing other modules.

### QApplication Instance

Always use the QtCompat.get_application() method to get the QApplication instance:

```python
from src.v5.ui.qt_compat import QtCompat
app = QtCompat.get_application()
```

## Contributing

When contributing to the V5 Visualization System, please follow these guidelines:

1. Use the compatibility layer for all Qt imports
2. Test your changes with both PyQt5 and PySide6
3. Document framework-specific behavior, if any
4. Add appropriate error handling for framework differences

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PySide6 development team
- PyQt5 development team
- Qt for Python project 