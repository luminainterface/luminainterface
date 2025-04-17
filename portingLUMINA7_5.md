# Porting Guide: LUMINA 7.5 Node System

This guide provides instructions for porting the LUMINA 7.5 node system to other projects or environments.

## System Overview

The LUMINA 7.5 node system is a flexible, asynchronous node-based processing framework built with PySide6. It consists of:

1. Base Node System (`src/v7_5/nodes/`)
   - Base node infrastructure
   - Node management and execution
   - Port and connection handling
   - Signal system for updates

2. Wiki Processing Implementation
   - Wikipedia integration
   - Asynchronous content processing
   - Auto-update functionality

## Dependencies

Required packages:
```
PySide6>=6.6.1
wikipedia>=1.4.0
asyncio
```

## Directory Structure

Minimum required structure:
```
your_project/
├── nodes/
│   ├── __init__.py
│   ├── base_node.py
│   ├── node_manager.py
│   ├── wiki_processor_node.py
│   └── example.py
├── requirements.txt
└── README.md
```

## Porting Steps

### 1. Core Components Setup

1. Create the base node infrastructure:
   ```python
   # nodes/base_node.py
   from typing import Dict, List, Any, Optional, Set
   from dataclasses import dataclass
   from uuid import UUID, uuid4
   from enum import Enum
   from PySide6.QtCore import QObject, Signal

   # Copy the NodePort, NodeType, NodeMetadata, and Node classes
   ```

2. Set up the node manager:
   ```python
   # nodes/node_manager.py
   from typing import Dict, List, Type, Optional
   from uuid import UUID
   import asyncio
   from PySide6.QtCore import QObject, Signal
   from .base_node import Node, NodePort

   # Copy the NodeManager class
   ```

### 2. Implementation Specific Components

1. Create your node implementations:
   ```python
   # nodes/your_node.py
   from .base_node import Node, NodeMetadata, NodeType

   class YourNode(Node):
       def __init__(self):
           metadata = NodeMetadata(
               name="Your Node",
               description="Your node description",
               category="Your Category",
               type=NodeType.PROCESSOR,
               color="#YOUR_COLOR",
               icon="🔧"
           )
           super().__init__(metadata)
           
           # Add your ports
           self.add_input_port("input_name", input_type, "description")
           self.add_output_port("output_name", output_type, "description")
           
       async def process(self) -> None:
           # Implement your processing logic
           pass
   ```

### 3. Integration Steps

1. Register your nodes with the manager:
   ```python
   manager = NodeManager()
   manager.register_node_type("your_node", YourNode)
   ```

2. Create and connect nodes:
   ```python
   node1 = manager.create_node("your_node")
   node2 = manager.create_node("another_node")
   
   manager.connect_nodes(
       node1.id, "output_port_name",
       node2.id, "input_port_name"
   )
   ```

3. Set up signal handlers:
   ```python
   def on_node_updated(node_id):
       node = manager.get_node(node_id)
       if node:
           # Handle node updates
           
   def on_error(error_msg):
       # Handle errors
           
   manager.node_updated.connect(on_node_updated)
   manager.error_occurred.connect(on_error)
   ```

4. Execute the node system:
   ```python
   await manager.execute()
   ```

### 4. UI Integration

If you're using the system with a UI:

1. Create node widgets:
   ```python
   from PySide6.QtWidgets import QWidget, QVBoxLayout

   class NodeWidget(QWidget):
       def __init__(self, node: Node, parent=None):
           super().__init__(parent)
           self.node = node
           self.setup_ui()
           
       def setup_ui(self):
           layout = QVBoxLayout(self)
           # Add your UI components
   ```

2. Handle node visualization:
   ```python
   class NodeGraphWidget(QWidget):
       def __init__(self, manager: NodeManager, parent=None):
           super().__init__(parent)
           self.manager = manager
           self.setup_ui()
           
       def setup_ui(self):
           # Implement node graph visualization
           pass
   ```

### 5. Best Practices

1. **Error Handling**:
   - Always use the error_occurred signal for error propagation
   - Implement proper cleanup in node destructors
   - Handle connection errors gracefully

2. **Async Operations**:
   - Use asyncio for asynchronous operations
   - Implement proper cancellation in cleanup methods
   - Handle task scheduling carefully

3. **Type Safety**:
   - Use type hints consistently
   - Implement port type checking
   - Validate connections before creation

4. **Resource Management**:
   - Implement cleanup methods for all nodes
   - Handle resource allocation/deallocation properly
   - Use context managers where appropriate

### 6. Example Implementation

```python
import asyncio
from PySide6.QtWidgets import QApplication
from nodes import NodeManager, YourNode

async def main():
    manager = NodeManager()
    manager.register_node_type("your_node", YourNode)
    
    node = manager.create_node("your_node")
    if not node:
        return
        
    # Set up signal handlers
    manager.node_updated.connect(your_update_handler)
    manager.error_occurred.connect(your_error_handler)
    
    # Configure node
    node.input_ports["your_input"].value = your_value
    
    # Execute
    await manager.execute()
    
    # Cleanup
    manager.stop()
    manager.clear()

if __name__ == "__main__":
    app = QApplication([])
    asyncio.run(main())
    app.exec()
```

## Common Issues and Solutions

1. **Cyclic Dependencies**
   - Problem: Nodes form a circular dependency
   - Solution: Use the NodeManager's cycle detection
   - Prevention: Design node graphs carefully

2. **Resource Leaks**
   - Problem: Resources not properly cleaned up
   - Solution: Implement cleanup methods
   - Prevention: Use context managers and proper error handling

3. **Type Mismatches**
   - Problem: Incompatible port connections
   - Solution: Use the built-in type checking
   - Prevention: Define clear port types and interfaces

4. **Signal Handling**
   - Problem: Signals not properly connected/disconnected
   - Solution: Use proper signal management
   - Prevention: Follow Qt signal/slot patterns

## Testing

1. Create unit tests for nodes:
   ```python
   import pytest
   from nodes import NodeManager, YourNode

   @pytest.mark.asyncio
   async def test_your_node():
       manager = NodeManager()
       node = YourNode()
       # Test node functionality
   ```

2. Test connection handling:
   ```python
   @pytest.mark.asyncio
   async def test_node_connections():
       manager = NodeManager()
       node1 = manager.create_node("node1")
       node2 = manager.create_node("node2")
       # Test connections
   ```

## Documentation

1. Document your nodes:
   ```python
   class YourNode(Node):
       """Your node description
       
       Attributes:
           attribute_name: description
           
       Ports:
           Inputs:
               - input_name: description
           Outputs:
               - output_name: description
       """
   ```

2. Document connections and dependencies

3. Maintain API documentation

## Support and Resources

- Reference the LUMINA 7.5 documentation
- Check the example implementations
- Use the testing framework
- Follow the best practices guide

## License and Attribution

When porting this system, ensure you:
1. Maintain original license notices
2. Provide attribution to the LUMINA 7.5 project
3. Document any modifications
4. Keep track of dependency licenses 