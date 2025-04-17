# Bridge Components

This document provides technical documentation for the bridge components that connect different modules of the Lumina Neural Network System.

## Overview

Bridge components serve as integration points between different subsystems of Lumina, facilitating:

1. Data transformation between different formats
2. Communication across different interface types
3. Seamless transition between system versions
4. Graceful degradation when components are missing
5. Unified access to distributed functionality

## Language Memory to V5 Visualization Bridge

### LanguageMemoryV5Bridge

**Location**: `src/language_memory_v5_bridge.py`

**Purpose**: Primary bridge connecting the Language Memory System with the V5 Fractal Echo Visualization system.

#### Key Features

- Initialization of required components for both systems
- Mock mode for testing without dependencies
- Simplified API for topic synthesis and visualization
- Statistics and metrics collection
- Socket-based communication for real-time updates

#### Class Structure

```python
class LanguageMemoryV5Bridge:
    """
    Bridge between Language Memory System and V5 Visualization System
    
    This class provides:
    1. Initialization of all necessary components
    2. Simplified API for visualization integration
    3. Connection between language memory and visualization system
    4. Data transformation between systems
    """
    
    def __init__(self, mock_mode: bool = False):
        """
        Initialize the Language Memory V5 Bridge
        
        Args:
            mock_mode: Use mock data instead of actual language memory system
        """
        # Components
        self.memory_system = None 
        self.socket_manager = None
        self.language_integration_plugin = None
        self.mock_mode = mock_mode
        self.v5_visualization_available = False
        
    def _initialize_components(self):
        """Initialize all required components"""
        # Initialize language memory synthesis
        # Initialize socket manager
        # Initialize and register plugins
        
    def synthesize_topic(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """
        Synthesize a topic using the language memory system
        
        Args:
            topic: The topic to synthesize
            depth: The search depth (1-5)
            
        Returns:
            Synthesis results with visualization data
        """
```

#### Usage Example

```python
# Initialize the bridge
bridge = LanguageMemoryV5Bridge()

# Synthesize a topic
result = bridge.synthesize_topic("neural networks", depth=3)

# Check if visualization is available
if bridge.is_visualization_available():
    # Get visualization plugin for UI integration
    plugin = bridge.get_visualization_plugin()
    
    # Get visualization data
    visualization_data = result.get("visualization_data")
```

### LanguageMemoryIntegrationPlugin

**Location**: `src/v5/language_memory_integration.py`

**Purpose**: V5 plugin that processes language data for visualization.

#### Key Features

- Implementation of the V5 plugin interface
- Transformation of language data into visual networks
- Fractal pattern generation based on language metrics
- Caching for performance optimization
- Mock data generation for testing

#### Class Structure

```python
class LanguageMemoryIntegrationPlugin:
    """
    V5 plugin for language memory visualization integration
    
    This plugin:
    1. Processes language memory data for visualization
    2. Generates network visualizations of topics and relationships
    3. Creates fractal pattern data based on language metrics
    4. Provides statistics and metrics about language memory
    """
    
    def __init__(self, plugin_id="language_memory_integration", mock_mode=False):
        """
        Initialize the language memory integration plugin
        
        Args:
            plugin_id: Unique identifier for this plugin
            mock_mode: Use mock data instead of real language memory
        """
        self.plugin_id = plugin_id
        self.mock_mode = mock_mode
        self.socket = None
        self.language_memory_synthesis = None
        self.cache = {}
        
    def process_language_data(self, topic, depth=3):
        """
        Process language data for the given topic
        
        Args:
            topic: The topic to process
            depth: Search depth
            
        Returns:
            Visualization data
        """
        
    def _prepare_visualization_data(self, topic, synthesized_memory, stats):
        """
        Prepare visualization data from language memory
        
        Args:
            topic: Main topic
            synthesized_memory: Memory synthesis results
            stats: Statistics for the synthesis operation
            
        Returns:
            Formatted visualization data
        """
        
    def get_socket_descriptor(self):
        """
        Return socket descriptor for frontend integration
        
        Returns:
            Socket descriptor dictionary
        """
```

#### Socket Descriptor Format

```python
{
    "plugin_id": "language_memory_integration",
    "message_types": ["language_memory_update", "process_topic"],
    "data_format": "json",
    "subscription_mode": "push",
    "ui_components": [
        "memory_network_view",
        "fractal_pattern_view",
        "memory_stats_panel"
    ]
}
```

### VisualizationBridge

**Location**: `src/v5_integration/visualization_bridge.py`

**Purpose**: Singleton approach to the integration, focused on visualization panel creation.

#### Key Features

- Singleton access pattern through `get_visualization_bridge()`
- Component discovery and initialization
- Visualization panel creation
- Error handling and fallback mechanisms
- Interface for topic visualization

#### Class Structure

```python
class VisualizationBridge:
    """
    Bridge between Language Memory System and V5 Visualization
    
    This class provides:
    1. Initialization of all necessary components
    2. Simplified API for visualization integration
    3. Fallback mechanisms when components are missing
    4. Data transformation between systems
    """
    
    def __init__(self):
        """Initialize the visualization bridge"""
        self.memory_system = None
        self.socket_manager = None
        self.language_integration_plugin = None
        self.v5_visualization_available = False
        
    def visualize_topic(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """
        Visualize a topic using V5 visualization system
        
        Args:
            topic: The topic to visualize
            depth: The search depth (1-5)
            
        Returns:
            Visualization data or error information
        """
        
    def create_visualization_panel(self, panel_type: str) -> Optional[Any]:
        """
        Create a visualization panel of the specified type
        
        Args:
            panel_type: Type of panel to create
            
        Returns:
            The created panel or None if not available
        """
```

#### Singleton Access

```python
# Get the singleton instance
bridge = get_visualization_bridge()

# Check if visualization is available
if bridge.is_visualization_available():
    # Create a visualization panel
    panel = bridge.create_visualization_panel("fractal_pattern_panel")
    
    # Visualize a topic
    data = bridge.visualize_topic("consciousness", depth=4)
```

## FrontendSocketManager

**Location**: `src/v5/frontend_socket_manager.py`

**Purpose**: Core infrastructure for plugin registration and message delivery in V5.

#### Key Features

- Registration of visualization plugins
- Mapping of UI components to provider plugins
- Directed and broadcast message delivery
- Real-time communication between components
- Websocket support for remote interfaces

#### Class Structure

```python
class FrontendSocketManager:
    """
    Manages socket connections between backend components and frontend visualization.
    Core infrastructure for the V5 visualization system.
    """
    
    def __init__(self):
        """Initialize the socket manager"""
        self.plugins = {}
        self.ui_component_map = {}
        self.discovery = ConnectionDiscovery.get_instance()
        self.message_handlers = {}
        self.websocket_server = None
        
    def register_plugin(self, plugin):
        """
        Register a plugin for frontend integration
        
        Args:
            plugin: Plugin object to register
            
        Returns:
            Plugin descriptor
        """
        
    def send_message(self, plugin_id, message_type, data):
        """
        Send a message to a specific plugin
        
        Args:
            plugin_id: Target plugin ID
            message_type: Type of message
            data: Message data
            
        Returns:
            Success status
        """
        
    def broadcast_message(self, message_type, data):
        """
        Broadcast a message to all registered plugins
        
        Args:
            message_type: Type of message
            data: Message data
            
        Returns:
            List of (plugin_id, result) tuples
        """
        
    def get_plugin_for_component(self, component_name):
        """
        Get plugins that provide a specific UI component
        
        Args:
            component_name: Name of the component
            
        Returns:
            List of plugin objects
        """
        
    def start_websocket_server(self, host="127.0.0.1", port=5678):
        """
        Start a websocket server for remote connections
        
        Args:
            host: Server hostname
            port: Server port
            
        Returns:
            Success status
        """
```

#### Usage Example

```python
# Create and configure the socket manager
socket_manager = FrontendSocketManager()

# Register a plugin
plugin = LanguageMemoryIntegrationPlugin()
socket_manager.register_plugin(plugin)

# Get plugins providing a specific component
fractal_plugins = socket_manager.get_plugin_for_component("fractal_pattern_view")

# Send a message to a specific plugin
socket_manager.send_message(
    "language_memory_integration", 
    "process_topic", 
    {"topic": "neural networks", "depth": 3}
)

# Broadcast a message to all plugins
socket_manager.broadcast_message(
    "system_status_update",
    {"status": "ready", "timestamp": time.time()}
)
```

## Other Bridge Components

### language_memory_synthesis_integration.py

**Location**: `src/language_memory_synthesis_integration.py`

**Purpose**: Integrates language memory systems with synthesis capabilities, providing a unified API for all language operations.

### Central Language Node

**Location**: `src/central_language_node.py`

**Purpose**: Unified integration point for all language-related components, bridging language memory, training, LLM enhancement, and neural processing.

### Interface Compatibility Bridges

Bridges that maintain compatibility between different interface versions:

1. **v1-v2 Bridge**: Connects text-based and graphical interfaces
2. **PyQt5-PySide6 Migration Components**: Transition from PyQt5 to PySide6
3. **Breath Integration Bridge**: Connects breath input systems with neural processing

## Implementation Guidelines

When implementing new bridge components:

1. **Graceful Degradation**: Always include fallback mechanisms when components are missing
2. **Mock Mode**: Provide mock data generation for testing without dependencies
3. **Error Handling**: Implement comprehensive error handling with detailed logging
4. **Thread Safety**: Ensure thread-safe communication between components
5. **Documentation**: Clear documentation of component interfaces and message formats
6. **Performance**: Consider caching and optimization for performance-critical operations
7. **Testability**: Include test mode and verification utilities

## Next Steps

Planned enhancements for bridge components:

1. **Enhanced Language Memory V5 Bridge**: Improved integration with visualization
2. **V5 to ConsciousnessNode Bridge**: Connect visualization with consciousness components
3. **Distributed Component Discovery**: Automatic discovery of components across processes
4. **Improved Message Routing**: Advanced message routing and subscription features
5. **Performance Optimization**: Enhanced caching and parallel processing 