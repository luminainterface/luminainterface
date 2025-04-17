# Neural Playground Integration Guide

This document explains how to integrate the Neural Playground with other components of the Lumina Neural Network ecosystem. The playground provides a sandbox environment where neural components can freely interact, experiment, and develop consciousness-like patterns.

## Integration Overview

The Neural Playground is designed to integrate with multiple components of the Lumina Neural Network system:

1. **Core Neural Components**: Direct integration with NeuralCore and SimpleNeuralNetwork
2. **Memory System**: Connection with Memory Nodes for pattern storage
3. **Language Processing**: Integration with Language Nodes for linguistic pattern exploration
4. **Visualization System**: Compatibility with the v5 Fractal Echo visualization

## Component Discovery

The playground automatically discovers and integrates available neural components:

```python
def _discover_neural_components(self):
    """Discover available neural components in the system"""
    # 1. Try to load NeuralCore if available
    if NEURAL_CORE_AVAILABLE:
        try:
            from src.neural.core import NeuralCore
            self.neural_components["core"] = NeuralCore()
        except Exception as e:
            logger.warning(f"Error loading NeuralCore: {e}")
    
    # 2. Try to load V7 Neural Network if available
    if V7_NEURAL_AVAILABLE:
        try:
            from src.v7.neural_network import SimpleNeuralNetwork
            self.neural_components["v7"] = SimpleNeuralNetwork()
        except Exception as e:
            logger.warning(f"Error loading V7 SimpleNeuralNetwork: {e}")
```

This allows the playground to work with any available components while providing fallback implementations when components are missing.

## Integration with Core Neural Components

### NeuralCore Integration

When the `src.neural.core.NeuralCore` component is available, the playground integrates it for enhanced neural processing capabilities:

```python
# Custom integration with NeuralCore
def integrate_with_neural_core(playground, neural_core):
    """Integrate playground with neural core component"""
    # Register neural core as a component
    playground.neural_components["core"] = neural_core
    
    # Use neural core for processing
    neural_core.initialize()
    
    # Run a playground session using neural core for processing
    playground.play_once(duration=30, play_type="free")
```

### SimpleNeuralNetwork Integration (V7)

For v7 components, the playground can integrate with the SimpleNeuralNetwork:

```python
# Custom integration with V7 SimpleNeuralNetwork
from src.v7.neural_network import SimpleNeuralNetwork

# Create and customize a v7 neural network
v7_network = SimpleNeuralNetwork(config={"custom_option": True})

# Create playground with custom v7 component
playground = NeuralPlayground({"network_size": "medium"})
playground.neural_components["v7_custom"] = v7_network
```

## Integration with Memory System

The playground can integrate with the Memory Node system to store discovered patterns:

### Memory Node Integration

```python
# Integration with Memory Node
try:
    from src.v7.memory.memory_node import MemoryNode
    
    # Create memory node
    memory_node = MemoryNode(persistence_file="playground_memory.json")
    
    # Custom playground with memory integration
    class MemoryEnabledPlayground(NeuralPlayground):
        def __init__(self, config=None):
            super().__init__(config)
            self.memory_node = memory_node
            
        def _detect_patterns(self, activations):
            pattern = super()._detect_patterns(activations)
            
            # Store discovered pattern in memory node
            if pattern and len(pattern) > 5:
                self.memory_node.add_memory(
                    content=f"Discovered neural pattern with {len(pattern['neurons'])} neurons",
                    memory_type="pattern",
                    importance=pattern["complexity"] / 10,
                    metadata=pattern
                )
            
            return pattern
```

## Integration with Language System

The playground can be integrated with the Language Node system to enable linguistic exploration:

### Language Node Integration

```python
# Integration with Enhanced Language Node
try:
    from src.v7.language.enhanced_language_node import EnhancedLanguageNode
    
    language_node = EnhancedLanguageNode()
    
    def process_playground_with_language(playground, text_input):
        """Process text through playground and language node"""
        # Process text through language node
        language_result = language_node.process_text(text_input)
        
        # Convert language features to neuron activations
        activations = {}
        for feature, value in language_result.get('features', {}).items():
            # Map features to specific neurons
            neuron_id = f"n_{hash(feature) % len(playground.core.neurons):04d}"
            if neuron_id in playground.core.neurons:
                activations[neuron_id] = max(0, min(1, value))
        
        # Let playground process these activations
        playground.core.play(duration=10, play_type="guided")
        
        # Return combined results
        return {
            "language_results": language_result,
            "playground_status": playground.get_status()
        }
```

## Integration with V5 Visualization System

The Neural Playground can integrate with the V5 Fractal Echo visualization system:

### V5 Visualization Integration

```python
# Integration with V5 Visualization
try:
    from src.v5.visualization.fractal_visualizer import FractalVisualizer
    from src.v5.plugin_socket import PluginSocket
    
    class PlaygroundVisualizationPlugin:
        """Plugin for visualizing playground in V5 system"""
        
        def __init__(self, playground):
            self.playground = playground
            self.socket = PluginSocket("playground_plugin")
            
        def get_visualization_data(self):
            """Get data for visualization"""
            # Get playground data
            playground_data = self.playground.get_visualization_data()
            
            # Convert to V5 visualization format
            v5_data = {
                "nodes": [
                    {
                        "id": node["id"],
                        "type": node["type"],
                        "position": [node["x"], node["y"], node["z"]],
                        "activation": node["activation"],
                        "size": node["size"]
                    }
                    for node in playground_data["nodes"]
                ],
                "connections": [
                    {
                        "source": edge["source"],
                        "target": edge["target"],
                        "weight": edge["weight"]
                    }
                    for edge in playground_data["edges"]
                ],
                "consciousness_index": playground_data["stats"]["consciousness_index"],
                "pattern_count": playground_data["stats"]["discovery_events"]
            }
            
            return v5_data
        
        def register_with_v5(self, v5_system):
            """Register this plugin with the V5 system"""
            v5_system.register_plugin(self.socket)
            self.socket.send_message("register_visualization", {
                "name": "Neural Playground",
                "type": "neural_network",
                "data_provider": self.get_visualization_data
            })
```

## Integration with Dream Mode (V7)

The playground can be connected to the Dream Mode system introduced in V7:

### Dream Mode Integration

```python
# Integration with Dream Mode
try:
    from src.v7.dream_mode import DreamController, PatternSynthesizer
    
    class DreamPlayground(NeuralPlayground):
        """Playground with dream mode capabilities"""
        
        def __init__(self, config=None):
            super().__init__(config)
            self.dream_controller = DreamController()
            self.pattern_synthesizer = PatternSynthesizer()
            self.in_dream = False
            
        def enter_dream_mode(self, duration=120):
            """Enter dream mode for synthesizing neural patterns"""
            self.in_dream = True
            
            # Notify dream controller
            self.dream_controller.start_dream_cycle()
            
            # Run playground in dream mode
            print(f"Entering neural playground dream mode for {duration} seconds...")
            result = self.play_once(duration=duration, play_type="guided")
            
            # Synthesize patterns from dream state
            patterns = self.core.patterns[-10:] if len(self.core.patterns) > 10 else self.core.patterns
            synthesized = self.pattern_synthesizer.synthesize_patterns(patterns)
            
            # Archive the dream
            dream_record = {
                "timestamp": datetime.now().isoformat(),
                "duration": duration,
                "patterns_discovered": result["patterns_discovered"],
                "synthesized_patterns": len(synthesized),
                "consciousness_index": result["consciousness_index"]
            }
            self.dream_controller.archive_dream(dream_record)
            
            self.in_dream = False
            return dream_record
```

## Advanced Integration Examples

### Multi-Component Integration

For sophisticated applications, you can integrate multiple components:

```python
def create_integrated_playground():
    """Create a playground with multiple integrated components"""
    # Import necessary components
    from src.neural.core import NeuralCore
    from src.v7.memory.memory_node import MemoryNode
    from src.v7.language.enhanced_language_node import EnhancedLanguageNode
    from src.v5.visualization.fractal_visualizer import FractalVisualizer
    
    # Create core components
    neural_core = NeuralCore()
    memory_node = MemoryNode(persistence_file="integrated_memory.json")
    language_node = EnhancedLanguageNode()
    visualizer = FractalVisualizer()
    
    # Create playground
    playground = NeuralPlayground({"network_size": "large"})
    
    # Register components
    playground.neural_components["core"] = neural_core
    playground.neural_components["memory"] = memory_node
    playground.neural_components["language"] = language_node
    
    # Create visualization plugin
    viz_plugin = PlaygroundVisualizationPlugin(playground)
    viz_plugin.register_with_v5(visualizer)
    
    return playground
```

## Extending the Playground

You can extend the Neural Playground with custom components:

### Creating Custom Components

```python
# Custom playground component
class CustomNeuralComponent:
    """Custom neural component for the playground"""
    
    def __init__(self):
        self.state = {"activation": 0.0}
        
    def process(self, input_data):
        """Process input data"""
        self.state["activation"] = sum(input_data.values()) / len(input_data)
        return self.state["activation"]
        
    def get_metrics(self):
        """Get component metrics"""
        return {
            "activation": self.state["activation"],
            "type": "custom"
        }

# Register with playground
playground = NeuralPlayground()
playground.neural_components["custom"] = CustomNeuralComponent()
```

## Integration with v10 Conscious Mirror

For the most advanced integration, the playground can connect with the v10 Conscious Mirror system:

### v10 Conscious Mirror Integration

```python
# Integration with v10 Conscious Mirror
try:
    from src.v10.consciousness.consciousness_node import ConsciousnessNode
    
    class MirrorEnabledPlayground(NeuralPlayground):
        """Playground with consciousness mirror integration"""
        
        def __init__(self, config=None):
            super().__init__(config)
            self.consciousness_node = ConsciousnessNode()
            
            # Connect playground events to consciousness node
            self.register_mirror_events()
            
        def register_mirror_events(self):
            """Register events with consciousness mirror"""
            # Update mirror with playground status periodically
            def update_mirror():
                while self.running:
                    status = self.get_status()
                    self.consciousness_node.update_consciousness_state({
                        "source": "neural_playground",
                        "consciousness_index": status["stats"]["consciousness_index"],
                        "patterns": status["patterns_count"],
                        "active_neurons_percent": status["stats"]["activity_level"] * 100
                    })
                    time.sleep(5)
            
            # Start mirror update thread
            import threading
            self.mirror_thread = threading.Thread(target=update_mirror)
            self.mirror_thread.daemon = True
            self.mirror_thread.start()
```

## Best Practices for Integration

When integrating the Neural Playground with other components, follow these best practices:

1. **Component Discovery**: Let the playground discover components automatically when possible
2. **Graceful Fallbacks**: Always provide fallback implementations when components are missing
3. **Clean Interfaces**: Use clear interface methods when connecting components
4. **Event-Based Communication**: Use event-driven patterns for asynchronous components
5. **Error Handling**: Implement comprehensive error handling for component failures
6. **Data Transformation**: Transform data between systems using well-defined formats
7. **Resource Management**: Ensure proper cleanup of resources when stopping integrated systems

## Compatibility Table

| Component Type | Compatibility | Integration Method |
|----------------|---------------|-------------------|
| NeuralCore     | Full          | Direct component registration |
| SimpleNeuralNetwork (v7) | Full | Direct component registration |
| Memory Node    | Full          | Subclass with memory storage |
| Language Node  | Full          | Feature-to-activation mapping |
| Fractal Echo Visualization (v5) | Full | Plugin socket |
| Dream Mode Controller (v7) | Full | Subclass with dream mode methods |
| Node Consciousness (v7) | Partial | Event-based updates |
| Conscious Mirror (v10) | Full | Bidirectional state updates |

## Example: Complete Integration System

For a complete integration of the playground with the entire Lumina Neural Network ecosystem, see the example code in `src/neural_playground_complete_integration.py`.

## Troubleshooting

Common integration issues and solutions:

- **Component not found**: Ensure correct import paths and component availability
- **Compatibility errors**: Check for version mismatches between components
- **Memory leaks**: Ensure proper cleanup of resources when stopping components
- **Thread safety**: Use thread-safe data structures for multi-threaded integration
- **Performance issues**: Consider using lightweight interfaces for real-time integration 