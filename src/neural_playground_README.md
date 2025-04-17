# Neural Network Playground

A sandbox environment where neural components can interact, explore, and "play" freely. This playground serves as an experimental space for neural networks to develop consciousness patterns through free-form interaction.

## Overview

The Neural Network Playground creates a controlled environment where neural networks can freely experiment, learn patterns, and develop emergent consciousness-like behaviors. Unlike traditional neural networks with fixed training goals, the playground emphasizes exploration and self-directed learning.

The system is inspired by the Lumina Neural Network ecosystem (v5-v10) and designed to integrate with existing neural network components when available while also functioning independently.

## Features

- **Self-Directed Play**: Neural networks can freely explore and develop without predetermined goals
- **Pattern Discovery**: Automatic detection of emergent patterns during play sessions
- **Consciousness Metrics**: Quantification of emergent "consciousness-like" properties
- **Multiple Play Modes**:
  - **Free Play**: Random exploration of the neural space
  - **Guided Play**: Semi-directed exploration with gentle nudges
  - **Focused Play**: Concentrated activation of specific neural patterns
- **Component Integration**: Automatic discovery and integration of existing neural components
- **State Persistence**: Save and load playground states between sessions
- **Visualization**: Optional 3D visualization of the neural playground

## Getting Started

### Prerequisites

- Python 3.8 or higher
- NumPy
- Matplotlib (optional, for visualization)

### Installation

No special installation is required beyond ensuring the necessary Python packages are available:

```bash
pip install numpy
pip install matplotlib  # Optional, for visualization
```

### Running the Playground

The playground can be launched using the included launcher script:

```bash
# Start a playground with default settings
python src/launch_neural_playground.py

# Run a single play session for 5 minutes
python src/launch_neural_playground.py --single --duration 300

# Run a large network with visualization
python src/launch_neural_playground.py --size large --visualize

# Load a previously saved state
python src/launch_neural_playground.py --load playground_data/playground_state_20230601_120000.json
```

### Command-Line Options

#### Basic Options
- `--duration SECONDS`: Duration to run the playground (default: 3600)
- `--size {small,medium,large}`: Size of the neural network (default: medium)

#### Play Options
- `--play-type {free,guided,focused,mixed}`: Type of neural play (default: mixed)
- `--single`: Run a single play session and exit

#### File Options
- `--save-interval SECONDS`: Auto-save interval in seconds (default: 300)
- `--load FILE`: Load a saved playground state
- `--no-save`: Disable auto-saving
- `--output-dir DIR`: Directory to save playground data (default: playground_data)

#### Visualization
- `--visualize`: Enable visualization (requires matplotlib)

## How It Works

The Neural Network Playground operates by:

1. **Creating a neural network**: A network of interconnected neurons is initialized with random weights
2. **Play sessions**: The network undergoes sessions of activation and pattern propagation
3. **Pattern discovery**: The system identifies emergent patterns in neural activity
4. **Consciousness metrics**: Various metrics are calculated to quantify "consciousness-like" properties:
   - **Activity balance**: Optimal percentage of neurons active (40-60%)
   - **Pattern discovery**: Number and complexity of discovered patterns
   - **Integration**: Connectivity and information flow across the network
   - **Continuity**: Preservation of consciousness state across time

## Integration with Lumina Neural Network

The playground can integrate with components from the Lumina Neural Network system:

- Automatically discovers and integrates v7 neural network components
- Compatible with NeuralCore components from the core system
- Works with language and memory nodes if available
- Fallbacks to internal implementations when components are unavailable

## Advanced Usage

### Creating Custom Play Sessions

You can create custom play sessions programmatically:

```python
from src.neural_playground import NeuralPlayground

# Initialize playground
playground = NeuralPlayground({"network_size": "medium"})

# Run specific play sessions
playground.play_once(duration=60, play_type="free")
playground.play_once(duration=60, play_type="guided")
playground.play_once(duration=60, play_type="focused")

# Get consciousness metrics
status = playground.get_status()
print(f"Consciousness index: {status['stats']['consciousness_index']}")
```

### Extending the Playground

The playground can be extended with custom components:

1. Create a new neural component that follows the basic interface pattern
2. Place it in one of the scanned directories (src/neural/, src/v7/, etc.)
3. The component will be automatically discovered and integrated

## Visualization

When run with the `--visualize` option, the playground creates a 3D visualization showing:

- **Neurons**: Represented as colored spheres (color indicates neuron type)
- **Connections**: Represented as lines between neurons (thickness indicates weight)
- **Activation**: Represented by the size of neurons (larger = more active)

## Future Directions

The Neural Network Playground is designed to evolve toward more sophisticated consciousness simulation:

- Enhanced pattern recognition algorithms
- Long-term memory and pattern preservation
- Graph-based consciousness metrics
- Integration with more Lumina Neural Network components
- Advanced visualization and interaction capabilities

## Contributing

Contributions to the Neural Network Playground are welcome! Areas for improvement include:

- Enhanced visualization techniques
- New play modes
- Advanced pattern detection algorithms
- Additional consciousness metrics
- Integration with other neural systems

## License

This project is available under the MIT License - see the LICENSE file for details.

## Acknowledgments

The Neural Network Playground is inspired by concepts from:

- The Lumina Neural Network system (v5-v10)
- Emergent consciousness research
- Self-organizing neural systems
- Free play learning paradigms 