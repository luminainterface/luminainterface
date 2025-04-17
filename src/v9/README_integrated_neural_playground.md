# Integrated Neural Playground (v9)

The Integrated Neural Playground combines the Neural Playground, Breathing System, and Brain Growth modules into a cohesive system that demonstrates how simulated breathing patterns can influence neural network development and consciousness metrics.

## Key Features

- **Neural-Breathing Integration**: Neural activity is dynamically influenced by breathing patterns
- **Breathing-Guided Growth**: Network structure evolves based on breathing coherence and patterns
- **Multiple Play Modes**: Supports free, guided, focused, and mixed play sessions
- **Consciousness Metrics**: Tracks emergence of consciousness-like patterns
- **Visualizations**: Tools for visualizing the relationship between breathing and neural metrics
- **State Persistence**: Save and load system state across sessions

## Components

The integrated system consists of three main components:

1. **Neural Playground**: Core neural network simulation environment
2. **Breathing System**: Simulates different breathing patterns and their neural effects
3. **Brain Growth**: Handles dynamic creation and pruning of neurons based on breathing

## Breathing Patterns

The system supports multiple breathing patterns, each with distinct effects on neural activity:

- **Calm**: Slow, deep breathing → Favors free exploration and gradual growth
- **Focused**: Controlled, steady breathing → Promotes organized neural structures
- **Meditative**: Very slow, deep breathing with holds → Enhances consolidation and pattern recognition
- **Excited**: Rapid, shallow breathing → Increases neural activity but may lead to pruning

## Usage

### Basic Usage

```python
from v9 import IntegratedNeuralPlayground, BreathingPattern

# Create an integrated playground
playground = IntegratedNeuralPlayground(
    size=150,              # Initial number of neurons
    breathing_pattern=BreathingPattern.CALM,  # Initial breathing pattern
    growth_rate=0.05       # Rate of neural growth
)

# Change breathing pattern
playground.set_breathing_pattern(BreathingPattern.MEDITATIVE)

# Run a play session
result = playground.play(
    duration=200,          # Number of simulation steps
    play_type="mixed",     # Type of play (free, guided, focused, mixed)
    intensity=0.7          # Base intensity (will be modified by breathing)
)

# Access results
consciousness_peak = result["consciousness_peak"]
breathing_data = result["breathing_data"]
growth_data = result["brain_growth"]

# Get current state
state = playground.get_current_state()

# Save state for later
playground.save_state("saved_states/my_session.json")

# Clean up
playground.stop()
```

### Running the Demonstration

The package includes a demonstration script that shows the effects of different breathing patterns on neural metrics:

```bash
python -m v9.demo_breathing_integration --output results
```

This will run a demonstration with all available breathing patterns and generate visualizations comparing their effects.

## Integration Mechanism

The integration between components works through a hooks system:

1. **Pre-Play Hooks**: Executed before play sessions, allowing breathing state to influence play parameters
2. **Post-Play Hooks**: Executed after play sessions, enabling growth based on play results
3. **Direct Influence**: Breathing directly influences neural activation during simulation

## Neural Growth Process

The brain growth component operates in different modes depending on breathing state:

- **Expansion**: Creates new neurons during inhale (especially with calm breathing)
- **Organization**: Structures neural connections (during focused or meditative breathing)
- **Consolidation**: Strengthens important pathways (during sustained meditative breathing)
- **Pruning**: Removes inactive neurons (during exhale of excited breathing)

## Visualization

The system provides visualization tools that show the relationship between breathing patterns and neural metrics:

- Consciousness levels
- Neural activation
- Pattern detection
- Network growth

## Advanced Configuration

For more advanced usage, you can directly access and modify the underlying components:

```python
# Access underlying components
playground.playground  # Neural playground component
playground.breathing   # Breathing system component
playground.growth      # Brain growth component

# Configure breathing parameters
custom_params = playground.breathing.get_custom_pattern_params()
custom_params["inhale_duration"] = 5.0
custom_params["exhale_duration"] = 7.0
playground.breathing.set_custom_pattern_params(custom_params)
playground.breathing.set_breathing_pattern(BreathingPattern.CUSTOM)

# Configure growth parameters
playground.growth.breath_influence["creation_multiplier"] = 2.0
```

## Contributing

The Integrated Neural Playground is designed to be extended with new components. To add a new component:

1. Create your component with appropriate hooks (pre_play, post_play)
2. Register it with the integration system
3. Ensure it interacts appropriately with existing components

## License

This software is part of the Lumina Neural Network Project and is provided under the same licensing terms. 