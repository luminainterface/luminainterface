# Neural Network Visualization System

## Quick Start
1. Double-click `run_visualizer.bat` to start the visualization
2. Modify `src/frontend/ui/config/visualizer_config.json` to customize settings
3. Run the batch file again to apply changes

## Configuration
The visualizer can be customized through the configuration file located at:
`src/frontend/ui/config/visualizer_config.json`

### Configuration Options

#### Network Settings
```json
{
    "network": {
        "num_layers": 3,           // Number of network layers
        "nodes_per_layer": 4,      // Nodes per layer
        "animation_speed": 1.0,    // Overall animation speed
        "signal_frequency": 0.5,   // Signal generation frequency
        "complexity": 0.5,         // Network complexity
        "growth_rate": 0.1,        // System growth rate
        "stability_threshold": 0.7  // System stability threshold
    }
}
```

#### Appearance Settings
```json
{
    "appearance": {
        "node_size": 30,           // Size of nodes
        "connection_width": 2,     // Width of connections
        "background_color": "#1E1E1E",  // Background color
        "grid_enabled": true,      // Show/hide grid
        "node_colors": {           // Node color schemes
            "normal": "#3498db",
            "auto_learner": "#2ecc71",
            "logic_gate": "#e74c3c",
            "seed": "#f39c12",
            "system": "#9b59b6"
        },
        "connection_colors": {     // Connection color schemes
            "literal": "#3498db",
            "semantic": "#e74c3c",
            "hybrid": "#9b59b6",
            "auto_learner": "#2ecc71",
            "logic_gate": "#e67e22",
            "seed": "#f1c40f"
        },
        "growth_effects": {        // Growth visualization settings
            "ring_radius": 50,     // Growth ring radius
            "pulse_speed": 2.0,    // Growth pulse speed
            "stage_colors": {      // Growth stage colors
                "seed": "#f1c40f",
                "sprout": "#2ecc71",
                "sapling": "#3498db",
                "mature": "#9b59b6"
            }
        }
    }
}
```

#### Data Source Settings
```json
{
    "data_sources": {
        "autowiki": {
            "enabled": false,      // Enable/disable AutoWiki integration
            "update_interval": 1000 // Update frequency in milliseconds
        },
        "neural_seed": {
            "enabled": false,      // Enable/disable Neural Seed integration
            "update_interval": 1000
        },
        "external": {
            "enabled": false,      // Enable/disable external data source
            "update_interval": 1000
        },
        "system_grower": {         // System grower integration
            "enabled": true,       // Enable/disable system grower
            "update_interval": 500, // Update frequency
            "health_metrics": {    // Health monitoring
                "cpu_usage": true,
                "memory_usage": true,
                "process_count": true,
                "stability": true
            }
        }
    }
}
```

#### Animation Settings
```json
{
    "animation": {
        "base_frequency": 0.5,     // Base oscillation frequency
        "frequency_variance": 0.2, // Randomness in frequency
        "movement_speed": 1.0,     // Node movement speed
        "movement_radius": 10.0,   // Maximum movement radius
        "transition_speed": 0.1,   // State transition speed
        "growth_animation": {      // Growth animation settings
            "ring_duration": 1000,  // Growth ring duration (ms)
            "pulse_duration": 500,  // Growth pulse duration
            "stage_transition": 2000 // Stage transition duration
        }
    }
}
```

## Running the Visualizer

### Using Batch File
Simply double-click `run_visualizer.bat` to start the visualization with default settings. The batch file will:
1. Install required dependencies
2. Start the system grower backend
3. Launch the visualization
4. Handle cleanup on exit

### Command Line
```bash
python src/frontend/ui/visualizer_launcher.py
```

### Custom Configuration
1. Edit `src/frontend/ui/config/visualizer_config.json`
2. Save changes
3. Restart the visualizer

## Architecture

### Core Components

1. **Network2DWidget** (`src/frontend/ui/components/widgets/network_2d_widget.py`)
   - The main visualization engine
   - Handles node and connection rendering
   - Processes real-time data updates
   - Manages animation states and transitions
   - Implements growth visualization

2. **GrowthVisualizer** (`src/frontend/ui/components/growth_visualizer.py`)
   - Handles growth-specific visual effects
   - Manages growth rings and pulses
   - Tracks evolution markers
   - Displays health indicators

3. **SystemState** (`src/frontend/ui/components/system_state.py`)
   - Manages system growth state
   - Tracks growth stages
   - Monitors system health
   - Handles backend communication

4. **TestWindow** (`src/frontend/ui/test_window.py`)
   - Test interface for the visualization
   - Provides UI controls and metrics display
   - Acts as a testing environment

5. **VisualizerLauncher** (`src/frontend/ui/visualizer_launcher.py`)
   - Handles initialization and configuration
   - Manages application lifecycle
   - Loads and applies settings

### Data Flow
```
Configuration File
    ↓
VisualizerLauncher
    ↓
TestWindow (UI Layer)
    ↓
Network2DWidget (Core Animation)
    ↓
GrowthVisualizer (Growth Effects)
    ↓
SystemState (Growth Management)
    ↓
Data Sources (AutoWiki, Neural Seed, System Grower)
```

## Features

### Visualization Elements
- **Nodes**: Represent neural network units
  - Normal nodes
  - Auto-learner nodes
  - Logic gate nodes
  - Seed component nodes
  - System nodes

- **Connections**: Represent network relationships
  - Literal connections (blue)
  - Semantic connections (red)
  - Hybrid connections (purple)
  - Auto-learner connections (green)
  - Logic gate connections (orange)
  - Seed connections (brown)

### Growth Visualization
- **Growth Rings**: Animated rings showing node growth
- **Evolution Markers**: Visual indicators for stage changes
- **Health Indicators**: Real-time system health metrics
- **Stage Backgrounds**: Color-coded backgrounds for growth stages
- **Connection Pulses**: Animated pulses along strong connections

### Animation States
- **Connection Modes**:
  - IDLE: Normal state
  - ACTIVE: High activity
  - LEARNING: Training state
  - ERROR: Problem state

- **Growth Stages**:
  - SEED: Initial state (yellow)
  - SPROUT: Early development (green)
  - SAPLING: Intermediate growth (blue)
  - MATURE: Full development (purple)

### System Integration
- **System Grower**: Backend process for system evolution
- **Health Monitoring**: Real-time system metrics
- **Growth Tracking**: Stage progression visualization
- **Stability Analysis**: System stability indicators

## Troubleshooting

### Common Issues

1. **Configuration File Not Found**
   - Ensure `visualizer_config.json` exists in the correct location
   - Check file permissions
   - Verify JSON syntax is correct

2. **Visualization Not Starting**
   - Check Python installation
   - Verify required packages are installed
   - Check console for error messages

3. **Performance Issues**
   - Reduce network complexity
   - Increase update intervals
   - Disable unnecessary features
   - Check system resource usage

4. **Growth Visualization Issues**
   - Verify system grower is running
   - Check growth stage transitions
   - Monitor system health metrics
   - Adjust growth animation parameters

### Debugging
- Check the console output for error messages
- Verify configuration file syntax
- Monitor system resources
- Check data source connections
- Review growth stage transitions
- Monitor system health metrics

## Contributing
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Submit pull request

## License
[Specify your license here]

## Contact
[Your contact information] 