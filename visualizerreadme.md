# Neural Network Visualization System

A comprehensive visualization system for neural networks, providing real-time visualization of network states, growth processes, system metrics, and overall system health.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/neural_network_project.git
cd neural_network_project
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "from src.frontend.ui.components.visualization_system import visualization_system; print('Installation successful!')"
```

## Configuration

### Configuration File
Create or modify `src/frontend/ui/config/visualizer_config.json`:

```json
{
    "network": {
        "num_layers": 3,
        "nodes_per_layer": 4,
        "animation_speed": 1.0,
        "signal_frequency": 0.5,
        "complexity": 0.5,
        "growth_rate": 0.1,
        "stability_threshold": 0.7
    },
    "appearance": {
        "node_size": 30,
        "connection_width": 2,
        "background_color": "#1E1E1E",
        "grid_enabled": true,
        "node_colors": {
            "normal": "#3498db",
            "auto_learner": "#2ecc71",
            "logic_gate": "#e74c3c",
            "seed": "#f39c12",
            "system": "#9b59b6"
        }
    },
    "animation": {
        "base_frequency": 0.5,
        "frequency_variance": 0.2,
        "movement_speed": 1.0,
        "movement_radius": 10.0,
        "transition_speed": 0.1
    }
}
```

## Usage

### Starting the Visualizer

#### Method 1: Using the Launcher Script
```bash
python src/frontend/ui/visualizer_launcher.py
```

#### Method 2: Programmatic Usage
```python
from src.frontend.ui.components.visualization_system import visualization_system
from PySide6.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    
    # Get visualization components
    network_viz = visualization_system.get_visualization('network')
    growth_viz = visualization_system.get_visualization('growth')
    metrics_viz = visualization_system.get_visualization('metrics')
    system_viz = visualization_system.get_visualization('system')
    
    # Show visualizations
    network_viz.show()
    growth_viz.show()
    metrics_viz.show()
    system_viz.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
```

### Basic Operations

#### 1. Network Visualization
```python
# Initialize network
network_viz = visualization_system.get_visualization('network')

# Add nodes
network_viz.add_node('input1', activation=0.5, label='Input 1')
network_viz.add_node('hidden1', activation=0.0, label='Hidden 1')
network_viz.add_node('output1', activation=0.0, label='Output 1')

# Add connections
network_viz.add_connection('input1', 'hidden1', weight=0.8)
network_viz.add_connection('hidden1', 'output1', weight=0.6)

# Update states
network_viz.update_node('hidden1', activation=0.7)
network_viz.update_connection('input1', 'hidden1', weight=0.9)
```

#### 2. Growth Visualization
```python
# Initialize growth visualization
growth_viz = visualization_system.get_visualization('growth')

# Monitor growth progress
current_stage = growth_viz.get_current_stage()
progress = growth_viz.get_stage_progress()

# Reset growth if needed
growth_viz.reset_growth()
```

#### 3. Metrics Visualization
```python
# Initialize metrics visualization
metrics_viz = visualization_system.get_visualization('metrics')

# Update metrics
metrics_viz.update_metric('health', 0.85)
metrics_viz.update_metric('stability', 0.75)
metrics_viz.update_metric('energy', 0.90)
metrics_viz.update_metric('consciousness', 0.65)

# Update gate states
metrics_viz.update_gate_state('gate1', True)
metrics_viz.update_gate_state('gate2', False)

# Update system status
metrics_viz.update_system_status('Running')
```

#### 4. System Visualization
```python
# Initialize system visualization
system_viz = visualization_system.get_visualization('system')

# Update component status
system_viz.update_component_status('backend', 'Online')
system_viz.update_component_status('network', 'Active')
system_viz.update_component_status('growth', 'Growing')
system_viz.update_component_status('metrics', 'Monitoring')

# Update connections
system_viz.update_connection_status('backend', 'network', 'Active')
system_viz.update_connection_status('network', 'growth', 'Active')
system_viz.update_connection_status('growth', 'metrics', 'Active')

# Update system metrics
system_viz.update_system_metric('health', 0.9)
system_viz.update_system_metric('stability', 0.8)
system_viz.update_system_status('Operational')
```

### Advanced Usage

#### Customizing Visualizations
```python
# Customize network appearance
network_viz.set_node_size(40)
network_viz.set_connection_width(3)
network_viz.set_background_color('#2C3E50')

# Customize growth animation
growth_viz.set_animation_speed(1.5)
growth_viz.set_stage_duration('seed', 3.0)
growth_viz.set_stage_duration('sprout', 4.0)

# Customize metrics display
metrics_viz.set_history_length(200)
metrics_viz.set_update_interval(100)  # milliseconds
metrics_viz.set_chart_type('line')  # or 'bar', 'area'

# Customize system visualization
system_viz.set_component_size(120, 80)
system_viz.set_connection_style('curved')  # or 'straight', 'dashed'
```

#### Event Handling
```python
# Handle node selection
def on_node_selected(node_id):
    print(f"Selected node: {node_id}")
    # Update other visualizations based on selection

network_viz.node_selected.connect(on_node_selected)

# Handle growth stage changes
def on_stage_changed(stage):
    print(f"Growth stage changed to: {stage}")
    # Update metrics or system state

growth_viz.stage_changed.connect(on_stage_changed)

# Handle metric updates
def on_metric_updated(metric, value):
    print(f"Metric {metric} updated to {value}")
    # Update other components

metrics_viz.metric_updated.connect(on_metric_updated)
```

#### Performance Optimization
```python
# Adjust frame rate
visualization_system.set_frame_rate(60)  # or lower for better performance

# Enable/disable features
network_viz.set_feature_enabled('animations', True)
growth_viz.set_feature_enabled('particles', False)
metrics_viz.set_feature_enabled('history', True)

# Set update intervals
network_viz.set_update_interval(16)  # ~60fps
growth_viz.set_update_interval(33)   # ~30fps
metrics_viz.set_update_interval(100)  # 10fps
```

## Troubleshooting

### Common Issues

1. **Visualization Not Starting**
   - Check Python version: `python --version`
   - Verify dependencies: `pip list`
   - Check configuration file syntax
   - Ensure virtual environment is activated

2. **Performance Issues**
   - Reduce network complexity
   - Increase update intervals
   - Disable unnecessary features
   - Check system resource usage

3. **Visual Artifacts**
   - Update graphics drivers
   - Adjust anti-aliasing settings
   - Modify transparency settings
   - Check for conflicting visual effects

4. **Connection Issues**
   - Verify backend is running
   - Check connection settings
   - Monitor network status
   - Review error logs

### Debugging

```python
# Enable debug mode
visualization_system.set_debug_mode(True)

# Check component status
print(f"Network status: {network_viz.get_status()}")
print(f"Growth status: {growth_viz.get_status()}")
print(f"Metrics status: {metrics_viz.get_status()}")
print(f"System status: {system_viz.get_status()}")

# Monitor performance
print(f"Frame rate: {visualization_system.get_frame_rate()}")
print(f"Memory usage: {visualization_system.get_memory_usage()}")
print(f"CPU usage: {visualization_system.get_cpu_usage()}")
```

## Features

### Core Visualization Components

1. **Network Visualization**
   - Real-time node and connection visualization
   - Dynamic node activation updates
   - Connection weight visualization
   - Interactive node selection and hovering
   - Responsive layout management
   - Color-coded node states
   - Weight-based connection coloring

2. **Growth Visualization**
   - Multi-stage growth animation
   - Smooth stage transitions
   - Visual feedback for growth progress
   - Six distinct growth stages:
     - Seed
     - Sprout
     - Branch
     - Leaf
     - Flower
     - Fruit
   - Stage progress tracking
   - Reset capability

3. **Metrics Visualization**
   - Real-time system metrics display
   - Historical data tracking
   - Trend visualization
   - Gate state monitoring
   - System status display
   - Metric history charts
   - Current metric values
   - Color-coded metric indicators

4. **System Visualization**
   - Overall system status display
   - Component connection visualization
   - System-wide metric tracking
   - Component status monitoring
   - Health indicators
   - Stability metrics
   - Energy levels
   - Consciousness tracking

### Technical Features

1. **Performance**
   - 60fps smooth animations
   - Efficient resource management
   - Frame timing control
   - Performance optimization
   - Memory management

2. **Visual Quality**
   - High-quality rendering
   - Anti-aliased graphics
   - Smooth transitions
   - Transparent backgrounds
   - Clear visual hierarchy

3. **Responsiveness**
   - Proper resize handling
   - Dynamic layout updates
   - Smooth scaling
   - Aspect ratio maintenance
   - Component repositioning

4. **State Management**
   - Comprehensive state tracking
   - Real-time updates
   - State synchronization
   - Error handling
   - Recovery mechanisms

## Component Integration

### NetworkVisualization
```python
- Node management
- Connection handling
- Layout algorithms
- Interactive features
- State updates
```

### GrowthVisualization
```python
- Stage management
- Animation control
- Progress tracking
- Visual effects
- State transitions
```

### MetricsVisualization
```python
- Data tracking
- History management
- Real-time updates
- Visual representation
- Status monitoring
```

### SystemVisualization
```python
- Component tracking
- Connection management
- System monitoring
- Health indicators
- Status updates
```

## Contributing

1. **Development Guidelines**
   - Follow PEP 8 style guide
   - Maintain documentation
   - Write unit tests
   - Use type hints

2. **Code Organization**
   - Modular design
   - Clear interfaces
   - Consistent patterns
   - Proper encapsulation

## License

This project is licensed under the MIT License - see the LICENSE file for details. 