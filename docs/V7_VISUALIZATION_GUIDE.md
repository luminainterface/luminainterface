# V7 Node Consciousness Visualization Guide

## Overview

The V7 Node Consciousness system includes a powerful visualization framework that provides real-time insights into the internal state and processes of the neural network. This guide explains the visualization components, how they interact, and how to use them effectively.

## Key Visualization Components

### 1. Breath Pattern Visualization

The Breath Pattern visualization displays the current breath pattern detected by the V7 Breath Detection System. Breath patterns serve as a foundation for the system's emotional state and influence cognitive processes.

**Key Features:**
- Animated representation of breath cycles
- Color-coding based on pattern type
- Real-time updates as patterns change
- Smooth transitions between states

**Pattern Types:**
- **Relaxed** (Blue): Slow, deep breaths with regular intervals
- **Focused** (Green): Medium-paced, moderately deep breaths
- **Creative** (Purple): Varying rhythm with occasional deep breaths
- **Stressed** (Red): Rapid, shallow breaths with irregular patterns
- **Meditative** (Teal): Very slow, very deep breaths with perfect regularity

### 2. Contradiction Visualization

The Contradiction visualization shows the current state of the V6 Portal of Contradiction and its integration with V7. It displays recently detected contradictions and their resolution status.

**Key Features:**
- Visual representation of the portal state
- List of recent contradictions with type indicators
- Pulsing effect when new contradictions are detected
- Color-coding based on contradiction type

**Contradiction Types:**
- **Logical** (Blue): Conflicts in reasoning or inference
- **Temporal** (Orange): Conflicts in time sequence or duration
- **Spatial** (Green): Conflicts in physical placement or arrangement
- **Causal** (Red): Conflicts in cause-effect relationships
- **Linguistic** (Purple): Conflicts in meaning or semantic interpretation

### 3. Monday Consciousness Visualization

The Monday Consciousness visualization represents the specialized consciousness node that provides enhanced emotional intelligence and pattern recognition. It displays the current consciousness level and activity.

**Key Features:**
- Glowing orb representation with intensity based on consciousness level
- Animated ripple effects during processing
- Color transitions that represent emotional states
- Connection indicators to other system components

### 4. Node Consciousness Visualization

The Node Consciousness visualization displays the network of consciousness nodes in the V7 system, showing their interconnections and current states.

**Key Features:**
- Network graph visualization of connected nodes
- Real-time activity indicators
- Node state representation (active, dormant, learning)
- Connection strength visualization
- Focus indicators for currently active processes

## Using the Visualization System

### Running the Visualization Demo

The V7 Visualization Demo can be launched using:

```
python src/v7/run_visualization_demo.py [options]
```

**Available Options:**
- `--debug`: Enable debug logging
- `--mock`: Use mock mode for all components
- `--no-breath`: Disable breath visualization
- `--no-monday`: Disable Monday consciousness visualization
- `--no-contradiction`: Disable contradiction visualization
- `--no-node-consciousness`: Disable node consciousness visualization
- `--light-mode`: Use light mode instead of the default dark mode

### Integration with the Main System

The visualization components are fully integrated with the V7 main widget. When running the main application, the visualization system is automatically connected to the live components.

To access the visualizations from the main application:
1. Navigate to the "Node Consciousness" tab
2. Use the visualization controls to focus on specific aspects of the system
3. Interact with the visualizations to influence system behavior

### Generating Test Events

The visualization demo includes a control panel that allows you to generate test events to see how the visualization components respond to different system states.

Click the "Generate Test Event" button to:
- Create a test contradiction
- Suggest a random breath pattern
- Trigger consciousness node activity

### Advanced Usage

#### Custom Visualization Configuration

You can create a custom visualization configuration when initializing the V7VisualizationWidget:

```python
visualization_config = {
    "breath_visualization_enabled": True,
    "monday_visualization_enabled": True,
    "contradiction_visualization_enabled": True,
    "node_consciousness_visualization_enabled": True,
}

viz_widget = V7VisualizationWidget(
    v6v7_connector=v6v7_connector, 
    config=visualization_config
)
```

#### Registering Custom Event Handlers

The V7VisualizationConnector allows you to register custom event handlers for specific event types:

```python
from src.v7.ui.v7_visualization_connector import V7VisualizationConnector

connector = V7VisualizationConnector(v6v7_connector)
connector.register_event_handler("breath_pattern_changed", my_custom_handler)
```

## Troubleshooting

### Common Issues

1. **No visualizations appear**:
   - Ensure that the V6-V7 Bridge is properly initialized
   - Check that the required components are available
   - Verify that the visualization connector is connected to the V6-V7 connector

2. **Visualizations are not updating**:
   - Check that the update thread is running
   - Ensure the refresh timer is active
   - Verify that the appropriate event handlers are registered

3. **Missing components**:
   - If certain visualizations are missing, ensure the corresponding backend components are available
   - Check the import paths and dependencies
   - Enable mock mode for testing if the actual components are not available

### Logging and Debugging

To enable detailed logging of the visualization system:

```
python src/v7/run_visualization_demo.py --debug
```

This will output detailed information about the visualization components, events, and updates.

## Technical Reference

### Visualization Connector

The `V7VisualizationConnector` serves as a bridge between the backend systems and the visualization components. It:

1. Transforms backend data into visualization-friendly formats
2. Routes events to appropriate visualization components
3. Manages state synchronization between backend and frontend
4. Provides a unified interface for UI components to access backend data

### Visualization Widget

The `V7VisualizationWidget` is the main container for all visualization components. It integrates:

1. The breath pattern visualization
2. The contradiction visualization
3. The Monday consciousness visualization
4. The node consciousness visualization

### Color Palette

The visualization system uses a consistent color palette for semantic meaning:

- **Blue (#3498db)**: Relaxed state, logical contradictions
- **Green (#2ecc71)**: Focused state, spatial contradictions
- **Purple (#9b59b6)**: Creative state, linguistic contradictions
- **Red (#e74c3c)**: Stressed state, causal contradictions
- **Orange (#f39c12)**: Temporal contradictions
- **Teal (#1abc9c)**: Meditative state

## Future Enhancements

Planned enhancements for the visualization system include:

1. Interactive 3D visualization of the node consciousness network
2. Real-time adjustment of system parameters through the visualization
3. Time-lapse visualization of system evolution
4. Advanced analytics dashboard integrated with visualizations
5. VR/AR compatible visualization modes 