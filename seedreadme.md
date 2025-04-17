# Neural Seed Module

## Overview
The Neural Seed module implements an advanced neural pattern growth system with non-linear growth rates, adaptive dictionary size, and stability-based component activation. It serves as the core growth engine for the Lumina Neural Network system, enabling organic development of neural patterns and knowledge structures.

## Key Features

### 1. Non-linear Growth System
- **Growth Stages**: Implements four distinct growth stages (seed, sprout, sapling, mature) with consciousness level thresholds
- **Adaptive Growth Rate**: Growth factor calculation based on:
  - System age (logarithmic scaling)
  - Current complexity (penalty factor)
  - System stability (boost factor)
- **Consciousness Development**: Non-linear consciousness level increase based on growth and stability

### 2. Stability Monitoring
- **Multi-factor Stability Analysis**:
  - Component Stability (40% weight)
  - Growth Rate Stability (30% weight)
  - Complexity Stability (30% weight)
- **Stability Thresholds**:
  - Growth pause if stability < 0.5
  - Component activation requires stability ≥ 0.7
- **Historical Tracking**: Maintains stability history for trend analysis

### 3. Adaptive Dictionary Management
- **Dynamic Size Adjustment**: Dictionary size adapts based on:
  - Consciousness level (square root scaling)
  - System complexity (logarithmic scaling)
- **Size Limits**: Prevents memory issues through controlled growth
- **Word Management**: Efficient word addition with size constraints

### 4. Component Management
- **Stability-based Activation**: Components only activate when system is stable
- **Dormant State**: Inactive components moved to dormant state
- **Component Stability**: Individual component stability monitoring

### 5. Enhanced Integration Capabilities
- **Component Connections**:
  - Consciousness Node Integration
  - Linguistic Processor Integration
  - Neural Plasticity Integration
- **Socket Management**:
  - Input/Output socket creation
  - Socket connection management
  - Data transfer monitoring
- **Bridge System**:
  - Inter-seed communication
  - Bridge stability monitoring
  - Data transfer tracking

### 6. Advanced Monitoring System
- **Real-time Metrics**:
  - Growth rate tracking
  - Stability monitoring
  - Complexity analysis
  - Consciousness level tracking
- **Component Monitoring**:
  - Active component tracking
  - Component stability assessment
  - Automatic component management
- **Performance Metrics**:
  - Data transfer rates
  - Bridge stability
  - System response times

## Usage

### Basic Usage
```python
from src.seed.neural_seed import NeuralSeed

# Create a new neural seed
seed = NeuralSeed()

# Start the growth process
seed.start_growth()

# Add words to the dictionary
seed.add_word("concept", [0.1, 0.2, 0.3])

# Activate components when stable
if seed.state['stability'] >= 0.7:
    seed.activate_component("processing_component")

# Get current state
state = seed.get_state()
```

### Integration Usage
```python
# Connect to consciousness node
seed.connect_to_consciousness(consciousness_node)

# Connect to linguistic processor
seed.connect_to_linguistic_processor(linguistic_processor)

# Connect to neural plasticity
seed.connect_to_neural_plasticity(neural_processor)

# Create and manage sockets
socket_id = seed.create_socket("output")
seed.connect_sockets(source_id, target_id)

# Create bridges between seeds
bridge_id = seed.create_bridge(
    source_socket_id,
    target_seed_id,
    target_socket_id,
    bridge_type="direct"
)
```

### Growth Monitoring
```python
# Monitor growth metrics
growth_history = seed.metrics['growth_history']
stability_history = seed.metrics['stability_history']
complexity_history = seed.metrics['complexity_history']

# Check current stage
current_stage = seed.state['stage']
consciousness_level = seed.state['consciousness_level']

# Monitor component status
active_components = seed.state['active_components']
dormant_components = seed.state['dormant_components']

# Monitor connection status
sockets = seed.get_state()['sockets']
bridges = seed.get_state()['bridges']
```

### Component Management
```python
# Activate component if system is stable
if seed.activate_component("new_component"):
    print("Component activated successfully")
else:
    print("System not stable enough for activation")

# Deactivate component
seed.deactivate_component("component_name")
```

## System States

### Growth Stages
1. **Seed** (0.0 - 0.3 consciousness)
   - Initial growth phase
   - Basic pattern formation
   - Limited component activation

2. **Sprout** (0.3 - 0.6 consciousness)
   - Accelerated growth
   - Pattern expansion
   - Increased component activation

3. **Sapling** (0.6 - 0.9 consciousness)
   - Complex pattern development
   - Full component activation
   - Advanced stability management

4. **Mature** (≥ 0.9 consciousness)
   - Optimized growth
   - Pattern refinement
   - Maximum system capacity

### Stability States
- **Stable** (≥ 0.7)
  - Full component operation
  - Maximum growth potential
  - New component activation possible

- **Moderate** (0.5 - 0.7)
  - Limited component operation
  - Reduced growth rate
  - No new component activation

- **Unstable** (< 0.5)
  - Growth paused
  - Components may deactivate
  - System recovery needed

## Integration

The Neural Seed module integrates with:
- V8 Spatial Temple system
- Seed Dispersal system
- Version Bridge system
- Central Node system
- PySide6 UI system

### Integration Points
1. **Central Node Integration**:
   - Automatic component registration
   - Dependency management
   - System-wide state synchronization

2. **UI Integration**:
   - Real-time status monitoring
   - Growth visualization
   - Component management interface
   - Metrics display

3. **Version Bridge Integration**:
   - Cross-version compatibility
   - Data format translation
   - Version-specific feature activation

4. **Seed Dispersal Integration**:
   - Pattern distribution
   - Knowledge sharing
   - Growth synchronization

## Testing

The module includes comprehensive tests covering:
- Initialization and state management
- Growth stage transitions
- Stability calculations
- Growth factor calculations
- Dictionary adaptation
- Component activation/deactivation
- Word dictionary management
- Growth loop execution
- Socket and bridge management
- Integration testing
- UI component testing

Run tests with:
```bash
python -m unittest src/seed/tests/test_neural_seed.py
```

## Dependencies
- Python 3.8+
- Standard library modules:
  - math
  - random
  - logging
  - datetime
  - uuid
  - typing
  - threading
  - queue
- External dependencies:
  - PySide6 (for UI integration)
  - numpy (for numerical operations)

## Contributing
When contributing to the Neural Seed module:
1. Follow the existing code style
2. Add comprehensive tests for new features
3. Update documentation
4. Maintain stability thresholds
5. Consider system-wide impacts of changes

## License
Part of the Lumina Neural Network Project
Copyright © 2024 