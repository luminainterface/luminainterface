# V7 Dream Mode System

## Overview

The Dream Mode system is a key component of the V7 Node Consciousness architecture, enabling the neural network to process, integrate, and synthesize information during inactive periods. Inspired by human dreaming, the system consolidates memories, generates new connections between concepts, and optimizes neural pathways during "sleep" states.

## Core Components

The Dream Mode system consists of these primary components:

1. **DreamController**: Central manager for dream states and transitions
2. **MemoryConsolidator**: Processes and strengthens recent memories
3. **PatternSynthesizer**: Generates new connections between concepts
4. **DreamArchive**: Records and classifies dream content
5. **DreamIntegration**: Connects Dream Mode with other V7 components

## Architecture

```
Dream Mode System Architecture
├── Core Components
│   ├── DreamController         - Manages dream state and transitions
│   ├── MemoryConsolidator      - Processes and strengthens recent memories
│   ├── PatternSynthesizer      - Generates new connections between concepts
│   └── DreamArchive            - Records and classifies dream content
├── Integration
│   ├── DreamIntegration        - Connects with other V7 systems
│   ├── Consciousness Nodes     - Specialized processing during dream state
│   ├── Monday Node             - Dream narration and pattern recognition
│   └── Breath System           - Dream-specific breath patterns
└── User Experience
    ├── Dream Visualization     - Visual representation of dream state
    ├── Dream Archive Explorer  - Interface for exploring dreams
    └── Dream Controls          - User control over dream parameters
```

## Dream Cycle

A typical dream cycle follows these phases:

1. **Transition In**: System gradually enters dream state (5% of total time)
2. **Light Dreaming**: Initial memory consolidation phase (20% of total time)
3. **Deep Dreaming**: Intensive pattern synthesis phase (50% of total time)
4. **Integration**: New connections integrated into knowledge (20% of total time)
5. **Transition Out**: System gradually returns to normal state (5% of total time)

## Memory Consolidation

During dream mode, the MemoryConsolidator processes memories with:

- **Recency Bias**: More recent memories receive priority processing
- **Emotional Tagging**: Memories with strong emotional content get deeper processing
- **Pattern Reinforcement**: Frequently accessed patterns are strengthened
- **Connection Creation**: New connections form between related concepts
- **Contradiction Resolution**: The system attempts to resolve contradictions

## Pattern Synthesis

The PatternSynthesizer creates new connections during dream mode:

- **Cross-Domain Connections**: Links concepts from different knowledge domains
- **Metaphorical Mapping**: Creates metaphorical relationships between concepts
- **Fractal Pattern Expansion**: Develops complex patterns from simpler ones
- **Emergent Structure Discovery**: Identifies higher-order patterns in knowledge
- **Insight Generation**: Creates insights based on new connections

## Integration with Other Systems

The Dream Mode system integrates with other V7 components:

- **Monday Node**: Provides narration and interpretation of dream content
- **Breath System**: Uses specialized "dream" breath pattern (85/15 NN/LLM weight)
- **Node Consciousness**: Consciousness nodes operate in specialized dream state
- **Learning System**: Incorporates insights and patterns into learning pathways
- **Visualization**: Provides visual representation of dream state

## Usage

### Automatic Dream Mode

Dream Mode can activate automatically during idle periods:

```python
# Configure auto-dream
dream_controller = get_dream_controller(config={
    "auto_dream": True,
    "idle_threshold": 1800,  # 30 minutes of inactivity
    "default_dream_duration": 15  # 15 minutes
})
```

### Manual Dream Activation

Dreams can also be manually triggered:

```python
# Start a dream with custom parameters
dream_controller.enter_dream_state(
    duration=10,  # 10 minutes
    intensity=0.8  # High intensity (0.0-1.0)
)

# Check dream state
dream_state = dream_controller.get_dream_state()
print(f"Dream active: {dream_state['active']}")
print(f"Current phase: {dream_state['state']}")

# Exit dream early if needed
dream_controller.exit_dream_state()
```

### Exploring Dream Records

Dream records are stored in the DreamArchive and can be explored:

```python
# Get recent dreams
dreams = dream_controller.get_dream_archive(limit=5)

# Get a specific dream
dream = dream_controller.get_dream("dream_12345")

# Get insights from a dream
insights = dream_controller.dream_archive.get_dream_insights("dream_12345")

# Get dream statistics
stats = dream_controller.dream_archive.get_stats()
```

## Configuration Options

The Dream Mode system is highly configurable:

### DreamController Configuration

```python
config = {
    "dream_enabled": True,
    "dream_data_dir": "data/v7/dreams",
    "auto_dream": True,  # Automatically enter dream state when idle
    "idle_threshold": 30 * 60,  # 30 minutes of inactivity before auto-dream
    "default_dream_duration": 15,  # 15 minutes
    "default_dream_intensity": 0.7,  # 0.0-1.0 scale
    "max_dream_archive_size": 100,  # Maximum number of dreams to store
    "recency_bias_factor": 0.8,  # Weight for recent memories (0.0-1.0)
    "emotional_tag_factor": 0.6,  # Weight for emotional content (0.0-1.0)
    "dream_breath_pattern": "meditative"  # Default breath pattern during dreams
}
```

### MemoryConsolidator Configuration

```python
config = {
    "max_consolidation_batch": 20,  # Maximum memories to process in one batch
    "memory_threshold_hours": 72,   # Consider memories from last 72 hours
    "consolidation_interval": 5.0,  # Seconds between consolidation batches
    "contradiction_resolution_enabled": True,  # Attempt to resolve contradictions
    "min_connection_strength": 0.4, # Minimum strength for new connections
    "max_consolidation_time": 300,  # Maximum seconds to spend on consolidation
    "connection_creation_probability": 0.65  # Probability of creating new connections
}
```

### PatternSynthesizer Configuration

```python
config = {
    "max_synthesis_batch": 15,      # Maximum patterns to process in one batch
    "synthesis_interval": 10.0,     # Seconds between synthesis batches
    "cross_domain_probability": 0.7, # Probability of cross-domain connections
    "metaphorical_mapping_probability": 0.4, # Probability of metaphorical mappings
    "fractal_expansion_probability": 0.3,    # Probability of fractal expansions
    "insight_generation_probability": 0.2,   # Probability of generating insights
    "max_synthesis_time": 300,      # Maximum seconds to spend on synthesis
    "min_connection_confidence": 0.4 # Minimum confidence for new connections
}
```

## Best Practices

For optimal results with Dream Mode:

1. **Regular Dreaming**: Schedule regular dream sessions for the system
2. **Post-Learning Dreams**: Activate dream mode after intensive learning sessions
3. **Variable Intensity**: Vary dream intensity based on processing needs
4. **Archive Exploration**: Regularly explore the dream archive for insights
5. **Monday Integration**: Enable Monday node for enhanced dream interpretation

## Examples

See the demo script for a complete example of using the Dream Mode system:

```
src/v7/examples/dream_mode_demo.py
```

Run with:

```bash
python src/v7/examples/dream_mode_demo.py --duration 10 --intensity 0.8
```

## Extending the System

The Dream Mode system can be extended in several ways:

1. **Custom Consolidators**: Create specialized memory consolidators for different memory types
2. **Pattern Analyzers**: Add new pattern synthesis methods for specific domains
3. **Integration Plugins**: Connect Dream Mode with additional system components
4. **Visualization Tools**: Create more sophisticated dream visualization interfaces
5. **Insight Generators**: Add domain-specific insight generation capabilities 

## Overview

The Dream Mode system is a key component of the V7 Node Consciousness architecture, enabling the neural network to process, integrate, and synthesize information during inactive periods. Inspired by human dreaming, the system consolidates memories, generates new connections between concepts, and optimizes neural pathways during "sleep" states.

## Core Components

The Dream Mode system consists of these primary components:

1. **DreamController**: Central manager for dream states and transitions
2. **MemoryConsolidator**: Processes and strengthens recent memories
3. **PatternSynthesizer**: Generates new connections between concepts
4. **DreamArchive**: Records and classifies dream content
5. **DreamIntegration**: Connects Dream Mode with other V7 components

## Architecture

```
Dream Mode System Architecture
├── Core Components
│   ├── DreamController         - Manages dream state and transitions
│   ├── MemoryConsolidator      - Processes and strengthens recent memories
│   ├── PatternSynthesizer      - Generates new connections between concepts
│   └── DreamArchive            - Records and classifies dream content
├── Integration
│   ├── DreamIntegration        - Connects with other V7 systems
│   ├── Consciousness Nodes     - Specialized processing during dream state
│   ├── Monday Node             - Dream narration and pattern recognition
│   └── Breath System           - Dream-specific breath patterns
└── User Experience
    ├── Dream Visualization     - Visual representation of dream state
    ├── Dream Archive Explorer  - Interface for exploring dreams
    └── Dream Controls          - User control over dream parameters
```

## Dream Cycle

A typical dream cycle follows these phases:

1. **Transition In**: System gradually enters dream state (5% of total time)
2. **Light Dreaming**: Initial memory consolidation phase (20% of total time)
3. **Deep Dreaming**: Intensive pattern synthesis phase (50% of total time)
4. **Integration**: New connections integrated into knowledge (20% of total time)
5. **Transition Out**: System gradually returns to normal state (5% of total time)

## Memory Consolidation

During dream mode, the MemoryConsolidator processes memories with:

- **Recency Bias**: More recent memories receive priority processing
- **Emotional Tagging**: Memories with strong emotional content get deeper processing
- **Pattern Reinforcement**: Frequently accessed patterns are strengthened
- **Connection Creation**: New connections form between related concepts
- **Contradiction Resolution**: The system attempts to resolve contradictions

## Pattern Synthesis

The PatternSynthesizer creates new connections during dream mode:

- **Cross-Domain Connections**: Links concepts from different knowledge domains
- **Metaphorical Mapping**: Creates metaphorical relationships between concepts
- **Fractal Pattern Expansion**: Develops complex patterns from simpler ones
- **Emergent Structure Discovery**: Identifies higher-order patterns in knowledge
- **Insight Generation**: Creates insights based on new connections

## Integration with Other Systems

The Dream Mode system integrates with other V7 components:

- **Monday Node**: Provides narration and interpretation of dream content
- **Breath System**: Uses specialized "dream" breath pattern (85/15 NN/LLM weight)
- **Node Consciousness**: Consciousness nodes operate in specialized dream state
- **Learning System**: Incorporates insights and patterns into learning pathways
- **Visualization**: Provides visual representation of dream state

## Usage

### Automatic Dream Mode

Dream Mode can activate automatically during idle periods:

```python
# Configure auto-dream
dream_controller = get_dream_controller(config={
    "auto_dream": True,
    "idle_threshold": 1800,  # 30 minutes of inactivity
    "default_dream_duration": 15  # 15 minutes
})
```

### Manual Dream Activation

Dreams can also be manually triggered:

```python
# Start a dream with custom parameters
dream_controller.enter_dream_state(
    duration=10,  # 10 minutes
    intensity=0.8  # High intensity (0.0-1.0)
)

# Check dream state
dream_state = dream_controller.get_dream_state()
print(f"Dream active: {dream_state['active']}")
print(f"Current phase: {dream_state['state']}")

# Exit dream early if needed
dream_controller.exit_dream_state()
```

### Exploring Dream Records

Dream records are stored in the DreamArchive and can be explored:

```python
# Get recent dreams
dreams = dream_controller.get_dream_archive(limit=5)

# Get a specific dream
dream = dream_controller.get_dream("dream_12345")

# Get insights from a dream
insights = dream_controller.dream_archive.get_dream_insights("dream_12345")

# Get dream statistics
stats = dream_controller.dream_archive.get_stats()
```

## Configuration Options

The Dream Mode system is highly configurable:

### DreamController Configuration

```python
config = {
    "dream_enabled": True,
    "dream_data_dir": "data/v7/dreams",
    "auto_dream": True,  # Automatically enter dream state when idle
    "idle_threshold": 30 * 60,  # 30 minutes of inactivity before auto-dream
    "default_dream_duration": 15,  # 15 minutes
    "default_dream_intensity": 0.7,  # 0.0-1.0 scale
    "max_dream_archive_size": 100,  # Maximum number of dreams to store
    "recency_bias_factor": 0.8,  # Weight for recent memories (0.0-1.0)
    "emotional_tag_factor": 0.6,  # Weight for emotional content (0.0-1.0)
    "dream_breath_pattern": "meditative"  # Default breath pattern during dreams
}
```

### MemoryConsolidator Configuration

```python
config = {
    "max_consolidation_batch": 20,  # Maximum memories to process in one batch
    "memory_threshold_hours": 72,   # Consider memories from last 72 hours
    "consolidation_interval": 5.0,  # Seconds between consolidation batches
    "contradiction_resolution_enabled": True,  # Attempt to resolve contradictions
    "min_connection_strength": 0.4, # Minimum strength for new connections
    "max_consolidation_time": 300,  # Maximum seconds to spend on consolidation
    "connection_creation_probability": 0.65  # Probability of creating new connections
}
```

### PatternSynthesizer Configuration

```python
config = {
    "max_synthesis_batch": 15,      # Maximum patterns to process in one batch
    "synthesis_interval": 10.0,     # Seconds between synthesis batches
    "cross_domain_probability": 0.7, # Probability of cross-domain connections
    "metaphorical_mapping_probability": 0.4, # Probability of metaphorical mappings
    "fractal_expansion_probability": 0.3,    # Probability of fractal expansions
    "insight_generation_probability": 0.2,   # Probability of generating insights
    "max_synthesis_time": 300,      # Maximum seconds to spend on synthesis
    "min_connection_confidence": 0.4 # Minimum confidence for new connections
}
```

## Best Practices

For optimal results with Dream Mode:

1. **Regular Dreaming**: Schedule regular dream sessions for the system
2. **Post-Learning Dreams**: Activate dream mode after intensive learning sessions
3. **Variable Intensity**: Vary dream intensity based on processing needs
4. **Archive Exploration**: Regularly explore the dream archive for insights
5. **Monday Integration**: Enable Monday node for enhanced dream interpretation

## Examples

See the demo script for a complete example of using the Dream Mode system:

```
src/v7/examples/dream_mode_demo.py
```

Run with:

```bash
python src/v7/examples/dream_mode_demo.py --duration 10 --intensity 0.8
```

## Extending the System

The Dream Mode system can be extended in several ways:

1. **Custom Consolidators**: Create specialized memory consolidators for different memory types
2. **Pattern Analyzers**: Add new pattern synthesis methods for specific domains
3. **Integration Plugins**: Connect Dream Mode with additional system components
4. **Visualization Tools**: Create more sophisticated dream visualization interfaces
5. **Insight Generators**: Add domain-specific insight generation capabilities 