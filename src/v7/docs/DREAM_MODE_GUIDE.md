# Dream Mode User Guide

## Introduction

Dream Mode is a key feature of Lumina V7.0.0.2 that enables the system to process information, consolidate memories, and generate new connections during periods of low activity. This mimics the cognitive processes that occur during human dreaming, allowing for better integration of knowledge and creative insight generation.

## Understanding Dream Mode

Dreams in Lumina V7 consist of several phases:

1. **Initiation** - The system prepares for the dream state, reducing external processing
2. **Dream phases** - Multiple phases (typically 3) with varying intensity levels:
   - Early phase: Low-medium intensity, focused on memory consolidation
   - Middle phase: High intensity, focused on pattern synthesis
   - Late phase: Medium intensity, with balanced processing
3. **Awakening** - Gradual transition back to normal operation

During these phases, two primary processes occur:
- **Memory consolidation**: Reviewing, strengthening, and connecting recent memories
- **Pattern synthesis**: Creating new connections between concepts across domains

## Using Dream Mode

### Automatic Dreams

By default, Lumina V7 is configured to automatically enter dream states during periods of inactivity. You can control this behavior through configuration:

```python
# Disable automatic dream cycles
config = {
    "auto_dream_enabled": False
}

# With longer intervals between dreams (12 hours)
config = {
    "auto_dream_enabled": True,
    "dream_interval_hours": 12
}

# Get dream controller with this configuration
from src.v7.lumina_v7.core.dream_controller import get_dream_controller
dream_controller = get_dream_controller(config=config)
```

### Manual Dream Control

You can manually trigger and control dream states:

```python
from src.v7.lumina_v7.core.dream_controller import get_dream_controller

# Get the dream controller
dream_controller = get_dream_controller()

# Start a dream with default parameters
dream_controller.start_dream()

# Start a dream with custom parameters
dream_controller.start_dream(
    duration_minutes=20,  # 20-minute dream
    intensity=0.8         # Higher intensity (0.0-1.0)
)

# Check dream status
state = dream_controller.get_dream_state()
print(f"Dream active: {state['active']}")
print(f"Current phase: {state['phase']}")
print(f"Intensity: {state['intensity']}")

# End dream early if needed
dream_controller.end_dream(reason="user_request")
```

### Examining Dream Records

Dream records are archived and can be retrieved:

```python
# Get current dream if active
current_dream = dream_controller.get_current_dream()

# Get recent dream history (last 5 dreams)
recent_dreams = dream_controller.get_dream_history(limit=5)

# Use the dream archive for more advanced queries
from src.v7.lumina_v7.core.dream_archive import DreamArchive

archive = DreamArchive()

# Search for dreams by criteria
insightful_dreams = archive.search_dreams({
    "insights_count_min": 3,
    "since": "2023-07-01"
})

# Get monthly dream summary
monthly_stats = archive.get_monthly_summary()
```

## Configuration Options

Dream Mode has many configuration options that can be tuned:

### Dream Controller Options

| Option | Description | Default |
|--------|-------------|---------|
| `auto_dream_enabled` | Enable automatic dream cycles | `True` |
| `dream_interval_hours` | Hours between automatic dream cycles | `8` |
| `min_dream_duration` | Minimum dream duration in minutes | `10` |
| `max_dream_duration` | Maximum dream duration in minutes | `30` |
| `dream_phase_count` | Number of phases in a dream cycle | `3` |
| `memory_consolidation_weight` | Weight for memory consolidation | `0.6` |
| `pattern_synthesis_weight` | Weight for pattern synthesis | `0.4` |
| `auto_awakening` | Automatically awaken from dreams | `True` |
| `default_dream_intensity` | Default intensity for dreams | `0.7` |
| `dream_jitter` | Random variation in dream timing/intensity | `0.2` |
| `archive_dreams` | Save dream records to archive | `True` |

### Memory Consolidator Options

| Option | Description | Default |
|--------|-------------|---------|
| `max_consolidation_batch` | Maximum memories to process in one batch | `20` |
| `memory_threshold_hours` | Consider memories from last N hours | `72` |
| `contradiction_resolution_enabled` | Attempt to resolve contradictions | `True` |
| `min_connection_strength` | Minimum strength for new connections | `0.4` |

### Pattern Synthesizer Options

| Option | Description | Default |
|--------|-------------|---------|
| `max_synthesis_batch` | Maximum patterns to process in one batch | `15` |
| `cross_domain_probability` | Probability of cross-domain connections | `0.7` |
| `metaphorical_mapping_probability` | Probability of metaphorical mappings | `0.4` |
| `fractal_expansion_probability` | Probability of fractal expansions | `0.3` |
| `insight_generation_probability` | Probability of generating insights | `0.2` |

## Example Configurations

### Light Dreamer (minimal resource usage)

```python
config = {
    "min_dream_duration": 5,
    "max_dream_duration": 15,
    "dream_phase_count": 2,
    "max_consolidation_batch": 10,
    "max_synthesis_batch": 5,
    "dream_interval_hours": 12
}
```

### Deep Thinker (maximum insight generation)

```python
config = {
    "min_dream_duration": 20,
    "max_dream_duration": 60,
    "dream_phase_count": 5,
    "phase_intensity_curve": [0.3, 0.5, 1.0, 0.7, 0.4],
    "max_consolidation_batch": 30,
    "max_synthesis_batch": 25,
    "cross_domain_probability": 0.9,
    "insight_generation_probability": 0.4
}
```

## Troubleshooting

### Dream Mode Doesn't Start

- Check if another dream is already active
- Verify the dream controller is properly initialized
- Ensure no critical system processes are running that block dreams

### High Memory Usage During Dreams

- Reduce `max_consolidation_batch` and `max_synthesis_batch`
- Decrease dream duration and intensity
- Limit the number of archive dreams with `max_size` parameter

### Poor Quality Insights

- Increase dream intensity and duration
- Raise `cross_domain_probability` and `insight_generation_probability`
- Check if the system has enough varied knowledge to draw connections from

## Best Practices

1. **Balance frequency and depth** - Shorter, more frequent dreams often work better than rare, long dreams
2. **Adjust to your knowledge base** - Systems with more knowledge benefit from higher cross-domain settings
3. **Schedule appropriately** - Set dream cycles during expected low-usage periods
4. **Monitor dream records** - Regularly review the insights and connections being made
5. **Iterative tuning** - Adjust configuration based on the quality of insights generated 