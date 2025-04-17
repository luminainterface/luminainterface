# Lumina V6.5 Bridge Connector

The V6.5 Bridge Connector is a comprehensive integration solution that connects all previous version bridges (v1-2, v3-4, and v5) to the V7 Node Consciousness system. It serves as a central hub for inter-version communication, allowing seamless data flow between different system components.

## Features

- **Complete Integration**: Connects all previous version bridges into a unified system
- **Centralized Event Routing**: Routes events between different versions based on event type
- **Flexible Configuration**: Configurable through command-line arguments or config files
- **Mock Mode**: Includes mock implementations for testing and development
- **V7 Compatibility**: Fully compatible with the V7 Node Consciousness system

## Architecture

The V6.5 Bridge Connector integrates the following components:

1. **V1-V2 Bridge**: Connects the text interface (v1) with the graphical interface (v2)
2. **V3-V4 Connector**: Integrates the advanced interfaces with breath and glyph processing
3. **V5 Bridge**: Simplified language memory system integration
4. **V7 Connector**: Connection to the Node Consciousness system

```
+---------------+       +---------------+       +---------------+
| V1-V2 Bridge  |------>|   V6.5 Bridge |<----->| V7 Node       |
+---------------+       |   Connector   |       | Consciousness |
                        |               |       +---------------+
+---------------+       |               |
| V3-V4         |------>|               |
| Connector     |       |               |
+---------------+       |               |
                        |               |
+---------------+       |               |
| V5 Language   |------>|               |
| Memory Bridge |       |               |
+---------------+       +---------------+
```

## Using the V6.5 Bridge Connector

### From Command Line

To enable the V6.5 Bridge Connector when launching the V7 system, use the `--v65-bridge` flag:

```bash
python -m src.v7.lumina_v7.launch_v7 --v65-bridge
```

Additional options:
- `--mock`: Use mock implementations (for testing)
- `--debug`: Enable debug logging
- `--no-ui`: Run without UI components

### In Code

```python
from src.v7.lumina_v7 import initialize_v7, create_default_config
from src.v7.lumina_v7.core.v65_bridge_connector import create_v65_bridge_connector

# Create configuration
config = create_default_config()
config["v65_bridge_enabled"] = True

# Initialize V7 system
manager, context = initialize_v7(config)

# Create V6.5 Bridge Connector
v65_bridge = create_v65_bridge_connector(config)

# Connect to V7
if "connector" in context and context["connector"]:
    v65_bridge.connect_to_v7(context["connector"])

# Register event handlers
v65_bridge.register_handler("text_input", my_handler, source="v1v2_bridge")

# Start the bridge
v65_bridge.start()

# When finished
v65_bridge.stop()
```

## Event Routing

The V6.5 Bridge Connector routes events between different versions based on their type:

| Event Type | Source | Destinations |
|------------|--------|-------------|
| `text_input`, `text_command` | v1-v2 | v3-v4, v5 |
| `breath_state`, `breath_pattern` | v3-v4 | v5, v7 |
| `glyph_update`, `visual_pattern` | v3-v4 | v5 |
| `memory_query`, `memory_update` | v5 | v7 |
| `topic_update`, `language_pattern` | v5 | v7 |
| `node_request`, `node_response` | v7 | v5 |

## Example

See the demonstration script for a complete example:

```python
# Import required modules
from src.v7.lumina_v7 import initialize_v7, shutdown_v7, create_default_config
from src.v7.lumina_v7.core.v65_bridge_connector import create_v65_bridge_connector

# Create and configure the bridge
config = create_default_config()
config["v65_bridge_enabled"] = True
manager, context = initialize_v7(config)
v65_bridge = create_v65_bridge_connector(config)

# Connect to V7
if "connector" in context:
    v65_bridge.connect_to_v7(context["connector"])

# Register handlers and start
v65_bridge.register_handler("text_input", my_handler)
v65_bridge.start()

# Use the bridge
v65_bridge.emit_event("text_input", {"text": "Hello"}, source="v1v2_bridge")

# When done
v65_bridge.stop()
shutdown_v7(context)
```

For a full working demo, see `src/v7/lumina_v7/examples/v65_bridge_demo.py`.

## Troubleshooting

If the V6.5 Bridge Connector fails to initialize or connect to specific bridges:

1. Check that the appropriate bridge components are available
2. Verify the paths to bridge modules in your Python path
3. Run with `--debug` flag to see detailed logging information
4. Use `--mock` mode for testing if real components are not available

## Extending

To add a new bridge type to the V6.5 Bridge Connector:

1. Create a new initialization method in the `V65BridgeConnector` class
2. Add event routing rules for the new bridge in `_route_event_by_type`
3. Update the configuration with appropriate flags for the new bridge

## Technical Notes

- The V6.5 Bridge Connector uses an event-based architecture with a processing thread
- Events are queued and processed asynchronously
- Each bridge can be enabled/disabled independently
- The connector includes a MockBridge implementation for testing 