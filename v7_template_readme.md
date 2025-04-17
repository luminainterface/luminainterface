# LUMINA V7 PySide6 Plugin Template

A flexible, modern template for PySide6 applications with a 16:9 aspect ratio and plugin architecture, designed for LUMINA V7 component integration.

## Features

- **16:9 Aspect Ratio**: Optimal for modern displays
- **Plugin Architecture**: Easily extend with new components
- **Flexible Layout**: Dockable and tabbed interface components
- **Event System**: Inter-plugin communication through events
- **Template Plugins**: Ready-to-use templates for common plugin types

## Quick Start

To launch the template application:

```
.\run_v7_template_ui.bat
```

## Template Structure

The template provides a complete framework ready to accept plugins:

### Main Components

- **Main Window**: 16:9 optimized window with flexible layout
- **Plugin Manager**: Handles discovery, loading, and activation of plugins
- **Event System**: Enables communication between different plugins
- **UI Framework**: Docking system, tabs, and consistent styling

### Template Plugins

Three template plugin types are included for demonstration:

1. **Chat Plugin**: Framework for text-based interaction
2. **Visualization Plugin**: 16:9 visualization area with controls
3. **Memory Plugin**: Storage and retrieval interface

## Creating Plugins

A sample plugin is automatically created when running the application. To create your own plugins:

1. Create a new Python file in the `plugins` directory
2. Create a class named `Plugin` that inherits from `PluginInterface`
3. Implement required methods like `initialize`, `get_dock_widgets`, etc.
4. Restart the application or click "Refresh Plugins"

### Plugin Interface

All plugins must implement the following interface:

```python
class Plugin(PluginInterface):
    def __init__(self, app_context):
        # Initialize with application context
        
    def initialize(self) -> bool:
        # Initialize the plugin
        return True
        
    def get_dock_widgets(self) -> List[QDockWidget]:
        # Return list of dock widgets
        
    def get_tab_widgets(self) -> List[tuple]:
        # Return list of (name, widget) tuples
        
    def get_toolbar_actions(self) -> List[QAction]:
        # Return list of toolbar actions
        
    def get_menu_actions(self) -> Dict[str, List[QAction]]:
        # Return menu actions
        
    def shutdown(self) -> None:
        # Clean shutdown
```

## Integration with Existing Components

The template is designed to easily integrate with:

- **V7 System**: Neural network and core V7 functionality
- **Mistral Integration**: Language model capabilities
- **Memory Systems**: Both onsite and database memory integration
- **Visualization**: Neural activity and data visualization

## Plugin Examples for Integration

### Mistral Chat Plugin Example

```python
class Plugin(PluginInterface):
    def __init__(self, app_context):
        super().__init__(app_context)
        self.name = "Mistral Chat"
        # Initialize Mistral components
        
    def initialize(self):
        # Connect to Mistral API
        # Set up UI components
        return True
```

### V7 Visualization Plugin Example

```python
class Plugin(PluginInterface):
    def __init__(self, app_context):
        super().__init__(app_context)
        self.name = "V7 Visualization"
        # Initialize visualization components
        
    def initialize(self):
        # Set up neural visualization
        # Connect to data sources
        return True
```

## System Requirements

- Python 3.8 or higher
- PySide6 (installed automatically if missing)
- Additional requirements may be specified by individual plugins

## Development and Extension

The template is designed for easy extension. Key areas for customization:

- Add new plugin types in the main application
- Enhance the plugin manager for more advanced plugin management
- Create specialized plugins for specific V7 functionality
- Add communication bridges between different plugin types

## Future Enhancements

Planned enhancements for future versions:

- Hot-reloading of plugins
- Plugin dependency management
- Enhanced styling and themes
- Package management for plugins
- Socket-based integration for external components 