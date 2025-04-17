# LUMINA V7 Dashboard

A modern, configurable visualization and monitoring dashboard for the LUMINA V7 Neural Integration System.

![Dashboard Example](screenshots/dashboard_example.png) *(placeholder image)*

## Quick Start

1. **Install dependencies:**
   ```batch
   install_dashboard_dependencies.bat
   ```

2. **Run the dashboard:**
   ```batch
   run_qt_dashboard.bat
   ```

## Features

- Real-time neural network activity visualization
- Language processing metrics and integration monitoring
- System resource monitoring and performance metrics
- Dream mode pattern visualization
- Memory system visualization
- Configurable UI with themes and layout options

## Configuration

The dashboard uses several configuration files in the `config` directory:

- **dashboard_config.json** - Main configuration
- **visualization_config.json** - Visualization settings
- **panel_plugins.json** - Panel registration and loading
- **auth_config.json** - Authentication and access control

If these files don't exist, they'll be created automatically with default values when you run:

```bash
python src/visualization/create_default_configs.py
```

## Customization

You can customize the dashboard by editing the configuration files:

### Main Dashboard Settings

Edit `config/dashboard_config.json` to configure:
- Refresh rates and history limits
- Mock mode settings for testing
- Connection parameters for the V7 system
- Panel-specific settings
- UI preferences
- API and security settings

Example:
```json
{
  "dashboard": {
    "refresh_rate": 1000,
    "max_history": 3600
  },
  "ui": {
    "color_theme": "dark",
    "layout_type": "tabbed"
  }
}
```

### Visualization Settings

Edit `config/visualization_config.json` to configure:
- Visualization backends (PyQtGraph, Matplotlib)
- Color themes and styles
- Line and symbol styles
- 3D visualization settings

Example:
```json
{
  "global": {
    "preferred_backend": "pyqtgraph",
    "fallback_backend": "matplotlib"
  },
  "themes": {
    "dark": {
      "background": "#2d3436",
      "foreground": "#ecf0f1"
    }
  }
}
```

## Running with Mock Data

To run the dashboard with mock data (without connecting to a real LUMINA V7 system):

```batch
run_qt_dashboard.bat
```

The dashboard will automatically generate realistic mock data for testing and demonstration.

## Running with Real V7 System

Ensure the LUMINA V7 system is running, then:

1. Edit the `run_qt_dashboard.bat` file to set the correct connection port
2. Run the dashboard:
   ```batch
   run_qt_dashboard.bat
   ```

## API Access

The dashboard includes a REST API that can be enabled in the configuration. To enable:

1. Edit `config/dashboard_config.json` and set:
   ```json
   "api": {
     "enabled": true,
     "port": 5679
   }
   ```

2. Restart the dashboard

3. Access the API at `http://localhost:5679/api/v1/metrics`

## Security

By default, authentication is disabled. To enable:

1. Edit `config/auth_config.json` and set:
   ```json
   "auth_enabled": true
   ```

2. Configure users and permissions in the same file

3. Restart the dashboard

## System Requirements

- Python 3.8+
- PyQt5 or PySide6
- PyQtGraph (for enhanced visualizations)
- Matplotlib (as fallback visualization)
- psutil (for system monitoring)
- GPUtil (optional, for GPU monitoring)

## Troubleshooting

See the [panelsreadme.md](panelsreadme.md) file for detailed troubleshooting information.

## Further Documentation

For complete documentation, refer to:
- [panelsreadme.md](panelsreadme.md): Detailed panel documentation
- [v7readme.md](v7readme.md): LUMINA V7 system documentation 