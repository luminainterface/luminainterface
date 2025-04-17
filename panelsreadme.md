# LUMINA V7 Unified Dashboard Panels

This document provides information about the Qt-based dashboard panels for the LUMINA V7 Unified System launcher.

## Overview

The LUMINA V7 Dashboard provides a modern, graphical interface to monitor and interact with the LUMINA V7 Neural Integration System. The panels offer real-time visualizations of neural activity, language processing metrics, learning rates, and system resources.

![Dashboard Example](screenshots/dashboard_example.png) *(This is a placeholder - actual screenshots should be added)*

## Features

- **Real-time Neural Activity Monitoring**: Visualize consciousness network patterns
- **Language Integration Dashboard**: Track Mistral LLM integration metrics
- **System Resource Monitoring**: Monitor memory usage and processing load
- **Dream Mode Visualization**: View neural pattern generation during dream states
- **Breath Detection Panel**: Visualize breathing rhythm integration
- **Configuration Interface**: Adjust neural and LLM weights through GUI

## Installation Requirements

### Required Dependencies

- Python 3.8+ 
- PyQt5 or PySide6 (Qt for Python)
- PyQtGraph (for enhanced visualizations)
- Matplotlib (as fallback visualization)
- SQLite3 (for metrics database)
- psutil (for system monitoring)
- GPUtil (optional, for GPU monitoring)

### Installation Steps

1. Install Python dependencies:

```bash
pip install PyQt5 PyQtGraph matplotlib numpy pandas psutil
```

2. If using PySide6 instead of PyQt5:

```bash
pip install PySide6
```

3. For GPU monitoring (optional):

```bash
pip install GPUtil
```

4. Create necessary directories for LUMINA:

```bash
mkdir -p data/neural_metrics logs config
```

## Setting Up Dashboard Panels

The dashboard panels can be integrated with the main LUMINA V7 launcher in two ways:

### Option 1: Standalone Dashboard

Run the dashboard independently from the main LUMINA system:

```bash
python src/visualization/create_dashboard_qt.py
```

### Option 2: Integrated Dashboard

Edit your `run_qt_dashboard.bat` file to contain:

```batch
@echo off
echo Starting LUMINA V7 Dashboard...

:: Set environment variables
set PYTHONPATH=%CD%;%CD%\src
set DASHBOARD_PORT=5679
set METRICS_DB_PATH=data/neural_metrics.db
set V7_CONNECTION_PORT=5678

:: Create necessary directories
if not exist data\neural_metrics mkdir data\neural_metrics
if not exist logs mkdir logs
if not exist config mkdir config

:: Check for dependencies
python src/visualization/check_dashboard_requirements.py
if errorlevel 1 (
    echo Dashboard dependencies missing, please install them.
    echo Run: pip install PyQt5 PyQtGraph matplotlib numpy pandas
    pause
    exit /b 1
)

:: Start the dashboard
echo Launching Dashboard...
start /B python src/visualization/create_dashboard_qt.py --v7-port %V7_CONNECTION_PORT% --db-path %METRICS_DB_PATH%

:: Start the bridge to LUMINA V7
start /B python src/visualization/dashboard_v7_bridge.py --port %V7_CONNECTION_PORT% --db-path %METRICS_DB_PATH%

echo Dashboard is now running.
echo Close this window to shut down the dashboard.
pause
```

## V7 Connection Requirements

For the dashboard panels to properly connect with the main LUMINA V7 system, several specific connection requirements must be met:

### Socket Communication Architecture

The LUMINA V7 dashboard uses a client-server socket architecture:

1. **Server (Main V7 System)**:
   - Implemented in `dashboard_metrics_plugin.py`
   - Listens on port 5678 by default (configurable)
   - Responds to JSON-formatted metrics requests
   - Provides real-time data about system metrics

2. **Client (Dashboard Panels)**:
   - Implemented in `dashboard_v7_bridge.py`
   - Connects to the server socket on port 5678
   - Sends structured requests for specific metrics
   - Receives and processes JSON response data

3. **Communication Protocol**:
   - JSON-formatted messages
   - Request-response pattern
   - Defined message types (e.g., "metrics_request")
   - Standardized data structures for metrics

### Connection Testing Tools

The system includes specialized tools for testing socket connections:

1. **Test Server**:
   - `test_dashboard_connection.py` - A standalone server that simulates the V7 metrics server
   - Listens on port 5678 and responds with mock metrics data
   - Useful for testing dashboard connectivity without running the full V7 system

2. **Test Client**:
   - `test_dashboard_client.py` - A simple client that connects to the metrics server
   - Sends a metrics request and displays the response
   - Helps verify server functionality and data format

### Running the Connection Test Tools

To diagnose connection issues:

```bash
# Start the test server in one terminal
python test_dashboard_connection.py

# Run the test client in another terminal
python test_dashboard_client.py
```

The client should connect successfully and display mock metrics data from the test server.

### Common Connection Issues and Solutions

1. **Port Already in Use**:
   - Error: "Address already in use" when starting the server
   - Solution: Check if another process is using port 5678 with `netstat -ano | findstr "5678"`
   - Alternative: Configure a different port using the `--v7-port` parameter

2. **Server Not Running**:
   - Error: "Connection refused" when the dashboard tries to connect
   - Solution: Ensure the V7 system or test server is running before starting the dashboard
   - Check: Verify the server is listening using `netstat -an | findstr "LISTENING" | findstr "5678"`

3. **Socket Connection Timeout**:
   - Error: "Connection timed out" when trying to connect
   - Solution: Check for firewall issues or network configuration problems
   - Verify: Test with localhost connections only to rule out network issues

4. **Mock Mode Operation**:
   - When connections fail, the dashboard can run in mock mode
   - Enable with: `--mock=True` parameter
   - This generates simulated data for testing the UI without a functioning server

### Integration with LUMINA V7 System

To ensure proper integration:

1. **Plugin Registration**:
   - The dashboard_metrics_plugin must be properly registered with the V7 plugin system
   - Check the plugin registration in `run_lumina_v7_unified.bat`

2. **Component Discovery**:
   - The dashboard bridge attempts to discover available V7 components
   - Components should implement the standard metrics API

3. **Threading Considerations**:
   - Both server and client components use threading for non-blocking operation
   - Thread safety is maintained through mutex locks and thread-safe data structures

### Dashboard-V7 Communication Sequence

The standard communication sequence between the dashboard and V7 system:

1. V7 system starts and initializes the dashboard_metrics_plugin
2. The plugin starts a socket server on the configured port
3. Dashboard panels are launched and the dashboard_v7_bridge connects to the server
4. The bridge periodically sends metrics requests to the server
5. The server responds with current metrics data
6. The bridge processes the data and updates the dashboard UI

## Integration Roadmap with LUMINA V7 Unified System

This roadmap outlines how the Qt Dashboard panels integrate with the main `run_lumina_v7_unified.bat` launcher and the visual components needed for each stage of development.

### Phase 1: Basic Integration (Current)

**Components:**
- Separate launcher (`run_qt_dashboard.bat`) connecting to the main LUMINA V7 system
- Basic metrics visualization for neural activity and language processing
- Socket-based communication via `dashboard_v7_bridge.py`

**Required Visuals:**
- Overview dashboard with system status indicators
- Neural activity time-series charts
- Language processing metrics graphs
- System resource monitors

**Integration Method:**
- The dashboard connects to LUMINA V7 via port 5678 (configurable)
- Metrics are collected from the LUMINA V7 system and stored in SQLite
- The Qt dashboard visualizes data from the database

### Phase 2: Embedded Dashboard (Next Step)

**Components:**
- Direct integration with `run_lumina_v7_unified.bat` menu system
- Embedded dashboard option in the main LUMINA interface
- Shared configuration between systems

**Required Visuals:**
- Enhanced menu system in LUMINA with dashboard option
- Dream mode visualization panels
- Breath detection rhythm displays
- Memory system network graphs

**Integration Steps:**
1. Modify `run_lumina_v7_unified.bat` to include dashboard launch option:
```batch
:MAIN_MENU
echo.
echo LUMINA V7.0.0.3 - What would you like to run?
echo 1. Full LUMINA V7 Unified System
echo 2. Dashboard Only (Consciousness Network and AutoWiki)
echo 3. Neural-Language Integration Demo
echo 4. System Configuration
echo 5. Qt Dashboard Panels (NEW)
echo 6. Exit
```

2. Add dashboard option handler to the batch file:
```batch
if errorlevel 5 goto QtDashboard
// ... existing code ...

:QtDashboard
call run_qt_dashboard.bat
goto MAIN_MENU
```

### Phase 3: Fully Unified System (Future)

**Components:**
- Single unified launcher for both systems
- Integrated configuration management
- Shared data structures and real-time visualization

**Required Visuals:**
- Advanced 3D consciousness network visualization
- Interactive neural weight adjustment interface
- Dream pattern 3D topology maps
- Comprehensive system monitoring dashboard

**Integration Method:**
- Modify LUMINA template UI to incorporate dashboard panels directly
- Create shared communication framework between UI components
- Implement unified configuration system

**Implementation Plan:**
1. Create visualization plugins for the LUMINA plugin system:
```
set TEMPLATE_PLUGINS_DIRS=plugins;src\v7\plugins;src\plugins;src\visualization\plugins
set TEMPLATE_AUTO_LOAD_PLUGINS=mistral_neural_chat_plugin.py;consciousness_system_plugin.py;auto_wiki_plugin.py;dream_mode_plugin.py;breath_detection_plugin.py;dashboard_visualization_plugin.py
```

2. Develop Qt panel components that can be embedded in the template UI
3. Implement unified configuration and state management
4. Create seamless communication between neural network and visualization components

## Working with run_lumina_v7_unified.bat

The main LUMINA V7 launcher has several components that the dashboard panels need to visualize:

### 1. Consciousness Network System

This system requires:
- Neural activity visualization panels
- Connection strength indicators
- Pattern formation displays
- Learning rate graphs

### 2. Mistral Neural Chat Integration

This requires:
- Language model activity panels
- Neural-language integration metrics
- Conversation flow visualization
- Semantic understanding charts

### 3. Dream Mode System

This requires:
- Dream state pattern visualization
- Creative pattern formation panels
- Neural weight shift indicators
- Dream archive viewers

### 4. Breath Detection System

This requires:
- Breathing rhythm visualization
- Neural entrainment displays
- Consciousness-breath integration panels

### 5. Memory System Visualization

This requires:
- Memory retrieval/storage visualization
- Conversation node network graphs
- Auto-wiki integration displays
- Memory consolidation panels
- Real-time visualization of memory access patterns
- Interactive network graph of associative memory connections
- Memory usage statistics and optimization recommendations
- Semantic relationship mapping between memory nodes
- Color-coded memory type identification (episodic, semantic, procedural)
- Timeline view of memory formation and consolidation
- Search interface for exploring stored memories

### 6. Dream Mode Panel

- Visualizes dream pattern generation
- Shows neural weight shifting during dream states
- Displays creativity metrics during autonomous dreaming

### 7. Configuration Panel

- Adjust neural weights and LLM weights
- Toggle system features (Dream Mode, Breath Detection)
- Configure memory system parameters

### 4. System Metrics Panel

- Real-time monitoring of CPU, memory, and GPU usage
- Visual graphs of resource utilization over time
- Color-coded indicators for resource status
- System information display with hardware details
- Adaptive visualization using PyQtGraph, Matplotlib, or custom rendering
- Configurable refresh rate and time window
- Support for both real metrics and mock data generation
- Performance threshold alerts and notifications
- Historical data comparison with previous runs
- Export capabilities for metrics data
- Resource allocation recommendations
- Process-specific monitoring for LUMINA components

### 5. Memory System Panel

- Real-time visualization of memory access patterns
- Interactive network graph of associative memory connections
- Memory usage statistics and optimization recommendations
- Semantic relationship mapping between memory nodes
- Color-coded memory type identification (episodic, semantic, procedural)
- Timeline view of memory formation and consolidation
- Search interface for exploring stored memories

## Data Architecture

The dashboard panels rely on a SQLite database for storing metrics:

```
data/neural_metrics.db
├── neural_activity (table)
├── language_metrics (table)
├── learning_metrics (table)
└── system_metrics (table)
```

### Bridge Architecture

The dashboard communicates with the LUMINA V7 system through a bridge component:

1. `src/visualization/dashboard_v7_bridge.py` connects to the LUMINA V7 system
2. Metrics are collected via socket communication
3. Data is stored in the SQLite database
4. The Qt dashboard reads and visualizes this data

## Visualization Techniques

The LUMINA V7 Dashboard employs various visualization techniques to effectively represent neural network activity, language processing, and system metrics.

### Real-time Graphing

- **Time-Series Data**: Neural activity, system resources, and language metrics are displayed as scrolling time-series data
- **Update Strategies**: Efficient update methods that minimize redraw operations for smoother visuals
- **Buffering**: Data buffering techniques to handle burst data without UI freezing

### Neural Network Visualization

- **Node-Edge Graphs**: Interactive force-directed graphs showing neural connections
- **Heat Maps**: Intensity-based visualizations of neural layer activations
- **3D Projections**: Dimensional reduction techniques (PCA, t-SNE) for visualizing high-dimensional neural spaces

### Advanced Chart Types

- **Sankey Diagrams**: For visualizing neural pattern flows and language processing paths
- **Radar Charts**: For multi-dimensional metrics comparison
- **Bubble Charts**: For visualizing multiple parameters simultaneously

### Performance Optimizations

- **Downsampling**: Automatic reduction of data points for large datasets
- **Incremental Rendering**: Progressive loading of visualization elements
- **GPU Acceleration**: Optional GPU-based rendering for complex visualizations
- **LOD (Level of Detail)**: Dynamic adjustment of visual complexity based on frame rate

### Visualization Backends

The dashboard supports multiple visualization backends that can be configured per panel:

1. **PyQtGraph**: Optimized for real-time data, with support for:
   - Fast line plots and scatter plots
   - Interactive zooming and panning
   - Real-time updates with minimal overhead

2. **Matplotlib**: For publication-quality visualizations:
   - Extensive customization options
   - Superior export capabilities (PNG, SVG, PDF)
   - Rich text and annotation support

3. **Custom Rendering**: For specialized visualizations:
   - Direct Qt QPainter usage for custom rendering
   - OpenGL-based rendering for 3D network visualizations
   - WebGL integration for complex interactive network graphs

## Customization

The dashboard offers extensive customization options to adapt to different use cases and preferences.

### Configuration Files

- **`config/dashboard_config.json`**
  ```json
  {
    "refresh_rate": 1000,
    "max_history": 3600,
    "panels": {
      "overview": {"enabled": true, "position": "top-left"},
      "neural": {"enabled": true, "position": "top-right"},
      "language": {"enabled": true, "position": "bottom-left"},
      "system": {"enabled": true, "position": "bottom-right"},
      "dream": {"enabled": true, "position": "floating"}
    }
  }
  ```

- **`config/visualization_config.json`**
  ```json
  {
    "color_theme": "dark",
    "custom_colors": {
      "neural_activity": "#3498db",
      "language_processing": "#2ecc71",
      "system_metrics": "#e74c3c",
      "dream_patterns": "#9b59b6"
    },
    "plot_styles": {
      "line_width": 1.5,
      "symbol_size": 5,
      "grid_alpha": 0.3
    },
    "fonts": {
      "title": "Segoe UI, 12pt, bold",
      "axis": "Segoe UI, 10pt",
      "legend": "Segoe UI, 9pt"
    }
  }
  ```

- **`config/panel_plugins.json`**
  ```json
  {
    "plugins": [
      {"name": "OverviewPanel", "module": "panels.overview", "enabled": true},
      {"name": "NeuralPanel", "module": "panels.neural_activity", "enabled": true},
      {"name": "LanguagePanel", "module": "panels.language_processing", "enabled": true},
      {"name": "SystemPanel", "module": "panels.system_metrics", "enabled": true},
      {"name": "DreamPanel", "module": "panels.dream_mode", "enabled": true},
      {"name": "ThirdPartyPanel", "module": "plugins.custom_panel", "enabled": false}
    ]
  }
  ```

### Theme Customization

The dashboard supports multiple themes that affect all visual elements:

- **Built-in Themes**: Dark, Light, System (follows OS theme)
- **Custom Themes**: Create your own theme by copying and modifying an existing theme file
- **Adaptive Elements**: UI elements automatically adjust to theme changes

### Layout Options

- **Docking System**: Panels can be rearranged via drag-and-drop
- **Tab System**: Group related panels into tabs to save space
- **Floating Panels**: Detach panels into separate windows
- **Layout Presets**: Save and load different panel arrangements
- **Split Views**: Create multi-panel views with adjustable splitters

### Data Visualization Options

- **Adjust plot parameters**: Line styles, marker types, colors, transparency
- **Change visualization type**: Switch between line plots, bar charts, scatter plots
- **Custom aggregations**: Configure how data is grouped and summarized
- **Export formats**: Configure default formats for data and image exports

### Extending the Dashboard

Instructions for creating custom panels:

1. Create a new Python file in `src/visualization/panels/`:

```python
from .base_panel import BasePanel
from PyQt5 import QtWidgets, QtCore

class CustomPanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__("Custom Panel", parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        
        # Create your custom UI elements
        self.plot = self.create_plot_widget()
        self.controls = self.create_control_widget()
        
        # Add to layout
        self.layout.addWidget(self.plot)
        self.layout.addWidget(self.controls)
        
    def update_data(self, data):
        # Handle incoming data updates
        pass
        
    def create_plot_widget(self):
        # Create your visualization widget
        return plot_widget
        
    def create_control_widget(self):
        # Create UI controls
        return control_widget
```

2. Register your panel in `config/panel_plugins.json`
3. Add configuration to `config/dashboard_config.json`

## Troubleshooting

**Q: Dashboard shows "Connecting..." but never connects to LUMINA**  
A: Ensure the LUMINA V7 system is running and the ports match in configuration.

**Q: Visualizations are not showing**  
A: Check that PyQtGraph is installed. The system will fall back to Matplotlib if PyQtGraph is unavailable.

**Q: "Mock Mode" appears in panels**  
A: This indicates the dashboard couldn't connect to the actual LUMINA system and is generating sample data.

**Q: GPU metrics are not showing in System Metrics Panel**  
A: Ensure GPUtil is installed. You can install it with `pip install GPUtil`. The panel will automatically detect available GPUs.

**Q: System Metrics Panel shows high CPU/memory but system seems normal**  
A: In mock mode, the panel generates random data. Check if the panel is running in mock mode or verify with other system monitoring tools.

## Deployment Guidance

The LUMINA V7 Dashboard can be deployed in various environments, from development workstations to production servers. This section provides guidance for different deployment scenarios.

### Local Development Environment

For dashboard development and testing:

1. **Setup**:
   ```bash
   git clone https://github.com/yourusername/lumina-v7-dashboard.git
   cd lumina-v7-dashboard
   pip install -r requirements.txt
   python src/visualization/create_dashboard_qt.py --mock=True
   ```

2. **Development Tools**:
   - Use Qt Designer for UI layout creation (`designer.exe` in PyQt5 installation)
   - Create custom panels with hot-reloading: `python src/visualization/create_dashboard_qt.py --dev-mode=True`
   - Test with mock data generation to avoid dependencies on LUMINA V7 system

### Production Deployment

For integrating with the full LUMINA V7 system:

1. **System Requirements**:
   - CPU: 4+ cores recommended for smooth visualization rendering
   - RAM: 8GB minimum, 16GB recommended
   - GPU: Recommended for advanced 3D visualizations
   - Storage: 500MB for dashboard code, plus database storage for metrics

2. **Installation**:
   - Include dashboard components in the main LUMINA V7 installation
   - Configure data persistence in `config/dashboard_config.json`:
     ```json
     {
       "data_persistence": {
         "storage_type": "sqlite",
         "db_path": "data/neural_metrics.db",
         "retention_days": 30,
         "auto_cleanup": true
       }
     }
     ```

3. **Performance Tuning**:
   - Adjust refresh rates based on system capability: `--refresh-rate=2000` (ms)
   - Limit data points for slower systems: `--max-points=1000`
   - Disable GPU acceleration if causing issues: `--gpu-accel=False`

### Standalone Deployment

For running the dashboard on a separate system from LUMINA V7:

1. **Network Configuration**:
   - Configure IP and port in both systems:
     ```bash
     python src/visualization/create_dashboard_qt.py --host=192.168.1.100 --port=5678
     ```
   - Set up credentials for secure connection:
     ```bash
     python src/visualization/create_dashboard_qt.py --auth-key="your-secret-key"
     ```

2. **Remote Monitoring Setup**:
   - Create a secure tunnel for remote monitoring
   - Set up automatic reconnection logic: `--auto-reconnect=True --reconnect-interval=5000`
   - Configure failover to mock data: `--fallback-to-mock=True`

### Containerized Deployment

For Docker-based deployment:

1. **Dockerfile**:
   ```Dockerfile
   FROM python:3.9-slim
   
   # System dependencies for PyQt
   RUN apt-get update && apt-get install -y \
       libgl1-mesa-glx \
       libxkbcommon-x11-0 \
       libdbus-1-3 \
       libxcb-icccm4 \
       libxcb-image0 \
       libxcb-keysyms1 \
       libxcb-randr0 \
       libxcb-render-util0 \
       libxcb-shape0 \
       libxcb-xinerama0 \
       libxcb-xkb1 \
       xvfb \
       && rm -rf /var/lib/apt/lists/*
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   # For headless environments
   ENV QT_QPA_PLATFORM=offscreen
   
   CMD ["python", "src/visualization/create_dashboard_qt.py", "--headless=True", "--host=0.0.0.0", "--port=5678"]
   ```

2. **Docker Compose**:
   ```yaml
   version: '3'
   services:
     dashboard:
       build: .
       ports:
         - "5678:5678"
       volumes:
         - ./data:/app/data
         - ./config:/app/config
         - ./logs:/app/logs
       environment:
         - METRICS_DB_PATH=/app/data/neural_metrics.db
         - V7_CONNECTION_PORT=5678
   ```

3. **Headless Operation**:
   - Configure the dashboard for headless operation in containers
   - Use REST API endpoints for data access: `--enable-rest-api=True --api-port=5679`

## Security Considerations

### Authentication and Authorization

1. **Dashboard Access**:
   - Implement user authentication for dashboard access
   - Configure role-based access control in `config/auth_config.json`:
     ```json
     {
       "auth_enabled": true,
       "auth_method": "basic",
       "users": [
         {"username": "admin", "password_hash": "...", "role": "admin"},
         {"username": "viewer", "password_hash": "...", "role": "viewer"}
       ],
       "roles": {
         "admin": ["read", "write", "configure"],
         "viewer": ["read"]
       }
     }
     ```

2. **Communication Security**:
   - Enable TLS for dashboard-LUMINA communication:
     ```bash
     python src/visualization/create_dashboard_qt.py --use-tls=True --cert-file="path/to/cert.pem" --key-file="path/to/key.pem"
     ```
   - Implement token-based authentication for API endpoints

### Data Protection

1. **Sensitive Data Handling**:
   - Configure data anonymization for sensitive neural patterns
   - Implement data encryption for stored metrics:
     ```json
     {
       "data_protection": {
         "encrypt_db": true,
         "encryption_key_file": "config/encryption_key.bin",
         "anonymize_patterns": true
       }
     }
     ```

2. **Audit Logging**:
   - Enable comprehensive audit logging for security monitoring:
     ```bash
     python src/visualization/create_dashboard_qt.py --audit-log=True --audit-log-path="logs/security_audit.log"
     ```
   - Log all configuration changes and access attempts

### Network Security

1. **Firewall Configuration**:
   - Restrict dashboard access to trusted IP ranges
   - Configure the dashboard to use non-standard ports to avoid automated scanning

2. **Intrusion Detection**:
   - Monitor for unusual access patterns or abnormal data requests
   - Implement rate limiting for API endpoints:
     ```json
     {
       "api_security": {
         "rate_limit_enabled": true,
         "max_requests_per_minute": 60,
         "block_threshold": 100
       }
     }
     ```

### Update and Patch Management

1. **Dependency Security**:
   - Regularly scan and update dependencies for security vulnerabilities:
     ```bash
     pip-audit
     pip install --upgrade vulnerable-package
     ```
   - Implement a process for security patch deployment

2. **Secure Configuration**:
   - Provide secure default configurations
   - Validate configuration files against a security schema before loading

## Further Development

### Adding New Panels

To create new dashboard panels:

1. Create a new Python file in `src/visualization/panels/`
2. Implement a class that inherits from `BasePanel`
3. Register it in `config/panel_plugins.json`

### Extending Visualizations

The dashboard supports two visualization backends:

1. **PyQtGraph** - Fast, interactive plots suitable for real-time data
2. **Matplotlib** - More customizable plots with better export options

Choose the appropriate backend based on your visualization needs.

### API Integration Options

The LUMINA V7 Dashboard provides several integration points for external systems:

#### REST API Endpoints

The dashboard can be configured to expose REST API endpoints:

```bash
python src/visualization/create_dashboard_qt.py --enable-api=True --api-port=5679
```

Available endpoints:

- **GET /api/v1/metrics/system** - Retrieve current system metrics
- **GET /api/v1/metrics/neural** - Get neural activity data
- **GET /api/v1/metrics/language** - Get language processing metrics
- **GET /api/v1/metrics/history/{metric_type}/{timeframe}** - Get historical data
- **POST /api/v1/config/update** - Update dashboard configuration

Example usage:

```python
import requests
import json

# Get current system metrics
response = requests.get("http://localhost:5679/api/v1/metrics/system")
metrics = response.json()

# Update configuration
new_config = {
    "refresh_rate": 2000,
    "panels": {
        "system": {"enabled": True}
    }
}
requests.post("http://localhost:5679/api/v1/config/update", 
              json=new_config,
              headers={"Authorization": "Bearer your-api-key"})
```

#### WebSocket Streaming

For real-time updates, the dashboard supports WebSocket connections:

```javascript
// Client-side WebSocket connection
const socket = new WebSocket('ws://localhost:5679/ws/metrics');

socket.onopen = function(e) {
  console.log("Connection established");
  socket.send(JSON.stringify({
    "subscribe": ["system", "neural", "language"]
  }));
};

socket.onmessage = function(event) {
  const metrics = JSON.parse(event.data);
  // Update UI with new metrics
  updateDashboard(metrics);
};
```

Server configuration:

```bash
python src/visualization/create_dashboard_qt.py --enable-websocket=True --ws-port=5680
```

#### Database Integration

External systems can read the metrics database directly:

```python
import sqlite3

conn = sqlite3.connect('data/neural_metrics.db')
cursor = conn.cursor()

# Query recent system metrics
cursor.execute("""
    SELECT timestamp, cpu_usage, memory_usage, gpu_usage 
    FROM system_metrics 
    ORDER BY timestamp DESC 
    LIMIT 100
""")

metrics = cursor.fetchall()
conn.close()
```

### Roadmap for Future Development

The LUMINA V7 Dashboard has an ambitious roadmap for future enhancements:

#### Version 7.1 - Enhanced Visualization

- **3D Neural Network Visualization**
  - Interactive 3D view of neural connections
  - Zoom, rotate and explore network topology
  - Path tracing for neural signal propagation

- **Advanced Dream Mode Analytics**
  - Dream pattern comparison visualizations
  - Creativity index monitoring
  - Pattern emergence prediction

- **Augmented Reality Integration**
  - HoloLens/AR visualization of neural networks
  - Spatial mapping of consciousness networks
  - Gesture-based interaction with neural structures

#### Version 7.2 - Collaborative Features

- **Multi-User Dashboard**
  - Collaborative viewing of neural activity
  - Role-based access to different dashboard features
  - Shared annotations and bookmarks

- **Remote Monitoring Applications**
  - Mobile companion app for monitoring
  - Email/SMS alerts for significant neural events
  - Remote configuration capabilities

- **Integration with Research Tools**
  - Export neural patterns to research formats
  - Integration with academic analysis tools
  - Standardized data exchange formats

#### Version 7.3 - AI-Enhanced Dashboard

- **Automated Insight Generation**
  - AI-powered analysis of neural patterns
  - Anomaly detection and alerting
  - Pattern recognition and classification

- **Predictive Analytics**
  - Forecast neural system behavior
  - Predict resource needs
  - Identify optimization opportunities

- **Natural Language Interface**
  - Query dashboard with natural language
  - Voice commands for hands-free operation
  - Conversational UI for system interaction

### Contributing to Development

We welcome contributions to the LUMINA V7 Dashboard project:

1. **Code Contributions**:
   - Fork the repository
   - Create a feature branch
   - Submit a pull request with detailed description

2. **Documentation**:
   - Help improve guides and tutorials
   - Provide examples of custom panel implementation
   - Create video tutorials for setup and configuration

3. **Testing**:
   - Test on different platforms and configurations
   - Provide performance benchmarks
   - Report bugs and suggest improvements

For detailed contribution guidelines, please see `CONTRIBUTING.md` in the repository. 