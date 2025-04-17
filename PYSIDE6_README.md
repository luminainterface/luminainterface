# Language Memory GUI - PySide6 Version

This README provides installation and usage instructions for the PySide6 version of the Language Memory GUI, which integrates with the V5 Fractal Echo Visualization system.

> **Note**: This document is part of the Lumina Neural Network System documentation. For detailed information about the V5 Visualization System and its integration architecture, please refer to [V5readme.md](V5readme.md).

## Overview

The Language Memory GUI PySide6 version is a modern, Qt6-based replacement for the original tkinter implementation. It offers:

- Enhanced visual presentation
- Better integration with V5 visualization components
- Improved performance and responsiveness
- Modern UI elements and animations
- Support for light/dark themes

## Installation

### Prerequisites

1. Python 3.7 or higher
2. PySide6 (Qt6-based Python bindings)
3. Language Memory System core components

### Step 1: Install Required Packages

```bash
# Core dependencies
pip install PySide6

# Visualization dependencies (optional but recommended)
pip install numpy matplotlib networkx
```

### Step 2: Configure Environment (Optional)

For custom configurations, you can set the following environment variables:

```bash
# Windows PowerShell
$env:V5_VISUALIZATION_ENABLED = "1"
$env:V5_CONFIG_PATH = "config/custom_config.json"

# Linux/Mac
export V5_VISUALIZATION_ENABLED=1
export V5_CONFIG_PATH=config/custom_config.json
```

## Usage

### Running the Application

To start the PySide6 version of the Language Memory GUI:

```bash
python src/language_memory_gui_pyside.py
```

### Features

The application provides three main tabs:

1. **Memory Tab**
   - Store new memories with topics, emotions, and keywords
   - Search existing memories by topic, keyword, or text
   - View detailed memory content and metadata

2. **Synthesis Tab**
   - Synthesize knowledge around specific topics
   - Adjust search depth for broader or narrower synthesis
   - View synthesized knowledge with insights and relationships
   - (When V5 is available) Visualize memory patterns as fractals

3. **Statistics Tab**
   - View system-wide statistics
   - Monitor memory usage and patterns
   - Track synthesis operations and results

### V5 Visualization Integration

If the V5 Fractal Echo Visualization components are available, the PySide6 version will automatically:

1. Enable fractal visualization of language memory patterns
2. Show neural network representations of memory connections
3. Provide interactive exploration of memory relationships
4. Display node consciousness metrics in real-time

## Troubleshooting

### PySide6 Installation Issues

If you encounter issues installing PySide6:

```bash
# Try using the --force-reinstall flag
pip install --force-reinstall PySide6

# Alternatively, try installing from wheel
pip install --prefer-binary PySide6
```

### Missing Visualization Components

If V5 visualization components are not available:

1. Check that all required packages are installed
2. Verify that the V5 components are properly installed in `src/v5/`
3. Look for import error messages in the console output
4. Check the log file at `language_memory_pyside.log`

### Performance Issues

If you experience performance issues:

1. Adjust visualization settings in `src/v5/config.py`
2. Reduce fractal depth for complex visualizations
3. Close other resource-intensive applications
4. Use a system with dedicated graphics

## Migration from tkinter Version

If you're migrating from the tkinter version:

1. Both versions can coexist and use the same memory storage
2. Data stored by either version is accessible by the other
3. The original version can still be used via `python src/language_memory_gui.py`
4. Configuration settings may need to be updated for the PySide6 version

## Contributing

When contributing to the PySide6 version:

1. Follow the component architecture patterns
2. Maintain separation between UI and logic
3. Ensure graceful fallbacks for missing components
4. Add docstrings to explain component usage
5. Follow the PEP 8 style guide

## License

This software is part of the Lumina Neural Network project and is subject to the same licensing terms as the main project. 