# LUMINA V7.5 System - Setup Instructions

This document provides step-by-step instructions for setting up and running the LUMINA V7.5 Neural Network System.

## Prerequisites

- **Python 3.8+** - Download from [python.org](https://www.python.org/downloads/)
- **Git** (optional) - For version control
- **Windows, Linux, or macOS**

## Installation Steps

1. **Install Dependencies**

   Run the included installation script to set up all required dependencies:

   ```bash
   # On Windows
   install_requirements.bat

   # On Linux/macOS
   # First make the script executable
   chmod +x install_requirements.sh
   # Then run it
   ./install_requirements.sh
   ```

   Alternatively, you can manually install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**

   Run the component test script to verify that all components are working properly:

   ```bash
   python component_test.py
   ```

   This will check for:
   - Required Python packages
   - Core system modules
   - Component execution

   Fix any issues reported by the test before proceeding.

3. **System Directory Structure**

   The installation script should have created the necessary directory structure:

   ```
   LUMINA/
   ├── data/               # Data storage
   │   ├── neural/         # Neural network data
   │   │   └── ui/         # V7 holographic interface
   │   ├── memory/         # Memory storage
   │   ├── seed/           # Neural seed growth data
   │   ├── consciousness/  # Node consciousness data
   │   ├── autowiki/       # AutoWiki data storage
   │   └── breath/         # Breath detection patterns
   ├── logs/               # Log files
   ├── src/                # Source code
   │   ├── v7/             # V7 components
   │   │   └── ui/         # V7 holographic interface
   │   ├── v7.5/           # V7.5 components
   │   ├── v7_5/           # Alternative path for V7.5 components
   │   └── dashboard/      # Dashboard components
   └── docs/               # Documentation
   ```

## Running the System

The main entry point for the system is the `run_v7_holographic.bat` script, which provides a menu for launching different components:

```bash
# On Windows
run_v7_holographic.bat

# On Linux/macOS
# First make the script executable
chmod +x run_v7_holographic.sh
# Then run it
./run_v7_holographic.sh
```

### Available Options

1. **Start Complete Holographic System** - Launches all components, including the Holographic Interface, Dashboard, and Neural Seed system.
2. **Start Dashboard Panels** - Launches only the dashboard visualization panels.
3. **Start Unified System** - Launches the Holographic Interface and Dashboard.
4. **Start Neural Seed Dashboard** - Launches the Neural Seed system with a special dashboard focus.
5. **Start V7.5 Chat Interface** - Launches the chat interface for text-based interaction.
6. **Start V7.5 System Monitor** - Launches the system monitor to display real-time metrics.
7. **Start Database Connector** - Starts the database synchronization service.
8. **View Documentation** - Opens documentation files.

### Troubleshooting

If you encounter issues:

1. Check the `logs/` directory for detailed error logs.
2. Ensure all required dependencies are installed.
3. Run the component test script to identify specific issues.
4. Check the console output for error messages.

### Common Issues

- **Module not found errors**: Ensure all requirements are installed properly.
- **PySide6 issues**: Make sure the PySide6 package is installed correctly.
- **Path errors**: The system uses both `v7.5` and `v7_5` directory structures; ensure paths are consistent.

## Advanced Configuration

The `.env` file contains configuration options that can be adjusted:

- **LLM Settings**: Configure language model parameters
- **Feature Flags**: Enable/disable specific features
- **Neural Network Settings**: Configure the neural network
- **System Configuration**: Adjust system parameters

## References

- See `MASTERreadme.md` for a complete overview of the system architecture
- See `v7readme.md` for details on the V7 Neural Consciousness system
- See `panelsreadme.md` for information about the dashboard panels 