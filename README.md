# Lumina Neural Network Monitor

A comprehensive system monitoring tool with real-time visualization of system resources.

## Features

- Real-time CPU usage monitoring
- Memory usage tracking
- Disk I/O visualization
- Network traffic monitoring
- Process resource usage tracking
- Dark theme interface
- Tabbed interface for easy navigation

## Requirements

- Python 3.8 or higher
- PySide6
- psutil
- numpy
- pyqtgraph

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd lumina-monitor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python src/main.py
```

## Interface

The application provides a tabbed interface with the following sections:

1. **System Overview**
   - CPU usage graph
   - Memory usage graph

2. **Disk I/O**
   - Real-time disk read/write operations
   - Per-disk selection

3. **Network**
   - Network traffic visualization
   - Interface selection
   - Download/upload rates

4. **Processes**
   - Process resource usage
   - CPU and memory tracking
   - Process selection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 