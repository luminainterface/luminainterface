"""
LUMINA v7.5 - Integrated Neural Interface Module

This module provides a comprehensive PySide6-based frontend for the LUMINA system
that integrates all components from previous versions into a unified interface.
"""

import os
import sys
import argparse
from pathlib import Path

__version__ = '7.5.0'
__author__ = 'LUMINA Development Team'

# Set up environment to enable interlinking with other components
if os.environ.get('V7_5_INTERLINK', 'false').lower() == 'true':
    os.environ['ENABLE_NEURAL_LINGUISTIC'] = 'true'
    os.environ['ENABLE_CONSCIOUSNESS_INTEGRATION'] = 'true'
    os.environ['ENABLE_MEMORY_INTEGRATION'] = 'true'

# Import main classes for easy access from this module
from .lumina_frontend import (
    LuminaMainWindow,
    ChatboxWidget,
    SystemStatusWidget,
    main
)

# Make this module runnable with command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="LUMINA v7.5 - Integrated Neural Interface")
    parser.add_argument("--port", type=int, default=7500, help="Port for the interlink connection")
    parser.add_argument("--monitor-only", action="store_true", help="Run only the system monitor")
    parser.add_argument("--interlink", action="store_true", help="Enable interlinking with v7 components")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--refresh-rate", type=int, default=5, help="Refresh rate for the monitor in seconds")
    return parser.parse_args() 