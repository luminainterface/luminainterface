"""
LUMINA v7.5 Package
Contains the GUI and AutoWikiProcessor components

This is a Python-compatible package name for the v7.5 module.
Serves as an import redirect for the module with a period in its name.
"""

# Import key components from the actual v7.5 module
import sys
from pathlib import Path

# Add the parent directory to sys.path to enable relative imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import and re-export key components from v7.5
try:
    # Import with underscores instead of dots
    from src.v7_5.system_integration import get_system_integration
    # Cannot import from modules with dots in the name
    # We'll use our module copies in v7_5 instead
except ImportError:
    # Fallback to direct imports if the directory structure is different
    pass

__version__ = "7.5.0" 