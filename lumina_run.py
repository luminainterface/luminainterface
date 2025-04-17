#!/usr/bin/env python
"""
Lumina - A neural network-based interactive system with a text UI (Fixed Version)

This launcher script provides a simple way to start the Lumina system.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Main entry point for Lumina"""
    parser = argparse.ArgumentParser(description="Lumina Neural System")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Try to import and run the fixed Lumina UI
    try:
        from lumina_ui_fixed import LuminaApp
        app = LuminaApp()
        app.run()
    except ImportError as e:
        print(f"Error: {e}")
        print("Please make sure Textual is installed: pip install textual")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting Lumina: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 