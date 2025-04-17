#!/usr/bin/env python
"""
Lumina GUI Launcher - A launcher script for the Lumina GUI

This launcher script provides a simple way to start the Lumina GUI system.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Main entry point for Lumina GUI"""
    parser = argparse.ArgumentParser(description="Lumina GUI System")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Try to import and run the Lumina GUI
    try:
        # Check if PyQt5 is installed
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError:
            print("Error: PyQt5 is not installed.")
            print("Please install it with: pip install PyQt5")
            sys.exit(1)
            
        from lumina_gui import LuminaGUI
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        window = LuminaGUI()
        window.show()
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please make sure all required packages are installed: pip install PyQt5")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting Lumina GUI: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 