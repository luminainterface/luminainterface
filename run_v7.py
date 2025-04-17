#!/usr/bin/env python
"""
LUMINA V7.0.0.2 Launcher
A simplified direct launcher that avoids import path issues
"""

import sys
import os
import logging
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("v7_launcher")

def main():
    """Entry point for LUMINA V7"""
    print("=" * 50)
    print("  LUMINA V7.0.0.2 Launcher")
    print("=" * 50)
    print()
    
    # Set up Python path
    current_dir = Path('.').resolve()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        print(f"Added {current_dir} to Python path")
    
    try:
        print("Initializing LUMINA V7...")
        
        # Import from proper location
        print("Importing UI components...")
        from PySide6.QtWidgets import QApplication
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # We'll try importing directly from ui.v7_run rather than from the lumina_v7 core
        # This avoids the module structure issues
        print("Loading UI...")
        from src.v7.ui.v7_run import create_v7_main_window
        
        # Create and show main window
        main_window = create_v7_main_window()
        main_window.show()
        
        print("LUMINA V7.0.0.2 launched successfully!")
        return app.exec()
        
    except Exception as e:
        logger.error(f"Error launching LUMINA V7: {e}")
        traceback.print_exc()
        
        # Provide helpful error info
        if "No module named" in str(e):
            print("\nModule import error. Check that your project structure is correct.")
            print("Current Python path:")
            for p in sys.path:
                print(f"  - {p}")
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 