#!/usr/bin/env python3
"""
Standalone Mistral Chat App Launcher
This script directly launches the Mistral Chat application without going through the V7 launcher.
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mistral_launcher")

def main():
    """Main entry point for the standalone Mistral Chat app"""
    print("Starting Mistral Chat Application...")
    
    # Set up Python path
    current_dir = Path('.').resolve()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        print(f"Added {current_dir} to Python path")
    
    try:
        # Import PySide6
        from PySide6.QtWidgets import QApplication
        
        # Import the Mistral Chat window
        from src.v7.ui.mistral_pyside_app import MistralChatWindow
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # Create and show main window
        main_window = MistralChatWindow()
        main_window.show()
        
        print("Mistral Chat Application launched successfully!")
        return app.exec()
        
    except Exception as e:
        logger.error(f"Error launching Mistral Chat Application: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 