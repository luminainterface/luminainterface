#!/usr/bin/env python3
"""
Run script for the Mistral PySide6 application with onsite memory integration.

This script launches the Mistral Chat application with the onsite memory system.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def main():
    """Main entry point for the application"""
    print("Starting Mistral Chat Application with Onsite Memory...")
    
    # Try to import required components
    try:
        # Import PySide6
        from PySide6.QtWidgets import QApplication
        
        # Import the main window class
        from src.v7.ui.mistral_pyside_app import MistralChatWindow
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle("Fusion")
        
        # Create main window
        window = MistralChatWindow()
        window.show()
        
        # Run application
        return app.exec()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure you have the required packages installed:")
        print("pip install pyside6 mistralai")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 