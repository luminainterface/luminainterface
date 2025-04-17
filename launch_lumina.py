#!/usr/bin/env python3
"""
LUMINA v7.5 Launcher
Sets up the environment and launches the GUI
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up the Python environment and paths"""
    # Get the root directory
    root_dir = Path(__file__).parent.absolute()
    
    # Add src directory to Python path
    src_dir = root_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Add v7_5 directory to Python path
    v7_5_dir = src_dir / 'v7_5'
    if str(v7_5_dir) not in sys.path:
        sys.path.insert(0, str(v7_5_dir))
    
    # Create required directories
    dirs = ['assets', 'logs', 'assets/fonts', 'model_versions', 
            'config', 'database', 'spiderweb']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Set environment variables
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Print Python path for debugging
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")
    print()
    
    return root_dir

def main():
    """Main entry point"""
    root_dir = setup_environment()
    
    try:
        # Import and run the LUMINA client
        from v7_5.lumina_client import LuminaClient
        from PySide6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        window = LuminaClient()
        window.show()
        sys.exit(app.exec())
        
    except ImportError as e:
        print(f"[ERROR] Failed to import required modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to start LUMINA client: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 