#!/usr/bin/env python3
"""
Mistral Chat App Launcher

This script launches the Mistral Chat with Onsite Memory application,
ensuring proper environment setup and directory structure.
"""

import os
import sys
import subprocess
from pathlib import Path

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data",
        "data/onsite_memory",
        "data/db"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def setup_environment():
    """Set up environment variables and paths"""
    # Add the current directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Add the src directory to the Python path if it exists
    src_dir = os.path.join(current_dir, "src")
    if os.path.exists(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Check for API key in environment
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("Warning: MISTRAL_API_KEY environment variable not set.")
        print("Application will run in mock mode.")

def main():
    """Main entry point for the application launcher"""
    print("Initializing Mistral Chat with Onsite Memory...")
    
    # Set up environment
    setup_environment()
    
    # Ensure directories exist
    ensure_directories()
    
    # Import and run the application
    try:
        # First try direct import
        print("Starting application...")
        import simple_mistral_gui
        return simple_mistral_gui.main()
    except ImportError:
        # If that fails, try running as subprocess
        print("Import failed, trying subprocess...")
        try:
            result = subprocess.run(["python", "simple_mistral_gui.py"])
            return result.returncode
        except Exception as e:
            print(f"Error running application: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main()) 