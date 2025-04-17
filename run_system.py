#!/usr/bin/env python3
"""
System Launcher
Launches both the troubleshooter and the PySide6 frontend
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def main():
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Start the troubleshooter in a separate process
    troubleshooter_process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=current_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a moment for the troubleshooter to initialize
    time.sleep(2)
    
    # Start the PySide6 frontend
    frontend_process = subprocess.Popen(
        [sys.executable, "src/node_manager_ui.py"],
        cwd=current_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for both processes to complete
        troubleshooter_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        # Clean up processes on Ctrl+C
        troubleshooter_process.terminate()
        frontend_process.terminate()
        troubleshooter_process.wait()
        frontend_process.wait()

if __name__ == "__main__":
    main() 