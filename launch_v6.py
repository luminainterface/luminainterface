#!/usr/bin/env python
"""
V6 Portal of Contradiction Launcher

Simple launcher script to run the V6 Portal of Contradiction with the 
appropriate command line arguments.
"""

import os
import sys
import subprocess

def main():
    print("Starting V6 Portal of Contradiction...")
    
    cmd = [
        sys.executable,  # The current Python interpreter
        "v5_enhanced.py",
        "--fix-message-flow",
        "--pattern", "auto",
        "--debug",
        "--enable-breath",
        "--enable-glyphs",
        "--enable-mirror",
        "--mock"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"Error running application: {e}")
        input("Press Enter to exit...")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 