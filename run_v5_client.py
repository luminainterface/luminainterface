#!/usr/bin/env python3
"""
Run V5 Client

Simple launcher script for the V5 visualization system.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_v5_client")

def main():
    """Run the V5 client"""
    print("\n=== Launching V5 Fractal Echo Visualization ===\n")
    
    # Set environment variables
    os.environ["V5_QT_FRAMEWORK"] = "PySide6"
    
    try:
        # Launch direct_run.py
        logger.info("Launching direct_run.py script")
        return subprocess.call([sys.executable, "direct_run.py"])
    except Exception as e:
        logger.error(f"Error launching application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 