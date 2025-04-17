#!/usr/bin/env python3
"""
Debug imports script to check import paths and module loading
"""

import os
import sys
from pathlib import Path

def main():
    """Main debug function"""
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("\nPython path:")
    for p in sys.path:
        print(f"  - {p}")
    
    # Add project root to path
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        print(f"\nAdded {project_root} to Python path")
    
    # Check if src directory exists
    src_dir = project_root / "src"
    print(f"\nChecking if {src_dir} exists: {src_dir.exists()}")
    
    # Check if src/v7 directory exists
    v7_dir = src_dir / "v7"
    print(f"Checking if {v7_dir} exists: {v7_dir.exists()}")
    
    # Check if src/v7/onsite_memory.py exists
    memory_file = v7_dir / "onsite_memory.py"
    print(f"Checking if {memory_file} exists: {memory_file.exists()}")
    
    # List files in src/v7
    if v7_dir.exists():
        print("\nFiles in src/v7 directory:")
        for file in v7_dir.iterdir():
            print(f"  - {file.name}")
    
    # Try to import
    print("\nAttempting to import OnsiteMemory...")
    try:
        from src.v7.onsite_memory import OnsiteMemory
        print("Import successful!")
        
        # Create test instance
        memory = OnsiteMemory(
            data_dir="debug_memory",
            memory_file="debug.json"
        )
        print(f"Created OnsiteMemory instance with file: {memory.memory_file}")
        
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Other error: {e}")

if __name__ == "__main__":
    main() 