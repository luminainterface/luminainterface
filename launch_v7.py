#!/usr/bin/env python
"""
LUMINA V7.0.0.1 Direct Python Launcher

This script provides a direct entry point to LUMINA V7 from the project root.
"""

import sys
import os
import importlib.util
import importlib
from pathlib import Path
import traceback

def main():
    """Main launcher function"""
    print("=" * 50)
    print("  LUMINA V7.0.0.1 Direct Python Launcher")
    print("=" * 50)
    print()
    
    # Add current directory to path
    current_dir = Path('.').resolve()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        print(f"Added {current_dir} to Python path")
    
    # Try multiple approaches to launch
    approaches = [
        launch_direct_import,
        launch_via_module_finder,
        launch_via_cd
    ]
    
    for i, approach in enumerate(approaches):
        print(f"\nTrying launch approach #{i+1}...")
        try:
            success = approach()
            if success:
                print("Launch successful!")
                return 0
        except Exception as e:
            print(f"Launch approach #{i+1} failed: {e}")
            print(traceback.format_exc())
    
    print("\nAll launch approaches failed.")
    print("Please check your Python environment and file structure.")
    return 1

def launch_direct_import():
    """Try to import and run the launcher directly"""
    try:
        from src.v7.v7_launcher import main
        print("Successfully imported main function")
        main()
        return True
    except ImportError as e:
        print(f"Direct import failed: {e}")
        return False

def launch_via_module_finder():
    """Try to find and load the launcher module"""
    try:
        # Look for the v7_launcher.py file
        root_dir = Path('.').resolve()
        launcher_path = root_dir / "src" / "v7" / "v7_launcher.py"
        
        if not launcher_path.exists():
            print(f"Launcher file not found at {launcher_path}")
            return False
            
        print(f"Found launcher at {launcher_path}")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("v7_launcher", launcher_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run the main function
        print("Launching main function...")
        module.main()
        return True
    except Exception as e:
        print(f"Module finder approach failed: {e}")
        return False

def launch_via_cd():
    """Try changing directory and importing relatively"""
    try:
        # Save original directory and path
        original_dir = os.getcwd()
        original_path = sys.path.copy()
        
        # Change to src/v7 directory
        os.chdir(os.path.join("src", "v7"))
        
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        print(f"Changed directory to {os.getcwd()}")
        
        # Try importing the launcher
        if os.path.exists("v7_launcher.py"):
            print("Found v7_launcher.py, importing...")
            import v7_launcher
            v7_launcher.main()
            success = True
        else:
            print("v7_launcher.py not found in src/v7")
            success = False
            
        # Restore original directory and path
        os.chdir(original_dir)
        sys.path = original_path
        
        return success
    except Exception as e:
        print(f"CD approach failed: {e}")
        # Restore original directory
        try:
            os.chdir(original_dir)
        except:
            pass
        return False

if __name__ == "__main__":
    sys.exit(main()) 