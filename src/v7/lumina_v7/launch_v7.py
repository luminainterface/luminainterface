#!/usr/bin/env python
"""
LUMINA V7.0.0.2 Launch Script

This script serves as the entry point for the LUMINA V7 system.
It checks for dependencies and launches the main application.
"""

import sys
import os
import subprocess
import logging
import time
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    print(f"Added {root_dir} to Python path")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lumina_v7.launch")

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    # Check critical dependencies
    try:
        import PySide6
        logger.info(f"Found PySide6 {PySide6.__version__}")
    except ImportError as e:
        logger.error(f"PySide6 import error: {e}")
        missing_deps.append("PySide6")
    
    try:
        import numpy
        logger.info(f"Found NumPy {numpy.__version__}")
    except ImportError as e:
        logger.error(f"NumPy import error: {e}")
        missing_deps.append("numpy")
    
    # Optional dependencies
    try:
        import pandas
        logger.info(f"Found pandas {pandas.__version__}")
    except ImportError:
        logger.warning("pandas not available - some features may be limited")
    
    try:
        import matplotlib
        logger.info(f"Found matplotlib {matplotlib.__version__}")
    except ImportError:
        logger.warning("matplotlib not available - visualization may be limited")
    
    return missing_deps

def install_single_package(package_name):
    """Install a single package using pip"""
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name, "--upgrade"
        ])
        logger.info(f"{package_name} installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def install_dependencies(specific_packages=None):
    """Install required dependencies"""
    try:
        # Get path to requirements.txt
        root_dir = Path(__file__).parent.parent.parent.parent
        requirements_file = root_dir / "requirements.txt"
        
        if specific_packages:
            # Just install specific packages
            for package in specific_packages:
                install_single_package(package)
            return True
        
        if not requirements_file.exists():
            logger.error(f"Requirements file not found at {requirements_file}")
            # Try to install just PySide6 as a fallback
            if specific_packages and "PySide6" in specific_packages:
                return install_single_package("PySide6>=6.6.0")
            return False
        
        logger.info(f"Installing dependencies from {requirements_file}")
        
        # Run pip install
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        
        logger.info("Dependencies installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def show_minimal_ui():
    """Show a minimal UI without PySide6"""
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
        
        root = tk.Tk()
        root.title("LUMINA V7.0.0.2 - Dependency Issue")
        root.geometry("500x300")
        
        frame = ttk.Frame(root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="LUMINA V7.0.0.2", font=("Arial", 16, "bold")).pack(pady=10)
        ttk.Label(frame, text="Missing Required Dependencies").pack(pady=5)
        ttk.Label(frame, text="PySide6 is required to run the LUMINA V7 application").pack(pady=5)
        
        def install_clicked():
            install_button.config(state=tk.DISABLED)
            status_label.config(text="Installing PySide6...")
            root.update()
            
            success = install_dependencies(["PySide6>=6.6.0"])
            
            if success:
                status_label.config(text="Installation successful! Please restart the application.")
                messagebox.showinfo("Success", "PySide6 installed successfully. Please restart the application.")
            else:
                status_label.config(text="Installation failed. Please install manually.")
                install_button.config(state=tk.NORMAL)
        
        install_button = ttk.Button(frame, text="Install PySide6", command=install_clicked)
        install_button.pack(pady=10)
        
        status_label = ttk.Label(frame, text="")
        status_label.pack(pady=5)
        
        ttk.Button(frame, text="Exit", command=root.destroy).pack(pady=10)
        
        root.mainloop()
        return True
    except Exception as e:
        logger.error(f"Failed to create minimal UI: {e}")
        print("\n" + "=" * 80)
        print("Missing required dependencies: PySide6")
        print("Please install dependencies with:")
        print(f"  {sys.executable} -m pip install PySide6")
        print("=" * 80 + "\n")
        return False

def launch_gui(minimal=False):
    """Launch the V7 GUI application"""
    if minimal:
        return show_minimal_ui()
        
    try:
        # Print current Python path for debugging
        logger.info(f"Python path: {sys.path}")
        
        # Try direct import first
        try:
            from src.v7.v7_launcher import main
            logger.info("Successfully imported v7_launcher from src.v7")
        except ImportError:
            # Try alternative relative import path
            logger.info("Trying alternative import path...")
            # Get v7 launcher path
            v7_launcher_path = Path(__file__).parent.parent / "v7_launcher.py"
            
            if not v7_launcher_path.exists():
                v7_launcher_path = Path(__file__).parent.parent.parent / "v7_launcher.py"
                
            if v7_launcher_path.exists():
                logger.info(f"Found v7_launcher.py at {v7_launcher_path}")
                
                # Use importlib to load the module
                import importlib.util
                spec = importlib.util.spec_from_file_location("v7_launcher", v7_launcher_path)
                v7_launcher = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(v7_launcher)
                main = v7_launcher.main
            else:
                raise ImportError(f"Could not find v7_launcher.py in any expected location")
        
        # Run the main function
        logger.info("Launching main function from v7_launcher")
        main()
        return True
    except ImportError as e:
        logger.error(f"Failed to import launcher: {e}")
        return False
    except Exception as e:
        logger.error(f"Error launching GUI: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point for the LUMINA V7 application"""
    try:
        # Check for dependencies
        logger.info(f"Starting LUMINA V7.0.0.2 with Python {sys.version.split()[0]}")
        
        # Check for --install flag
        if "--install" in sys.argv:
            install_dependencies()
            return
        
        # Check for --minimal flag
        minimal_mode = "--minimal" in sys.argv
        
        # Check dependencies
        missing_deps = check_dependencies()
        
        if "PySide6" in missing_deps and not minimal_mode:
            print("\n" + "=" * 80)
            print("Missing PySide6 dependency")
            
            # Try launching minimal UI for installation
            print("Launching minimal UI for dependency installation...")
            print("=" * 80 + "\n")
            
            return launch_gui(minimal=True)
        
        elif missing_deps and not minimal_mode:
            print("\n" + "=" * 80)
            print(f"Missing dependencies: {', '.join(missing_deps)}")
            print("You can install them with:")
            print(f"  {sys.executable} {__file__} --install")
            print("=" * 80 + "\n")
            
            # Ask user if they want to install dependencies
            choice = input("Would you like to install dependencies now? (y/n): ")
            if choice.lower() in ["y", "yes"]:
                install_dependencies(missing_deps)
                print("Please restart the application.")
            return
        
        # Launch GUI
        return launch_gui(minimal=minimal_mode)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main() 