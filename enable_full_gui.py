#!/usr/bin/env python3
"""
Enable Full GUI Experience

This script ensures all requirements are met for the full V5 PySide6 GUI experience,
then launches the application.
"""

import os
import sys
import subprocess
import logging
import importlib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enable_full_gui")

def check_pyside6():
    """Check if PySide6 is installed and working"""
    try:
        import PySide6
        from PySide6.QtWidgets import QApplication
        logger.info(f"PySide6 is installed (version: {PySide6.__version__})")
        return True
    except ImportError:
        logger.error("PySide6 is not installed")
        return False

def install_pyside6():
    """Install PySide6 if not already installed"""
    logger.info("Attempting to install PySide6...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PySide6>=6.2.0"])
        logger.info("PySide6 installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install PySide6: {e}")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    dependencies = {
        "PySide6": "PySide6",
        "networkx": "networkx",
        "numpy": "numpy",
        "matplotlib": "matplotlib"
    }
    
    missing = []
    for name, package in dependencies.items():
        try:
            importlib.import_module(package)
            logger.info(f"{name} is installed")
        except ImportError:
            logger.warning(f"{name} is not installed")
            missing.append(package)
    
    return missing

def install_dependencies(packages):
    """Install missing dependencies"""
    if not packages:
        return True
    
    logger.info(f"Installing missing dependencies: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Set up environment variables for GUI"""
    # Force PySide6 usage
    os.environ["V5_QT_FRAMEWORK"] = "PySide6"
    
    # Set QT_QPA_PLATFORM for possible headless environments
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        logger.info("No display detected, setting QT_QPA_PLATFORM=offscreen")
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    
    logger.info("Environment setup complete")

def main():
    """Main function to enable and launch the full GUI experience"""
    print("\n===== Enabling Full V5 PySide6 GUI Experience =====\n")
    
    # Check if PySide6 is installed
    if not check_pyside6():
        print("PySide6 is required but not installed.")
        choice = input("Would you like to install it now? (y/n): ")
        if choice.lower() == 'y':
            if not install_pyside6():
                print("Failed to install PySide6. Please install it manually.")
                return 1
        else:
            print("Cannot continue without PySide6.")
            return 1
    
    # Check for other dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"Some dependencies are missing: {', '.join(missing_deps)}")
        choice = input("Would you like to install them now? (y/n): ")
        if choice.lower() == 'y':
            if not install_dependencies(missing_deps):
                print("Failed to install dependencies. Please install them manually.")
                return 1
        else:
            print("Continuing without all dependencies. Some features may not work.")
    
    # Set up environment
    setup_environment()
    
    # Launch the direct run script
    print("\nLaunching V5 PySide6 GUI...\n")
    try:
        result = subprocess.call([sys.executable, "direct_run.py"])
        return result
    except Exception as e:
        logger.error(f"Failed to launch GUI: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 