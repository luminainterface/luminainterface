#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Dashboard Requirements Script
==================================

Verifies that all required packages for the LUMINA V7 Dashboard are installed.
Optionally can install missing packages if requested.
"""

import sys
import os
import importlib
import logging
import subprocess
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/requirements_check.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DashboardRequirementsCheck")

# Required packages
REQUIRED_PACKAGES = [
    {"name": "PyQt5", "import_name": "PyQt5", "install_name": "PyQt5"},
    {"name": "NumPy", "import_name": "numpy", "install_name": "numpy"},
    {"name": "Matplotlib", "import_name": "matplotlib", "install_name": "matplotlib"},
]

# PySide6 alternative for PyQt5
PYSIDE6_PACKAGES = [
    {"name": "PySide6", "import_name": "PySide6", "install_name": "PySide6"},
    {"name": "NumPy", "import_name": "numpy", "install_name": "numpy"},
    {"name": "Matplotlib", "import_name": "matplotlib", "install_name": "matplotlib"},
]

# Optional but recommended packages
OPTIONAL_PACKAGES = [
    {"name": "PyQtGraph", "import_name": "pyqtgraph", "install_name": "pyqtgraph"},
    {"name": "Pandas", "import_name": "pandas", "install_name": "pandas"},
]

def check_package(package_info):
    """
    Check if a package is installed
    
    Args:
        package_info: Dictionary with package information
        
    Returns:
        bool: True if installed, False otherwise
    """
    package_name = package_info["name"]
    import_name = package_info["import_name"]
    
    try:
        importlib.import_module(import_name)
        logger.info(f"{package_name} is installed")
        return True
    except ImportError:
        logger.warning(f"{package_name} is not installed")
        return False

def install_package(package_info):
    """
    Install a package using pip
    
    Args:
        package_info: Dictionary with package information
        
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    package_name = package_info["name"]
    install_name = package_info["install_name"]
    
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install {package_name}")
        return False

def check_requirements(install_missing=False, check_optional=True, gui_framework="PyQt5"):
    """
    Check if all required packages are installed
    
    Args:
        install_missing: Whether to install missing packages
        check_optional: Whether to check optional packages
        gui_framework: Which GUI framework to use ("PyQt5" or "PySide6")
        
    Returns:
        bool: True if all required packages are installed, False otherwise
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    missing_required = []
    missing_optional = []
    
    # Select appropriate package set based on GUI framework
    if gui_framework.lower() == "pyside6":
        required_packages = PYSIDE6_PACKAGES
        logger.info("Using PySide6 as GUI framework")
    else:
        required_packages = REQUIRED_PACKAGES
        logger.info("Using PyQt5 as GUI framework")
    
    # Check required packages
    logger.info("Checking required packages...")
    for package_info in required_packages:
        if not check_package(package_info):
            missing_required.append(package_info)
            
    # Check optional packages if requested
    if check_optional:
        logger.info("Checking optional packages...")
        for package_info in OPTIONAL_PACKAGES:
            if not check_package(package_info):
                missing_optional.append(package_info)
    
    # Install missing packages if requested
    if install_missing and (missing_required or missing_optional):
        logger.info("Installing missing packages...")
        
        # Install required packages
        for package_info in missing_required:
            install_package(package_info)
            
        # Install optional packages
        for package_info in missing_optional:
            install_package(package_info)
        
        # Recheck required packages
        still_missing = []
        for package_info in missing_required:
            if not check_package(package_info):
                still_missing.append(package_info["name"])
                
        if still_missing:
            logger.error(f"Failed to install required packages: {', '.join(still_missing)}")
            return False
    
    # Return True if all required packages are installed
    return len(missing_required) == 0 or install_missing

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Check dashboard requirements")
    
    parser.add_argument(
        "--install", 
        action="store_true",
        help="Install missing packages"
    )
    
    parser.add_argument(
        "--skip-optional", 
        action="store_true",
        help="Skip checking optional packages"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--gui-framework",
        choices=["PyQt5", "PySide6"],
        default="PyQt5",
        help="GUI framework to use (PyQt5 or PySide6)"
    )
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check requirements
    if check_requirements(
        install_missing=args.install, 
        check_optional=not args.skip_optional,
        gui_framework=args.gui_framework
    ):
        logger.info("All required packages are installed")
        sys.exit(0)
    else:
        logger.error("Some required packages are missing")
        sys.exit(1)

if __name__ == "__main__":
    main() 