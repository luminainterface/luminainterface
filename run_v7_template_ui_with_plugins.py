#!/usr/bin/env python3
"""
Run the V7 Template UI with All Plugins (Consciousness, Mistral, Neural Network)

This script launches the V7 Template UI with all integrated plugins for a
comprehensive neural network experience with consciousness monitoring.
"""

import os
import sys
import logging
import subprocess
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/v7_launcher.log", mode="a")
    ]
)
logger = logging.getLogger("V7IntegratedLauncher")

def ensure_directories():
    """Create necessary directories for the system"""
    directories = [
        "logs",
        "plugins",
        "data",
        "data/consciousness", 
        "data/mistral", 
        "data/neural",
        "icons"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"Created directory: {directory}")

def check_dependencies():
    """Check and install required dependencies"""
    requirements = {
        "PySide6": "PySide6",
        "numpy": "numpy",
    }
    
    for module, package in requirements.items():
        try:
            __import__(module)
            logger.info(f"Dependency check: {module} is installed")
        except ImportError:
            logger.warning(f"Dependency check: {module} is not installed, installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"Installed {package}")

def set_environment_variables():
    """Set environment variables for plugin configuration"""
    env_vars = {
        "TEMPLATE_PLUGINS_ENABLED": "true",
        "TEMPLATE_PLUGINS_DIRS": "plugins;src/v7/plugins;src/plugins",
        "TEMPLATE_AUTO_LOAD_PLUGINS": "consciousness_system_plugin.py;mistral_plugin.py;neural_network_plugin.py;memory_system_plugin.py",
        "TEMPLATE_TITLE": "LUMINA V7 Integrated Consciousness System",
        "PYTHONPATH": f"{os.getcwd()};{os.getcwd()}/src"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set environment variable: {key}={value}")

def run_template_app():
    """Run the template application with plugins"""
    logger.info("Launching LUMINA V7 Integrated Consciousness System")
    
    try:
        # Try to use v7_pyside6_template module
        from v7_pyside6_template import main as run_template
        logger.info("Running using v7_pyside6_template module")
        return run_template()
    except ImportError:
        logger.warning("Could not import v7_pyside6_template, trying alternative methods")
        
        # Try to use template launcher from src
        try:
            from src.v7.template.launcher import launch_template_ui
            logger.info("Running using launcher from src.v7.template")
            
            return launch_template_ui(
                plugins_dir="plugins",
                load_plugins=[
                    "consciousness_system_plugin.py", 
                    "mistral_plugin.py", 
                    "neural_network_plugin.py"
                ],
                theme="dark"
            )
        except ImportError:
            logger.warning("Could not import launcher from src, trying batch file")
            
            # Try to run batch file
            if os.name == "nt":
                batch_options = [
                    "run_v7_template_ui_with_plugins.bat",
                    "run_v7_template_ui.bat"
                ]
                
                for batch_file in batch_options:
                    if Path(batch_file).exists():
                        logger.info(f"Running batch file: {batch_file}")
                        return os.system(batch_file)
            
            # Direct command line execution
            logger.info("Running v7_pyside6_template.py directly")
            cmd = [
                sys.executable, 
                "v7_pyside6_template.py", 
                "--plugins-enabled", 
                "--auto-load-plugins"
            ]
            return subprocess.call(cmd)

def main():
    """Main entry point for the launcher"""
    parser = argparse.ArgumentParser(description="Launch V7 Template with integrated plugins")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency checking")
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Welcome message
    print("===================================")
    print("  LUMINA V7 Integrated System Launcher")
    print("===================================")
    
    # Setup environment
    ensure_directories()
    
    if not args.skip_deps:
        check_dependencies()
    
    set_environment_variables()
    
    # Run the template application
    try:
        exit_code = run_template_app()
        if exit_code != 0:
            logger.error(f"Application exited with code {exit_code}")
            print(f"\nLaunch failed with exit code {exit_code}")
            print("Check logs/v7_launcher.log for details")
            return exit_code
        
        logger.info("Application closed successfully")
        print("\nLUMINA V7 Integrated System closed successfully.")
        return 0
    except Exception as e:
        logger.exception(f"Error running template application: {e}")
        print(f"\nError running template application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 