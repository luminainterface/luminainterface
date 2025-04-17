#!/usr/bin/env python
"""
LUMINA V7.5 Holographic Frontend Launcher
Handles directory structure differences and ensures proper imports
"""

import os
import sys
import argparse
import importlib.util
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'holographic_frontend.log'), mode='a')
    ]
)
logger = logging.getLogger('HolographicLauncher')

def check_module_exists(module_path):
    """Check if a module exists at the given path."""
    try:
        spec = importlib.util.find_spec(module_path)
        return spec is not None
    except (ImportError, AttributeError):
        return False

def main():
    """Main function to handle arguments and launch the appropriate frontend."""
    parser = argparse.ArgumentParser(description='LUMINA V7.5 Holographic Frontend Launcher')
    parser.add_argument('--gui-framework', default='PySide6', help='GUI framework to use (PySide6 or PyQt6)')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode with simulated data')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--port', type=int, default=5678, help='Port for backend communication')
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Check GUI framework
    gui_framework = args.gui_framework
    try:
        importlib.import_module(gui_framework.lower())
        logger.info(f"Using {gui_framework} as GUI framework")
    except ImportError:
        logger.warning(f"{gui_framework} not installed. Falling back to PySide6.")
        try:
            importlib.import_module('pyside6')
            gui_framework = 'PySide6'
            logger.info("Using PySide6 as GUI framework")
        except ImportError:
            logger.error("Neither PySide6 nor the specified framework is installed. Please install with: pip install PySide6")
            print("Error: GUI framework not available. Please install PySide6 with: pip install PySide6")
            sys.exit(1)
    
    # Set environment variable for other components to use
    os.environ['GUI_FRAMEWORK'] = gui_framework
    
    # Check which module structure is being used (src.v7.5 vs src.v7_5)
    v7_path = None
    
    # First check for src.v7.5 style (with dot)
    if check_module_exists('src.v7.5'):
        v7_path = 'src.v7.5'
        logger.info("Found v7.5 module with dot notation (src.v7.5)")
    # Then check for src.v7_5 style (with underscore)
    elif check_module_exists('src.v7_5'):
        v7_path = 'src.v7_5'
        logger.info("Found v7.5 module with underscore notation (src.v7_5)")
    # Fallback to v7 if v7.5 is not available
    elif check_module_exists('src.v7'):
        v7_path = 'src.v7'
        logger.info("Falling back to v7 module (src.v7)")
    else:
        logger.error("Could not find any v7 or v7.5 modules. Please ensure the project structure is correct.")
        print("Error: Could not find required modules. Please check your installation.")
        sys.exit(1)
    
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs(os.path.join('data', 'holographic'), exist_ok=True)
    
    # Try to import and run the holographic frontend
    try:
        mock_arg = '--mock' if args.mock else ''
        port_arg = f'--port {args.port}'
        gui_arg = f'--gui-framework {gui_framework}'
        
        # Try v7.5 holographic_frontend first
        try:
            logger.info(f"Attempting to import {v7_path}.ui.holographic_frontend")
            holographic_module = importlib.import_module(f'{v7_path}.ui.holographic_frontend')
            if hasattr(holographic_module, 'run_holographic_frontend'):
                logger.info("Using v7.5 holographic frontend direct function call")
                holographic_module.run_holographic_frontend(
                    mock=args.mock,
                    port=args.port,
                    gui_framework=gui_framework
                )
                return
            elif hasattr(holographic_module, 'HolographicMainWindow'):
                logger.info("Using v7.5 holographic frontend class initialization")
                app_module = importlib.import_module(f'{gui_framework.lower()}.QtWidgets')
                app = app_module.QApplication([])
                window = holographic_module.HolographicMainWindow(
                    mock=args.mock,
                    port=args.port,
                    gui_framework=gui_framework
                )
                window.show()
                sys.exit(app.exec())
                return
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not import v7.5 UI directly: {e}")
        
        # If that fails, try v7 holographic_frontend
        try:
            logger.info("Attempting to import src.v7.ui.holographic_frontend")
            holographic_module = importlib.import_module('src.v7.ui.holographic_frontend')
            if hasattr(holographic_module, 'run_holographic_frontend'):
                logger.info("Using v7 holographic frontend direct function call")
                holographic_module.run_holographic_frontend(
                    mock=args.mock,
                    port=args.port,
                    gui_framework=gui_framework
                )
                return
            elif hasattr(holographic_module, 'HolographicMainWindow'):
                logger.info("Using v7 holographic frontend class initialization")
                app_module = importlib.import_module(f'{gui_framework.lower()}.QtWidgets')
                app = app_module.QApplication([])
                window = holographic_module.HolographicMainWindow(
                    mock=args.mock,
                    port=args.port,
                    gui_framework=gui_framework
                )
                window.show()
                sys.exit(app.exec())
                return
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not import v7 UI directly: {e}")
            
        # If all imports fail, use the command line fallback
        import subprocess
        cmd = [sys.executable, '-m', f'{v7_path}.ui.holographic_frontend']
        if args.mock:
            cmd.append('--mock')
        cmd.extend(['--port', str(args.port)])
        cmd.extend(['--gui-framework', gui_framework])
        
        logger.info(f"Using subprocess to launch holographic frontend: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    except Exception as e:
        logger.error(f"Error launching holographic frontend: {e}", exc_info=True)
        print(f"Error launching LUMINA V7.5 Holographic Frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 