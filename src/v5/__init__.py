"""
V5 Fractal Echo Visualization System

This package contains the core components of the V5 Fractal Echo Visualization System,
which provides visualization for the Lumina Neural Network.

It can be launched as a module using:
    python -m src.v5.main

Or imported and launched programmatically:
    from src.v5 import launch_v5
    launch_v5()
"""

import sys
import logging
import importlib
from pathlib import Path

# Export key components
from src.v5.frontend_socket_manager import FrontendSocketManager
from src.v5.v5_plugin import V5Plugin

# Setup logging
logger = logging.getLogger("V5")

def launch_v5(mock_mode=False, debug=False):
    """
    Launch the V5 Visualization System
    
    Args:
        mock_mode (bool): Whether to use mock/simulated data
        debug (bool): Whether to enable debug logging
        
    Returns:
        bool: True if launched successfully
    """
    logger.info("Launching V5 Visualization System")
    
    try:
        # Import main module
        main_module = importlib.import_module("src.v5.main")
        
        # Add command line arguments for mock and debug if needed
        if mock_mode and "--mock" not in sys.argv:
            sys.argv.append("--mock")
        if debug and "--debug" not in sys.argv:
            sys.argv.append("--debug")
        
        # Launch main function
        if hasattr(main_module, "main"):
            # Start in new process
            import multiprocessing
            
            def run_main():
                try:
                    main_module.main()
                except Exception as e:
                    logger.error(f"Error in V5 main: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # Start process
            process = multiprocessing.Process(target=run_main, daemon=True)
            process.start()
            
            logger.info(f"V5 system launched successfully (PID: {process.pid})")
            return True
        else:
            logger.error("V5 main module does not have a main() function")
            return False
    except Exception as e:
        logger.error(f"Error launching V5 system: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False 