#!/usr/bin/env python3
"""
V5 Lumina Launcher

This script launches the integrated V5 Lumina system, connecting:
1. The Language Memory System
2. V5 Fractal Echo Visualization
3. Neural Linguistic Processor
4. Frontend components
"""

import os
import sys
import logging
import argparse
import time
import subprocess
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("v5_lumina_launch.log")
    ]
)
logger = logging.getLogger("v5-lumina-launcher")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="V5 Lumina Launcher")
    
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--ui", choices=["v5", "text", "none"], default="v5", 
                        help="UI mode (v5, text, none)")
    parser.add_argument("--no-memory", action="store_true", 
                        help="Disable language memory system")
    parser.add_argument("--no-neural", action="store_true", 
                        help="Disable neural linguistic processor")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
    parser.add_argument("--no-start", action="store_true", 
                        help="Initialize but don't start the system")
    parser.add_argument("--status", action="store_true", 
                        help="Print system status and exit")
    
    return parser.parse_args()


def launch_integration_system(args):
    """Launch the V5 Lumina integration system"""
    from v5_lumina_integration import get_integration_system
    
    # Configure based on args
    config = {
        "mock_mode": args.mock,
        "test_mode": args.test,
        "disable_memory": args.no_memory,
        "disable_neural": args.no_neural,
        "ui_mode": args.ui
    }
    
    # Get the integration system
    integration = get_integration_system(config)
    
    # Initialize the system
    if not integration.initialize():
        logger.error("Failed to initialize V5 Lumina system")
        return False
    
    # Start the system unless --no-start is specified
    if not args.no_start:
        if not integration.start_system():
            logger.error("Failed to start V5 Lumina system")
            return False
    
    return integration


def launch_ui(args, integration=None):
    """Launch the appropriate UI based on args"""
    if args.ui == "none":
        return None
    
    ui_process = None
    
    if args.ui == "v5":
        # Launch V5 visualization UI
        try:
            cmd = [sys.executable, "src/ui/v5_run.py"]
            
            if args.mock:
                cmd.append("--mock")
            
            logger.info(f"Launching V5 UI: {' '.join(cmd)}")
            ui_process = subprocess.Popen(cmd)
            logger.info(f"V5 UI launched with PID: {ui_process.pid}")
        except Exception as e:
            logger.error(f"Failed to launch V5 UI: {e}")
    
    elif args.ui == "text":
        # Launch text UI
        try:
            cmd = [sys.executable, "lumina_run.py"]
            
            if args.mock:
                cmd.append("--mock")
            
            logger.info(f"Launching Text UI: {' '.join(cmd)}")
            ui_process = subprocess.Popen(cmd)
            logger.info(f"Text UI launched with PID: {ui_process.pid}")
        except Exception as e:
            logger.error(f"Failed to launch Text UI: {e}")
    
    return ui_process


def run_test_sequence(integration):
    """Run a test sequence to verify system functionality"""
    from v5_lumina_integration import run_test_sequence as run_tests
    
    logger.info("Running test sequence")
    run_tests(integration)
    logger.info("Test sequence complete")


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info("Launching V5 Lumina system")
    logger.info(f"Configuration: UI={args.ui}, Mock={args.mock}, Test={args.test}, "
                f"No-Memory={args.no_memory}, No-Neural={args.no_neural}")
    
    # Launch the integration system
    integration = launch_integration_system(args)
    if not integration:
        logger.error("Failed to launch integration system")
        sys.exit(1)
    
    # Print status if requested
    if args.status:
        status = integration.get_status()
        print(json.dumps(status, indent=2))
    
    # Run test sequence if test mode is enabled
    if args.test:
        run_test_sequence(integration)
        if not args.ui == "none":
            logger.info("Test complete, stopping system")
            integration.stop_system()
            sys.exit(0)
    
    # Launch the UI if not in test mode or if UI is explicitly requested with test
    ui_process = None
    if not args.test or args.ui != "none":
        ui_process = launch_ui(args, integration)
    
    # Main loop
    try:
        logger.info("V5 Lumina system running. Press Ctrl+C to exit.")
        
        # Keep running until interrupted
        while True:
            # Check if UI process is still running
            if ui_process and ui_process.poll() is not None:
                logger.warning(f"UI process exited with code: {ui_process.returncode}")
                
                # Attempt to restart UI if it crashed
                if ui_process.returncode != 0:
                    logger.info("Attempting to restart UI")
                    ui_process = launch_ui(args, integration)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping system...")
        
        # Stop UI process
        if ui_process:
            logger.info("Stopping UI process")
            ui_process.terminate()
            ui_process.wait(timeout=5)
        
        # Stop integration system
        if integration:
            logger.info("Stopping integration system")
            integration.stop_system()
        
        logger.info("V5 Lumina system shutdown complete")


if __name__ == "__main__":
    main() 