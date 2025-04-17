#!/usr/bin/env python3
"""
LUMINA V7 Main Execution Script

This script initializes and runs the LUMINA V7 system, including:
- Neural Network System
- User Interface
- Enhanced Language Integration
- System Managers

Usage:
    python run_v7.py [options]

Options:
    --no-gui        Run without GUI
    --debug         Enable debug logging
    --help          Show this help message
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/v7_runtime.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

# Check for required dependencies
try:
    # UI dependencies
    try:
        from PySide6.QtWidgets import QApplication
        logger.info("Using PySide6 for UI")
        USING_PYSIDE6 = True
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
            logger.info("Using PyQt5 for UI")
            USING_PYSIDE6 = False
        except ImportError:
            logger.error("Neither PySide6 nor PyQt5 is installed. UI will not be available.")
            USING_PYSIDE6 = None
    
    # Import system components
    from src.v7.system.system_manager import SystemManager
    from src.v7.neural_network.neural_network_manager import NeuralNetworkManager
    from src.v7.language.enhanced_language_integration import EnhancedLanguageIntegration
    
    # Import UI components if available
    if USING_PYSIDE6 is not None:
        from src.v7.ui.ui_manager import UIManager
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    IMPORTS_SUCCESSFUL = False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LUMINA V7 Neural Network System")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def show_splash_screen():
    """Display ASCII art splash screen"""
    splash = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   ██╗     ██╗   ██╗███╗   ███╗██╗███╗   ██╗ █████╗      ║
    ║   ██║     ██║   ██║████╗ ████║██║████╗  ██║██╔══██╗     ║
    ║   ██║     ██║   ██║██╔████╔██║██║██╔██╗ ██║███████║     ║
    ║   ██║     ██║   ██║██║╚██╔╝██║██║██║╚██╗██║██╔══██║     ║
    ║   ███████╗╚██████╔╝██║ ╚═╝ ██║██║██║ ╚████║██║  ██║     ║
    ║   ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝     ║
    ║                                                          ║
    ║   V7.0.0.2 - Advanced Neural Network System             ║
    ║   © 2023 LUMINA Labs                                     ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(splash)
    logger.info("LUMINA V7.0.0.2 starting...")

def initialize_system():
    """Initialize system components"""
    try:
        # Create system manager
        logger.info("Initializing system manager...")
        system_manager = SystemManager()
        
        # Initialize neural network
        logger.info("Initializing neural network manager...")
        nn_manager = NeuralNetworkManager(system_manager)
        system_manager.register_component("neural_network", nn_manager)
        
        # Initialize enhanced language integration
        logger.info("Initializing enhanced language integration...")
        language_integration = EnhancedLanguageIntegration(system_manager)
        system_manager.register_component("language", language_integration)
        
        logger.info("System initialization complete.")
        return system_manager
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return None

def run_headless(system_manager):
    """Run system in headless mode (no GUI)"""
    logger.info("Running in headless mode")
    
    try:
        # Start neural network
        system_manager.get_component("neural_network").start()
        
        # Main loop
        while True:
            system_manager.update()
            time.sleep(0.1)  # Prevent CPU overuse
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        system_manager.shutdown()

def run_with_gui(system_manager):
    """Run system with GUI"""
    logger.info("Initializing GUI")
    
    try:
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create UI manager
        ui_manager = UIManager(system_manager)
        ui_manager.show()
        
        # Set up auto-updating connections for language chat panel
        language_integration = system_manager.get_component("language")
        if language_integration and "language_chat" in ui_manager.panels:
            language_panel = ui_manager.panels["language_chat"]
            if hasattr(language_panel, "set_language_integration"):
                logger.info("Connecting language integration to chat panel")
                language_panel.set_language_integration(language_integration)
        
        # Start the application
        exit_code = app.exec() if USING_PYSIDE6 else app.exec_()
        
        # Clean shutdown
        system_manager.shutdown()
        return exit_code
    except Exception as e:
        logger.error(f"GUI error: {e}")
        system_manager.shutdown()
        return 1

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Display splash screen
    show_splash_screen()
    
    # Check if imports were successful
    if not IMPORTS_SUCCESSFUL:
        logger.error("Critical imports failed. Exiting.")
        return 1
    
    # Initialize system
    system_manager = initialize_system()
    if not system_manager:
        logger.error("Failed to initialize system. Exiting.")
        return 1
    
    # Run in headless mode or with GUI
    if args.no_gui or USING_PYSIDE6 is None:
        run_headless(system_manager)
    else:
        return run_with_gui(system_manager)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 