#!/usr/bin/env python3
"""
V7 Visualization System Runner

This script launches the V7 Self-Learning Visualization System with appropriate
command-line options and configurations.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import required modules
try:
    from PySide6 import QtWidgets
except ImportError:
    print("Error: PySide6 not found. Please install PySide6.")
    sys.exit(1)

# Import V7 components
try:
    from .main_widget import V7MainWidget
    from .v7_socket_manager import V7SocketManager, MockKnowledgePlugin
except ImportError as e:
    # Fall back to direct import if relative import fails
    try:
        from main_widget import V7MainWidget
        from v7_socket_manager import V7SocketManager, MockKnowledgePlugin
    except ImportError as e:
        print(f"Error importing V7 components: {e}")
        sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("v7_visualization.log")
    ]
)
logger = logging.getLogger("V7Visualization")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="V7 Self-Learning Visualization System")
    
    parser.add_argument("--mock-plugins", action="store_true",
                        help="Use mock plugins for development and testing")
    
    parser.add_argument("--components", type=str, default="all",
                        help="Comma-separated list of components to load (default: all)")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    parser.add_argument("--integration-mode", action="store_true",
                        help="Run in integration mode with V5 components")
    
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration for visualizations")
    
    return parser.parse_args()

def setup_socket_manager(args=None):
    """Set up the socket manager with appropriate plugins"""
    socket_manager = V7SocketManager()
    
    # If no args provided or using mock plugins, use mock plugins
    use_mock = args.mock_plugins if args else True
    
    # If using mock plugins, register them
    if use_mock:
        logger.info("Using mock plugins for development")
        
        # Register a mock knowledge plugin
        mock_knowledge = MockKnowledgePlugin()
        socket_manager.register_knowledge_plugin(mock_knowledge)
        
        # Check if we should load specific domains
        if args and hasattr(args, 'components'):
            components = args.components.lower().split(",")
            if "all" not in components:
                logger.info(f"Loading specific components: {components}")
                # Here we would load only the requested components
    
    # Try to find and register real plugins
    else:
        logger.info("Looking for available plugins...")
        
        # Here we would search for and register actual plugins
        # This is a placeholder for the actual implementation
        try:
            plugin_count = 0
            # Load knowledge plugins
            # Load learning controllers
            # Load AutoWiki plugins
            
            if plugin_count == 0:
                logger.warning("No plugins found. Falling back to mock plugins.")
                mock_knowledge = MockKnowledgePlugin()
                socket_manager.register_knowledge_plugin(mock_knowledge)
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")
            logger.info("Falling back to mock plugins")
            mock_knowledge = MockKnowledgePlugin()
            socket_manager.register_knowledge_plugin(mock_knowledge)
    
    return socket_manager

def load_configuration(args):
    """Load configuration file if specified"""
    config = {}
    
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
    
    return config

def create_v7_main_window(mock_plugins=True, debug=False, no_gpu=False):
    """
    Create and configure the V7 main window.
    This function is used by the direct launcher.
    
    Args:
        mock_plugins: Whether to use mock plugins
        debug: Whether to enable debug logging
        no_gpu: Whether to disable GPU acceleration
        
    Returns:
        The configured V7MainWidget
    """
    # Set log level based on debug flag
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Process environment variables
    try:
        # Check for .env file
        env_file = Path(project_root) / '.env'
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=env_file)
                logger.info(f"Loaded environment from {env_file}")
            except ImportError:
                logger.warning("python-dotenv not installed, skipping .env file")
    except Exception as e:
        logger.error(f"Error processing environment: {e}")
    
    # If requested, disable GPU acceleration
    if no_gpu:
        logger.info("GPU acceleration disabled")
        os.environ["QT_OPENGL"] = "software"
    
    # Create args-like object for setup_socket_manager
    class Args:
        pass
    args = Args()
    args.mock_plugins = mock_plugins
    args.components = "all"
    
    # Set up the socket manager
    socket_manager = setup_socket_manager(args)
    
    # Create main widget
    main_widget = V7MainWidget(socket_manager)
    
    # Configure widget
    # Default size
    main_widget.resize(1600, 900)
    
    # Set window title
    main_widget.setWindowTitle("LUMINA V7.0.0.2")
    
    logger.info("V7 Main Window created successfully")
    
    return main_widget

def main():
    """Main function to run the visualization system"""
    # Parse command-line arguments
    args = parse_args()
    
    # Set log level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load configuration
    config = load_configuration(args)
    
    # Set up the socket manager
    socket_manager = setup_socket_manager(args)
    
    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("V7 Self-Learning Visualization")
    
    # If requested, disable GPU acceleration
    if args.no_gpu:
        logger.info("GPU acceleration disabled")
        os.environ["QT_OPENGL"] = "software"
    
    # Set stylesheet from config if available
    if "stylesheet" in config:
        try:
            with open(config["stylesheet"], 'r') as f:
                app.setStyleSheet(f.read())
            logger.info(f"Applied stylesheet from {config['stylesheet']}")
        except Exception as e:
            logger.error(f"Error applying stylesheet: {e}")
    
    # Create main widget
    main_widget = V7MainWidget(socket_manager)
    
    # Configure widget based on arguments and config
    if "window_size" in config:
        try:
            width = config["window_size"]["width"]
            height = config["window_size"]["height"]
            main_widget.resize(width, height)
            logger.info(f"Set window size to {width}x{height}")
        except Exception as e:
            logger.error(f"Error setting window size: {e}")
    else:
        # Default size
        main_widget.resize(1600, 900)
    
    # Set window title
    main_widget.setWindowTitle("LUMINA V7.0.0.2")
    
    # Show the main widget
    main_widget.show()
    
    logger.info("V7 Visualization System started")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 