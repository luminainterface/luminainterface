#!/usr/bin/env python3
"""
Main entry point for the V5 Fractal Echo Visualization System.
"""

import sys
import logging
import atexit
import os
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import Qt compatibility layer
try:
    from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui, Qt
    HAS_QT = True
except ImportError:
    HAS_QT = False
    print("WARNING: Qt libraries not found. GUI will not be available.")

# Import bridge system components
try:
    from version_bridge_manager import VersionBridgeManager
    HAS_BRIDGE_SYSTEM = True
except ImportError:
    HAS_BRIDGE_SYSTEM = False
    print("WARNING: Bridge system not found. Continuing without bridge integration.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="V5 Fractal Echo Visualization System")
    parser.add_argument("--mock", action="store_true", help="Run with mock data")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI (for testing)")
    parser.add_argument("--no-bridge", action="store_true", help="Run without bridge system")
    return parser.parse_args()

def setup_logging(debug=False):
    """Configure logging for the application."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / "v5_visualization.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger("V5-Main")

def main():
    """Main entry point for the application."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.debug)
    logger.info("Starting V5 Fractal Echo Visualization System")
    
    # Check if Qt is available
    if not HAS_QT and not args.no_gui:
        logger.error("Qt libraries not found and --no-gui not specified. Cannot continue.")
        return 1
    
    # Initialize Qt application if GUI is enabled
    app = None
    if not args.no_gui and HAS_QT:
        # Import UI components here to avoid errors if Qt is not available
        from src.v5.ui.main_widget import V5MainWidget
        from src.v5.frontend_socket_manager import FrontendSocketManager
        from src.v5.initialize_plugins import initialize_plugins, fix_component_mapping, establish_forced_connections
        from src.v5.db_manager import DatabaseManager
        
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("V5 Fractal Echo Visualization")
        app.setApplicationVersion("1.0.0")
        
        # Initialize database manager
        db_manager = DatabaseManager.get_instance()
        
        # Register cleanup handler
        atexit.register(cleanup_resources, db_manager)
        
        # Create socket manager
        socket_manager = FrontendSocketManager()
        
        # Initialize bridge system if available
        bridge_manager = None
        if HAS_BRIDGE_SYSTEM and not args.no_bridge:
            try:
                logger.info("Initializing Version Bridge Manager")
                
                # Try to load bridge configuration
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                          "config", "bridge_config.json")
                
                # Initialize Version Bridge Manager with mock mode
                bridge_config = {"mock_mode": args.mock}
                bridge_manager = VersionBridgeManager(bridge_config)
                
                # Start the bridge manager
                bridge_success = bridge_manager.start()
                if bridge_success:
                    logger.info("Version Bridge Manager started successfully")
                else:
                    logger.warning("Version Bridge Manager failed to start")
                
                # Connect socket manager to bridge components
                if hasattr(socket_manager, "connect_to_bridge_manager"):
                    socket_manager.connect_to_bridge_manager(bridge_manager)
                    logger.info("Socket manager connected to bridge manager")
                
                logger.info("Version Bridge Manager initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing bridge system: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Initialize plugins
        try:
            # Try automatic plugin discovery first
            mock_enabled = args.mock
            
            if mock_enabled:
                logger.info("Running with MOCK DATA enabled")
            
            discovered_plugins = socket_manager.discover_plugins()
            if not discovered_plugins:
                logger.info("Initializing plugins manually")
                plugins = initialize_plugins(socket_manager, mock_mode=mock_enabled)
                
                # Fix component mapping
                socket_manager = fix_component_mapping(socket_manager)
                
                # Establish forced connections
                socket_manager = establish_forced_connections(socket_manager, plugins)
        except Exception as e:
            logger.error(f"Error initializing plugins: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Create main widget
        main_widget = V5MainWidget(socket_manager)
        main_widget.show()
        
        # Start periodic plugin discovery if supported
        try:
            socket_manager.start_plugin_discovery()
        except Exception as e:
            logger.error(f"Error starting plugin discovery: {str(e)}")

        # Connect to language memory system through bridge if available
        if bridge_manager:
            try:
                # Try to initialize language memory connection through bridge
                language_bridge = bridge_manager.get_component("language_memory_v5_bridge")
                if language_bridge:
                    logger.info("Found Language Memory V5 Bridge component")
                    result = language_bridge.get_available_topics()
                    if result:
                        logger.info(f"Successfully connected to Language Memory System: {len(result)} topics available")
                
                # Register V5MainWidget as an event handler for bridge messages
                if hasattr(main_widget, "handle_bridge_message"):
                    for message_type in ["memory_update", "topic_update", "fractal_pattern"]:
                        bridge_manager.register_message_handler("v5_language", message_type, 
                                                            main_widget.handle_bridge_message)
                    logger.info("Registered V5MainWidget as bridge message handler")
            except Exception as e:
                logger.error(f"Error connecting to language memory system through bridge: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Execute application
        logger.info("Entering Qt application main loop")
        exit_code = app.exec_()
        
        # Clean up bridge manager if it exists
        if bridge_manager:
            try:
                bridge_manager.stop()
                logger.info("Bridge manager stopped")
            except Exception as e:
                logger.error(f"Error stopping bridge manager: {str(e)}")
        
        # Log application exit
        logger.info("Application shutdown complete")
        
        # Return exit code
        return exit_code
    
    else:
        # Headless operation (no GUI)
        logger.info("Running in headless mode (no GUI)")
        
        # Initialize bridge system if available
        bridge_manager = None
        if HAS_BRIDGE_SYSTEM and not args.no_bridge:
            try:
                logger.info("Initializing Version Bridge Manager")
                
                # Initialize Version Bridge Manager with mock mode
                bridge_config = {"mock_mode": args.mock}
                bridge_manager = VersionBridgeManager(bridge_config)
                
                # Start the bridge manager
                bridge_success = bridge_manager.start()
                if bridge_success:
                    logger.info("Version Bridge Manager started successfully")
                    
                    # Keep the application running until interrupted
                    try:
                        logger.info("Headless mode active. Press Ctrl+C to exit.")
                        import time
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        logger.info("Received keyboard interrupt. Shutting down...")
                    finally:
                        # Clean up bridge manager
                        bridge_manager.stop()
                        logger.info("Bridge manager stopped")
                else:
                    logger.error("Version Bridge Manager failed to start")
                    return 1
                
            except Exception as e:
                logger.error(f"Error initializing bridge system: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                return 1
        else:
            logger.error("Headless mode requires bridge system. Cannot continue.")
            return 1
        
        # Log application exit
        logger.info("Application shutdown complete")
        return 0

def cleanup_resources(db_manager=None):
    """Clean up resources before application exit."""
    logger = logging.getLogger("V5-Main")
    
    # Close database connection
    if db_manager:
        try:
            db_manager.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")

if __name__ == "__main__":
    sys.exit(main()) 