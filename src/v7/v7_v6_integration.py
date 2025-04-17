"""
V6-V7 Integration Script

This script provides the integration layer between V6 Portal of Contradiction
and V7 Node Consciousness, focusing on backend connectivity.

It serves as a launching point for the combined system with all bridges in place.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("v6v7_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("V6V7Integration")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="V6-V7 Integration Layer")
    
    # General options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--mock', action='store_true', help='Use mock mode for all components')
    
    # Version options
    parser.add_argument('--v6-only', action='store_true', help='Only load V6 components')
    parser.add_argument('--v7-only', action='store_true', help='Only load V7 components')
    
    # Feature toggles
    parser.add_argument('--no-breath', action='store_true', help='Disable breath integration')
    parser.add_argument('--no-monday', action='store_true', help='Disable Monday consciousness')
    parser.add_argument('--no-contradiction', action='store_true', help='Disable contradiction handling')
    parser.add_argument('--no-node-consciousness', action='store_true', help='Disable node consciousness')
    
    # UI options
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no UI)')
    parser.add_argument('--ui-port', type=int, default=8080, help='Port for web UI')
    
    return parser.parse_args()

def configure_logging(debug_mode):
    """Configure logging based on debug mode"""
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)

def load_qt_framework():
    """Load the Qt framework based on what's available"""
    try:
        # Try to load from V5 compatibility layer
        from src.v5.ui.qt_compat import QtWidgets, QtCore, QtGui
        logger.info("Using Qt compatibility layer from V5")
        return QtWidgets, QtCore, QtGui
    except ImportError:
        # Fall back to PySide6
        try:
            from PySide6 import QtWidgets, QtCore, QtGui
            logger.info("Using PySide6 directly")
            return QtWidgets, QtCore, QtGui
        except ImportError:
            # Fall back to PyQt5
            try:
                from PyQt5 import QtWidgets, QtCore, QtGui
                logger.info("Using PyQt5 directly")
                return QtWidgets, QtCore, QtGui
            except ImportError:
                logger.error("No Qt framework found. Please install PySide6 or PyQt5.")
                return None, None, None

def main():
    """Main entry point for the integration layer"""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    configure_logging(args.debug)
    
    # Log startup
    logger.info("Starting V6-V7 Integration Layer")
    logger.info(f"Arguments: {args}")
    
    # Create integration configuration
    config = {
        "mock_mode": args.mock,
        "v6_enabled": not args.v7_only,
        "v7_enabled": not args.v6_only,
        "breath_integration_enabled": not args.no_breath,
        "contradiction_handling_enabled": not args.no_contradiction,
        "monday_integration_enabled": not args.no_monday,
        "node_consciousness_enabled": not args.no_node_consciousness,
        "headless": args.headless,
        "ui_port": args.ui_port
    }
    
    # Create bridge component
    try:
        from src.v7.v6_v7_bridge import create_v6v7_bridge
        bridge = create_v6v7_bridge(config)
        logger.info("✅ V6-V7 Bridge created successfully")
    except Exception as e:
        logger.error(f"Failed to create V6-V7 Bridge: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)
    
    # Check bridge status
    bridge_status = bridge.get_status()
    logger.info(f"Bridge status: {bridge_status}")
    
    # Start all bridge components
    bridge.start_all_components()
    logger.info("✅ All bridge components started")
    
    # Launch UI if not headless
    if not args.headless:
        try:
            # Get Qt framework
            QtWidgets, QtCore, QtGui = load_qt_framework()
            if QtWidgets is None:
                logger.warning("Running in headless mode due to missing Qt framework")
                args.headless = True
            else:
                # Create application
                app = QtWidgets.QApplication(sys.argv)
                app.setApplicationName("V6-V7 Integration")
                
                # Decide which UI to load
                if args.v7_only:
                    # Load V7 UI
                    try:
                        from src.v7.ui.main_widget import V7MainWidget
                        main_widget = V7MainWidget(
                            bridge.get_component("socket_manager", "v7"), 
                            v6v7_connector=bridge
                        )
                        main_widget.setWindowTitle("V7 Node Consciousness")
                        main_widget.resize(1280, 800)
                        main_widget.show()
                        logger.info("✅ V7 UI loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load V7 UI: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                elif args.v6_only:
                    # Load V6 UI
                    try:
                        from src.v6.ui.main_widget import V6MainWidget
                        main_widget = V6MainWidget(
                            bridge.get_component("socket_manager", "v6")
                        )
                        main_widget.setWindowTitle("V6 Portal of Contradiction")
                        main_widget.resize(1280, 800)
                        main_widget.show()
                        logger.info("✅ V6 UI loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load V6 UI: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                else:
                    # Load combined UI
                    try:
                        # Attempt to load V7 UI with V6 integration
                        from src.v7.ui.main_widget import V7MainWidget
                        main_widget = V7MainWidget(
                            bridge.get_component("socket_manager", "v7"), 
                            v6v7_connector=bridge
                        )
                        main_widget.setWindowTitle("V6-V7 Integration")
                        main_widget.resize(1280, 800)
                        main_widget.show()
                        logger.info("✅ Combined UI loaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to load combined UI: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        
                        # Fallback to V6 UI
                        try:
                            from src.v6.ui.main_widget import V6MainWidget
                            main_widget = V6MainWidget(
                                bridge.get_component("socket_manager", "v6")
                            )
                            main_widget.setWindowTitle("V6 Portal of Contradiction")
                            main_widget.resize(1280, 800)
                            main_widget.show()
                            logger.info("✅ Fallback to V6 UI successful")
                        except Exception as e2:
                            logger.error(f"Failed to load fallback V6 UI: {e2}")
                            args.headless = True
                
                # Emit system ready event after a short delay
                if not args.headless:
                    QtCore.QTimer.singleShot(1000, lambda: bridge.send_message(
                        "system_ready", 
                        {"timestamp": time.time(), "version": "v6-v7-integration"}
                    ))
                    
                    # Run application
                    logger.info("Running Qt application...")
                    sys.exit(app.exec())
        except Exception as e:
            logger.error(f"Error in UI initialization: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            args.headless = True
    
    # Headless mode - keep running until interrupted
    if args.headless:
        logger.info("Running in headless mode")
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        finally:
            # Shutdown bridge
            bridge.shutdown()
            logger.info("Bridge shutdown complete")

# Run the integration layer if this script is executed directly
if __name__ == "__main__":
    main() 