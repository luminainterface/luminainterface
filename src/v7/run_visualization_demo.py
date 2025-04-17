#!/usr/bin/env python
"""
V7 Visualization Demo Launcher

This script launches a demonstration of the V7 visualization system
that shows the integration between V6 Portal of Contradiction,
V7 Node Consciousness, and the breath detection system.
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
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("v7.visualization_demo")

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="V7 Visualization Demo")
    
    # General options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--mock', action='store_true', help='Use mock mode for all components')
    
    # Feature toggles
    parser.add_argument('--no-breath', action='store_true', help='Disable breath visualization')
    parser.add_argument('--no-monday', action='store_true', help='Disable Monday consciousness visualization')
    parser.add_argument('--no-contradiction', action='store_true', help='Disable contradiction visualization')
    parser.add_argument('--no-node-consciousness', action='store_true', help='Disable node consciousness visualization')
    
    # Visualization options
    parser.add_argument('--dark-mode', action='store_true', help='Use dark mode (default)')
    parser.add_argument('--light-mode', action='store_true', help='Use light mode')
    
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

def create_dark_palette(QtGui):
    """Create a dark palette for the application"""
    palette = QtGui.QPalette()
    
    # Set color groups
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
    
    return palette

def initialize_breath_contradiction_bridge(mock_mode=False):
    """Initialize the breath contradiction bridge"""
    try:
        from src.v7.breath_contradiction_bridge import BreathContradictionBridge
        
        # Create bridge with appropriate configuration
        config = {
            "mock_mode": mock_mode,
            "auto_sync": True,
        }
        
        bridge = BreathContradictionBridge(config)
        
        # Start the bridge
        bridge.start()
        logger.info("✅ Breath Contradiction Bridge initialized and started")
        return bridge
    except Exception as e:
        logger.error(f"❌ Failed to initialize Breath Contradiction Bridge: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def initialize_v6v7_bridge(mock_mode=False):
    """Initialize the V6-V7 bridge"""
    try:
        # Try to use the bridge from the v7 module
        if hasattr(sys.modules.get("src.v7", None), "v6_v7_bridge"):
            from src.v7.v6_v7_bridge import create_v6v7_bridge
            
            # Create bridge with appropriate configuration
            config = {
                "mock_mode": mock_mode,
                "v6_enabled": True,
                "v7_enabled": True,
                "breath_integration_enabled": True,
                "contradiction_handling_enabled": True,
                "monday_integration_enabled": True,
                "node_consciousness_enabled": True
            }
            
            bridge = create_v6v7_bridge(config)
            logger.info("✅ V6V7 Bridge initialized from v7 module")
            return bridge
        
        # Alternative approach - use the connector
        from src.v7.v6_v7_connector import V6V7Connector
        
        # Create connector with appropriate configuration
        config = {
            "mock_mode": mock_mode,
            "v6_enabled": True,
            "v7_enabled": True,
            "contradiction_processor_enabled": True,
            "node_consciousness_enabled": True,
            "monday_integration_enabled": True,
            "auto_wiki_enabled": True
        }
        
        connector = V6V7Connector(config)
        connector.initialize()
        
        logger.info("✅ V6V7 Connector initialized as alternative")
        return connector
    
    except Exception as e:
        logger.error(f"❌ Failed to initialize V6V7 Bridge: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def main():
    """Main entry point for the visualization demo"""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    configure_logging(args.debug)
    
    # Log startup
    logger.info("Starting V7 Visualization Demo")
    logger.info(f"Arguments: {args}")
    
    # Create visualization configuration
    visualization_config = {
        "breath_visualization_enabled": not args.no_breath,
        "monday_visualization_enabled": not args.no_monday,
        "contradiction_visualization_enabled": not args.no_contradiction,
        "node_consciousness_visualization_enabled": not args.no_node_consciousness,
    }
    
    # Initialize bridges/connectors
    v6v7_connector = initialize_v6v7_bridge(args.mock)
    breath_bridge = initialize_breath_contradiction_bridge(args.mock)
    
    # Initialize Qt framework
    QtWidgets, QtCore, QtGui = load_qt_framework()
    if QtWidgets is None:
        logger.error("Failed to load Qt framework. Cannot display visualization.")
        sys.exit(1)
    
    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("V7 Visualization Demo")
    
    # Set dark mode
    if not args.light_mode:
        # Apply dark palette
        app.setPalette(create_dark_palette(QtGui))
        app.setStyle("Fusion")
    
    # Create and display the visualization widget
    try:
        from src.v7.ui.v7_visualization_widget import V7VisualizationWidget
        
        # Create the widget with the bridge
        viz_widget = V7VisualizationWidget(
            v6v7_connector=v6v7_connector, 
            config=visualization_config
        )
        
        # Set up window
        viz_widget.setWindowTitle("V7 Node Consciousness Visualization")
        viz_widget.resize(1024, 768)
        viz_widget.show()
        
        logger.info("✅ Visualization widget created and displayed")
        
        # Update after a short delay to give components time to initialize
        QtCore.QTimer.singleShot(1000, viz_widget.refresh)
        
        # Schedule periodic refreshes
        refresh_timer = QtCore.QTimer()
        refresh_timer.timeout.connect(viz_widget.refresh)
        refresh_timer.start(2000)  # Refresh every 2 seconds
        
        # Create a "test events" function to generate interesting visualizations
        def generate_test_events():
            if breath_bridge:
                logger.info("Generating test contradiction for visualization")
                breath_bridge.create_test_contradiction()
            
            # Try to suggest breath patterns if possible
            try:
                if breath_bridge and breath_bridge.breath_detector:
                    import random
                    patterns = ["relaxed", "focused", "creative", "stressed", "meditative"]
                    pattern = random.choice(patterns)
                    weight = random.uniform(0.6, 0.9)
                    logger.info(f"Suggesting breath pattern: {pattern} (weight: {weight:.2f})")
                    breath_bridge.breath_detector.suggest_pattern(pattern, weight)
            except Exception as e:
                logger.debug(f"Could not suggest breath pattern: {e}")
        
        # Create a button to trigger test events
        test_button = QtWidgets.QPushButton("Generate Test Event")
        test_button.clicked.connect(generate_test_events)
        
        # Add button to a small control panel
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.addWidget(test_button)
        
        control_panel.setWindowTitle("V7 Visualization Controls")
        control_panel.resize(200, 100)
        control_panel.show()
        
        # Schedule a test event after startup
        QtCore.QTimer.singleShot(3000, generate_test_events)
        
        # Run the application
        logger.info("Running Qt application event loop...")
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Failed to create visualization widget: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

# Run the demo if this script is executed directly
if __name__ == "__main__":
    main() 