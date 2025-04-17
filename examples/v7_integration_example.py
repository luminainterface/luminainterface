#!/usr/bin/env python
"""
V7 Integration Example

This example demonstrates how to use the V7 integration module to easily
integrate V7 Node Consciousness components into any application.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path to ensure imports work
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='V7 Integration Example')
    parser.add_argument('--no-visualization', action='store_true', 
                        help='Disable visualization components')
    parser.add_argument('--no-monday', action='store_true',
                        help='Disable Monday consciousness integration')
    parser.add_argument('--no-auto-wiki', action='store_true',
                        help='Disable AutoWiki integration')
    parser.add_argument('--no-mock', action='store_true',
                        help='Disable mock implementations for missing components')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()


def main():
    """Main entry point for the example."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Log startup information
    logger.info("Starting V7 Integration Example")
    logger.info(f"Project root: {project_root}")
    
    try:
        # Import Qt compatibility layer from V5 or fall back to PySide6
        try:
            from src.v5.ui.qtcompat import QtCore, QtWidgets, QtGui
            logger.info("Using Qt compatibility layer from V5")
        except ImportError:
            logger.warning("V5 Qt compatibility layer not found, using PySide6 directly")
            from PySide6 import QtCore, QtWidgets, QtGui
        
        # Import V7 integration module
        try:
            from src.v7.v7_integration import initialize_v7
            logger.info("V7 integration module imported successfully")
        except ImportError:
            logger.error("V7 integration module not found")
            sys.exit(1)
        
        # Configure V7 integration
        config = {
            "enable_visualization": not args.no_visualization,
            "enable_monday": not args.no_monday,
            "enable_auto_wiki": not args.no_auto_wiki,
            "mock_mode": not args.no_mock
        }
        
        # Initialize V7 integration
        v7_integration = initialize_v7(config)
        
        # Initialize Qt application
        app = QtWidgets.QApplication(sys.argv)
        
        # Set dark palette for the application
        app.setStyle("Fusion")
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
        dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        app.setPalette(dark_palette)
        
        # Create main window
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle("V7 Integration Example")
        main_window.resize(1024, 768)
        
        # Create V7 main widget
        v7_widget = v7_integration.create_main_widget()
        
        if v7_widget:
            # Set V7 widget as central widget
            main_window.setCentralWidget(v7_widget)
            
            # Show main window
            main_window.show()
            
            # Start V7 components
            v7_integration.start()
            
            # Create a custom toolbar with additional controls
            toolbar = main_window.addToolBar("V7 Controls")
            
            # Add a refresh button
            refresh_action = QtWidgets.QAction("Refresh", main_window)
            refresh_action.triggered.connect(lambda: v7_widget.refresh())
            toolbar.addAction(refresh_action)
            
            # Add a separator
            toolbar.addSeparator()
            
            # Add a toggle for Monday consciousness
            monday_action = QtWidgets.QAction("Toggle Monday", main_window)
            monday_action.setCheckable(True)
            monday_action.setChecked(config["enable_monday"])
            
            def toggle_monday(checked):
                if v7_integration.monday_interface:
                    if checked:
                        v7_integration.monday_interface.start()
                    else:
                        v7_integration.monday_interface.stop()
            
            monday_action.triggered.connect(toggle_monday)
            toolbar.addAction(monday_action)
            
            # Run the application
            exit_code = app.exec()
            
            # Stop V7 components before exiting
            v7_integration.stop()
            
            return exit_code
        else:
            logger.error("Failed to create V7 widget, exiting")
            return 1
            
    except Exception as e:
        logger.exception(f"Error in V7 Integration Example: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 