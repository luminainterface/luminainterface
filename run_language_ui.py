#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Language System - PySide6 UI Application

This script launches a standalone PySide6 application for interacting
with the Enhanced Language System components.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("language_ui")

# Check for PySide6
try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget
    from PySide6.QtCore import Qt
except ImportError:
    logger.error("PySide6 is not installed. Please install it using 'pip install pyside6'")
    sys.exit(1)

# Import the language integration
try:
    from src.language.pyside6_integration import (
        get_language_pyside6_integration, 
        create_language_ui_panel
    )
except ImportError as e:
    logger.error(f"Error importing language integration: {e}")
    logger.error("Make sure the language system is properly installed")
    sys.exit(1)

class LanguageSystemWindow(QMainWindow):
    """Main window for the Enhanced Language System UI"""
    
    def __init__(self, config=None, mock_mode=False):
        """Initialize the main window"""
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Enhanced Language System")
        self.resize(1000, 800)
        
        # Initialize the language integration
        self.integration = get_language_pyside6_integration(
            config=config,
            mock_mode=mock_mode
        )
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Create language UI panel
        self.language_panel = create_language_ui_panel(
            parent=central_widget,
            integration=self.integration
        )
        layout.addWidget(self.language_panel)
        
        # Start status timer
        self.integration.start_status_timer(3000)  # Update every 3 seconds
        
        logger.info("Enhanced Language System UI initialized")
    
    def closeEvent(self, event):
        """Handle close event"""
        # Stop status timer
        if self.integration:
            self.integration.stop_status_timer()
        
        # Accept the event to close the window
        event.accept()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Language System UI")
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode without actual language components"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for data storage"
    )
    
    parser.add_argument(
        "--llm-weight",
        type=float,
        default=0.5,
        help="Initial LLM weight (0.0-1.0)"
    )
    
    parser.add_argument(
        "--nn-weight",
        type=float,
        default=0.5,
        help="Initial neural network weight (0.0-1.0)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Validate weights
    if not 0.0 <= args.llm_weight <= 1.0:
        logger.error(f"Invalid LLM weight: {args.llm_weight}. Must be between 0.0 and 1.0.")
        return 1
    
    if not 0.0 <= args.nn_weight <= 1.0:
        logger.error(f"Invalid NN weight: {args.nn_weight}. Must be between 0.0 and 1.0.")
        return 1
    
    # Create configuration
    config = {
        "data_dir": args.data_dir,
        "llm_weight": args.llm_weight,
        "nn_weight": args.nn_weight
    }
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create main window
    window = LanguageSystemWindow(
        config=config,
        mock_mode=args.mock
    )
    window.show()
    
    # Run the application
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 